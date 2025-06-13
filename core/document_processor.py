"""
Procesador de documentos PDF para APU
Con capacidades avanzadas de extracción, chunking y metadata
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
import pdfplumber
from datetime import datetime
import re

from config.settings import DOCUMENT_CONFIG, PROCESSED_DIR, DOCUMENT_METADATA_FIELDS
from utils.logger import logger
from utils.helpers import (
    generate_document_id,
    extract_metadata_smart,
    clean_text,
    save_json,
    get_file_info
)

class DocumentProcessor:
    """Procesador de documentos PDF con capacidades avanzadas"""
    
    def __init__(self):
        self.chunk_size = DOCUMENT_CONFIG["chunk_size"]
        self.chunk_overlap = DOCUMENT_CONFIG["chunk_overlap"]
        self.separators = DOCUMENT_CONFIG["separators"]
        self.min_chunk_size = DOCUMENT_CONFIG["min_chunk_size"]
        
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Procesa un archivo PDF completo
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Diccionario con documento procesado y metadata
        """
        logger.info(f"Procesando PDF: {pdf_path.name}")
        
        try:
            # Extraer texto usando múltiples métodos
            text, pages_content = self._extract_text_from_pdf(pdf_path)
            
            if not text or len(text.strip()) < 100:
                raise ValueError("No se pudo extraer suficiente texto del PDF")
            
            # Obtener información del archivo
            file_info = get_file_info(pdf_path)
            
            # Extraer metadata inteligente
            metadata = extract_metadata_smart(text, pdf_path.name)
            metadata.update({
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "processed_date": datetime.now().isoformat(),
                "total_pages": len(pages_content),
                "file_info": file_info
            })
            
            # Generar ID único - FIX: Asegurar que se genera correctamente
            doc_id = generate_document_id(text, pdf_path.name)
            if not doc_id:
                # Fallback si falla la generación
                doc_id = f"doc_{pdf_path.stem}_{int(datetime.now().timestamp())}"
            
            # Crear chunks con contexto
            chunks = self._create_smart_chunks(text, pages_content, metadata, doc_id)
            
            # Resultado final
            result = {
                "doc_id": doc_id,
                "metadata": metadata,
                "full_text": text,
                "chunks": chunks,
                "stats": {
                    "total_chunks": len(chunks),
                    "total_chars": len(text),
                    "total_words": len(text.split()),
                    "avg_chunk_size": sum(len(c["content"]) for c in chunks) / len(chunks) if chunks else 0
                }
            }
            
            # Guardar resultado procesado
            self._save_processed_document(result)
            
            logger.info(f"PDF procesado exitosamente: {len(chunks)} chunks creados")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando PDF {pdf_path.name}: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extrae texto de un PDF usando múltiples métodos
        
        Returns:
            Tupla (texto completo, lista de páginas con contenido)
        """
        pages_content = []
        full_text = ""
        
        # Método 1: pdfplumber (mejor para tablas y layout complejo)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    
                    # Extraer tablas si existen
                    tables = page.extract_tables()
                    table_text = ""
                    if tables:
                        for table in tables:
                            # Convertir tabla a texto
                            table_text += "\n" + self._table_to_text(table) + "\n"
                    
                    combined_text = page_text + table_text
                    
                    if combined_text.strip():
                        pages_content.append({
                            "page_num": i + 1,
                            "content": clean_text(combined_text),
                            "has_tables": bool(tables)
                        })
                        full_text += combined_text + "\n\n"
                        
        except Exception as e:
            logger.warning(f"Error con pdfplumber, intentando con PyPDF2: {e}")
            
            # Método 2: PyPDF2 como fallback
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text.strip():
                            pages_content.append({
                                "page_num": i + 1,
                                "content": clean_text(page_text),
                                "has_tables": False
                            })
                            full_text += page_text + "\n\n"
                            
            except Exception as e:
                logger.error(f"Error con PyPDF2: {e}")
                raise
        
        return clean_text(full_text), pages_content
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """Convierte una tabla a texto formateado"""
        if not table:
            return ""
        
        # Convertir tabla a texto con formato
        lines = []
        for row in table:
            # Filtrar celdas None
            row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
            lines.append(row_text)
        
        return "\n".join(lines)
    
    def _create_smart_chunks(self, text: str, pages_content: List[Dict], 
                           metadata: Dict, doc_id: str) -> List[Dict[str, Any]]:
        """
        Crea chunks inteligentes preservando contexto y estructura
        
        Args:
            text: Texto completo del documento
            pages_content: Contenido de páginas
            metadata: Metadata del documento
            doc_id: ID del documento
        """
        chunks = []
        
        # Detectar secciones del documento
        sections = self._detect_sections(text)
        
        for section in sections:
            section_text = section["content"]
            section_metadata = {
                **metadata,
                "section": section["title"],
                "section_level": section["level"]
            }
            
            # Crear chunks para esta sección
            if len(section_text) <= self.chunk_size:
                # Sección pequeña, un solo chunk
                chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                chunks.append({
                    "content": section_text,
                    "metadata": section_metadata,
                    "chunk_id": chunk_id,
                    "position": len(chunks)
                })
            else:
                # Dividir sección en chunks
                section_chunks = self._split_text_into_chunks(section_text)
                
                for i, chunk_text in enumerate(section_chunks):
                    chunk_metadata = {
                        **section_metadata,
                        "chunk_index_in_section": i,
                        "total_chunks_in_section": len(section_chunks)
                    }
                    
                    chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                    chunks.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata,
                        "chunk_id": chunk_id,
                        "position": len(chunks)
                    })
        
        # Si no se detectaron secciones, usar método simple
        if not chunks:
            simple_chunks = self._split_text_into_chunks(text)
            for i, chunk_text in enumerate(simple_chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunks.append({
                    "content": chunk_text,
                    "metadata": metadata,
                    "chunk_id": chunk_id,
                    "position": i
                })
        
        return chunks
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Detecta secciones en el documento (Introduction, Methods, etc.)
        """
        sections = []
        
        # Patrones para detectar secciones
        section_patterns = [
            # Numeradas: 1. Introduction, II. Methods, etc.
            r'^(?P<num>[IVX]+\.|[0-9]+\.?)\s+(?P<title>[A-Z][A-Za-z\s]+)$',
            # Solo título en mayúsculas
            r'^(?P<title>(?:ABSTRACT|INTRODUCTION|BACKGROUND|METHODS?|METHODOLOGY|RESULTS?|DISCUSSION|CONCLUSION|REFERENCES|ACKNOWLEDGMENTS?))$',
            # Título con formato
            r'^(?P<title>(?:Abstract|Introduction|Background|Methods?|Methodology|Results?|Discussion|Conclusion|References|Acknowledgments?))(?:\s|$)',
            # Secciones de apuntes
            r'^(?P<title>(?:RESPUESTAS|QUIZ|REPASO|EJERCICIOS?|PROBLEMAS?|TEORIA|PRACTICA))(?:\s|$)',
        ]
        
        lines = text.split('\n')
        current_section = {"title": "Document Start", "content": "", "level": 0, "start_line": 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Verificar si es un título de sección
            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Guardar sección anterior si tiene contenido
                    if current_section["content"].strip():
                        sections.append(current_section)
                    
                    # Iniciar nueva sección
                    title = match.group("title") if "title" in match.groupdict() else line
                    current_section = {
                        "title": title.strip(),
                        "content": "",
                        "level": 1 if "num" in match.groupdict() else 2,
                        "start_line": i
                    }
                    break
            else:
                # No es título, agregar al contenido actual
                current_section["content"] += line + "\n"
        
        # Agregar última sección
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Divide texto en chunks usando separadores inteligentes
        """
        chunks = []
        
        # Intentar dividir por párrafos primero
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Si el párrafo es muy largo, dividirlo
            if len(paragraph) > self.chunk_size:
                # Guardar chunk actual si existe
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Dividir párrafo largo
                sentences = self._split_into_sentences(paragraph)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                # Verificar si cabe en el chunk actual
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    # Guardar chunk actual y empezar nuevo
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        # Agregar último chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Aplicar overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones"""
        # Patrón simple para detectar fin de oración
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Aplica overlap entre chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # Primer chunk: agregar inicio del siguiente
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    overlap_text = next_chunk[:self.chunk_overlap]
                    overlapped_chunks.append(chunk + "\n[...] " + overlap_text)
                else:
                    overlapped_chunks.append(chunk)
            elif i == len(chunks) - 1:
                # Último chunk: agregar final del anterior
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                overlapped_chunks.append(overlap_text + " [...]\n" + chunk)
            else:
                # Chunks intermedios: overlap de ambos lados
                prev_chunk = chunks[i - 1]
                next_chunk = chunks[i + 1]
                prev_overlap = prev_chunk[-self.chunk_overlap:]
                next_overlap = next_chunk[:self.chunk_overlap]
                overlapped_chunks.append(
                    prev_overlap + " [...]\n" + chunk + "\n[...] " + next_overlap
                )
        
        return overlapped_chunks
    
    def _save_processed_document(self, document: Dict[str, Any]) -> None:
        """Guarda documento procesado"""
        try:
            output_path = PROCESSED_DIR / f"{document['doc_id']}.json"
            save_json(document, output_path)
            logger.info(f"Documento guardado: {output_path}")
        except Exception as e:
            logger.error(f"Error guardando documento procesado: {e}")
    
    def process_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Procesa todos los PDFs en un directorio
        
        Args:
            directory: Directorio con PDFs
            
        Returns:
            Lista de documentos procesados
        """
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar")
        
        processed_docs = []
        errors = []
        
        for pdf_file in pdf_files:
            try:
                doc = self.process_pdf(pdf_file)
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error procesando {pdf_file.name}: {e}")
                errors.append({"file": pdf_file.name, "error": str(e)})
        
        # Resumen
        logger.info(f"Procesamiento completado: {len(processed_docs)} exitosos, {len(errors)} errores")
        
        if errors:
            logger.error(f"Errores encontrados: {errors}")
        
        return processed_docs