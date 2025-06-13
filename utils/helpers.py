"""
Funciones auxiliares para APU
"""
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

def generate_document_id(content: str, filename: str) -> str:
    """
    Genera un ID único para un documento
    
    Args:
        content: Contenido del documento
        filename: Nombre del archivo
        
    Returns:
        ID único del documento
    """
    hash_content = f"{filename}_{content[:1000]}"
    return hashlib.md5(hash_content.encode()).hexdigest()

def extract_ieee_metadata(text: str) -> Dict[str, Any]:
    """
    Extrae metadata de un documento IEEE
    
    Args:
        text: Texto del documento
        
    Returns:
        Diccionario con metadata extraída
    """
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": [],
        "date": None,
    }
    
    # Extraer título (usualmente en las primeras líneas)
    title_match = re.search(r'^([^\n]{10,200})(?:\n|$)', text, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    # Extraer autores (patrones comunes en IEEE)
    authors_patterns = [
        r'(?:Authors?|by)\s*:?\s*([^\n]+)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
    ]
    
    for pattern in authors_patterns:
        match = re.search(pattern, text[:1000], re.MULTILINE | re.IGNORECASE)
        if match:
            authors_text = match.group(1)
            # Separar por comas o "and"
            authors = re.split(r',|and', authors_text)
            metadata["authors"] = [a.strip() for a in authors if a.strip()]
            break
    
    # Extraer abstract
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)\s*[-—:]?\s*([^\n]+(?:\n(?!\s*(?:Keywords|KEYWORDS|I\.|1\.|Introduction))[^\n]+)*)',
        text,
        re.IGNORECASE | re.MULTILINE
    )
    if abstract_match:
        metadata["abstract"] = abstract_match.group(1).strip()
    
    # Extraer keywords
    keywords_match = re.search(
        r'(?:Keywords|KEYWORDS|Index Terms)\s*[-—:]?\s*([^\n]+)',
        text,
        re.IGNORECASE
    )
    if keywords_match:
        keywords_text = keywords_match.group(1)
        keywords = re.split(r'[,;]', keywords_text)
        metadata["keywords"] = [k.strip() for k in keywords if k.strip()]
    
    # Buscar fecha
    date_patterns = [
        r'(\d{1,2})\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            try:
                # Intentar parsear la fecha
                date_str = match.group(0)
                # Aquí podrías usar dateutil.parser para más flexibilidad
                metadata["date"] = date_str
            except:
                pass
            break
    
    return metadata

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando caracteres problemáticos
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    # Eliminar caracteres de control
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text

def format_source_citation(doc_metadata: Dict[str, Any], page: Optional[int] = None) -> str:
    """
    Formatea una cita de fuente
    
    Args:
        doc_metadata: Metadata del documento
        page: Número de página (opcional)
        
    Returns:
        Cita formateada
    """
    parts = []
    
    if doc_metadata.get("authors"):
        authors = doc_metadata["authors"]
        if len(authors) > 2:
            parts.append(f"{authors[0]} et al.")
        else:
            parts.append(", ".join(authors))
    
    if doc_metadata.get("title"):
        parts.append(f'"{doc_metadata["title"]}"')
    
    if doc_metadata.get("date"):
        parts.append(f"({doc_metadata['date']})")
    
    if page:
        parts.append(f"p. {page}")
    
    return ", ".join(parts) if parts else "Fuente no especificada"

def save_json(data: Any, filepath: Path) -> None:
    """
    Guarda datos en formato JSON
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: Path) -> Any:
    """
    Carga datos desde un archivo JSON
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Datos cargados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Trunca texto a una longitud máxima
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def get_file_info(filepath: Path) -> Dict[str, Any]:
    """
    Obtiene información de un archivo
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Diccionario con información del archivo
    """
    stat = filepath.stat()
    return {
        "name": filepath.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": filepath.suffix.lower(),
    }