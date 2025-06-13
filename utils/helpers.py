"""
Módulo de utilidades para la extracción y manejo de metadata de documentos académicos
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
    try:
        hash_content = f"{filename}_{content[:1000]}"
        return hashlib.md5(hash_content.encode('utf-8')).hexdigest()
    except Exception:
        # Fallback si hay problemas
        return f"doc_{filename}_{int(datetime.now().timestamp())}"

def extract_class_notes_metadata_enhanced(text: str, filename: str) -> Dict[str, Any]:
    """
    Extrae metadata específica de apuntes de clase con patrones mejorados
    
    Args:
        text: Texto del documento
        filename: Nombre del archivo
        
    Returns:
        Diccionario con metadata extraída
    """
    metadata = {
        "title": "",
        "authors": [],
        "document_type": "class_notes",
        "course_week": None,
        "date": None,
        "institution": "",
        "email": "",
        "topics_covered": [],
        "sections": [],
        "quiz_questions": 0,
        "file_name": filename,
        "academic_level": "university"
    }
    
    # 1. EXTRAER TÍTULO Y SEMANA
    title_patterns = [
        r'Apuntes\s+Semana\s+(\d+)\s*[-–]\s*(\d{2}/\d{2}/\d{4})',
        r'Apuntes\s+Semana\s+(\d+)',
        r'Semana\s+(\d+)\s*[-–]\s*(.+?)(?:\n|$)',
        r'^(.+?Semana.+?)(?:\n|$)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            if len(match.groups()) >= 2:
                week_num = match.group(1)
                date_or_title = match.group(2)
                metadata["course_week"] = int(week_num)
                metadata["title"] = f"Apuntes Semana {week_num}"
                
                # Si el segundo grupo es una fecha
                if re.match(r'\d{2}/\d{2}/\d{4}', date_or_title):
                    metadata["date"] = date_or_title
            else:
                metadata["title"] = match.group(0)
            break
    
    # 2. EXTRAER AUTOR
    author_patterns = [
        r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?',  # Nombres completos
        r'Estudiante:\s*([^\n]+)',
        r'Autor:\s*([^\n]+)',
        r'Por:\s*([^\n]+)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$'  # Línea completa con nombre
    ]
    
    # Buscar en las primeras 500 caracteres donde suele estar el autor
    text_header = text[:500]
    
    for pattern in author_patterns:
        matches = re.findall(pattern, text_header, re.MULTILINE)
        if matches:
            # Filtrar nombres que parezcan reales (al menos 2 palabras, no muy cortos)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                
                words = match.strip().split()
                if len(words) >= 2 and all(len(word) >= 2 for word in words):
                    # Verificar que no sea parte del contenido académico
                    if not any(term in match.lower() for term in ['respuesta', 'pregunta', 'quiz', 'repaso', 'encoder']):
                        metadata["authors"] = [match.strip()]
                        break
            if metadata["authors"]:
                break
    
    # 3. EXTRAER FECHA (patrones adicionales)
    if not metadata["date"]:
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\d{4}-\d{1,2}-\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text[:200])  # Buscar en el header
            if match:
                metadata["date"] = match.group(1)
                break
        
        # Extraer del nombre del archivo si no se encuentra
        if not metadata["date"]:
            filename_date_patterns = [
                r'(\d{6})',  # DDMMYY
                r'(\d{8})',  # DDMMYYYY
                r'(\d{2}_\d{2}_\d{4})',  # DD_MM_YYYY
                r'(\d{4}_\d{2}_\d{2})'   # YYYY_MM_DD
            ]
            
            for pattern in filename_date_patterns:
                match = re.search(pattern, filename)
                if match:
                    raw_date = match.group(1)
                    # Intentar formatear la fecha
                    try:
                        if len(raw_date) == 6:  # DDMMYY
                            formatted_date = f"{raw_date[:2]}/{raw_date[2:4]}/20{raw_date[4:]}"
                        elif len(raw_date) == 8:  # DDMMYYYY
                            formatted_date = f"{raw_date[:2]}/{raw_date[2:4]}/{raw_date[4:]}"
                        else:
                            formatted_date = raw_date.replace('_', '/')
                        metadata["date"] = formatted_date
                    except:
                        metadata["date"] = raw_date
                    break
    
    # 4. EXTRAER INSTITUCIÓN
    institution_patterns = [
        r'(Instituto Tecnológico[^,\n]*)',
        r'(Universidad[^,\n]*)',
        r'(TEC[^,\n]*)',
        r'(Tecnológico de Costa Rica)'
    ]
    
    for pattern in institution_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["institution"] = match.group(1).strip()
            break
    
    # 5. EXTRAER EMAIL
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    if email_match:
        metadata["email"] = email_match.group(1)
    
    # 6. EXTRAER TEMAS CUBIERTOS
    topics = []
    
    # Buscar en secciones principales
    topic_patterns = [
        r'(?:II\.|2\.)\s+REPASO\s*\n\s*A\.\s+([^\n]+)',
        r'(?:III\.|3\.)\s+([A-Z][A-Za-z\s]+)(?:\n|$)',
        r'(?:IV\.|4\.)\s+([A-Z][A-Za-z\s]+)(?:\n|$)',
        r'(?:V\.|5\.)\s+([A-Z][A-Za-z\s]+)(?:\n|$)',
    ]
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        topics.extend(matches)
    
    # Buscar términos técnicos específicos
    technical_terms = [
        'autoencoder', 'encoder', 'decoder', 'u-net', 'convolución', 'pooling',
        'inception', 'neural network', 'deep learning', 'machine learning',
        'redes neuronales', 'inteligencia artificial'
    ]
    
    for term in technical_terms:
        if term.lower() in text.lower():
            topics.append(term.title())
    
    # Eliminar duplicados y limpiar
    metadata["topics_covered"] = list(set([topic.strip() for topic in topics if topic.strip()]))
    
    # 7. EXTRAER SECCIONES
    sections = []
    section_patterns = [
        r'(?:I\.|1\.)\s+([A-Z][^.\n]+)',
        r'(?:II\.|2\.)\s+([A-Z][^.\n]+)',
        r'(?:III\.|3\.)\s+([A-Z][^.\n]+)',
        r'(?:IV\.|4\.)\s+([A-Z][^.\n]+)',
        r'(?:V\.|5\.)\s+([A-Z][^.\n]+)',
        r'(?:VI\.|6\.)\s+([A-Z][^.\n]+)',
        r'(?:VII\.|7\.)\s+([A-Z][^.\n]+)',
        r'(?:VIII\.|8\.)\s+([A-Z][^.\n]+)',
        r'(?:IX\.|9\.)\s+([A-Z][^.\n]+)',
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        sections.extend(matches)
    
    metadata["sections"] = [section.strip() for section in sections]
    
    # 8. CONTAR PREGUNTAS DE QUIZ
    quiz_patterns = [
        r'\d+\)\s+[^?\n]+\?',  # Preguntas numeradas
        r'Pregunta\s+\d+',
        r'Respuesta:'
    ]
    
    quiz_count = 0
    for pattern in quiz_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if pattern == r'Respuesta:':
            quiz_count = max(quiz_count, len(matches))
        else:
            quiz_count += len(matches)
    
    metadata["quiz_questions"] = quiz_count
    
    # 9. INFORMACIÓN ADICIONAL
    metadata["word_count"] = len(text.split())
    metadata["char_count"] = len(text)
    metadata["has_formulas"] = bool(re.search(r'[=+\-*/∑∫]', text))
    metadata["has_code"] = bool(re.search(r'[{}();]', text))
    metadata["language"] = "spanish"
    metadata["processed_date"] = datetime.now().isoformat()
    metadata["extraction_confidence"] = _calculate_confidence(metadata)
    
    return metadata

def _calculate_confidence(metadata: Dict[str, Any]) -> float:
    """Calcula un score de confianza de la extracción de metadata"""
    confidence = 0.0
    max_score = 10.0
    
    if metadata.get("authors"): confidence += 2.0
    if metadata.get("date"): confidence += 1.5
    if metadata.get("course_week"): confidence += 1.5
    if metadata.get("institution"): confidence += 1.0
    if metadata.get("email"): confidence += 1.0
    if metadata.get("topics_covered"): confidence += 1.5
    if metadata.get("sections"): confidence += 1.0
    if metadata.get("quiz_questions", 0) > 0: confidence += 0.5
    
    return min(1.0, confidence / max_score)

def _detect_document_type_enhanced(text: str, filename: str) -> str:
    """Detección mejorada del tipo de documento"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    class_indicators = [
        ("apuntes", 3), ("semana", 2), ("quiz", 2), ("respuestas", 2), 
        ("repaso", 2), ("estudiante", 1), ("tec.cr", 2), ("instituto tecnológico", 3)
    ]
    
    ieee_indicators = [
        ("ieee", 3), ("conference", 2), ("abstract", 2), ("methodology", 2),
        ("results", 1), ("conclusion", 1), ("references", 1)
    ]
    
    class_score = sum(weight for term, weight in class_indicators 
                     if term in text_lower or term in filename_lower)
    ieee_score = sum(weight for term, weight in ieee_indicators 
                    if term in text_lower)
    
    if class_score >= 4:
        return "class_notes"
    elif ieee_score >= 5:
        return "ieee_paper"
    else:
        return "generic"

def extract_metadata_smart(text: str, filename: str) -> Dict[str, Any]:
    """
    Función principal para extraer metadata inteligentemente
    """
    doc_type = _detect_document_type_enhanced(text, filename)
    
    if doc_type == "class_notes":
        return extract_class_notes_metadata_enhanced(text, filename)
    elif doc_type == "ieee_paper":
        return _extract_ieee_metadata(text)
    else:
        return _extract_generic_metadata(text, filename)

def _extract_ieee_metadata(text: str) -> Dict[str, Any]:
    """Extrae metadata de papers IEEE"""
    metadata = {"document_type": "ieee_paper"}
    
    # Título
    lines = text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 20 and not line.isupper():
            metadata["title"] = line
            break
    
    # Autores
    author_patterns = [
        r'(?:Authors?|by)\s*:?\s*([^\n]+)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, text[:1000], re.MULTILINE | re.IGNORECASE)
        if match:
            authors_text = match.group(1)
            authors = re.split(r',|and', authors_text)
            metadata["authors"] = [a.strip() for a in authors if a.strip()]
            break
    
    # Abstract
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)\s*[-—:]?\s*([^\n]+(?:\n(?!\s*(?:Keywords|KEYWORDS|I\.|1\.|Introduction))[^\n]+)*)',
        text, re.IGNORECASE | re.MULTILINE
    )
    if abstract_match:
        metadata["abstract"] = abstract_match.group(1).strip()
    
    return metadata

def _extract_generic_metadata(text: str, filename: str) -> Dict[str, Any]:
    """Extrae metadata genérica"""
    metadata = {"document_type": "generic"}
    
    clean_filename = filename.replace("_", " ").replace(".pdf", "")
    metadata["title"] = clean_filename
    
    name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    names = re.findall(name_pattern, text[:1000])
    if names:
        metadata["authors"] = names[:3]
    
    return metadata

def clean_text(text: str) -> str:
    """Limpia el texto eliminando caracteres problemáticos"""
    if not text:
        return ""
    
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def format_source_citation(doc_metadata: Dict[str, Any], page: Optional[int] = None) -> str:
    """Formatea una cita de fuente"""
    parts = []
    
    if doc_metadata.get("title"):
        parts.append(doc_metadata["title"])
    
    if doc_metadata.get("authors"):
        authors = doc_metadata["authors"]
        if len(authors) > 2:
            parts.append(f"por {authors[0]} et al.")
        else:
            parts.append(f"por {', '.join(authors)}")
    
    if doc_metadata.get("institution"):
        parts.append(doc_metadata["institution"])
    
    if doc_metadata.get("date"):
        parts.append(f"({doc_metadata['date']})")
    
    if page:
        parts.append(f"p. {page}")
    
    return " - ".join(parts) if parts else "Documento sin título"

def save_json(data: Any, filepath: Path) -> None:
    """Guarda datos en formato JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error guardando JSON: {e}")

def load_json(filepath: Path) -> Any:
    """Carga datos desde un archivo JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando JSON: {e}")
        return {}

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Trunca texto a una longitud máxima"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def get_file_info(filepath: Path) -> Dict[str, Any]:
    """Obtiene información de un archivo"""
    try:
        stat = filepath.stat()
        return {
            "name": filepath.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": filepath.suffix.lower(),
        }
    except Exception as e:
        return {"name": filepath.name, "error": str(e)}

# Compatibilidad con código existente
def extract_ieee_metadata(text: str) -> Dict[str, Any]:
    """Función de compatibilidad"""
    return extract_metadata_smart(text, "unknown.pdf")