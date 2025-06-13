"""
Funciones auxiliares mejoradas para extraer metadata de diferentes tipos de documentos
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

def extract_metadata_smart(text: str, filename: str) -> Dict[str, Any]:
    """
    Extrae metadata de diferentes tipos de documentos (IEEE, apuntes, etc.)
    
    Args:
        text: Texto del documento
        filename: Nombre del archivo
        
    Returns:
        Diccionario con metadata extraída
    """
    metadata = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": [],
        "date": None,
        "document_type": "unknown",
        "institution": "",
        "course": "",
        "email": ""
    }
    
    # Detectar tipo de documento
    doc_type = _detect_document_type(text, filename)
    metadata["document_type"] = doc_type
    
    if doc_type == "class_notes":
        metadata.update(_extract_class_notes_metadata(text, filename))
    elif doc_type == "ieee_paper":
        metadata.update(_extract_ieee_metadata(text))
    elif doc_type == "thesis":
        metadata.update(_extract_thesis_metadata(text))
    else:
        metadata.update(_extract_generic_metadata(text, filename))
    
    return metadata

def _detect_document_type(text: str, filename: str) -> str:
    """Detecta el tipo de documento"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Indicadores de apuntes de clase
    class_indicators = [
        "apuntes", "notes", "clase", "semana", "quiz", "respuestas",
        "instituto tecnologico", "escuela", "estudiante", "tec.cr"
    ]
    
    # Indicadores de paper IEEE
    ieee_indicators = [
        "ieee", "conference", "proceedings", "abstract", "introduction",
        "methodology", "results", "conclusion", "references"
    ]
    
    # Indicadores de tesis
    thesis_indicators = [
        "tesis", "thesis", "universidad", "university", "grado", "maestria",
        "doctorado", "phd", "bachelor", "master"
    ]
    
    # Contar indicadores
    class_count = sum(1 for indicator in class_indicators if indicator in text_lower or indicator in filename_lower)
    ieee_count = sum(1 for indicator in ieee_indicators if indicator in text_lower)
    thesis_count = sum(1 for indicator in thesis_indicators if indicator in text_lower)
    
    # Determinar tipo
    if class_count >= 2:
        return "class_notes"
    elif ieee_count >= 3:
        return "ieee_paper"
    elif thesis_count >= 2:
        return "thesis"
    else:
        return "generic"

def _extract_class_notes_metadata(text: str, filename: str) -> Dict[str, Any]:
    """Extrae metadata específica de apuntes de clase"""
    metadata = {}
    
    # Extraer título de apuntes
    title_patterns = [
        r'Apuntes del? (.+?)(?:\n|$)',
        r'Notes (.+?)(?:\n|$)',
        r'Clase (.+?)(?:\n|$)',
        r'Semana (.+?)(?:\n|$)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["title"] = f"Apuntes: {match.group(1).strip()}"
            break
    
    if not metadata.get("title"):
        # Extraer del nombre del archivo
        clean_filename = filename.replace("_", " ").replace(".pdf", "")
        metadata["title"] = f"Apuntes: {clean_filename}"
    
    # Extraer autor/estudiante
    author_patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)',  # Nombre completo
        r'Estudiante:\s*([^\n]+)',
        r'Autor:\s*([^\n]+)',
        r'Por:\s*([^\n]+)'
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, text)
        if match:
            author_name = match.group(1).strip()
            if len(author_name.split()) >= 2:  # Al menos nombre y apellido
                metadata["authors"] = [author_name]
                break
    
    # Extraer institución
    institution_patterns = [
        r'(Instituto Tecnol[oó]gico[^,\n]*)',
        r'(Universidad[^,\n]*)',
        r'(Escuela[^,\n]*)',
        r'(TEC[^,\n]*)'
    ]
    
    for pattern in institution_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata["institution"] = match.group(1).strip()
            break
    
    # Extraer email
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
    if email_match:
        metadata["email"] = email_match.group(1)
    
    # Extraer fecha del nombre del archivo
    date_patterns = [
        r'(\d{6})',  # DDMMYY
        r'(\d{8})',  # DDMMYYYY
        r'(\d{2}_\d{2}_\d{4})',  # DD_MM_YYYY
        r'(\d{4}_\d{2}_\d{2})'   # YYYY_MM_DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            metadata["date"] = match.group(1)
            break
    
    # Extraer curso/materia
    if "semana" in filename.lower():
        course_match = re.search(r'(\d+)_Semana', filename)
        if course_match:
            metadata["course"] = f"Semana {course_match.group(1)}"
    
    return metadata

def _extract_ieee_metadata(text: str) -> Dict[str, Any]:
    """Extrae metadata de papers IEEE (versión mejorada)"""
    metadata = {}
    
    # Extraer título (primeras líneas significativas)
    lines = text.split('\n')
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) > 20 and not line.isupper() and not line.startswith('http'):
            metadata["title"] = line
            break
    
    # Extraer autores
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
    
    return metadata

def _extract_thesis_metadata(text: str) -> Dict[str, Any]:
    """Extrae metadata de tesis"""
    metadata = {}
    
    # Implementar extracción específica para tesis
    # (similar a IEEE pero con patrones específicos de tesis)
    
    return metadata

def _extract_generic_metadata(text: str, filename: str) -> Dict[str, Any]:
    """Extrae metadata genérica"""
    metadata = {}
    
    # Título genérico del nombre del archivo
    clean_filename = filename.replace("_", " ").replace(".pdf", "")
    metadata["title"] = clean_filename
    
    # Buscar cualquier nombre que parezca autor
    name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    names = re.findall(name_pattern, text[:1000])
    if names:
        metadata["authors"] = names[:3]  # Máximo 3 nombres
    
    return metadata

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando caracteres problemáticos
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not text:
        return ""
    
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
    
    # Título
    if doc_metadata.get("title"):
        parts.append(doc_metadata["title"])
    
    # Autores
    if doc_metadata.get("authors"):
        authors = doc_metadata["authors"]
        if len(authors) > 2:
            parts.append(f"por {authors[0]} et al.")
        else:
            parts.append(f"por {', '.join(authors)}")
    
    # Institución (para apuntes)
    if doc_metadata.get("institution"):
        parts.append(doc_metadata["institution"])
    
    # Fecha
    if doc_metadata.get("date"):
        parts.append(f"({doc_metadata['date']})")
    
    # Página
    if page:
        parts.append(f"p. {page}")
    
    return " - ".join(parts) if parts else "Documento sin título"

def save_json(data: Any, filepath: Path) -> None:
    """
    Guarda datos en formato JSON
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error guardando JSON: {e}")

def load_json(filepath: Path) -> Any:
    """
    Carga datos desde un archivo JSON
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Datos cargados
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando JSON: {e}")
        return {}

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
    if not text or len(text) <= max_length:
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
        return {
            "name": filepath.name,
            "error": str(e)
        }

# Compatibilidad con el código existente
def extract_ieee_metadata(text: str) -> Dict[str, Any]:
    """Función de compatibilidad"""
    return extract_metadata_smart(text, "unknown.pdf")