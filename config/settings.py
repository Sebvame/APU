"""
Configuración central para APU
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Crear directorios si no existen
for dir_path in [DOCUMENTS_DIR, PROCESSED_DIR, FAISS_INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración de la aplicación
APP_CONFIG = {
    "name": os.getenv("APP_NAME", "APU - Apuntes IA"),
    "version": os.getenv("APP_VERSION", "1.0.0"),
    "debug": os.getenv("DEBUG_MODE", "False").lower() == "true",
}

# Configuración de Ollama
# En config/settings.py, agregar:
OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "mistral:7b"),  # Cambiado
    "temperature": 0.2,  # Aumentado ligeramente para creatividad
    "max_tokens": 8184,  # Aumentado para respuestas más completas
    "timeout": 240,      # Aumentado
}

# Configuración de embeddings
EMBEDDINGS_CONFIG = {
    "model_name": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
    "device": "cpu",  # Cambiar a "cuda" si tienes GPU
    "encode_kwargs": {"normalize_embeddings": True},
}

# Configuración de procesamiento de documentos
DOCUMENT_CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
    "separators": ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
    "min_chunk_size": 100,
}

# Configuración de búsqueda MEJORADA
SEARCH_CONFIG = {
    "max_results": int(os.getenv("MAX_SEARCH_RESULTS", "8")),  # Aumentado de 5 a 8
    "similarity_threshold": 0.3,  # Reducido de 0.5 a 0.3 para ser menos restrictivo
    "rerank": True,
    "use_mmr": True,  # Maximum Marginal Relevance para diversidad
    "mmr_lambda": 0.7,  # Balance entre relevancia y diversidad
    "contextual_boost": {
        "title_match": 1.3,  # Boost si coincide con título
        "author_match": 1.2,  # Boost si coincide con autor
        "recent_document": 1.1,  # Boost para documentos recientes
        "section_relevance": {
            "abstract": 1.4,
            "introduction": 1.2,
            "conclusion": 1.3,
            "results": 1.1
        }
    },
    "adaptive_threshold": True,  # Ajustar threshold según calidad de resultados
    "min_threshold": 0.1,  # Threshold mínimo absoluto
    "max_threshold": 0.7   # Threshold máximo
}

# Configuración de Tavily
TAVILY_CONFIG = {
    "api_key": os.getenv("TAVILY_API_KEY", ""),
    "search_depth": "advanced",
    "max_results": 5,
}

# Configuración de UI
UI_CONFIG = {
    "theme": os.getenv("STREAMLIT_THEME", "dark"),
    "max_chat_history": int(os.getenv("MAX_CHAT_HISTORY", "50")),
    "show_sources": True,
    "enable_feedback": True,
}

# Prompts del sistema MEJORADOS
SYSTEM_PROMPTS = {
    "main_agent": """Eres APU (Apuntes IA), un asistente especializado en ayudar a estudiantes 
a consultar y entender sus apuntes académicos y documentos educativos.

Tu objetivo es proporcionar respuestas precisas y útiles basándote en los documentos disponibles.

Instrucciones:
1. SIEMPRE basa tus respuestas en la información de los documentos
2. Cita las fuentes específicas cuando proporciones información
3. Si no encuentras información relevante, indícalo claramente
4. Mantén un tono académico pero accesible
5. Estructura tus respuestas de forma clara y organizada
6. Adapta tu respuesta al tipo de documento (apuntes de clase, papers académicos, etc.)
7. Incluye detalles relevantes como autores, fechas, instituciones cuando estén disponibles

Tipos de documentos que puedes encontrar:
- Apuntes de clase
- Papers académicos IEEE
- Documentos de tesis
- Material educativo general

Para apuntes de clase, incluye información sobre:
- Estudiante/autor
- Institución educativa
- Fecha de los apuntes
- Tema de la clase

Para papers académicos, incluye:
- Autores
- Abstract/resumen
- Metodología
- Resultados principales
""",
    
    "rag_prompt": """Utiliza la siguiente información para responder la pregunta del usuario.
Si la información no es suficiente, indícalo claramente.

Contexto: {context}

Pregunta: {question}

Respuesta:""",
    
    "web_search_prompt": """El usuario ha solicitado una búsqueda en internet.
Busca información relevante y actual sobre: {query}

Proporciona un resumen claro y cita las fuentes.""",
}

# Metadata fields para diferentes tipos de documentos
DOCUMENT_METADATA_FIELDS = {
    "class_notes": [
        "title", "authors", "date", "institution", "course", "email", 
        "document_type", "file_name", "processed_date"
    ],
    "ieee_paper": [
        "title", "authors", "date", "conference", "abstract", "keywords", 
        "doi", "pages", "document_type", "file_name"
    ],
    "thesis": [
        "title", "authors", "date", "university", "degree", "department",
        "advisor", "abstract", "keywords", "document_type"
    ],
    "generic": [
        "title", "authors", "date", "document_type", "file_name"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO" if not APP_CONFIG["debug"] else "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "apu.log",
}