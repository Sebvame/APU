"""
Configuración central para APU (Apuntes IA)
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
OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "temperature": 0.1,  # Baja temperatura para respuestas más precisas
    "max_tokens": 2048,
    "timeout": 60,
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

# Configuración de búsqueda
SEARCH_CONFIG = {
    "max_results": int(os.getenv("MAX_SEARCH_RESULTS", "5")),
    "similarity_threshold": 0.5,
    "rerank": True,
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

# Prompts del sistema
SYSTEM_PROMPTS = {
    "main_agent": """Eres APU (Apuntes IA), un asistente especializado en ayudar a estudiantes 
a consultar y entender sus apuntes académicos en formato IEEE.

Tu objetivo es proporcionar respuestas precisas y útiles basándote en los documentos disponibles.

Instrucciones:
1. Siempre basa tus respuestas en la información de los documentos
2. Cita las fuentes específicas cuando proporciones información
3. Si no encuentras información relevante, indícalo claramente
4. Mantén un tono académico pero accesible
5. Estructura tus respuestas de forma clara y organizada

Herramientas disponibles:
- search_documents: Para buscar información en los apuntes
- web_search: Para buscar información en internet (solo cuando el usuario lo solicite explícitamente)
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

# Metadata fields para documentos IEEE
IEEE_METADATA_FIELDS = [
    "title",
    "authors",
    "date",
    "conference",
    "abstract",
    "keywords",
    "doi",
    "pages",
    "file_name",
    "section",
    "subsection",
]

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO" if not APP_CONFIG["debug"] else "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "apu.log",
}