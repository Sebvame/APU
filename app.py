"""
APU - Apuntes IA
Aplicación principal con interfaz estilo NotebookLM
"""
import streamlit as st
from pathlib import Path
import time
from datetime import datetime
import os
import warnings

# Suprimir warnings de torch y huggingface
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Configuración de página debe ser lo primero
st.set_page_config(
    page_title="APU - Apuntes IA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports después de configuración
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingsManager
from core.vector_store import VectorStore
from tools.rag_tool import RAGTool
from tools.web_search_tool import WebSearchTool
from agents.main_agent import APUAgent
from agents.memory import MemoryManager
from ui.components import (
    render_chat_message,
    render_source_card,
    render_stats_card,
    render_upload_section
)
from config.settings import DOCUMENTS_DIR, APP_CONFIG
from utils.logger import logger

# Inicialización de estado de sesión
def init_session_state():
    """Inicializa el estado de la sesión"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.agent = None
        st.session_state.memory_manager = None
        st.session_state.processing = False
        st.session_state.documents_loaded = False
        st.session_state.vector_store = None
        st.session_state.embeddings_manager = None
        st.session_state.show_sources = True
        st.session_state.current_sources = []
        st.session_state.web_search_enabled = False  # Nueva opción

@st.cache_resource
def initialize_system():
    """Inicializa el sistema (solo una vez)"""
    try:
        with st.spinner("🚀 Inicializando APU..."):
            # Embeddings manager
            embeddings_manager = EmbeddingsManager()
            
            # Vector store
            vector_store = VectorStore(embeddings_manager.embedding_dim)
            
            # Intentar cargar índice existente
            if vector_store.load():
                st.success("✅ Índice cargado exitosamente")
            else:
                st.info("📂 No se encontró índice previo. Carga documentos para comenzar.")
            
            # Tools
            rag_tool = RAGTool(embeddings_manager, vector_store)
            web_search_tool = WebSearchTool()
            
            # Agent
            agent = APUAgent(rag_tool, web_search_tool)
            
            # Memory manager
            memory_manager = MemoryManager()
            
            return {
                "embeddings_manager": embeddings_manager,
                "vector_store": vector_store,
                "agent": agent,
                "memory_manager": memory_manager,
                "rag_tool": rag_tool
            }
    
    except Exception as e:
        st.error(f"Error inicializando sistema: {str(e)}")
        logger.error(f"Error en inicialización: {e}")
        return None

def process_documents(uploaded_files, system_components):
    """Procesa documentos subidos"""
    if not uploaded_files:
        return
    
    processor = DocumentProcessor()
    processed_docs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Guardar archivo temporalmente
            temp_path = DOCUMENTS_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Procesar
            status_text.text(f"Procesando: {uploaded_file.name}")
            doc = processor.process_pdf(temp_path)
            processed_docs.append(doc)
            
            # Actualizar progreso
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
            logger.error(f"Error procesando archivo: {e}")
    
    if processed_docs:
        # Generar embeddings
        status_text.text("Generando embeddings...")
        embeddings_dict = system_components["embeddings_manager"].encode_documents(processed_docs)
        
        # Agregar al vector store
        status_text.text("Indexando documentos...")
        system_components["vector_store"].add_documents(processed_docs, embeddings_dict)
        
        status_text.text("✅ Procesamiento completado")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return True
    
    return False

def detect_web_search_intent(message: str) -> bool:
    """Detecta si el usuario quiere buscar en internet"""
    web_keywords = [
        "buscar en internet", "busca en internet", "búsqueda web", "buscar web",
        "buscar online", "información actualizada", "noticias recientes", 
        "últimas noticias", "información actual", "buscar google",
        "search web", "web search", "internet search"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in web_keywords)

def main():
    """Función principal de la aplicación"""
    # Inicializar estado
    init_session_state()
    
    # CSS personalizado para estilo NotebookLM
    st.markdown("""
    <style>
    /* Suprimir warnings de torch */
    .stAlert[data-testid="stAlert"] {
        display: none;
    }
    
    /* Estilo general */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Mensajes */
    .user-message {
        background-color: #1a1a1a;
        border-left: 3px solid #4a9eff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    
    .assistant-message {
        background-color: #0f0f0f;
        border-left: 3px solid #00ff88;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
    }
    
    /* Fuentes */
    .source-card {
        background-color: #1a1a1a;
        border: 1px solid #333;
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .source-card:hover {
        background-color: #252525;
        border-color: #4a9eff;
    }
    
    /* Input */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0f0f0f;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #4a9eff;'>📚 APU - Apuntes IA</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Tu asistente inteligente para consultar apuntes académicos</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Panel de Control")
        
        # Inicializar sistema si no está hecho
        if not st.session_state.initialized:
            system_components = initialize_system()
            if system_components:
                st.session_state.agent = system_components["agent"]
                st.session_state.memory_manager = system_components["memory_manager"]
                st.session_state.vector_store = system_components["vector_store"]
                st.session_state.embeddings_manager = system_components["embeddings_manager"]
                st.session_state.rag_tool = system_components["rag_tool"]
                st.session_state.initialized = True
                
                # Crear sesión
                st.session_state.session_id = st.session_state.memory_manager.create_session()
        
        # Estadísticas
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            st.markdown("### 📊 Estadísticas")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documentos", stats["total_documents"])
            with col2:
                st.metric("Chunks", stats["total_chunks"])
        
        # Cargar documentos
        st.markdown("### 📄 Documentos")
        
        uploaded_files = st.file_uploader(
            "Cargar PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Sube documentos en formato IEEE"
        )
        
        if uploaded_files and st.button("🔄 Procesar Documentos", type="primary"):
            with st.spinner("Procesando documentos..."):
                if process_documents(uploaded_files, {
                    "embeddings_manager": st.session_state.embeddings_manager,
                    "vector_store": st.session_state.vector_store
                }):
                    st.success("✅ Documentos procesados exitosamente")
                    st.session_state.documents_loaded = True
                    st.rerun()
        
        # Opciones
        st.markdown("### ⚙️ Opciones")
        st.session_state.show_sources = st.checkbox("Mostrar fuentes", value=True)
        
        # Opción de búsqueda web
        st.session_state.web_search_enabled = st.checkbox(
            "🌐 Permitir búsqueda web", 
            value=False,
            help="Permite que APU busque en internet cuando sea necesario"
        )
        
        if st.session_state.web_search_enabled:
            st.info("💡 **Tip**: Para buscar en internet, usa frases como 'buscar en internet' o 'información actualizada'")
        
        # Exportar conversación
        if st.session_state.messages:
            st.markdown("### 💾 Exportar")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 JSON"):
                    path = st.session_state.memory_manager.export_session(
                        st.session_state.session_id, "json"
                    )
                    st.success(f"Exportado: {Path(path).name}")
            with col2:
                if st.button("📝 Markdown"):
                    path = st.session_state.memory_manager.export_session(
                        st.session_state.session_id, "markdown"
                    )
                    st.success(f"Exportado: {Path(path).name}")
        
        # Limpiar conversación
        if st.button("🗑️ Limpiar Conversación"):
            st.session_state.messages = []
            st.session_state.current_sources = []
            if st.session_state.agent:
                st.session_state.agent.clear_memory()
            st.rerun()
    
    # Área principal de chat
    chat_container = st.container()
    
    with chat_container:
        # Mostrar mensajes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Mostrar fuentes si están disponibles
                if message["role"] == "assistant" and "sources" in message:
                    if st.session_state.show_sources and message["sources"]:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in message["sources"]:
                                if source["type"] == "document":
                                    st.markdown(f"📄 **{source['title']}**")
                                elif source["type"] == "web":
                                    st.markdown(f"🌐 [{source.get('url', 'Búsqueda web')}]({source.get('url', '#')})")
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí...", disabled=not st.session_state.initialized):
        # Verificar si hay documentos cargados (solo para búsqueda local)
        if not st.session_state.documents_loaded and st.session_state.vector_store.get_stats()["total_documents"] == 0:
            # Verificar si es una búsqueda web
            if detect_web_search_intent(prompt) and st.session_state.web_search_enabled:
                # Permitir búsqueda web sin documentos
                pass
            else:
                st.warning("⚠️ No hay documentos cargados. Por favor, carga algunos PDFs primero o activa la búsqueda web.")
                return
        
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Guardar en memoria
        st.session_state.memory_manager.add_message(
            st.session_state.session_id,
            "user",
            prompt
        )
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Pensando..."):
                try:
                    # Detectar si necesita búsqueda web
                    use_web_search = (
                        st.session_state.web_search_enabled and 
                        detect_web_search_intent(prompt)
                    )
                    
                    if use_web_search:
                        # Mostrar que está buscando en internet
                        message_placeholder.markdown("🌐 Buscando en internet...")
                        
                        # Realizar búsqueda web
                        web_results = st.session_state.agent.search_web(prompt)
                        
                        # Actualizar respuesta con resultados web
                        full_response = f"## Búsqueda en Internet\n\n{web_results}"
                        
                        sources = [{"type": "web", "url": "Internet search results"}]
                        
                        # Metadatos para memoria
                        metadata = {"tool_used": "web_search"}
                        
                    else:
                        # Búsqueda normal en documentos
                        response = st.session_state.agent.chat(prompt, st.session_state.session_id)
                        full_response = response["answer"]
                        sources = response.get("sources", [])
                        metadata = {"tool_used": "search_documents"}
                    
                    # Mostrar respuesta
                    message_placeholder.markdown(full_response)
                    
                    # Guardar respuesta en memoria
                    st.session_state.memory_manager.add_message(
                        st.session_state.session_id,
                        "assistant",
                        full_response,
                        metadata=metadata
                    )
                    
                    # Guardar mensaje en estado
                    message_data = {
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    }
                    st.session_state.messages.append(message_data)
                    
                    # Mostrar fuentes
                    if st.session_state.show_sources and sources:
                        with sources_placeholder.expander("📚 Fuentes consultadas", expanded=True):
                            for source in sources:
                                if source["type"] == "document":
                                    st.markdown(f"📄 **{source['title']}**")
                                elif source["type"] == "web":
                                    st.markdown(f"🌐 **Búsqueda en Internet**")
                    
                    # Generar preguntas de seguimiento solo para búsquedas en documentos
                    if not use_web_search:
                        followup_questions = st.session_state.agent.generate_followup_questions(
                            full_response, prompt
                        )
                        
                        if followup_questions:
                            st.markdown("---")
                            st.markdown("**💡 Preguntas sugeridas:**")
                            for q in followup_questions:
                                if st.button(f"❓ {q}", key=f"followup_{hash(q)}"):
                                    # Simular nuevo prompt
                                    st.session_state.messages.append({"role": "user", "content": q})
                                    st.rerun()
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    logger.error(f"Error generando respuesta: {e}")
                    
                    # Guardar error en memoria
                    st.session_state.memory_manager.add_message(
                        st.session_state.session_id,
                        "assistant",
                        error_msg,
                        metadata={"error": str(e)}
                    )
    
    # Footer con información
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Versión**: {APP_CONFIG['version']}")
    with col2:
        if st.session_state.session_id:
            st.markdown(f"**Sesión**: `{st.session_state.session_id[:8]}...`")
    with col3:
        if st.session_state.agent:
            st.markdown(f"**Modelo**: {st.session_state.agent.llm.model}")
        else:
            st.markdown("**Modelo**: No inicializado")

if __name__ == "__main__":
    main()