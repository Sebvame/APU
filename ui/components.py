"""
Componentes de UI reutilizables para APU
"""
import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

def render_chat_message(message: Dict[str, Any], show_timestamp: bool = True):
    """
    Renderiza un mensaje de chat con estilo NotebookLM
    
    Args:
        message: Diccionario con información del mensaje
        show_timestamp: Si mostrar timestamp
    """
    role = message.get("role", "user")
    content = message.get("content", "")
    timestamp = message.get("timestamp", datetime.now().isoformat())
    
    # Estilo según rol
    if role == "user":
        avatar = "👤"
        css_class = "user-message"
        name = "Tú"
    else:
        avatar = "🤖"
        css_class = "assistant-message"
        name = "APU"
    
    # Container del mensaje
    with st.container():
        col1, col2 = st.columns([1, 20])
        
        with col1:
            st.markdown(avatar)
        
        with col2:
            # Header del mensaje
            if show_timestamp:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M")
                st.markdown(f"**{name}** · {time_str}")
            else:
                st.markdown(f"**{name}**")
            
            # Contenido con formato
            formatted_content = format_message_content(content)
            st.markdown(f'<div class="{css_class}">{formatted_content}</div>', 
                       unsafe_allow_html=True)

def format_message_content(content: str) -> str:
    """
    Formatea el contenido del mensaje para mejor visualización
    
    Args:
        content: Contenido del mensaje
        
    Returns:
        Contenido formateado
    """
    # Resaltar código inline
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
    
    # Resaltar bloques de código
    content = re.sub(
        r'```(\w+)?\n(.*?)```',
        lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
        content,
        flags=re.DOTALL
    )
    
    # Convertir saltos de línea
    content = content.replace('\n', '<br>')
    
    # Resaltar elementos importantes
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    
    return content

def render_source_card(source: Dict[str, Any], expanded: bool = False):
    """
    Renderiza una tarjeta de fuente
    
    Args:
        source: Información de la fuente
        expanded: Si mostrar expandido
    """
    source_type = source.get("type", "document")
    
    if source_type == "document":
        icon = "📄"
        title = source.get("title", "Documento sin título")
        
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                    <div>
                        <strong>{title}</strong><br>
                        <small style="color: #888;">Documento local</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if expanded and "metadata" in source:
                with st.expander("Ver detalles"):
                    metadata = source["metadata"]
                    if metadata.get("authors"):
                        st.write(f"**Autores**: {', '.join(metadata['authors'])}")
                    if metadata.get("date"):
                        st.write(f"**Fecha**: {metadata['date']}")
                    if metadata.get("section"):
                        st.write(f"**Sección**: {metadata['section']}")
    
    elif source_type == "web":
        icon = "🌐"
        url = source.get("url", "#")
        domain = url.split('/')[2] if '/' in url else url
        
        st.markdown(f"""
        <div class="source-card">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                <div>
                    <a href="{url}" target="_blank" style="color: #4a9eff;">
                        {domain}
                    </a><br>
                    <small style="color: #888;">Búsqueda web</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_stats_card(stats: Dict[str, Any]):
    """
    Renderiza una tarjeta de estadísticas
    
    Args:
        stats: Diccionario con estadísticas
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📚 Documentos",
            value=stats.get("total_documents", 0),
            help="Total de documentos en la base de conocimiento"
        )
    
    with col2:
        st.metric(
            label="📝 Fragmentos",
            value=stats.get("total_chunks", 0),
            help="Total de fragmentos indexados"
        )
    
    with col3:
        size_mb = stats.get("index_size_mb", 0) + stats.get("metadata_size_mb", 0)
        st.metric(
            label="💾 Tamaño",
            value=f"{size_mb:.1f} MB",
            help="Tamaño total del índice"
        )

def render_upload_section():
    """Renderiza la sección de carga de archivos"""
    st.markdown("""
    <div style="
        border: 2px dashed #4a9eff;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: rgba(74, 158, 255, 0.1);
    ">
        <h3>📄 Arrastra tus PDFs aquí</h3>
        <p>o haz clic para seleccionar archivos</p>
        <small>Formatos soportados: PDF (preferiblemente IEEE)</small>
    </div>
    """, unsafe_allow_html=True)

def render_welcome_screen():
    """Renderiza la pantalla de bienvenida"""
    st.markdown("""
    <div style="
        max-width: 600px;
        margin: 50px auto;
        text-align: center;
        padding: 40px;
    ">
        <h1 style="color: #4a9eff; font-size: 48px;">
            🎓 Bienvenido a APU
        </h1>
        <p style="font-size: 20px; color: #888; margin: 20px 0;">
            Tu asistente inteligente para consultar apuntes académicos
        </p>
        
        <div style="
            background-color: #1a1a1a;
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
        ">
            <h3>🚀 Cómo empezar</h3>
            <ol style="text-align: left; color: #ccc;">
                <li>Carga tus documentos PDF en la barra lateral</li>
                <li>Espera a que se procesen e indexen</li>
                <li>¡Comienza a hacer preguntas!</li>
            </ol>
        </div>
        
        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        ">
            <div style="
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #333;
            ">
                <span style="font-size: 30px;">🔍</span>
                <h4>Búsqueda Inteligente</h4>
                <small>Encuentra información relevante en tus apuntes</small>
            </div>
            
            <div style="
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #333;
            ">
                <span style="font-size: 30px;">📊</span>
                <h4>Análisis Contextual</h4>
                <small>Comprende conceptos complejos</small>
            </div>
            
            <div style="
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #333;
            ">
                <span style="font-size: 30px;">🌐</span>
                <h4>Búsqueda Web</h4>
                <small>Complementa con información actualizada</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_thinking_animation():
    """Renderiza una animación de pensamiento"""
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    
    .thinking-dots span {
        animation: pulse 1.5s infinite;
        font-size: 30px;
    }
    
    .thinking-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .thinking-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    </style>
    
    <div class="thinking-dots" style="text-align: center; padding: 20px;">
        <span>💭</span>
        <span>💭</span>
        <span>💭</span>
    </div>
    """, unsafe_allow_html=True)

def render_error_message(error: str, suggestion: str = None):
    """
    Renderiza un mensaje de error con estilo
    
    Args:
        error: Mensaje de error
        suggestion: Sugerencia para resolver el error
    """
    st.markdown(f"""
    <div style="
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    ">
        <h4 style="color: #ff4444;">❌ Error</h4>
        <p>{error}</p>
        {f'<p style="color: #888;"><strong>Sugerencia:</strong> {suggestion}</p>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)

def render_session_summary(summary: Dict[str, Any]):
    """
    Renderiza un resumen de sesión
    
    Args:
        summary: Diccionario con resumen de sesión
    """
    st.markdown("### 📊 Resumen de Sesión")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏱️ Duración", f"{summary.get('duration_minutes', 0)} min")
        st.metric("💬 Mensajes", summary.get('total_messages', 0))
    
    with col2:
        st.metric("📄 Docs consultados", summary.get('documents_accessed', 0))
        tools = summary.get('tools_usage', {})
        st.metric("🔍 Búsquedas", tools.get('search_documents', 0))
    
    with col3:
        st.metric("🌐 Búsquedas web", tools.get('web_search', 0))
        topics = summary.get('topics', [])
        if topics:
            st.write("**Temas:**")
            for topic in topics[:3]:
                st.write(f"• {topic}")

def create_download_button(data: str, filename: str, label: str = "Descargar"):
    """
    Crea un botón de descarga estilizado
    
    Args:
        data: Datos a descargar
        filename: Nombre del archivo
        label: Etiqueta del botón
    """
    st.download_button(
        label=f"⬇️ {label}",
        data=data,
        file_name=filename,
        mime="text/plain",
        help=f"Descargar {filename}"
    )