# 📚 APU - Apuntes IA

APU (Apuntes IA) es un asistente conversacional inteligente diseñado para ayudar a estudiantes a consultar y comprender sus apuntes académicos en formato IEEE. Utiliza técnicas de Retrieval-Augmented Generation (RAG) para combinar búsqueda semántica con generación de lenguaje natural.

![APU Logo](https://img.shields.io/badge/APU-Apuntes%20IA-blue?style=for-the-badge&logo=book)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## 🌟 Características

- **🔍 Búsqueda Inteligente**: Encuentra información relevante en tus documentos usando búsqueda semántica
- **💬 Chat Natural**: Interfaz conversacional estilo NotebookLM
- **📊 Análisis Contextual**: Comprende y explica conceptos complejos
- **🌐 Búsqueda Web**: Complementa con información actualizada de internet (cuando lo solicites)
- **📝 Citas y Referencias**: Siempre indica las fuentes de información
- **💾 Memoria de Sesión**: Mantiene el contexto de la conversación
- **📤 Exportación**: Descarga tus conversaciones en JSON o Markdown

## 🚀 Instalación Rápida

### Prerrequisitos

- Python 3.8 o superior
- [Ollama](https://ollama.ai) instalado y ejecutándose
- 8GB RAM mínimo (16GB recomendado)

### Pasos de Instalación

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/APU.git
cd APU
```



El instalador:
- ✅ Verificará tu entorno
- ✅ Instalará las dependencias
- ✅ Configurará las variables de entorno
- ✅ Descargará el modelo Llama 3.2:3b

1. **Configura Tavily (opcional)**

Para búsquedas web, obtén una API key gratuita en [tavily.com](https://tavily.com) y agrégala al archivo `.env`:
```
TAVILY_API_KEY=tu_api_key_aqui
```

## 🎮 Uso

1. **Inicia la aplicación**
```bash
python app.py
```

2. **Abre tu navegador**

Navega a [http://localhost:8501](http://localhost:8501)

3. **Carga tus documentos**

- Usa la barra lateral para cargar PDFs
- Los documentos se procesarán automáticamente
- Verás las estadísticas actualizarse

4. **¡Comienza a preguntar!**

Ejemplos de preguntas:
- "¿Qué dice el documento sobre machine learning?"
- "Resume la metodología del paper"
- "¿Cuáles son las conclusiones principales?"
- "Busca en internet información actualizada sobre [tema]"

## 📁 Estructura del Proyecto

```
APU/
├── app.py              # Aplicación principal Streamlit
├── config/
│   └── settings.py     # Configuraciones centrales
├── core/
│   ├── document_processor.py  # Procesamiento de PDFs
│   ├── embeddings.py         # Gestión de embeddings
│   └── vector_store.py       # Almacén vectorial FAISS
├── agents/
│   ├── main_agent.py   # Agente orquestador
│   └── memory.py       # Gestión de memoria
├── tools/
│   ├── rag_tool.py     # Herramienta RAG
│   └── web_search_tool.py  # Búsqueda web
├── ui/
│   └── components.py   # Componentes de interfaz
├── utils/
│   ├── logger.py       # Sistema de logging
│   └── helpers.py      # Funciones auxiliares
└── data/
    ├── documents/      # PDFs originales
    ├── processed/      # Documentos procesados
    └── faiss_index/    # Índice vectorial
```

## 🛠️ Configuración Avanzada

### Variables de Entorno

Edita el archivo `.env` para personalizar:

```env
# Modelo de IA
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Embeddings
EMBEDDINGS_MODEL=all-MiniLM-L6-v2

# Procesamiento
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# UI
STREAMLIT_THEME=dark
MAX_CHAT_HISTORY=50
```

### Modelos Alternativos

Puedes usar otros modelos de Ollama:
```bash
# Modelos más grandes (mejor calidad)
ollama pull llama3.2:7b
ollama pull mistral:7b

# Modelos más pequeños (más rápidos)
ollama pull phi3:mini
```

## 🔧 Solución de Problemas

### Ollama no se conecta
```bash
# Verifica que Ollama esté ejecutándose
ollama list

# Reinicia Ollama
ollama serve
```

### Error de memoria
- Reduce `CHUNK_SIZE` en `.env`
- Usa un modelo más pequeño
- Procesa menos documentos a la vez

### Documentos no se procesan
- Verifica que sean PDFs válidos
- Revisa los logs en `apu.log`
- Asegúrate de que tengan texto extraíble


## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Tecnologías principales

- [Langchain](https://langchain.com) por el framework de agentes
- [Streamlit](https://streamlit.io) por la interfaz web
- [Ollama](https://ollama.ai) por los modelos locales
- [FAISS](https://github.com/facebookresearch/faiss) por la búsqueda vectorial

## 📞 Contacto

¿Preguntas? ¿Problemas? 
- 📧 Email: tu-email@ejemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/APU/issues)

---

