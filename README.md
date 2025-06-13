# 📚 APU - Apuntes IA

**Asistente Inteligente para Consultar Apuntes Académicos**

APU es una aplicación avanzada de RAG (Retrieval-Augmented Generation) especializada en procesar y consultar apuntes de clase, documentos académicos y papers de investigación. Utiliza embeddings semánticos y modelos de lenguaje locales para proporcionar respuestas precisas basadas en tus documentos.

## 🌟 Características Principales

- **🔍 Búsqueda Semántica Avanzada**: Encuentra información relevante usando similitud vectorial
- **📄 Procesamiento Inteligente de PDFs**: Extrae metadata específica de apuntes de clase
- **🤖 RAG con Modelos Locales**: Usa Ollama para respuestas sin enviar datos a terceros
- **🌐 Búsqueda Web Opcional**: Complementa con información actualizada de internet
- **📊 Gestión Visual de Documentos**: Interfaz intuitiva para manejar tus archivos
- **💾 Memoria de Conversación**: Mantiene contexto entre preguntas
- **⚡ Threshold Adaptativo**: Ajusta automáticamente la precisión de búsqueda

## 🛠️ Requisitos del Sistema

- **Python**: 3.10 o superior
- **Sistema Operativo**: Windows, macOS, o Linux
- **RAM**: Mínimo 8GB (recomendado 16GB para modelos grandes)
- **Espacio en Disco**: Al menos 10GB libres
- **Internet**: Para descargar modelos inicialmente

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd APU
```

### 2. Crear y Activar Entorno Virtual

#### En Windows (PowerShell):
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Si tienes problemas de permisos, ejecuta primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### En macOS/Linux:
```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate
```

### 3. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias principales
pip install streamlit>=1.29.0
pip install sentence-transformers>=2.2.2
pip install faiss-cpu>=1.7.4
pip install langchain>=0.1.0
pip install langchain-community>=0.0.20
pip install numpy>=1.26.0
pip install PyPDF2>=3.0.0
pip install pdfplumber>=0.10.0
pip install python-dotenv>=1.0.0
pip install requests>=2.31.0

# Dependencias opcionales para búsqueda web
pip install tavily-python>=0.3.0
```

### 4. Instalar y Configurar Ollama

#### Descargar Ollama:
- **Windows/macOS**: Ve a [https://ollama.ai/download](https://ollama.ai/download)
- **Linux**: 
  ```bash
  curl -fsSL https://ollama.ai/install.sh | sh
  ```

#### Iniciar Ollama:
```bash
# Iniciar servidor Ollama (mantener esta terminal abierta)
ollama serve
```

#### Descargar Modelo Recomendado:
```bash
# En otra terminal, descargar modelo (recomendado para mejor rendimiento)
ollama pull mistral:7b

# Alternativas según tus recursos:
ollama pull llama3.1:7b     # Buen balance calidad/velocidad
ollama pull llama3.2:3b     # Más rápido, menos recursos
ollama pull phi3:7b         # Optimizado para Q&A
```

### 5. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Configuración de la Aplicación
APP_NAME=APU - Apuntes IA
APP_VERSION=1.0.0
DEBUG_MODE=False

# Configuración Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# Configuración de Procesamiento
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_SEARCH_RESULTS=8

# Búsqueda Web (Opcional)
TAVILY_API_KEY=tu_api_key_aqui

# Configuración UI
STREAMLIT_THEME=dark
MAX_CHAT_HISTORY=50
```

### 6. Crear Estructura de Directorios

```bash
# Los directorios se crean automáticamente al ejecutar la app
# Pero puedes crearlos manualmente si lo prefieres:

mkdir -p data/documents
mkdir -p data/processed  
mkdir -p data/faiss_index
mkdir -p data/sessions
```

## ▶️ Ejecutar la Aplicación

### 1. Verificar que Ollama esté ejecutándose:
```bash
# En una terminal, mantener ejecutando:
ollama serve
```

### 2. Iniciar APU:
```bash
# En otra terminal, con el entorno virtual activado:
streamlit run app.py
```

### 3. Acceder a la Aplicación:
- Abrir navegador en: **http://localhost:8502**
- La aplicación se cargará automáticamente

## 📖 Cómo Usar APU

### 1. **Cargar Documentos**

1. Ve al **sidebar** → "📄 Documentos"
2. Haz clic en **"Browse files"**, alternativamente puedes arrastrar y soltar los archivos sobre la sección.
3. Selecciona tus archivos PDF (apuntes de clase, papers, etc.)
4. Presiona **"🔄 Procesar Documnentos"**


### 2. **Realizar Consultas**

#### Búsqueda en Documentos (por defecto):
```
¿Qué es una regresión lineal?
¿Qué es un autoencoder?
¿Quién es el autor del documento de la Semana #?
Explica las diferencias entre encoder y decoder
¿Cuáles son los tipos de autoencoders mencionados?
```

#### Búsqueda Web (opcional):
```
Buscar en internet las últimas noticias sobre IA
Información actualizada sobre U-Net en 2024
Buscar web ejemplos de autoencoders recientes
```

### 3. **Funciones Avanzadas**

#### Exportar Conversaciones:
- **JSON**: Para análisis posterior
- **Markdown**: Para documentación

#### Configurar Búsqueda:
- **Mostrar fuentes**: Ver documentos consultados
- **Búsqueda web**: Habilitar consultas en internet
- **Limpiar conversación**: Reiniciar contexto

### 4. **Ejemplos de Metadata Extraída**

APU extrae automáticamente información como:

```json
{
  "title": "Apuntes Semana #",
  "authors": ["José José Josares"],  
  "document_type": "class_notes",
  "course_week": 13,
  "date": "20/05/2025",
  "institution": "Instituto Tecnológico de Costa Rica",
  "topics_covered": ["Autoencoder", "U-Net", "Deep Learning"],
  "sections": ["RESPUESTAS DEL QUIZ", "REPASO", "ENCODER"],
  "quiz_questions": 4,
  "extraction_confidence": 0.85
}
```

## ⚙️ Configuración Avanzada

### Cambiar Modelo de Ollama:
```bash
# Descargar modelo diferente
ollama pull llama3.1:8b

# Actualizar .env
OLLAMA_MODEL=llama3.1:8b
```

### Optimizar Rendimiento:
```env
# Para documentos largos
CHUNK_SIZE=700
CHUNK_OVERLAP=100

# Para más resultados
MAX_SEARCH_RESULTS=12

# Para mejor precisión
SIMILARITY_THRESHOLD=0.2
```

### Habilitar Búsqueda Web:
1. Obtener API key gratuita en [https://tavily.com](https://tavily.com)
2. Agregar en `.env`: `TAVILY_API_KEY=tu_api_key`
3. Activar en la interfaz: ☑️ "🌐 Permitir búsqueda web"
4. Iniciar el prompt/consulta con "Busca en internet..."

## 🔧 Solución de Problemas

### Error: "No se pudo conectar con Ollama"
```bash
# Verificar que Ollama esté ejecutándose
ollama serve

# Verificar conexión
curl http://localhost:11434/api/version

# Verificar modelo descargado
ollama list
```

### Error: "Timeout del modelo"
- Cambiar a un modelo más rápido: `ollama pull llama3.2:3b`
- Aumentar timeout en `.env`: `OLLAMA_TIMEOUT=512`

### Error: "Memoria insuficiente"
- Usar modelo más pequeño: `phi3:3b` o `llama3.2:1b`
- Cerrar otras aplicaciones que consuman RAM

### Error: "No se encontraron documentos"
1. Verificar que los PDFs estén en `data/documents/`
2. Procesarlos usando la interfaz web
3. Verificar que el índice se haya creado en `data/faiss_index/`

### Problemas con Streamlit:
```bash
# Limpiar caché
streamlit cache clear

# Ejecutar en puerto diferente
streamlit run app.py --server.port 8503
```

## 📊 Estadísticas y Monitoreo

La aplicación muestra en tiempo real:
- **Documentos indexados**
- **Fragmentos de texto** procesados
- **Tamaño del índice** vectorial
- **Calidad promedio** de embeddings
- **Sesión actual** y historial

