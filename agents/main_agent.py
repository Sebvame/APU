"""
Agente principal con manejo de contexto y timeouts
"""
from typing import List, Dict, Any, Optional
import requests
import json
import time

from config.settings import OLLAMA_CONFIG, SYSTEM_PROMPTS
from utils.logger import logger

class ImprovedOllamaClient:
    """Cliente Ollama mejorado con mejor manejo de timeouts y contexto"""
    
    def __init__(self):
        self.base_url = OLLAMA_CONFIG["host"]
        self.model = OLLAMA_CONFIG["model"]
        self.temperature = OLLAMA_CONFIG["temperature"]
        self.max_tokens = OLLAMA_CONFIG["max_tokens"]
        self.timeout = 120  # Aumentado a 2 minutos
        
        # Verificar conexión
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            if response.status_code == 200:
                logger.info(f"Conectado a Ollama: {self.base_url}")
                # Verificar modelo disponible
                self._check_model_availability()
            else:
                raise Exception(f"Ollama no responde: {response.status_code}")
        except Exception as e:
            logger.error(f"Error conectando a Ollama: {e}")
            raise RuntimeError(f"No se pudo conectar con Ollama en {self.base_url}")
    
    def _check_model_availability(self):
        """Verifica si el modelo está disponible y sugiere mejores opciones"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                logger.info(f"Modelos disponibles: {available_models}")
                
                # Sugerir mejores modelos si están disponibles
                better_models = [
                    "llama3.1:8b", "llama3.1:7b", "llama3.2:8b", 
                    "mistral:7b", "codellama:7b", "phi3:7b"
                ]
                
                for better_model in better_models:
                    if any(better_model in model for model in available_models):
                        logger.warning(f"Modelo recomendado disponible: {better_model}")
                        logger.warning(f"Considera cambiar OLLAMA_MODEL en .env a: {better_model}")
                        break
                
        except Exception as e:
            logger.warning(f"No se pudo verificar modelos disponibles: {e}")
    
    def invoke(self, prompt: str, max_retries: int = 2) -> str:
        """Invoca el modelo Ollama con reintentos y mejor manejo de errores"""
        
        for attempt in range(max_retries + 1):
            try:
                # Optimizar prompt para el modelo
                optimized_prompt = self._optimize_prompt(prompt)
                
                payload = {
                    "model": self.model,
                    "prompt": optimized_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "top_p": 0.9,  # Mejor calidad de respuesta
                        "repeat_penalty": 1.1,  # Evitar repeticiones
                        "num_ctx": 4096,  # Contexto más largo
                    }
                }
                
                logger.info(f"Enviando prompt a Ollama (intento {attempt + 1}/{max_retries + 1})")
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Ollama respondió en {elapsed_time:.2f} segundos")
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "")
                    
                    if answer.strip():
                        return answer
                    else:
                        raise Exception("Respuesta vacía del modelo")
                else:
                    raise Exception(f"Error HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout en intento {attempt + 1}. Reintentando...")
                if attempt < max_retries:
                    time.sleep(2)  # Esperar antes de reintentar
                    continue
                else:
                    return self._generate_timeout_response(prompt)
                    
            except Exception as e:
                logger.error(f"Error en intento {attempt + 1}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    return f"Error: No se pudo generar respuesta después de {max_retries + 1} intentos. {str(e)}"
        
        return "Error: No se pudo generar respuesta"
    
    def _optimize_prompt(self, prompt: str) -> str:
        """Optimiza el prompt para mejor rendimiento del modelo"""
        # Truncar contexto si es muy largo
        if len(prompt) > 8000:
            logger.warning("Prompt muy largo, truncando contexto")
            # Mantener el sistema prompt y la pregunta, truncar el medio
            lines = prompt.split('\n')
            system_lines = lines[:20]  # Primeras 20 líneas (sistema)
            user_lines = lines[-10:]   # Últimas 10 líneas (pregunta)
            
            truncated_prompt = '\n'.join(system_lines) + '\n\n[...contexto truncado...]\n\n' + '\n'.join(user_lines)
            return truncated_prompt
        
        return prompt
    
    def _generate_timeout_response(self, prompt: str) -> str:
        """Genera una respuesta básica cuando hay timeout"""
        if "qué es" in prompt.lower() or "what is" in prompt.lower():
            return """**Respuesta Parcial** (Timeout del modelo)

Lo siento, el modelo tardó demasiado en responder. Basándome en la información encontrada en los documentos, puedo decir que:

- Se encontraron referencias relacionadas con tu pregunta en los apuntes
- Para obtener una respuesta más completa, intenta:
  1. Hacer una pregunta más específica
  2. Buscar términos más concretos
  3. Verificar que Ollama esté funcionando correctamente

**Sugerencia**: Considera usar un modelo más rápido como `llama3.2:1b` para respuestas más rápidas, o `llama3.1:8b` para mayor calidad."""

        return """**Timeout del Modelo**

El modelo tardó demasiado en procesar tu consulta. Esto puede deberse a:

1. **Contexto muy largo**: Los documentos encontrados son muy extensos
2. **Modelo sobrecargado**: `llama3.2:3b` puede ser lento para consultas complejas
3. **Recursos limitados**: El sistema puede estar bajo carga

**Recomendaciones**:
- Intenta una pregunta más específica
- Considera usar un modelo más eficiente
- Verifica que Ollama esté funcionando correctamente"""

class APUAgent:
    """Agente principal de APU mejorado"""
    
    def __init__(self, rag_tool, web_search_tool):
        self.rag_tool = rag_tool
        self.web_search_tool = web_search_tool
        self.llm = ImprovedOllamaClient()
        self.conversation_history = []
        
        logger.info("Agente APU mejorado inicializado")
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario con mejor manejo de contexto
        
        Args:
            message: Mensaje del usuario
            session_id: ID de sesión
            
        Returns:
            Respuesta del agente
        """
        try:
            logger.info(f"Procesando mensaje: {message[:100]}...")
            
            # Buscar en documentos primero
            search_results = self.rag_tool.run(message)
            logger.info(f"Resultados de búsqueda obtenidos: {search_results}...")
            
            # Construir prompt optimizado
            prompt = self._build_optimized_prompt(message, search_results)
            
            # Generar respuesta
            response = self.llm.invoke(prompt)
            
            # Procesar y enriquecer respuesta
            enhanced_response = self._enhance_response(response, search_results, message)
            
            # Agregar a historial (mantener solo últimas 5 conversaciones)
            self.conversation_history.append({
                "user": message,
                "assistant": enhanced_response,
                "sources": [{"type": "document", "title": "Documentos locales"}]
            })
            
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
            
            result = {
                "answer": enhanced_response,
                "intermediate_steps": [("search_documents", search_results)],
                "sources": [{"type": "document", "title": "Documentos locales"}],
                "session_id": session_id
            }
            
            logger.info("Respuesta generada exitosamente")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            return {
                "answer": f"Lo siento, ocurrió un error al procesar tu pregunta: {str(e)}",
                "intermediate_steps": [],
                "sources": [],
                "error": str(e)
            }
    
    def _build_optimized_prompt(self, user_message: str, search_results: str) -> str:
        """Construye un prompt optimizado para el modelo"""
        
        # Contexto del historial (solo últimas 2 conversaciones)
        history_context = ""
        if self.conversation_history:
            history_context = "\n\nContexto de conversación reciente:\n"
            for conv in self.conversation_history[-2:]:
                history_context += f"P: {conv['user'][:100]}...\n"
                history_context += f"R: {conv['assistant'][:200]}...\n\n"
        
        # Extraer información clave de los resultados
        key_info = self._extract_key_information(search_results)
        logger.info(f"Información clave extraída: {key_info}...")
        prompt = f"""Eres APU, un asistente académico especializado. Analiza la información y responde de forma clara y estructurada.

HISTORIAL DE CONVERSACIÓN:
{history_context}

INFORMACIÓN ENCONTRADA:
{key_info}

PREGUNTA: {user_message}

INSTRUCCIONES:
1. Responde basándote SOLO en la información proporcionada de los documentos
2. Si es una definición, comienza con una explicación clara
3. Usa formato markdown para organizar la respuesta
4. Incluye ejemplos si están disponibles
5. Mantén un tono académico pero accesible
6. Prioriza la información encontrada en los documentos sobre el historial de conversación

RESPUESTA:"""
        
        return prompt
    
    def _extract_key_information(self, search_results: str) -> str:
        """Extrae información clave de los resultados de búsqueda"""
        # Simplificar resultados largos
        if len(search_results) > 2000:
            # Extraer solo las partes más relevantes
            lines = search_results.split('\n')
            key_lines = []
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['🟢', '🟡', 'score:', 'relevancia']):
                    key_lines.append(line)
                elif line.strip().startswith('```') or line.strip().endswith('```'):
                    key_lines.append(line)
                elif len(line.strip()) > 50 and not line.startswith('📄'):
                    key_lines.append(line)
            
            return '\n'.join(key_lines[:20])  # Máximo 20 líneas
        
        return search_results
    
    def _enhance_response(self, response: str, search_results: str, question: str) -> str:
        """Mejora la respuesta con información adicional"""
        # Si la respuesta es muy corta, agregar contexto
        if len(response.strip()) < 100:
            response += "\n\n*Nota: Esta respuesta se basa en la información disponible en los documentos cargados.*"
        
        # Si hay timeout, agregar sugerencias
        if "timeout" in response.lower() or "error" in response.lower():
            response += f"\n\n**Información encontrada**: Se localizaron {len(search_results.split('📄')) - 1} documentos relacionados con tu consulta."
        
        return response
    
    def search_web(self, query: str) -> str:
        """Busca en internet (funcionalidad opcional)"""
        try:
            web_results = self.web_search_tool.run(query)
            
            if "❌" not in web_results and "Error" not in web_results:
                # Crear prompt simplificado para web
                prompt = f"""Analiza esta información de internet sobre: {query}

{web_results[:1500]}

Proporciona un resumen claro y organizado en español."""
                
                processed_response = self.llm.invoke(prompt)
                return f"{processed_response}\n\n---\n\n**Fuente**: Búsqueda en Internet"
            else:
                return web_results
                
        except Exception as e:
            logger.error(f"Error en búsqueda web: {e}")
            return f"Error al realizar búsqueda web: {str(e)}"
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.conversation_history = []
        logger.info("Memoria de conversación limpiada")
    
    def generate_followup_questions(self, response: str, original_query: str) -> List[str]:
        """Genera preguntas de seguimiento inteligentes"""
        try:
            # Preguntas basadas en el tipo de consulta
            if "qué es" in original_query.lower():
                return [
                    f"¿Cómo funciona {original_query.split('qué es')[-1].strip('?').strip()}?",
                    "¿Puedes dar ejemplos prácticos?",
                    "¿Cuáles son sus ventajas y desventajas?"
                ]
            elif "cómo" in original_query.lower():
                return [
                    "¿Puedes explicar esto paso a paso?",
                    "¿Hay algún ejemplo específico?",
                    "¿Qué problemas puede resolver esto?"
                ]
            else:
                return [
                    "¿Podrías profundizar en este tema?",
                    "¿Hay conceptos relacionados que debería conocer?",
                    "¿Puedes dar más detalles?"
                ]
                
        except Exception as e:
            logger.error(f"Error generando preguntas de seguimiento: {e}")
            return [
                "¿Podrías explicar esto con más detalle?",
                "¿Hay ejemplos prácticos relacionados?"
            ]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la conversación"""
        return {
            "total_messages": len(self.conversation_history),
            "recent_topics": [conv["user"][:50] + "..." for conv in self.conversation_history[-3:]],
            "tools_used": ["search_documents"],
            "model_used": self.llm.model
        }