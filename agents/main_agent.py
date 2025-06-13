"""
Agente principal de APU simplificado (sin LangChain agents)
"""
from typing import List, Dict, Any, Optional
import requests
import json

from config.settings import OLLAMA_CONFIG, SYSTEM_PROMPTS
from utils.logger import logger

class SimpleOllamaClient:
    """Cliente simplificado para Ollama"""
    
    def __init__(self):
        self.base_url = OLLAMA_CONFIG["host"]
        self.model = OLLAMA_CONFIG["model"]
        self.temperature = OLLAMA_CONFIG["temperature"]
        self.max_tokens = OLLAMA_CONFIG["max_tokens"]
        
        # Verificar conexión
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info(f"Conectado a Ollama: {self.base_url}")
            else:
                raise Exception(f"Ollama no responde: {response.status_code}")
        except Exception as e:
            logger.error(f"Error conectando a Ollama: {e}")
            raise RuntimeError(f"No se pudo conectar con Ollama en {self.base_url}")
    
    def invoke(self, prompt: str) -> str:
        """Invoca el modelo Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise Exception(f"Error en Ollama: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error invocando Ollama: {e}")
            return f"Error: {str(e)}"

class APUAgent:
    """Agente principal de APU simplificado"""
    
    def __init__(self, rag_tool, web_search_tool):
        self.rag_tool = rag_tool
        self.web_search_tool = web_search_tool
        self.llm = SimpleOllamaClient()
        self.conversation_history = []
        
        logger.info("Agente APU simplificado inicializado")
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario
        
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
            
            # Construir prompt para el LLM
            prompt = self._build_prompt(message, search_results)
            
            # Generar respuesta
            response = self.llm.invoke(prompt)
            
            # Agregar a historial
            self.conversation_history.append({
                "user": message,
                "assistant": response,
                "sources": [{"type": "document", "title": "Documentos locales"}]
            })
            
            # Mantener solo las últimas 10 conversaciones
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            result = {
                "answer": response,
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
    
    def _build_prompt(self, user_message: str, search_results: str) -> str:
        """Construye el prompt para el LLM"""
        
        # Contexto del historial
        history_context = ""
        if self.conversation_history:
            history_context = "\n\nHistorial reciente:\n"
            for conv in self.conversation_history[-3:]:  # Solo últimas 3
                history_context += f"Usuario: {conv['user']}\n"
                history_context += f"APU: {conv['assistant']}\n\n"
        
        prompt = f"""
{SYSTEM_PROMPTS["main_agent"]}

{history_context}

Información encontrada en los documentos:
{search_results}

Pregunta del usuario: {user_message}

Instrucciones:
1. Basa tu respuesta en la información encontrada en los documentos
2. Si la información no es suficiente, indícalo claramente
3. Mantén un tono académico pero accesible
4. Estructura tu respuesta de forma clara
5. Cita las fuentes cuando sea relevante

Respuesta:
"""
        
        return prompt
    
    def search_web(self, query: str) -> str:
        """Busca en internet (funcionalidad opcional)"""
        try:
            # Realizar búsqueda web
            web_results = self.web_search_tool.run(query)
            
            # Si hay resultados, construir prompt para el LLM
            if "❌" not in web_results and "Error" not in web_results:
                prompt = f"""
{SYSTEM_PROMPTS["main_agent"]}

El usuario ha solicitado una búsqueda en internet sobre: {query}

Resultados de la búsqueda web:
{web_results}

Instrucciones:
1. Analiza y resume la información encontrada en internet
2. Presenta la información de manera clara y organizada
3. Mantén un tono informativo pero advierte sobre la veracidad
4. Incluye las fuentes mencionadas

Respuesta basada en búsqueda web:
"""
                
                # Generar respuesta procesada por el LLM
                processed_response = self.llm.invoke(prompt)
                
                # Combinar respuesta del LLM con resultados originales
                return f"{processed_response}\n\n---\n\n**Resultados detallados:**\n{web_results}"
            else:
                # Si hay error en la búsqueda, devolver el mensaje original
                return web_results
                
        except Exception as e:
            logger.error(f"Error en búsqueda web: {e}")
            return f"Error al realizar búsqueda web: {str(e)}"
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.conversation_history = []
        logger.info("Memoria de conversación limpiada")
    
    def generate_followup_questions(self, response: str, original_query: str) -> List[str]:
        """Genera preguntas de seguimiento"""
        # Implementación simple
        questions = [
            "¿Podrías explicar esto con más detalle?",
            "¿Hay ejemplos prácticos relacionados?",
            "¿Qué otros aspectos debería considerar?"
        ]
        return questions[:2]  # Solo 2 preguntas
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la conversación"""
        return {
            "total_messages": len(self.conversation_history),
            "recent_topics": [conv["user"][:50] + "..." for conv in self.conversation_history[-3:]],
            "tools_used": ["search_documents"]
        }