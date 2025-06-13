"""
Agente principal de APU que orquesta las herramientas
"""
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

from config.settings import OLLAMA_CONFIG, SYSTEM_PROMPTS, UI_CONFIG
from tools.rag_tool import RAGTool
from tools.web_search_tool import WebSearchTool
from utils.logger import logger

class APUAgent:
    """Agente principal de APU"""
    
    def __init__(self, rag_tool: RAGTool, web_search_tool: WebSearchTool):
        self.rag_tool = rag_tool
        self.web_search_tool = web_search_tool
        
        # Inicializar LLM
        self.llm = self._initialize_llm()
        
        # Crear prompt del agente
        self.prompt = self._create_agent_prompt()
        
        # Inicializar memoria
        self.memory = ConversationBufferWindowMemory(
            k=UI_CONFIG["max_chat_history"],
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Crear agente
        self.agent = self._create_agent()
        
        logger.info("Agente APU inicializado")
    
    def _initialize_llm(self) -> Ollama:
        """Inicializa el modelo Ollama"""
        try:
            llm = Ollama(
                base_url=OLLAMA_CONFIG["host"],
                model=OLLAMA_CONFIG["model"],
                temperature=OLLAMA_CONFIG["temperature"],
                num_predict=OLLAMA_CONFIG["max_tokens"],
                timeout=OLLAMA_CONFIG["timeout"],
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=True
            )
            
            # Verificar conexión
            llm.invoke("test")
            logger.info(f"LLM inicializado: {OLLAMA_CONFIG['model']}")
            
            return llm
            
        except Exception as e:
            logger.error(f"Error inicializando LLM: {e}")
            raise RuntimeError(
                f"No se pudo conectar con Ollama en {OLLAMA_CONFIG['host']}. "
                "Asegúrate de que Ollama esté ejecutándose."
            )
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Crea el prompt del agente"""
        template = """
{system_prompt}

Historial de conversación:
{chat_history}

Herramientas disponibles:
{tools}

Nombres de herramientas: {tool_names}

Instrucciones importantes:
1. SIEMPRE usa la herramienta search_documents primero para buscar en los documentos locales
2. SOLO usa web_search si el usuario lo solicita explícitamente o si no encuentras información en los documentos
3. Proporciona respuestas detalladas basándote en la información encontrada
4. Cita las fuentes cuando proporciones información
5. Si no encuentras información relevante, indícalo claramente

Formato de respuesta:
Debes responder siguiendo este formato EXACTO:

Question: la pregunta del usuario
Thought: qué necesitas hacer
Action: el nombre de la herramienta a usar
Action Input: el input para la herramienta
Observation: el resultado de la herramienta
... (este patrón Thought/Action/Action Input/Observation puede repetirse N veces)
Thought: Ya tengo la información necesaria
Final Answer: la respuesta final detallada para el usuario

Pregunta del usuario: {input}
{agent_scratchpad}
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["system_prompt", "chat_history", "tools", "tool_names", "input", "agent_scratchpad"],
            partial_variables={
                "system_prompt": SYSTEM_PROMPTS["main_agent"]
            }
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Crea el agente ejecutor"""
        # Lista de herramientas
        tools = [self.rag_tool, self.web_search_tool]
        
        # Crear agente ReAct
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )
        
        # Crear ejecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario
        
        Args:
            message: Mensaje del usuario
            session_id: ID de sesión para mantener contexto
            
        Returns:
            Respuesta del agente
        """
        try:
            logger.info(f"Procesando mensaje: {message[:100]}...")
            
            # Ejecutar agente
            response = self.agent.invoke({
                "input": message
            })
            
            # Extraer información relevante
            result = {
                "answer": response.get("output", ""),
                "intermediate_steps": response.get("intermediate_steps", []),
                "sources": self._extract_sources(response),
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
    
    def _extract_sources(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae las fuentes usadas en la respuesta"""
        sources = []
        
        #