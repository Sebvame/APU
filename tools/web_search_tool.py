"""
Herramienta de búsqueda web usando Tavily
"""
from typing import List, Dict, Any, Optional
import os

from config.settings import TAVILY_CONFIG
from utils.logger import logger

# Importación condicional de Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily no está disponible. Instala tavily-python para usar búsquedas web.")

class WebSearchTool:
    """Herramienta para búsqueda en internet usando Tavily - Versión Simplificada"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = """Busca información actualizada en internet. 
        Úsala SOLO cuando el usuario solicite explícitamente buscar en internet o 
        cuando necesites información que no está en los documentos locales.
        NO la uses por defecto."""
        
        self.api_key = TAVILY_CONFIG.get("api_key", "")
        self.client = None
        
        if TAVILY_AVAILABLE and self.api_key:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Cliente Tavily inicializado correctamente")
            except Exception as e:
                logger.error(f"Error inicializando Tavily: {e}")
                self.client = None
        else:
            if not TAVILY_AVAILABLE:
                logger.warning("Tavily no está instalado")
            if not self.api_key:
                logger.warning("API key de Tavily no configurada")
    
    def run(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Ejecuta búsqueda web
        
        Args:
            query: Consulta de búsqueda
            max_results: Número máximo de resultados
            
        Returns:
            String formateado con resultados
        """
        try:
            # Verificar disponibilidad
            if not self.client:
                return self._get_unavailable_message()
            
            logger.info(f"Búsqueda web: {query}")
            
            # Ejecutar búsqueda
            search_params = {
                "query": query,
                "search_depth": TAVILY_CONFIG.get("search_depth", "advanced"),
                "max_results": max_results or TAVILY_CONFIG.get("max_results", 5)
            }
            
            response = self.client.search(**search_params)
            
            # Procesar resultados
            if not response or "results" not in response:
                return "No se encontraron resultados en la búsqueda web."
            
            # Formatear resultados
            formatted_results = self._format_results(response["results"], query)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda web: {e}")
            return f"Error al realizar búsqueda web: {str(e)}"
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Formatea los resultados de búsqueda web
        
        Args:
            results: Resultados de Tavily
            query: Query original
            
        Returns:
            String formateado
        """
        if not results:
            return "No se encontraron resultados relevantes."
        
        formatted_parts = []
        
        # Header
        formatted_parts.append(f"🌐 **Resultados de búsqueda web para**: \"{query}\"\n")
        formatted_parts.append(f"Se encontraron {len(results)} resultados:\n")
        
        # Formatear cada resultado
        for i, result in enumerate(results, 1):
            title = result.get("title", "Sin título")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score", 0)
            
            # Formatear resultado individual
            formatted_parts.append(f"\n**{i}. {title}**")
            
            if url:
                formatted_parts.append(f"🔗 {url}")
            
            # Relevancia
            if score > 0.8:
                relevance = "🟢 Alta relevancia"
            elif score > 0.5:
                relevance = "🟡 Relevancia media"
            else:
                relevance = "🔴 Baja relevancia"
            
            formatted_parts.append(f"{relevance} (Score: {score:.2f})")
            
            # Contenido
            if content:
                # Limitar longitud del contenido
                if len(content) > 300:
                    content = content[:300] + "..."
                formatted_parts.append(f"\n{content}\n")
            
            formatted_parts.append("---")
        
        # Agregar disclaimer
        formatted_parts.append("\n⚠️ **Nota**: Estos resultados provienen de internet. ")
        formatted_parts.append("Verifica la información con fuentes confiables.")
        
        return "\n".join(formatted_parts)
    
    def _get_unavailable_message(self) -> str:
        """Mensaje cuando la búsqueda web no está disponible"""
        message = ["❌ La búsqueda web no está disponible.\n"]
        
        if not TAVILY_AVAILABLE:
            message.append("- Tavily no está instalado. Ejecuta: pip install tavily-python")
        
        if not self.api_key:
            message.append("- API key de Tavily no configurada.")
            message.append("  1. Obtén una API key gratuita en: https://tavily.com")
            message.append("  2. Agrégala en el archivo .env: TAVILY_API_KEY=tu_api_key")
        
        message.append("\nPor ahora, solo puedo buscar en los documentos locales.")
        
        return "\n".join(message)
    
    def search_academic(self, query: str) -> str:
        """
        Búsqueda especializada en contenido académico
        
        Args:
            query: Consulta de búsqueda
            
        Returns:
            Resultados formateados
        """
        # Agregar términos académicos a la búsqueda
        academic_query = f"{query} academic paper research IEEE ACM scholar"
        
        return self.run(academic_query)
    
    def fact_check(self, statement: str) -> str:
        """
        Verifica un hecho o afirmación
        
        Args:
            statement: Afirmación a verificar
            
        Returns:
            Resultados de verificación
        """
        fact_check_query = f"fact check verify {statement}"
        
        results = self.run(fact_check_query, max_results=3)
        
        # Agregar contexto
        formatted = f"🔍 **Verificación de hechos**:\n\n"
        formatted += f"Afirmación: \"{statement}\"\n\n"
        formatted += results
        
        return formatted
    
    def get_recent_info(self, topic: str, days: int = 30) -> str:
        """
        Obtiene información reciente sobre un tema
        
        Args:
            topic: Tema a buscar
            days: Días de antigüedad máxima
            
        Returns:
            Información reciente
        """
        # Agregar filtros de tiempo a la búsqueda
        time_query = f"{topic} latest recent news updates {days} days"
        
        results = self.run(time_query)
        
        # Formatear con contexto temporal
        formatted = f"📅 **Información reciente sobre**: {topic}\n"
        formatted += f"(Últimos {days} días)\n\n"
        formatted += results
        
        return formatted