"""
Herramienta RAG para búsqueda en documentos - VERSION SIMPLIFICADA
"""
from typing import List, Dict, Any, Optional

from core.embeddings import EmbeddingsManager
from core.vector_store import VectorStore
from config.settings import SEARCH_CONFIG
from utils.logger import logger
from utils.helpers import format_source_citation, truncate_text

class RAGTool:
    """Herramienta para búsqueda en documentos usando RAG - Versión Simplificada"""
    
    def __init__(self, embeddings_manager: EmbeddingsManager, vector_store: VectorStore):
        self.name = "search_documents"
        self.description = """Busca información en los documentos académicos disponibles. 
        Úsala cuando necesites encontrar información específica de los apuntes.
        Retorna los fragmentos más relevantes con sus fuentes."""
        
        self.embeddings_manager = embeddings_manager
        self.vector_store = vector_store
        logger.info("Herramienta RAG inicializada")
    
    def run(self, query: str, top_k: Optional[int] = None, 
             filter_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ejecuta la búsqueda RAG
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados
            filter_metadata: Filtros de metadata
            
        Returns:
            String formateado con los resultados
        """
        try:
            logger.info(f"Búsqueda RAG: {query}")
            
            # Codificar query
            query_embedding = self.embeddings_manager.encode_text(query)
            
            # Buscar en vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k or SEARCH_CONFIG["max_results"],
                threshold=SEARCH_CONFIG["similarity_threshold"],
                filter_metadata=filter_metadata
            )
            
            if not results:
                return "No se encontraron resultados relevantes en los documentos."
            
            # Formatear resultados
            formatted_results = self._format_results(results, query)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda RAG: {e}")
            return f"Error al buscar en los documentos: {str(e)}"
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Formatea los resultados de búsqueda
        
        Args:
            results: Lista de resultados
            query: Query original
            
        Returns:
            String formateado con los resultados
        """
        formatted_parts = []
        
        # Agrupar por documento
        results_by_doc = {}
        for result in results:
            doc_id = result.get("doc_id", "unknown")
            if doc_id not in results_by_doc:
                results_by_doc[doc_id] = []
            results_by_doc[doc_id].append(result)
        
        # Formatear cada documento
        for doc_id, doc_results in results_by_doc.items():
            # Obtener metadata del documento
            doc_metadata = doc_results[0].get("doc_metadata", {})
            
            # Crear cita
            citation = format_source_citation(doc_metadata)
            
            # Header del documento
            formatted_parts.append(f"\n📄 **{citation}**")
            
            # Chunks del documento
            for i, result in enumerate(doc_results, 1):
                score = result.get("score", 0)
                content = result.get("content", "")
                section = result.get("metadata", {}).get("section", "")
                
                # Indicador de relevancia
                if score > 0.8:
                    relevance = "🟢 Alta relevancia"
                elif score > 0.6:
                    relevance = "🟡 Relevancia media"
                else:
                    relevance = "🔴 Baja relevancia"
                
                # Formatear chunk
                formatted_parts.append(f"\n{relevance} (Score: {score:.2f})")
                
                if section:
                    formatted_parts.append(f"Sección: {section}")
                
                # Contenido (truncado si es muy largo)
                if len(content) > 500:
                    content = truncate_text(content, 500)
                
                formatted_parts.append(f"```\n{content}\n```")
        
        # Agregar resumen
        summary = f"\n\n📊 **Resumen de búsqueda**\n"
        summary += f"- Query: \"{query}\"\n"
        summary += f"- Documentos encontrados: {len(results_by_doc)}\n"
        summary += f"- Fragmentos relevantes: {len(results)}\n"
        
        return summary + "\n".join(formatted_parts)
    
    def search_with_context(self, query: str, context_window: int = 1) -> List[Dict[str, Any]]:
        """
        Busca con contexto adicional (chunks anteriores y posteriores)
        
        Args:
            query: Consulta de búsqueda
            context_window: Número de chunks de contexto a cada lado
            
        Returns:
            Resultados con contexto expandido
        """
        # Búsqueda inicial
        query_embedding = self.embeddings_manager.encode_text(query)
        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=SEARCH_CONFIG["max_results"]
        )
        
        # Expandir con contexto
        expanded_results = []
        seen_chunks = set()
        
        for result in initial_results:
            doc_id = result["doc_id"]
            chunk_index = result["chunk_index"]
            
            # Obtener chunks del documento
            doc_chunks = self.vector_store.get_document_chunks(doc_id)
            
            # Agregar chunks de contexto
            start_idx = max(0, chunk_index - context_window)
            end_idx = min(len(doc_chunks), chunk_index + context_window + 1)
            
            for i in range(start_idx, end_idx):
                chunk = doc_chunks[i]
                chunk_id = chunk["chunk_id"]
                
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    
                    # Marcar si es el chunk principal o contexto
                    chunk["is_main"] = (i == chunk_index)
                    chunk["original_score"] = result["score"] if chunk["is_main"] else 0
                    
                    expanded_results.append(chunk)
        
        return expanded_results
    
    def get_document_summary(self, doc_id: str) -> str:
        """
        Obtiene un resumen de un documento específico
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Resumen del documento
        """
        chunks = self.vector_store.get_document_chunks(doc_id)
        
        if not chunks:
            return "Documento no encontrado"
        
        # Metadata del documento
        doc_metadata = chunks[0]["metadata"].get("doc_metadata", {})
        
        summary_parts = []
        
        # Título y autores
        if doc_metadata.get("title"):
            summary_parts.append(f"**Título**: {doc_metadata['title']}")
        
        if doc_metadata.get("authors"):
            authors = ", ".join(doc_metadata["authors"])
            summary_parts.append(f"**Autores**: {authors}")
        
        # Abstract
        if doc_metadata.get("abstract"):
            abstract = truncate_text(doc_metadata["abstract"], 300)
            summary_parts.append(f"**Abstract**: {abstract}")
        
        # Keywords
        if doc_metadata.get("keywords"):
            keywords = ", ".join(doc_metadata["keywords"])
            summary_parts.append(f"**Keywords**: {keywords}")
        
        # Estadísticas
        summary_parts.append(f"\n**Estadísticas**:")
        summary_parts.append(f"- Total de secciones: {len(chunks)}")
        summary_parts.append(f"- Fecha: {doc_metadata.get('date', 'No especificada')}")
        
        return "\n".join(summary_parts)