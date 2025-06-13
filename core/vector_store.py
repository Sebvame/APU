"""
Almacén vectorial mejorado usando FAISS para APU
#         Codifica un lote de textos
"""
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
import re

from config.settings import FAISS_INDEX_DIR, SEARCH_CONFIG
from utils.logger import logger
from utils.helpers import save_json, load_json

class VectorStore:
    """Almacén vectorial basado en FAISS con búsqueda mejorada"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = {}
        self.doc_mapping = {}  # chunk_id -> doc_id
        self.chunks_data = []  # Lista de chunks en orden
        self.index_path = FAISS_INDEX_DIR / "faiss.index"
        self.metadata_path = FAISS_INDEX_DIR / "metadata.pkl"
        self.mapping_path = FAISS_INDEX_DIR / "mapping.json"
        
        # Inicializar índice
        self._initialize_index()
    
    def _initialize_index(self):
        """Inicializa el índice FAISS"""
        # Usar IndexFlatIP para similitud coseno (producto interno con vectores normalizados)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Agregar wrapper para IDs
        self.index = faiss.IndexIDMap(self.index)
        
        logger.info(f"Índice FAISS inicializado. Dimensión: {self.embedding_dim}")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings_dict: Dict[str, Dict[str, np.ndarray]]):
        """
        Agrega documentos al índice
        
        Args:
            documents: Lista de documentos procesados
            embeddings_dict: Diccionario con embeddings por documento
        """
        logger.info(f"Agregando {len(documents)} documentos al índice")
        
        all_embeddings = []
        all_ids = []
        
        for doc in documents:
            doc_id = doc["doc_id"]
            
            if doc_id not in embeddings_dict:
                logger.warning(f"No se encontraron embeddings para documento: {doc_id}")
                continue
            
            doc_embeddings = embeddings_dict[doc_id]["chunk_embeddings"]
            
            # Agregar cada chunk
            for i, (chunk, embedding) in enumerate(zip(doc["chunks"], doc_embeddings)):
                chunk_id = chunk["chunk_id"]
                
                # Guardar metadata enriquecida
                self.metadata[chunk_id] = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "doc_metadata": doc["metadata"],
                    "embedding_quality": self._assess_embedding_quality(embedding),
                    "content_features": self._extract_content_features(chunk["content"])
                }
                
                # Mapeo para búsqueda rápida
                self.doc_mapping[chunk_id] = doc_id
                self.chunks_data.append(chunk)
                
                # Agregar a listas para indexación
                all_embeddings.append(embedding)
                all_ids.append(len(all_ids))  # ID secuencial
        
        if all_embeddings:
            # Convertir a numpy array
            embeddings_array = np.array(all_embeddings).astype('float32')
            ids_array = np.array(all_ids).astype('int64')
            
            # Normalizar embeddings para similitud coseno
            faiss.normalize_L2(embeddings_array)
            
            # Agregar al índice
            self.index.add_with_ids(embeddings_array, ids_array)
            
            logger.info(f"Agregados {len(all_embeddings)} chunks al índice. Total: {self.index.ntotal}")
        
        # Guardar índice
        self.save()
    
    def _assess_embedding_quality(self, embedding: np.ndarray) -> float:
        """Evalúa la calidad del embedding basado en su distribución"""
        try:
            # Calcular métricas de calidad
            variance = np.var(embedding)
            norm = np.linalg.norm(embedding)
            sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
            
            # Score de calidad (0-1)
            quality_score = min(1.0, variance * norm * (1 - sparsity))
            return float(quality_score)
        except:
            return 0.5  # Score neutro si hay error
    
    def _extract_content_features(self, content: str) -> Dict[str, Any]:
        """Extrae características del contenido para mejorar la búsqueda"""
        features = {
            "length": len(content),
            "word_count": len(content.split()),
            "has_numbers": bool(re.search(r'\d', content)),
            "has_formulas": bool(re.search(r'[=+\-*/]', content)),
            "has_questions": bool(re.search(r'\?', content)),
            "has_lists": bool(re.search(r'^\s*[-*•]\s', content, re.MULTILINE)),
            "has_code": bool(re.search(r'[{}();]', content)),
            "language_indicators": {
                "spanish": len(re.findall(r'\b(?:el|la|de|en|y|a|que|es|se|no|te|lo|le|da|su|por|son|con|para|una|tiene|más|ser|hacer|poder|decir|todo|tener|su|grande|pequeño|primero|mucho|muy|después|tiempo|muy|tanto|cada|día|vida|vez|caso|forma|mundo|sobre|todo|país|ejemplo|durante|nuevo|mismo|gobierno|nuestro|otro|trabajo|vida|puede|bien|año|entre|está|durante|hacen|años)\b', content, re.IGNORECASE)),
                "english": len(re.findall(r'\b(?:the|be|to|of|and|a|in|that|have|i|it|for|not|on|with|he|as|you|do|at|this|but|his|by|from|they|she|or|an|will|my|one|all|would|there|their)\b', content, re.IGNORECASE))
            }
        }
        
        return features
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = None,
              threshold: float = None,
              filter_metadata: Optional[Dict[str, Any]] = None,
              use_adaptive_threshold: bool = True) -> List[Dict[str, Any]]:
        """
        Busca en el índice con algoritmos mejorados
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados
            threshold: Umbral de similitud
            filter_metadata: Filtros de metadata
            use_adaptive_threshold: Usar threshold adaptativo
            
        Returns:
            Lista de resultados
        """
        if self.index.ntotal == 0:
            logger.warning("Índice vacío")
            return []
        
        # Usar configuración por defecto si no se especifica
        top_k = top_k or SEARCH_CONFIG["max_results"]
        initial_threshold = threshold or SEARCH_CONFIG["similarity_threshold"]
        
        # Preparar query
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Búsqueda inicial con más resultados para filtrar
        search_k = min(top_k * 5, self.index.ntotal)
        
        # Búsqueda
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Procesar resultados
        results = []
        similarities = distances[0]
        
        # Threshold adaptativo
        if use_adaptive_threshold and SEARCH_CONFIG.get("adaptive_threshold", False):
            adaptive_threshold = self._calculate_adaptive_threshold(similarities, initial_threshold)
            logger.debug(f"Threshold adaptativo: {adaptive_threshold} (original: {initial_threshold})")
        else:
            adaptive_threshold = initial_threshold
        
        for i, (similarity, idx) in enumerate(zip(similarities, indices[0])):
            if idx == -1:  # FAISS retorna -1 para resultados no válidos
                continue
                
            # Aplicar threshold
            if similarity < adaptive_threshold:
                continue
            
            # Obtener chunk
            chunk_data = self.chunks_data[idx]
            chunk_id = chunk_data["chunk_id"]
            
            # Obtener metadata
            metadata = self.metadata.get(chunk_id, {})
            
            # Aplicar filtros de metadata si existen
            if filter_metadata:
                if not self._matches_filters(metadata, filter_metadata):
                    continue
            
            # Calcular score final con boosts
            final_score = self._calculate_boosted_score(
                similarity, metadata, chunk_data["content"]
            )
            
            # Crear resultado
            result = {
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id"),
                "content": chunk_data["content"],
                "score": final_score,
                "original_score": float(similarity),
                "metadata": metadata.get("metadata", {}),
                "doc_metadata": metadata.get("doc_metadata", {}),
                "chunk_index": metadata.get("chunk_index", 0),
                "embedding_quality": metadata.get("embedding_quality", 0.5)
            }
            
            results.append(result)
        
        # Ordenar por score final
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Aplicar MMR si está habilitado
        if SEARCH_CONFIG.get("use_mmr", False) and len(results) > 1:
            results = self._apply_mmr(results, top_k)
        else:
            results = results[:top_k]
        
        # Re-rankear si está configurado
        if SEARCH_CONFIG.get("rerank", False):
            results = self._rerank_results(results, query_embedding[0])
        
        logger.info(f"Búsqueda completada: {len(results)} resultados (threshold: {adaptive_threshold:.3f})")
        return results
    
    def _calculate_adaptive_threshold(self, similarities: np.ndarray, base_threshold: float) -> float:
        """Calcula threshold adaptativo basado en la distribución de similitudes"""
        valid_similarities = similarities[similarities > 0]
        
        if len(valid_similarities) == 0:
            return base_threshold
        
        # Estadísticas de similitudes
        mean_sim = np.mean(valid_similarities)
        std_sim = np.std(valid_similarities)
        max_sim = np.max(valid_similarities)
        
        # Threshold adaptativo
        if max_sim > 0.8:  # Hay resultados muy relevantes
            adaptive_threshold = max(base_threshold, mean_sim - std_sim)
        elif max_sim > 0.6:  # Resultados moderadamente relevantes
            adaptive_threshold = max(base_threshold * 0.8, mean_sim - 1.5 * std_sim)
        else:  # Resultados poco relevantes
            adaptive_threshold = max(SEARCH_CONFIG["min_threshold"], base_threshold * 0.6)
        
        # Aplicar límites
        adaptive_threshold = max(SEARCH_CONFIG["min_threshold"], 
                               min(SEARCH_CONFIG["max_threshold"], adaptive_threshold))
        
        return adaptive_threshold
    
    def _calculate_boosted_score(self, similarity: float, metadata: Dict, content: str) -> float:
        """Calcula score con boosts contextuales"""
        score = similarity
        boost_config = SEARCH_CONFIG.get("contextual_boost", {})
        
        # Boost por sección
        section = metadata.get("metadata", {}).get("section", "").lower()
        section_boosts = boost_config.get("section_relevance", {})
        for section_name, boost_factor in section_boosts.items():
            if section_name in section:
                score *= boost_factor
                break
        
        # Boost por calidad de embedding
        embedding_quality = metadata.get("embedding_quality", 0.5)
        score *= (0.8 + 0.4 * embedding_quality)  # Factor entre 0.8 y 1.2
        
        # Boost por características del contenido
        content_features = metadata.get("content_features", {})
        if content_features.get("has_formulas", False):
            score *= 1.1  # Boost para contenido técnico
        if content_features.get("word_count", 0) > 100:
            score *= 1.05  # Boost para contenido más completo
        
        return min(1.0, score)  # Limitar a 1.0
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Verifica si el metadata coincide con los filtros"""
        for key, value in filters.items():
            if metadata.get("doc_metadata", {}).get(key) != value:
                return False
        return True
    
    def _apply_mmr(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Aplica Maximum Marginal Relevance para diversificar resultados"""
        if len(results) <= 1:
            return results
        
        lambda_param = SEARCH_CONFIG.get("mmr_lambda", 0.7)
        selected = []
        remaining = results.copy()
        
        # Seleccionar el primer resultado (más relevante)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -1
            
            for i, candidate in enumerate(remaining):
                # Score de relevancia
                relevance_score = candidate["score"]
                
                # Score de diversidad (mínima similitud con ya seleccionados)
                diversity_score = 1.0
                for selected_result in selected:
                    # Similitud basada en contenido (aproximación)
                    content_similarity = self._approximate_content_similarity(
                        candidate["content"], selected_result["content"]
                    )
                    diversity_score = min(diversity_score, 1 - content_similarity)
                
                # Score MMR
                mmr_score = lambda_param * relevance_score + (1 - lambda_param) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _approximate_content_similarity(self, content1: str, content2: str) -> float:
        """Aproxima similitud entre contenidos usando características simples"""
        # Implementación simple usando palabras comunes
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _rerank_results(self, results: List[Dict[str, Any]], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Re-rankea resultados considerando metadata y contexto
        """
        for result in results:
            # El score ya incluye boosts, mantener el orden
            pass
        
        return results
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene todos los chunks de un documento
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Lista de chunks del documento
        """
        chunks = []
        
        for chunk_id, metadata in self.metadata.items():
            if metadata.get("doc_id") == doc_id:
                chunk_data = next((c for c in self.chunks_data if c["chunk_id"] == chunk_id), None)
                if chunk_data:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk_data["content"],
                        "metadata": metadata
                    })
        
        # Ordenar por índice
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        
        return chunks
    
    def save(self):
        """Guarda el índice y metadata"""
        try:
            # Guardar índice FAISS
            faiss.write_index(self.index, str(self.index_path))
            
            # Guardar metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    "metadata": self.metadata,
                    "chunks_data": self.chunks_data,
                    "embedding_dim": self.embedding_dim
                }, f)
            
            # Guardar mapping
            save_json(self.doc_mapping, self.mapping_path)
            
            logger.info("Índice y metadata guardados exitosamente")
            
        except Exception as e:
            logger.error(f"Error guardando índice: {e}")
            raise
    
    def load(self):
        """Carga el índice y metadata"""
        try:
            # Cargar índice FAISS
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Índice cargado. Total vectores: {self.index.ntotal}")
            
            # Cargar metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data["metadata"]
                    self.chunks_data = data["chunks_data"]
                    self.embedding_dim = data["embedding_dim"]
            
            # Cargar mapping
            if self.mapping_path.exists():
                self.doc_mapping = load_json(self.mapping_path)
            
            logger.info("Índice y metadata cargados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando índice: {e}")
            return False
    
    def clear(self):
        """Limpia el índice"""
        self._initialize_index()
        self.metadata.clear()
        self.doc_mapping.clear()
        self.chunks_data.clear()
        logger.info("Índice limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice"""
        unique_docs = len(set(self.doc_mapping.values()))
        
        # Calcular estadísticas de calidad
        embedding_qualities = [
            metadata.get("embedding_quality", 0.5) 
            for metadata in self.metadata.values()
        ]
        avg_embedding_quality = np.mean(embedding_qualities) if embedding_qualities else 0.0
        
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": unique_docs,
            "embedding_dim": self.embedding_dim,
            "avg_embedding_quality": round(avg_embedding_quality, 3),
            "index_size_mb": self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
            "metadata_size_mb": self.metadata_path.stat().st_size / (1024 * 1024) if self.metadata_path.exists() else 0,
            "search_config": {
                "threshold": SEARCH_CONFIG["similarity_threshold"],
                "max_results": SEARCH_CONFIG["max_results"],
                "adaptive_threshold": SEARCH_CONFIG.get("adaptive_threshold", False)
            }
        }