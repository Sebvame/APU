"""
Almacén vectorial usando FAISS para APU
"""
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json

from config.settings import FAISS_INDEX_DIR, SEARCH_CONFIG
from utils.logger import logger
from utils.helpers import save_json, load_json

class VectorStore:
    """Almacén vectorial basado en FAISS"""
    
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
                
                # Guardar metadata
                self.metadata[chunk_id] = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "doc_metadata": doc["metadata"]
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
    
    def search(self, query_embedding: np.ndarray, 
              top_k: int = None,
              threshold: float = None,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Busca en el índice
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados
            threshold: Umbral de similitud
            filter_metadata: Filtros de metadata
            
        Returns:
            Lista de resultados
        """
        if self.index.ntotal == 0:
            logger.warning("Índice vacío")
            return []
        
        # Usar configuración por defecto si no se especifica
        top_k = top_k or SEARCH_CONFIG["max_results"]
        threshold = threshold or SEARCH_CONFIG["similarity_threshold"]
        
        # Preparar query
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Buscar más resultados para poder filtrar
        search_k = min(top_k * 3, self.index.ntotal)
        
        # Búsqueda
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Procesar resultados
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS retorna -1 para resultados no válidos
                continue
                
            # Convertir distancia a similitud (para IndexFlatIP ya es similitud)
            similarity = float(dist)
            
            # Aplicar threshold
            if similarity < threshold:
                continue
            
            # Obtener chunk
            chunk_data = self.chunks_data[idx]
            chunk_id = chunk_data["chunk_id"]
            
            # Obtener metadata
            metadata = self.metadata.get(chunk_id, {})
            
            # Aplicar filtros de metadata si existen
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if metadata.get("doc_metadata", {}).get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Crear resultado
            result = {
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id"),
                "content": chunk_data["content"],
                "score": similarity,
                "metadata": metadata.get("metadata", {}),
                "doc_metadata": metadata.get("doc_metadata", {}),
                "chunk_index": metadata.get("chunk_index", 0)
            }
            
            results.append(result)
            
            # Limitar resultados
            if len(results) >= top_k:
                break
        
        # Re-rankear si está configurado
        if SEARCH_CONFIG.get("rerank", False):
            results = self._rerank_results(results, query_embedding[0])
        
        logger.info(f"Búsqueda completada: {len(results)} resultados")
        return results
    
    def _rerank_results(self, results: List[Dict[str, Any]], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Re-rankea resultados considerando metadata y contexto
        """
        for result in results:
            # Boost por metadata relevante
            boost = 1.0
            
            # Boost si es del abstract o introducción
            if result["metadata"].get("section", "").lower() in ["abstract", "introduction"]:
                boost *= 1.2
            
            # Boost si tiene keywords que coinciden
            # (esto requeriría análisis adicional del query)
            
            # Aplicar boost
            result["score"] *= boost
        
        # Re-ordenar por score
        results.sort(key=lambda x: x["score"], reverse=True)
        
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
        
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": unique_docs,
            "embedding_dim": self.embedding_dim,
            "index_size_mb": self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
            "metadata_size_mb": self.metadata_path.stat().st_size / (1024 * 1024) if self.metadata_path.exists() else 0
        }