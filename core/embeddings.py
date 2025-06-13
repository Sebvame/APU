"""
Sistema de embeddings para APU usando Sentence Transformers
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from config.settings import EMBEDDINGS_CONFIG
from utils.logger import logger

class EmbeddingsManager:
    """Gestor de embeddings para documentos"""
    
    def __init__(self):
        self.model_name = EMBEDDINGS_CONFIG["model_name"]
        self.device = EMBEDDINGS_CONFIG["device"]
        self.encode_kwargs = EMBEDDINGS_CONFIG["encode_kwargs"]
        
        logger.info(f"Inicializando modelo de embeddings: {self.model_name}")
        
        # Verificar disponibilidad de CUDA
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU")
            self.device = "cpu"
        
        # Cargar modelo
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Modelo cargado. Dimensión de embeddings: {self.embedding_dim}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Codifica un texto individual
        
        Args:
            text: Texto a codificar
            
        Returns:
            Vector de embeddings
        """
        if not text or not text.strip():
            logger.warning("Texto vacío recibido para encoding")
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                **self.encode_kwargs
            )
            return embedding
        except Exception as e:
            logger.error(f"Error codificando texto: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Codifica un batch de textos
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño del batch
            
        Returns:
            Array de embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Codificando batch de {len(texts)} textos")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                **self.encode_kwargs
            )
            
            logger.info(f"Batch codificado exitosamente. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error codificando batch: {e}")
            # Intentar codificar uno por uno como fallback
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text(text))
            return np.array(embeddings)
    
    def encode_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Codifica una lista de documentos con sus chunks
        
        Args:
            documents: Lista de documentos procesados
            
        Returns:
            Diccionario con embeddings por documento
        """
        all_embeddings = {}
        
        for doc in documents:
            doc_id = doc["doc_id"]
            logger.info(f"Codificando documento: {doc_id}")
            
            # Extraer textos de los chunks
            chunk_texts = [chunk["content"] for chunk in doc["chunks"]]
            
            if not chunk_texts:
                logger.warning(f"Documento sin chunks: {doc_id}")
                continue
            
            # Codificar chunks
            chunk_embeddings = self.encode_batch(chunk_texts)
            
            # Codificar metadata importante
            metadata_texts = []
            
            # Título
            if doc["metadata"].get("title"):
                metadata_texts.append(f"Title: {doc['metadata']['title']}")
            
            # Abstract
            if doc["metadata"].get("abstract"):
                metadata_texts.append(f"Abstract: {doc['metadata']['abstract']}")
            
            # Keywords
            if doc["metadata"].get("keywords"):
                keywords_text = "Keywords: " + ", ".join(doc["metadata"]["keywords"])
                metadata_texts.append(keywords_text)
            
            # Codificar metadata si existe
            metadata_embeddings = None
            if metadata_texts:
                metadata_embeddings = self.encode_batch(metadata_texts)
            
            all_embeddings[doc_id] = {
                "chunk_embeddings": chunk_embeddings,
                "metadata_embeddings": metadata_embeddings,
                "embedding_dim": self.embedding_dim
            }
        
        return all_embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula similitud coseno entre dos embeddings
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        # Normalizar si no están normalizados
        if not self.encode_kwargs.get("normalize_embeddings", False):
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Producto punto (equivale a similitud coseno con vectores normalizados)
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def find_similar(self, query_embedding: np.ndarray, 
                    embeddings: np.ndarray,
                    top_k: int = 5,
                    threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Encuentra los embeddings más similares
        
        Args:
            query_embedding: Embedding de consulta
            embeddings: Array de embeddings para buscar
            top_k: Número de resultados
            threshold: Umbral mínimo de similitud
            
        Returns:
            Lista de resultados con índices y scores
        """
        if len(embeddings) == 0:
            return []
        
        # Normalizar query si es necesario
        if not self.encode_kwargs.get("normalize_embeddings", False):
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            # Normalizar embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / norms
        else:
            embeddings_norm = embeddings
        
        # Calcular similitudes
        similarities = np.dot(embeddings_norm, query_embedding)
        
        # Filtrar por threshold
        valid_indices = np.where(similarities >= threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Obtener top k
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]
        
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(similarities[idx])
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "encode_kwargs": self.encode_kwargs
        }