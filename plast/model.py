"""
This module defines the PLASTModel class, which manages the loading and access of
various components of a PLAST model, including the Word2Vec model, embeddings,
mean embedding, IDF values, and UMAP mapper. Each component is loaded lazily to
optimize resource usage.
"""

import pickle
from typing import Any, Dict, Optional
import gensim


class PLASTModel:
    """
    Class representing a PLAST model with lazy loading.

    :param config: Configuration dictionary with file paths for model components.
    :type config: dict
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PLASTModel object.

        :param config: Configuration dictionary with file paths for model components.
        :type config: dict
        """
        self.config: Dict[str, Any] = config
        self._model: Optional[gensim.models.Word2Vec] = None
        self._embeddings: Optional[Dict[str, Any]] = None
        self._mean_embedding: Optional[Any] = None
        self._idf: Optional[Dict[str, float]] = None
        self._umap_mapper: Optional[Any] = None

    @property
    def model(self) -> gensim.models.Word2Vec:
        """
        Returns the Word2Vec model instance. If the model has not been loaded yet,
        it loads the model from the path specified in the configuration.

        :returns: Loaded Word2Vec model instance.
        :rtype: gensim.models.Word2Vec
        """
        if self._model is None:
            self._model = gensim.models.Word2Vec.load(self.config["model"])
        return self._model

    @property
    def embeddings(self) -> Dict[str, Any]:
        """
        Returns the precomputed embeddings dictionary. If the embeddings have not been
        loaded yet, it loads them from the path specified in the configuration.

        :returns: Dictionary of embeddings.
        :rtype: dict
        """
        if self._embeddings is None:
            with open(self.config["embeddings"], "rb") as f:
                self._embeddings = pickle.load(f)
        return self._embeddings

    @property
    def mean_embedding(self) -> Any:
        """
        Returns the mean embedding vector. If it has not been loaded yet,
        it loads it from the path specified in the configuration.

        :returns: Mean embedding vector.
        :rtype: Any
        """
        if self._mean_embedding is None:
            with open(self.config["mean_embedding"], "rb") as f:
                self._mean_embedding = pickle.load(f)
        return self._mean_embedding

    @property
    def idf(self) -> Dict[str, float]:
        """
        Returns the IDF dictionary. If it has not been loaded yet,
        it loads it from the path specified in the configuration.

        :returns: Dictionary of IDF values.
        :rtype: dict
        """
        if self._idf is None:
            with open(self.config["idf"], "rb") as f:
                self._idf = pickle.load(f)
        return self._idf

    @property
    def umap_mapper(self) -> Any:
        """
        Returns the UMAP mapper instance. If it has not been loaded yet,
        it loads it from the path specified in the configuration.

        :returns: UMAP mapper object.
        :rtype: Any
        """
        if self._umap_mapper is None:
            with open(self.config["umap_mapper"], "rb") as f:
                self._umap_mapper = pickle.load(f)
        return self._umap_mapper
