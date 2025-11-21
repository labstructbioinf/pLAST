"""
This module defines the PLASTModel class, which manages the loading and access of
various components of a PLAST model, including the Word2Vec model, embeddings,
mean embedding, IDF values, and UMAP mapper. Each component is loaded lazily to
optimize resource usage.
"""

import pickle
import gensim


class PLASTModel:
    """
    Class representing a PLAST model with lazy loading.
    """

    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._embeddings = None
        self._mean_embedding = None
        self._idf = None
        self._umap_mapper = None

    @property
    def model(self):
        """
        Returns the Word2Vec model instance. If the model has not been loaded yet,
        it loads the model from the path specified in the configuration.
        """
        if self._model is None:
            self._model = gensim.models.Word2Vec.load(self.config["model"])
        return self._model

    @property
    def embeddings(self):
        """
        Returns the precomputed embeddings dictionary. If the embeddings have not been
        loaded yet, it loads them from the path specified in the configuration.
        """
        if self._embeddings is None:
            with open(self.config["embeddings"], "rb") as f:
                self._embeddings = pickle.load(f)
        return self._embeddings

    @property
    def mean_embedding(self):
        """
        Returns the mean embedding vector. If it has not been loaded yet,
        it loads it from the path specified in the configuration.
        """
        if self._mean_embedding is None:
            with open(self.config["mean_embedding"], "rb") as f:
                self._mean_embedding = pickle.load(f)
        return self._mean_embedding

    @property
    def idf(self):
        """
        Returns the IDF dictionary. If it has not been loaded yet,
        it loads it from the path specified in the configuration.
        """
        if self._idf is None:
            with open(self.config["idf"], "rb") as f:
                self._idf = pickle.load(f)
        return self._idf

    @property
    def umap_mapper(self):
        """
        Returns the UMAP mapper instance. If it has not been loaded yet,
        it loads it from the path specified in the configuration.
        """
        if self._umap_mapper is None:
            with open(self.config["umap_mapper"], "rb") as f:
                self._umap_mapper = pickle.load(f)
        return self._umap_mapper
