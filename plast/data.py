"""
This module loads and manages all data required for pLAST, including models, metadata,
cluster mappings, and GenBank features. It provides lazy loading for efficient resource usage.
"""
import pickle
from typing import Union
import pandas as pd
from plast.exceptions import DataLoadingError
from plast.model import PLASTModel


class PLASTData:
    """
    Class storing all data required for pLAST.
    """

    def __init__(
        self,
        config_dict: Union[dict, None] = None,
    ):
        self.config = config_dict or {}
        self.models = {}
        self.metadata = None
        self.cluster_mapping = None
        self.gbfeatures = None
        print("Loading pLAST data...")

    def __getattribute__(self, name):
        # Lazy loading
        if name == "metadata":
            if super().__getattribute__("metadata") is None:
                self._load_metadata()
        elif name == "cluster_mapping":
            if super().__getattribute__("cluster_mapping") is None:
                self.cluster_mapping = pd.read_csv(
                    self.config["cluster_mapping"],
                    sep="\t",
                    header=None,
                    names=["accession", "locus_tag", "cluster_id"],
                    index_col=["accession", "locus_tag"],
                )
        elif name == "gbfeatures":
            if super().__getattribute__("gbfeatures") is None:
                with open(self.config["gbfeatures"], "rb") as f:
                    self.gbfeatures = pickle.load(f)

        return super().__getattribute__(name)

    def _load_metadata(self):
        self.metadata = (
            pd.read_csv(self.config["metadata"], sep="\t")
            .sort_values("accession")
            .set_index("accession")
        )

    def get_model(self, model_name: str):
        """
        Get the model by name, loading it if necessary.
        """
        if model_name not in self.models:
            if model_name not in self.config["models"]:
                raise DataLoadingError(
                    f"Model {model_name} not found in configuration."
                )
            self.models[model_name] = PLASTModel(self.config["models"][model_name])
        return self.models[model_name]
