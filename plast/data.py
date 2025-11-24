"""
This module loads and manages all data required for pLAST, including models, metadata,
cluster mappings, and GenBank features. It provides lazy loading for efficient resource usage.
"""

import pickle
from typing import Dict, Any, Optional
import pandas as pd
from plast.exceptions import DataLoadingError
from plast.model import PLASTModel


class PLASTData:
    """
    Stores all data required for pLAST, including models, metadata, cluster mappings,
    and GenBank features. Provides lazy loading for efficient resource usage.

    :param config_dict: Configuration dictionary with file paths and model info.
    :type config_dict: dict or None
    """

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PLASTData object.

        :param config_dict: Configuration dictionary with file paths and model info.
        :type config_dict: dict or None
        """
        self.config: Dict[str, Any] = config_dict or {}
        self.models: Dict[str, PLASTModel] = {}
        self.metadata: Optional[pd.DataFrame] = None
        self.cluster_mapping: Optional[pd.DataFrame] = None
        self.gbfeatures: Optional[Any] = None
        print("Loading pLAST data...")

    def __getattribute__(self, name: str) -> Any:
        """
        Provides lazy loading for metadata, cluster_mapping, and gbfeatures.

        :param name: Attribute name.
        :type name: str
        :returns: Attribute value, loading if necessary.
        :rtype: Any
        """
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

    def _load_metadata(self) -> None:
        """
        Loads metadata from the configured file path and sets it as an indexed DataFrame.

        :returns: None
        """
        self.metadata = (
            pd.read_csv(self.config["metadata"], sep="\t")
            .sort_values("accession")
            .set_index("accession")
        )

    def get_model(self, model_name: str) -> PLASTModel:
        """
        Get the model by name, loading it if necessary.

        :param model_name: Name of the model to retrieve.
        :type model_name: str
        :returns: Loaded PLASTModel object.
        :rtype: PLASTModel
        :raises DataLoadingError: If model is not found in configuration.
        """
        if model_name not in self.models:
            if model_name not in self.config["models"]:
                raise DataLoadingError(
                    f"Model {model_name} not found in configuration."
                )
            self.models[model_name] = PLASTModel(self.config["models"][model_name])
        return self.models[model_name]
