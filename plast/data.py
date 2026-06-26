"""
This module loads and manages all data required for pLAST, including models, metadata,
cluster mappings, and GenBank features. It provides lazy loading for efficient resource usage.
"""

import gzip
import pickle
import sys
import threading
from typing import Dict, Any, Optional, Iterable, List
import numpy as np
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
        self.metadata_indexes: Dict[str, pd.Index] = {}
        self.metadata_aliases: Dict[str, Dict[str, str]] = {}
        self.cluster_mapping: Optional[pd.DataFrame] = None
        self.cluster_mappings: Dict[str, pd.DataFrame] = {}
        self.cluster_lookups: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.model_search_indexes: Dict[str, Dict[str, Any]] = {}
        self.model_umap_indexes: Dict[str, Dict[str, Any]] = {}
        self.gbfeatures: Optional[Any] = None
        self._cache_lock = threading.RLock()
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
                self.cluster_mapping = self.get_cluster_mapping()
        elif name == "gbfeatures":
            if super().__getattribute__("gbfeatures") is None:
                self.gbfeatures = self._load_pickle(self.config["gbfeatures"])
        return super().__getattribute__(name)

    def _ensure_metadata_loaded(self) -> None:
        if super().__getattribute__("metadata") is None:
            self._load_metadata()

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [item for item in value if item is not None]
        return [value]

    @staticmethod
    def _without_version(accession: str) -> str:
        return accession.split(".", 1)[0]

    @staticmethod
    def _first_existing_column(metadata: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
        for name in names:
            if name in metadata.columns:
                return name
        return None

    @staticmethod
    def _split_accessions(accession: str) -> List[str]:
        return [
            alias.strip()
            for alias in str(accession).split(",")
            if alias.strip()
        ]

    @staticmethod
    def _split_metadata_values(value: Any) -> List[str]:
        return [
            item.strip()
            for item in str(value).replace(";", ",").split(",")
            if item.strip()
        ]

    @staticmethod
    def _source_names_from_accession_prefixes(prefixes: Iterable[str]) -> set:
        names = set()
        for prefix in prefixes:
            name = str(prefix).strip().rstrip("_")
            if name:
                names.add(name.lower())
        return names

    @classmethod
    def _value_matches_any(cls, value: Any, wanted_values: Iterable[str]) -> bool:
        wanted = {str(item).strip().lower() for item in wanted_values if item}
        return any(
            item.lower() in wanted
            for item in cls._split_metadata_values(value)
        )

    def _accession_alias_candidates(
        self, accession: str, strip_prefixes: Optional[Iterable[str]] = None
    ) -> List[str]:
        strip_prefixes = [str(prefix) for prefix in self._as_list(strip_prefixes)]
        values = [str(accession)] + self._split_accessions(str(accession))
        candidates: List[str] = []

        for value in values:
            for candidate in (value, self._without_version(value)):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            for prefix in strip_prefixes:
                if value.startswith(prefix):
                    stripped = value[len(prefix) :]
                    for candidate in (stripped, self._without_version(stripped)):
                        if candidate and candidate not in candidates:
                            candidates.append(candidate)
        return candidates

    def _accession_matches_prefix(self, accession: str, prefixes: Iterable[str]) -> bool:
        wanted = tuple(str(prefix) for prefix in prefixes)
        return any(
            alias.startswith(wanted)
            for alias in self._split_accessions(str(accession))
        )

    @staticmethod
    def _load_pickle(path: str) -> Any:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
            return pickle.load(f)

    def _load_metadata(self) -> None:
        """
        Loads metadata from the configured file path and builds in-memory indexes.

        :returns: None
        """
        metadata = pd.read_csv(
            self.config["metadata"],
            sep="\t",
            compression="infer",
            low_memory=False,
        )
        rename_columns = {}
        if "accession" not in metadata.columns and "Plasmid_ID" in metadata.columns:
            rename_columns["Plasmid_ID"] = "accession"
        if "database" not in metadata.columns:
            database_column = self._first_existing_column(
                metadata, ("Data_Source", "data_source", "source", "Source")
            )
            if database_column is not None:
                rename_columns[database_column] = "database"
        if "topology" not in metadata.columns and "Topology" in metadata.columns:
            rename_columns["Topology"] = "topology"
        if rename_columns:
            metadata = metadata.rename(columns=rename_columns)
        if "accession" not in metadata.columns:
            raise DataLoadingError("Metadata must contain an 'accession' column.")

        metadata["accession"] = metadata["accession"].astype(str)
        self.metadata = metadata.sort_values("accession").set_index(
            "accession", drop=False
        )
        self._build_metadata_indexes()

    def _metadata_index_config(self) -> Dict[str, Dict[str, Any]]:
        configured = dict(self.config.get("metadata_indexes", {}))
        configured.setdefault("__all__", {})
        default_index = self.config.get("default_metadata_index")
        if default_index:
            configured.setdefault(default_index, {})
        for model_config in self.config.get("models", {}).values():
            index_name = model_config.get("metadata_index")
            if index_name:
                configured.setdefault(index_name, {})
        return configured

    def _build_metadata_indexes(self) -> None:
        metadata = self.metadata
        for index_name, index_config in self._metadata_index_config().items():
            indexed = metadata

            prefixes = self._as_list(
                index_config.get(
                    "accession_prefixes", index_config.get("accession_prefix")
                )
            )
            if prefixes:
                accession_mask = indexed["accession"].map(
                    lambda accession: self._accession_matches_prefix(
                        accession, prefixes
                    )
                )
                source_mask = pd.Series(False, index=indexed.index)
                source_names = self._source_names_from_accession_prefixes(prefixes)
                source_column = self._first_existing_column(
                    indexed,
                    ("database", "Data_Source", "data_source", "source", "Source"),
                )
                if source_column is not None and source_names:
                    source_mask = indexed[source_column].map(
                        lambda source: self._value_matches_any(source, source_names)
                    )
                mask = accession_mask | source_mask
                if mask.any():
                    indexed = indexed.loc[mask]
                elif not index_config.get("legacy_unprefixed_fallback", True):
                    indexed = indexed.loc[mask]

            databases = self._as_list(
                index_config.get(
                    "databases",
                    index_config.get("database", index_config.get("data_source")),
                )
            )
            if databases:
                db_column = self._first_existing_column(
                    indexed,
                    ("database", "Data_Source", "data_source", "source", "Source"),
                )
                if db_column is not None:
                    wanted = {str(database).lower() for database in databases}
                    indexed = indexed.loc[
                        indexed[db_column].map(
                            lambda database: self._value_matches_any(
                                database, wanted
                            )
                        )
                    ]

            topologies = self._as_list(index_config.get("topology"))
            if topologies:
                if "topology" not in indexed.columns:
                    raise DataLoadingError(
                        "Metadata topology filter requires a 'topology' column."
                    )
                wanted = {str(topology).lower() for topology in topologies}
                indexed = indexed.loc[
                    indexed["topology"].astype(str).str.lower().isin(wanted)
                ]

            self.metadata_indexes[index_name] = indexed.index
            self.metadata_aliases[index_name] = self._build_alias_map(
                indexed.index, index_config
            )

    def _build_alias_map(
        self, accessions: Iterable[str], index_config: Dict[str, Any]
    ) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        strip_prefixes = index_config.get("strip_accession_prefixes")
        if strip_prefixes is None:
            strip_prefixes = index_config.get("strip_accession_prefix")
        if strip_prefixes is True:
            strip_prefixes = index_config.get(
                "accession_prefixes", index_config.get("accession_prefix")
            )
        strip_prefixes = self._as_list(strip_prefixes)
        database_prefixes = [
            str(database)
            for database in self._as_list(
                index_config.get(
                    "databases",
                    index_config.get("database", index_config.get("data_source")),
                )
            )
        ]

        for accession in accessions:
            accession = str(accession)
            for candidate in self._accession_alias_candidates(
                accession, strip_prefixes
            ):
                aliases.setdefault(candidate, accession)
                for database_prefix in database_prefixes:
                    aliases.setdefault(f"{database_prefix}_{candidate}", accession)
        return aliases

    def get_metadata_index_name(self, model_name: Optional[str] = None) -> str:
        """
        Resolve the metadata index configured for a model.
        """
        if model_name and model_name in self.config.get("models", {}):
            index_name = self.config["models"][model_name].get("metadata_index")
            if index_name:
                return index_name
        return self.config.get("default_metadata_index", "__all__")

    def resolve_accession(
        self,
        accession: str,
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
        allow_fallbacks: bool = True,
    ) -> Optional[str]:
        """
        Resolve a model/browser accession to the row accession used in metadata.
        """
        self._ensure_metadata_loaded()
        accession = str(accession).strip()
        if not accession:
            return None
        candidates = [accession, self._without_version(accession)]
        index_name = metadata_index or self.get_metadata_index_name(model_name)
        aliases = self.metadata_aliases.get(index_name, {})
        for candidate in candidates:
            if candidate in aliases:
                return aliases[candidate]

        if not allow_fallbacks:
            return None

        # Fallbacks for old configs without metadata_indexes.
        for candidate in candidates:
            if candidate in self.metadata.index:
                return candidate
        for candidate in candidates:
            for aliases in self.metadata_aliases.values():
                if candidate in aliases:
                    return aliases[candidate]
        return None

    def accession_in_metadata_index(
        self,
        accession: str,
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
    ) -> bool:
        """
        Return True only if the accession resolves in the requested metadata index.
        """
        index_name = metadata_index or self.get_metadata_index_name(model_name)
        return (
            self.resolve_accession(
                accession,
                model_name,
                index_name,
                allow_fallbacks=False,
            )
            is not None
        )

    def filter_ids_for_metadata_index(
        self,
        plasmid_ids: Iterable[str],
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
    ) -> List[str]:
        """
        Keep only ids that belong to a model/browser metadata index.
        """
        return [
            plasmid_id
            for plasmid_id in plasmid_ids
            if self.accession_in_metadata_index(
                plasmid_id, model_name, metadata_index
            )
        ]

    def candidate_accessions(
        self,
        accession: str,
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
    ) -> List[str]:
        """
        Return likely accession aliases for metadata, gbfeatures, mappings and embeddings.
        """
        index_name = metadata_index or self.get_metadata_index_name(model_name)
        index_config = self._metadata_index_config().get(index_name, {})
        strip_prefixes = index_config.get("strip_accession_prefixes")
        if strip_prefixes is None:
            strip_prefixes = index_config.get("strip_accession_prefix")
        if strip_prefixes is True:
            strip_prefixes = index_config.get(
                "accession_prefixes", index_config.get("accession_prefix")
            )
        strip_prefixes = self._as_list(strip_prefixes)
        database_prefixes = [
            str(database)
            for database in self._as_list(
                index_config.get(
                    "databases",
                    index_config.get("database", index_config.get("data_source")),
                )
            )
        ]

        candidates: List[str] = []
        for candidate in self._accession_alias_candidates(
            str(accession), strip_prefixes
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
            for database_prefix in database_prefixes:
                prefixed = f"{database_prefix}_{candidate}"
                if candidate and prefixed not in candidates:
                    candidates.append(prefixed)

        resolved = self.resolve_accession(accession, model_name, metadata_index)
        if resolved:
            for candidate in self._accession_alias_candidates(
                resolved, strip_prefixes
            ):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
                for database_prefix in database_prefixes:
                    prefixed = f"{database_prefix}_{candidate}"
                    if candidate and prefixed not in candidates:
                        candidates.append(prefixed)
        return candidates

    def get_metadata_row(
        self,
        plasmid_id: str,
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
    ) -> pd.Series:
        """
        Return one metadata row resolved through the configured in-memory index.
        """
        resolved_id = self.resolve_accession(plasmid_id, model_name, metadata_index)
        if resolved_id is None:
            raise KeyError(plasmid_id)
        row = self.metadata.loc[resolved_id].copy()
        row.name = plasmid_id
        return row

    def get_metadata_for_ids(
        self,
        plasmid_ids: Iterable[str],
        model_name: Optional[str] = None,
        metadata_index: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return metadata rows for ids without copying the full metadata table.
        """
        self._ensure_metadata_loaded()
        resolved_ids = []
        row_ids = []
        for plasmid_id in plasmid_ids:
            resolved_id = self.resolve_accession(
                plasmid_id, model_name, metadata_index
            )
            if resolved_id is not None:
                resolved_ids.append(resolved_id)
                row_ids.append(plasmid_id)
        if not row_ids:
            return pd.DataFrame(columns=self.metadata.columns, index=[])
        result = self.metadata.loc[np.asarray(resolved_ids, dtype=object)].copy()
        result.index = row_ids
        return result

    @staticmethod
    def _normalize_embedding_matrix(matrix: np.ndarray) -> tuple:
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        zero_norm_mask = row_norms[:, 0] == 0
        safe_norms = np.where(zero_norm_mask[:, None], 1.0, row_norms)
        return matrix / safe_norms, zero_norm_mask

    def get_search_index(self, model_name: str) -> Dict[str, Any]:
        """
        Return a cached, model-specific in-memory search index.

        The index is filtered to the metadata scope configured for the model and
        stores NumPy matrices so each query can rank plasmids with vectorized
        operations instead of looping over all embeddings in Python.
        """
        if model_name in self.model_search_indexes:
            return self.model_search_indexes[model_name]

        with self._cache_lock:
            if model_name in self.model_search_indexes:
                return self.model_search_indexes[model_name]

            self._ensure_metadata_loaded()
            model = self.get_model(model_name)
            ids = []
            vectors = []
            embedding_positions = []
            skipped = 0
            vector_size = 0
            for position, (plasmid_id, embedding) in enumerate(model.embeddings.items()):
                vector = np.asarray(embedding, dtype=np.float32)
                if vector_size == 0:
                    vector_size = int(vector.shape[0])
                if not self.accession_in_metadata_index(plasmid_id, model_name):
                    skipped += 1
                    continue
                ids.append(plasmid_id)
                vectors.append(vector)
                embedding_positions.append(position)

            if vectors:
                matrix = np.vstack(vectors).astype(np.float32, copy=False)
            else:
                matrix = np.empty((0, vector_size), dtype=np.float32)

            matrix_norm, zero_norm_mask = self._normalize_embedding_matrix(matrix)
            matrix_sq_norms = np.einsum("ij,ij->i", matrix, matrix)
            index = {
                "ids": ids,
                "matrix": matrix,
                "matrix_norm": matrix_norm.astype(np.float32, copy=False),
                "matrix_sq_norms": matrix_sq_norms.astype(np.float32, copy=False),
                "zero_norm_mask": zero_norm_mask,
                "embedding_positions": np.asarray(embedding_positions, dtype=np.int64),
                "id_to_embedding_position": {
                    plasmid_id: position
                    for plasmid_id, position in zip(ids, embedding_positions)
                },
                "skipped_by_metadata_index": skipped,
            }
            self.model_search_indexes[model_name] = index
            return index

    def get_model_ids_and_umap_coords(self, model_name: str) -> tuple:
        """
        Return cached plasmid ids and UMAP coordinates for a model metadata scope.
        """
        if model_name in self.model_umap_indexes:
            cached = self.model_umap_indexes[model_name]
            return cached["ids"], cached["umap_coords"]

        with self._cache_lock:
            if model_name in self.model_umap_indexes:
                cached = self.model_umap_indexes[model_name]
                return cached["ids"], cached["umap_coords"]

            search_index = self.get_search_index(model_name)
            model = self.get_model(model_name)
            umap_coords = np.asarray(model.umap_mapper)
            if len(model.embeddings) != len(umap_coords):
                raise DataLoadingError(
                    f"Mismatch between embeddings ({len(model.embeddings)}) "
                    f"and UMAP coordinates ({len(umap_coords)})"
                )
            positions = search_index["embedding_positions"]
            scoped_coords = (
                umap_coords[positions] if len(positions) else umap_coords[:0]
            )
            index = {
                "ids": search_index["ids"],
                "umap_coords": scoped_coords,
            }
            self.model_umap_indexes[model_name] = index
            return index["ids"], index["umap_coords"]

    def warm_search_indexes(
        self, model_names: Optional[Iterable[str]] = None
    ) -> None:
        """
        Preload metadata, embeddings and vectorized search indexes into RAM.
        """
        if model_names is None:
            names = list(self.config.get("models", {}).keys())
        else:
            names = [str(model_name) for model_name in self._as_list(model_names)]
        self._ensure_metadata_loaded()
        for model_name in names:
            self.get_search_index(model_name)

    def _cluster_mapping_path(self, model_name: Optional[str] = None) -> str:
        mapping_path = None
        if model_name and model_name in self.config.get("models", {}):
            mapping_path = self.config["models"][model_name].get("cluster_mapping")
        mapping_path = mapping_path or self.config.get("cluster_mapping")
        if not mapping_path:
            raise DataLoadingError("Cluster mapping path is not configured.")
        return mapping_path

    @staticmethod
    def _normalize_cluster_mapping_columns(mapping: pd.DataFrame) -> pd.DataFrame:
        mapping = mapping.rename(
            columns={
                "plasmid_id": "accession",
                "protein_id": "locus_tag",
            }
        )
        required_columns = {"accession", "locus_tag", "cluster_id"}
        missing_columns = required_columns.difference(mapping.columns)
        if missing_columns:
            raise DataLoadingError(
                "Cluster mapping is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        return mapping[["accession", "locus_tag", "cluster_id"]].dropna(
            subset=["accession", "locus_tag", "cluster_id"]
        )

    def _iter_cluster_mapping_chunks(
        self, mapping_path: str, chunksize: Optional[int] = None
    ):
        chunksize = int(
            chunksize or self.config.get("cluster_mapping_chunksize", 500_000)
        )
        read_kwargs = {
            "sep": "\t",
            "compression": "infer",
            "dtype": str,
            "chunksize": chunksize,
        }
        reader = pd.read_csv(mapping_path, **read_kwargs)
        try:
            first_chunk = next(reader)
        except StopIteration:
            return

        try:
            yield self._normalize_cluster_mapping_columns(first_chunk)
            for chunk in reader:
                yield self._normalize_cluster_mapping_columns(chunk)
        except DataLoadingError:
            headerless_reader = pd.read_csv(
                mapping_path,
                header=None,
                names=["accession", "locus_tag", "cluster_id"],
                usecols=[0, 1, 2],
                **read_kwargs,
            )
            for chunk in headerless_reader:
                yield self._normalize_cluster_mapping_columns(chunk)

    def get_cluster_mapping(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Return the MMseqs cluster mapping for a specific model.
        """
        mapping_path = self._cluster_mapping_path(model_name)
        if mapping_path not in self.cluster_mappings:
            mapping_chunks = self._iter_cluster_mapping_chunks(mapping_path)
            mapping = pd.concat(mapping_chunks, ignore_index=True)
            mapping["accession"] = mapping["accession"].astype(str)
            mapping["locus_tag"] = mapping["locus_tag"].astype(str)
            mapping["cluster_id"] = mapping["cluster_id"].astype(str)
            self.cluster_mappings[mapping_path] = (
                mapping.set_index(["accession", "locus_tag"]).sort_index()
            )
        return self.cluster_mappings[mapping_path]

    def get_cluster_lookup(
        self, model_name: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Return a lightweight accession/protein -> cluster lookup for MMseqs results.

        The lookup avoids pandas MultiIndex access in the request path, which removes
        lexsort warnings and reduces CPU/RAM overhead for repeated single-key lookups.
        """
        mapping_path = self._cluster_mapping_path(model_name)
        if mapping_path in self.cluster_lookups:
            return self.cluster_lookups[mapping_path]

        with self._cache_lock:
            if mapping_path in self.cluster_lookups:
                return self.cluster_lookups[mapping_path]

            lookup: Dict[str, Dict[str, str]] = {}
            if mapping_path in self.cluster_mappings:
                mapping_iter = [self.cluster_mappings[mapping_path].reset_index()]
            else:
                mapping_iter = self._iter_cluster_mapping_chunks(mapping_path)

            for chunk in mapping_iter:
                for accession, locus_tag, cluster_id in chunk.itertuples(
                    index=False, name=None
                ):
                    accession = sys.intern(str(accession))
                    cluster_id = sys.intern(str(cluster_id))
                    accession_lookup = lookup.setdefault(accession, {})
                    accession_lookup.setdefault(str(locus_tag), cluster_id)

            self.cluster_lookups[mapping_path] = lookup
            return lookup

    def get_mmseqs_db(self, model_name: str, padded: bool = False) -> str:
        """
        Return the MMseqs database path configured for a model.
        """
        model_config = self.config.get("models", {}).get(model_name, {})
        if padded:
            db_path = model_config.get("mmseqs_db_padded") or self.config.get(
                "mmseqs_db_padded"
            )
            if db_path:
                return db_path
        db_path = model_config.get("mmseqs_db") or self.config.get("mmseqs_db")
        if not db_path:
            raise DataLoadingError(
                f"MMseqs database path is not configured for {model_name}."
            )
        return db_path

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
