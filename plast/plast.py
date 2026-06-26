"""
This module implements the core classes and methods for the PLAST (Plasmid Search Tool)
"""

from collections.abc import Mapping
import time
from pathlib import Path
from typing import Any, Callable, Union
import subprocess
import tempfile
import json
import numpy as np
import pandas as pd
from plast.exceptions import NotFoundError
from plast.data import PLASTData
from plast.parsers import read_gbff, read_m8, read_hmmscan_output, read_fasta
from plast.utils import (
    is_gpu_available,
    warm_page_cache,
)

EXTERNAL_LOG_CHUNK_SIZE = 12000


class PLAST:
    """
    Class representing a plasmid query for pLAST.

    :param data: Reference PLASTData object.
    :type data: PLASTData
    :param model: Model name to use.
    :type model: str or None
    :param logger: Optional logger instance.
    :type logger: Any, optional
    """

    def __init__(self, data: PLASTData, model: Union[str, None] = None, logger=None):
        """
        Initialize the PLAST query object.

        :param data: Reference PLASTData object.
        :type data: PLASTData
        :param model: Model name to use.
        :type model: str or None
        :param logger: Optional logger instance.
        :type logger: Any, optional
        """
        self.name = ""
        self.data = data
        self.model = model if model is not None else ""
        self.parsed = None
        self.length = -1
        self.vector = None
        self.embedding = None
        self.results = None
        self.search_cache = {}
        self.file = None
        self.logger = logger
        self.disable_logs = logger is None
        self.query_context = {}
        self.debug(f"Initialized PLAST(name='{self.name}', model='{self.model}')")

    def copy(self) -> "PLAST":
        """
        Create a copy of the PLAST object.

        :returns: A copy of the PLAST object.
        :rtype: PLAST
        """
        self.debug(
            f"Copying PLAST(name='{self.name}', model='{self.model}', length={self.length})"
        )
        new_plast = PLAST(data=self.data, model=self.model, logger=self.logger)
        new_plast.name = self.name
        new_plast.parsed = self.parsed.copy() if self.parsed is not None else None
        new_plast.length = self.length
        new_plast.vector = self.vector.copy() if self.vector is not None else None
        new_plast.embedding = (
            self.embedding.copy() if self.embedding is not None else None
        )
        new_plast.results = self.results.copy() if self.results is not None else None
        new_plast.search_cache = self.search_cache.copy()
        new_plast.file = self.file
        new_plast.disable_logs = self.disable_logs
        new_plast.query_context = self.query_context.copy()
        return new_plast

    def to_dict(self) -> dict:
        """
        Return the query as a dictionary.

        :returns: Dictionary representation of the PLAST object.
        :rtype: dict
        """
        return {
            "name": self.name,
            "model": self.model,
            "parsed": (
                self.parsed.to_dict(orient="records")
                if self.parsed is not None
                else None
            ),
            "length": self.length,
            "vector": self.vector,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "results": self.results,
            "search_cache": self.search_cache,
            "query_context": self.query_context,
        }

    @staticmethod
    def from_dict(data: dict, plast_data: PLASTData, logger=None) -> "PLAST":
        """
        Load the query from a dictionary.

        :param data: Dictionary containing PLAST data.
        :type data: dict
        :param plast_data: Reference PLASTData object.
        :type plast_data: PLASTData
        :param logger: Optional logger instance.
        :type logger: Any, optional
        :returns: Loaded PLAST object.
        :rtype: PLAST
        """
        plast = PLAST(data=plast_data, logger=logger)
        plast.name = data.get("name", "")
        plast.model = data.get("model", "")
        plast.parsed = pd.DataFrame(data["parsed"]) if data.get("parsed") else None
        plast.length = data.get("length", -1)
        plast.vector = data.get("vector", None)
        plast.embedding = np.array(data["embedding"]) if data.get("embedding") else None
        plast.results = data.get("results", None)
        plast.search_cache = data.get("search_cache") or {}
        plast.query_context = data.get("query_context") or {}
        return plast

    def to_json(self, filename: Union[str, None] = None) -> Union[str, None]:
        """
        Return the query as a JSON object or write to file.

        :param filename: Optional filename to write JSON.
        :type filename: str or None
        :returns: JSON string if filename is None, else None.
        :rtype: str or None
        :raises Exception: On file write error.
        """
        self.debug(f"Serializing PLAST to JSON (filename={filename})")
        data = {
            "name": self.name,
            "model": self.model,
            "parsed": (
                self.parsed.to_dict(orient="records")
                if self.parsed is not None
                else None
            ),
            "length": self.length,
            "vector": self.vector,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "results": self.results,
            "search_cache": self.search_cache,
            "query_context": self.query_context,
        }
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                self.log(f"Wrote PLAST JSON to {filename}")
            except Exception as e:
                self.error(f"Failed to write JSON to '{filename}': {e}")
                raise
        else:
            return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    def from_json(json_data: Union[str, Path], data: PLASTData) -> "PLAST":
        """
        Load the query from a JSON string or Path.

        :param json_data: JSON string or Path to JSON file.
        :type json_data: str or pathlib.Path
        :param data: Reference PLASTData object.
        :type data: PLASTData
        :returns: Loaded PLAST object.
        :rtype: PLAST
        :raises ValueError: If input is not valid.
        """
        # NOTE: static method; logs will be available once PLAST instance is created
        if isinstance(json_data, Path):
            with open(json_data, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        elif isinstance(json_data, str):
            json_data = json.loads(json_data)
        else:
            raise ValueError("Input must be a JSON string or a Path object.")

        plast = PLAST(data=data)
        plast.name = json_data.get("name", "")
        plast.model = json_data.get("model", "")
        plast.parsed = (
            pd.DataFrame(json_data["parsed"]) if json_data.get("parsed") else None
        )
        plast.length = json_data.get("length", -1)
        plast.vector = json_data.get("vector", None)
        plast.embedding = (
            np.array(json_data["embedding"]) if json_data.get("embedding") else None
        )
        plast.results = json_data.get("results", None)
        plast.search_cache = json_data.get("search_cache") or {}
        plast.query_context = json_data.get("query_context") or {}
        plast.debug(
            f"Deserialized PLAST from JSON (name='{plast.name}', model='{plast.model}', "
            f"length={plast.length})"
        )
        return plast

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message via self.logger if available, else stdout.

        :param message: Message to log.
        :type message: str
        :param level: Logging level.
        :type level: str
        """
        msg = message if isinstance(message, str) else repr(message)
        if self.logger:
            try:
                log_fn = getattr(self.logger, level)
            except AttributeError:
                log_fn = getattr(self.logger, "info", None)
            try:
                if log_fn:
                    log_fn(msg)
            except (TypeError, ValueError):
                # Swallow logging errors to not interrupt workflow
                pass
        elif not self.disable_logs:
            print(f"{level.upper()}: {msg}")

    def error(self, message: str) -> None:
        """
        Log an error message.

        :param message: Error message.
        :type message: str
        """
        self.log(message, level="error")

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        :param message: Debug message.
        :type message: str
        """
        self.log(message, level="debug")

    def _log_external_output(self, tool: str, stream: str, output: Any) -> None:
        """Store stdout/stderr from external tools in the job log."""
        if not output:
            return
        if isinstance(output, (bytes, bytearray)):
            text = output.decode("utf-8", errors="replace")
        else:
            text = str(output)
        text = text.strip()
        if not text:
            return

        if len(text) <= EXTERNAL_LOG_CHUNK_SIZE:
            self.log(f"{tool} {stream}:\n{text}")
            return

        total = (len(text) + EXTERNAL_LOG_CHUNK_SIZE - 1) // EXTERNAL_LOG_CHUNK_SIZE
        for idx, start in enumerate(range(0, len(text), EXTERNAL_LOG_CHUNK_SIZE), 1):
            chunk = text[start : start + EXTERNAL_LOG_CHUNK_SIZE]
            self.log(f"{tool} {stream} ({idx}/{total}):\n{chunk}")

    def tmp_dir(self) -> Path:
        tmp_dir = Path(self.data.config.get("tmp_dir", "tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    @staticmethod
    def _mmseqs_target_to_mapping_key(target_id: str) -> Union[tuple, None]:
        if "::" not in target_id:
            return None
        accession, protein_id = target_id.split("::", 1)
        if not accession or not protein_id:
            return None
        return accession, protein_id

    @staticmethod
    def _lookup_cluster_id(cluster_mapping, mapping_key: tuple) -> Union[str, None]:
        if isinstance(cluster_mapping, Mapping):
            value = None
            accession_lookup = cluster_mapping.get(mapping_key[0])
            if isinstance(accession_lookup, Mapping):
                value = accession_lookup.get(mapping_key[1])
            if value is None:
                value = cluster_mapping.get(mapping_key)
            if value is None or pd.isna(value):
                return None
            return str(value)

        try:
            value = cluster_mapping.loc[mapping_key]
        except KeyError:
            return None

        if isinstance(value, pd.DataFrame):
            if value.empty or "cluster_id" not in value.columns:
                return None
            value = value["cluster_id"].iloc[0]
        elif isinstance(value, pd.Series):
            value = value["cluster_id"] if "cluster_id" in value.index else value.iloc[0]

        if pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _cluster_model_token(cluster_id: str, token_prefix: str = "") -> str:
        token = str(cluster_id)
        if token_prefix and not token.startswith(token_prefix):
            return f"{token_prefix}{token}"
        return token

    @staticmethod
    def _unique(values: list) -> list:
        seen = set()
        out = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @staticmethod
    def _normalise_one(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _feature_vector_for_accession(self, accession: str) -> list:
        """
        Return the ordered cluster/annotation vector for a database plasmid.
        """
        accession_candidates = self.data.candidate_accessions(accession, self.model)
        feature_id = next(
            (
                candidate
                for candidate in accession_candidates
                if candidate in self.data.gbfeatures
            ),
            None,
        )
        if feature_id is None:
            return []

        model_config = self.data.config.get("models", {}).get(self.model, {})
        if model_config.get("type") == "mmseqs2":
            cluster_mapping = self.data.get_cluster_lookup(self.model)
            cluster_accessions = self._unique(
                accession_candidates + self.data.candidate_accessions(feature_id, self.model)
            )
            vector = []
            for prot in self.data.gbfeatures[feature_id].keys():
                cluster_id = None
                for cluster_accession in cluster_accessions:
                    cluster_id = self._lookup_cluster_id(
                        cluster_mapping, (cluster_accession, prot)
                    )
                    if cluster_id is not None:
                        break
                vector.append(cluster_id)
            return vector

        return [
            str(prot["eggnog"]) if prot["eggnog"] is not None else None
            for prot in self.data.gbfeatures[feature_id].values()
        ]

    def _encoded_token_matrix(
        self,
        vector: list,
        transform: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Encode an ordered architecture vector into weighted per-token vectors.
        """
        model = self.data.get_model(self.model)
        model_config = self.data.config.get("models", {}).get(self.model, {})
        token_prefix = model_config.get("cluster_token_prefix", "")
        vocabulary = model.model.wv.key_to_index
        mean_embedding = np.asarray(model.mean_embedding, dtype=np.float32)
        vecs_w, weights = [], []
        n_fallback = 0

        for og in vector:
            raw_og = str(og) if og is not None else None
            token = (
                self._cluster_model_token(raw_og, token_prefix)
                if raw_og is not None
                else None
            )
            if token is not None and token in vocabulary:
                if transform:
                    w = float(model.idf.get(raw_og, model.idf.get(token, 1.0)))
                    vecs_w.append(model.model.wv.get_vector(token, norm=True) * w)
                    weights.append(w)
                else:
                    vecs_w.append(model.model.wv.get_vector(token, norm=True))
                    weights.append(1.0)
            else:
                vecs_w.append(mean_embedding)
                weights.append(1.0)
                n_fallback += 1

        if not vecs_w:
            return (
                np.empty((0, mean_embedding.shape[0]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                0,
            )
        return (
            np.vstack(vecs_w).astype(np.float32, copy=False),
            np.asarray(weights, dtype=np.float32),
            n_fallback,
        )

    @staticmethod
    def _best_sliding_window_score(
        query_embedding: np.ndarray,
        target_vecs_w: np.ndarray,
        target_weights: np.ndarray,
        window_size: int,
        circular: bool = True,
    ) -> Union[dict[str, Any], None]:
        n_tokens = int(target_vecs_w.shape[0])
        if n_tokens == 0:
            return None

        window_size = min(max(1, int(window_size)), n_tokens)
        window_count = n_tokens if circular else max(1, n_tokens - window_size + 1)
        pad = window_size - 1 if circular and window_size > 1 else 0
        if pad:
            vecs = np.vstack([target_vecs_w, target_vecs_w[:pad]])
            weights = np.concatenate([target_weights, target_weights[:pad]])
        else:
            vecs = target_vecs_w
            weights = target_weights

        zero_vec = np.zeros((1, vecs.shape[1]), dtype=np.float32)
        cumsum_vecs = np.vstack([zero_vec, np.cumsum(vecs, axis=0)])
        cumsum_weights = np.concatenate(
            [np.zeros((1,), dtype=np.float32), np.cumsum(weights)]
        )
        starts = np.arange(window_count)
        ends = starts + window_size
        window_sums = cumsum_vecs[ends] - cumsum_vecs[starts]
        weight_sums = cumsum_weights[ends] - cumsum_weights[starts]
        safe_weights = np.where(weight_sums == 0, 1.0, weight_sums)
        window_embeddings = window_sums / safe_weights[:, None]

        query_embedding = PLAST._normalise_one(query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            scores = np.ones(window_count, dtype=np.float32)
        else:
            window_norms = np.linalg.norm(window_embeddings, axis=1, keepdims=True)
            zero_norm = window_norms[:, 0] == 0
            window_embeddings = window_embeddings / np.where(
                zero_norm[:, None], 1.0, window_norms
            )
            scores = window_embeddings @ query_embedding
            if np.any(zero_norm):
                scores = scores.copy()
                scores[zero_norm] = 1.0

        best_idx = int(np.argmax(scores))
        best_start = int(starts[best_idx])
        best_end = best_start + window_size - 1
        wraps = bool(circular and best_start + window_size > n_tokens)
        if circular:
            best_end = best_end % n_tokens

        return {
            "score": float(scores[best_idx]),
            "window_start": best_start,
            "window_end": best_end,
            "window_size": int(window_size),
            "target_orf_count": n_tokens,
            "window_wraps": wraps,
        }

    @staticmethod
    def get_by_accession(
        plast_id: str,
        data: PLASTData,
        model: str,
        load_features: bool = True,
    ) -> "PLAST":
        """
        Get a PLAST object by its accession ID.

        :param plast_id: Accession ID.
        :type plast_id: str
        :param data: Reference PLASTData object.
        :type data: PLASTData
        :param model: Model name.
        :type model: str
        :param load_features: Whether to load ordered ORF features and annotations.
        :type load_features: bool
        :returns: PLAST object for the accession.
        :rtype: PLAST
        :raises NotFoundError: If accession not found.
        """
        accession_id = plast_id.strip().split(".", 1)[0]
        accession_candidates = data.candidate_accessions(accession_id, model)
        model_data = data.get_model(model)
        embedding_id = next(
            (
                candidate
                for candidate in accession_candidates
                if candidate in model_data.embeddings
            ),
            None,
        )
        if embedding_id is None:
            raise NotFoundError(
                f"Embedding for accession {accession_id} not found in model {model}."
            )
        plast = PLAST(data=data, model=model)
        plast.name = embedding_id
        plast.length = int(data.get_metadata_row(embedding_id, model)["length"])
        plast.embedding = model_data.embeddings[embedding_id]
        plast.parsed = pd.DataFrame()
        plast.vector = []

        if not load_features:
            return plast

        try:
            feature_id = next(
                (
                    candidate
                    for candidate in accession_candidates
                    if candidate in data.gbfeatures
                ),
                None,
            )
            if feature_id is None:
                return plast

            plast.parsed = (
                pd.DataFrame.from_dict(data.gbfeatures[feature_id], orient="index")
                .rename_axis("locus_tag")
                .reset_index()
            )
            plast.name = feature_id
            if data.config["models"][model]["type"] == "mmseqs2":
                cluster_mapping = data.get_cluster_lookup(model)
                cluster_accessions = []
                for candidate in data.candidate_accessions(feature_id, model):
                    if candidate not in cluster_accessions:
                        cluster_accessions.append(candidate)

                plast.vector = []
                for prot in data.gbfeatures[feature_id].keys():
                    cluster_id = None
                    for cluster_accession in cluster_accessions:
                        cluster_id = PLAST._lookup_cluster_id(
                            cluster_mapping, (cluster_accession, prot)
                        )
                        if cluster_id is not None:
                            break
                    plast.vector.append(cluster_id)
            else:
                plast.vector = [
                    str(prot["eggnog"]) if prot["eggnog"] is not None else None
                    for prot in data.gbfeatures[feature_id].values()
                ]
        except MemoryError:
            plast.parsed = pd.DataFrame()
            plast.vector = []
            plast.log(
                (
                    "gbfeatures could not be loaded due to memory limits; "
                    f"using stored embedding for accession {accession_id}"
                ),
                level="warning",
            )
        return plast

    def load_gbff(self, path_or_stream: Union[Path, any]) -> "PLAST":
        """
        Load a GBFF file or stream and populate parsed, name, and length.

        :param path_or_stream: Path to GBFF file or stream.
        :type path_or_stream: pathlib.Path or file-like
        :returns: Self.
        :rtype: PLAST
        :raises OSError: On file read error.
        :raises ValueError: On parsing error.
        """
        start_t = time.perf_counter()
        if isinstance(path_or_stream, Path):
            self.log(f"Reading GBFF from file: {path_or_stream}")
            try:
                with open(path_or_stream, "rb") as file:
                    self.parsed, self.name, self.length = read_gbff(file)
            except (OSError, ValueError) as e:
                self.error(f"Failed to read GBFF file '{path_or_stream}': {e}")
                raise
        else:
            self.log("Reading GBFF from stream")
            try:
                self.parsed, self.name, self.length = read_gbff(path_or_stream)
            except (ValueError, TypeError) as e:
                self.error(f"Failed to read GBFF from stream: {e}")
                raise
        if "locus_tag" not in self.parsed.columns:
            self.debug("No locus_tag column found, creating synthetic locus tags")
            self.parsed["locus_tag"] = [
                f"locus_tag_{i}" for i in range(len(self.parsed))
            ]
        cds_count = (
            int((self.parsed["type"] == "CDS").sum())
            if "type" in self.parsed
            else len(self.parsed)
        )
        self.log(
            f"Loaded GBFF: name='{self.name}', length={self.length}, "
            f"CDS={cds_count}, total_records={len(self.parsed)}"
        )
        self.debug(f"load_gbff elapsed: {time.perf_counter() - start_t:.3f}s")
        return self

    def load_nt_fasta(self, input_fasta: Union[str, any]) -> "PLAST":
        """
        Load a nucleotide FASTA, run Prodigal to predict ORFs, and store results in self.parsed.

        :param input_fasta: FASTA file path or string.
        :type input_fasta: str or file-like
        :returns: Self.
        :rtype: PLAST
        :raises OSError: On file read error.
        :raises ValueError: On parsing error.
        :raises UnicodeDecodeError: On encoding error.
        """
        start_t = time.perf_counter()
        try:
            seqs = list(read_fasta(input_fasta).values())
            self.length = len(seqs[0]) if seqs else 0
            self.debug(f"FASTA length detected: {self.length} bp")
        except (OSError, ValueError, UnicodeDecodeError) as e:
            self.error(f"Failed to read FASTA '{input_fasta}': {e}")
            raise

        prodigal = "prodigal"
        with tempfile.NamedTemporaryFile(
            prefix="prodigal_in_",
            suffix=".fasta",
            dir=self.tmp_dir(),
            mode="w+",
        ) as input_file:
            try:
                input_file.write(input_fasta)
            except (TypeError, OSError):
                # If input_fasta is a path, write file contents
                try:
                    with open(input_fasta, "r", encoding="utf-8") as fh:
                        input_file.write(fh.read())
                except (OSError, UnicodeDecodeError) as e:
                    self.error(f"Could not write input FASTA to temp file: {e}")
                    raise
            input_file.flush()
            with tempfile.NamedTemporaryFile(
                prefix="prodigal_out_",
                suffix=".faa",
                dir=self.tmp_dir(),
                mode="w+",
            ) as output_file:

                cmd = (
                    f"{prodigal} -i {input_file.name}"
                    f" -a {output_file.name}"
                    f" -g 11 -q"
                )
                if self.length < 200000:
                    cmd += " -p meta"
                self.debug(f"Running Prodigal: {cmd}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, check=True, capture_output=True, text=True
                    )
                    self._log_external_output("Prodigal", "stdout", result.stdout)
                    self._log_external_output("Prodigal", "stderr", result.stderr)
                except subprocess.CalledProcessError as e:
                    self.error(f"Prodigal failed with return code {e.returncode}")
                    self.error(f"Command: {cmd}")
                    if e.stdout:
                        self.error(f"Stdout: {e.stdout}")
                    if e.stderr:
                        self.error(f"Stderr: {e.stderr}")
                    raise

                records = []
                current_sequence = []
                current_data = {}

                with open(output_file.name, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if current_data:
                                current_data["translation"] = "".join(current_sequence)
                                records.append(current_data)
                            parts = line.split(" # ")
                            current_data = {
                                "locus_tag": parts[0][1:],
                                "start": int(parts[1]),
                                "end": int(parts[2]),
                                "strand": int(parts[3]),
                                "type": "CDS",
                                "gene": "",
                                "product": "",
                            }
                            current_sequence = []
                        else:
                            current_sequence.append(line)

                if current_data:
                    current_data["translation"] = "".join(current_sequence)
                    records.append(current_data)

                self.parsed = pd.DataFrame(records)

        cds_count = len(self.parsed) if self.parsed is not None else 0
        self.log(
            f"Prodigal ORF prediction completed: CDS={cds_count}, "
            f"plasmid_length={self.length} bp"
        )
        self.debug(f"load_nt_fasta elapsed: {time.perf_counter() - start_t:.3f}s")
        return self

    def assign_mmseqs_clusters(self, threads: int = 1, use_gpu: bool = True) -> "PLAST":
        """
        Assign clusters using MMseqs2.

        :param threads: Number of threads to use.
        :type threads: int
        :param use_gpu: Whether to use GPU if available.
        :type use_gpu: bool
        :returns: Self.
        :rtype: PLAST
        :raises Exception: On MMseqs2 failure.
        """
        if self.parsed is None:
            self.error("assign_mmseqs_clusters called but 'parsed' is None")
            return self

        start_t = time.perf_counter()
        analysis_df = (
            self.parsed[self.parsed["type"] == "CDS"]
            .sort_values("start")[["locus_tag", "translation"]]
            .set_index("locus_tag")
        )
        gpu_enabled = use_gpu and is_gpu_available()
        self.log(
            f"Running MMseqs2 easy-search on {len(analysis_df)} CDS "
            f"(threads={threads}, gpu={gpu_enabled})"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.tmp_dir(), prefix="mmseqs_search_"
        ) as faa_input:
            for locus, sequence in analysis_df.iterrows():
                faa_input.write(f">{locus}\n{sequence['translation']}\n")
            faa_input.flush()
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.tmp_dir(), prefix="mmseqs_output_"
            ) as mmseqs_output:
                with tempfile.TemporaryDirectory(
                    dir=self.tmp_dir()
                ) as temp_dir:
                    cmd = [
                        "mmseqs",
                        "easy-search",
                        faa_input.name,
                        self.data.get_mmseqs_db(self.model, padded=gpu_enabled),
                        mmseqs_output.name,
                        temp_dir,
                        "--search-type",
                        "3",
                        "-c",
                        "0.8",
                        "--min-seq-id",
                        "0.3",
                        "-s",
                        "7.5",
                        "--threads",
                        str(threads),
                    ] + (["--gpu", "1"] if gpu_enabled else [])
                    self.debug(f"MMseqs2 command: {' '.join(cmd)}")
                    try:
                        sb = subprocess.run(
                            cmd,
                            capture_output=True,
                            check=True,
                        )
                        self._log_external_output("MMseqs2", "stdout", sb.stdout)
                        self._log_external_output("MMseqs2", "stderr", sb.stderr)
                    except subprocess.CalledProcessError as e:
                        # Break long message and avoid broad except
                        try:
                            out = (
                                e.output.decode("ASCII", errors="ignore")
                                if e.output
                                else ""
                            )
                        except (AttributeError, UnicodeError):
                            out = ""
                        try:
                            err = (
                                e.stderr.decode("ASCII", errors="ignore")
                                if e.stderr
                                else ""
                            )
                        except (AttributeError, UnicodeError):
                            err = ""
                        self.error(f"MMseqs2 search failed: {e}")
                        if out:
                            self.error(f"stdout: {out}")
                        if err:
                            self.error(f"stderr: {err}")
                        raise

                    cluster_mapping = self.data.get_cluster_lookup(self.model)
                    target_ids = read_m8(
                        mmseqs_output.name, res_per_query=1
                    ).reindex(analysis_df.index)["target_id"]
                    self.vector = []
                    for target_id in target_ids:
                        cluster_id = None
                        if pd.notna(target_id):
                            mapping_key = self._mmseqs_target_to_mapping_key(
                                str(target_id)
                            )
                            if mapping_key is not None:
                                cluster_id = self._lookup_cluster_id(
                                    cluster_mapping, mapping_key
                                )
                            if cluster_id is None:
                                self.debug(
                                    f"No cluster mapping found for MMseqs target_id={target_id}"
                                )
                        self.vector.append(cluster_id)
                    n_none = sum(1 for v in self.vector if v is None)
                    self.log(
                        f"MMseqs2 assignment completed: assigned={len(self.vector) - n_none}, "
                        f"missing={n_none}, total={len(self.vector)}"
                    )
                    self.debug(
                        f"assign_mmseqs_clusters elapsed: {time.perf_counter() - start_t:.3f}s"
                    )
                    return self

    def assign_eggnog_annot(self, processes: int = 4) -> "PLAST":
        """
        Assign HMMscan annotations to the query plasmid by running multiple hmmscan
        processes in parallel (each using one CPU core) on subsets of the CDS translations.

        :param processes: Number of parallel processes.
        :type processes: int
        :returns: Self.
        :rtype: PLAST
        """
        db = Path(self.data.config["hmmscan_db"])
        files = [db] + [
            db.with_suffix(db.suffix + s) for s in (".h3m", ".h3i", ".h3f", ".h3p")
        ]
        warm_page_cache(
            files,
            threshold=0.9,
            state_dir=self.tmp_dir(),
            logger=self.logger,
        )

        if self.parsed is None:
            self.error("assign_eggnog_annot called but 'parsed' is None")
            return self

        start_t = time.perf_counter()
        analysis_df = (
            self.parsed[self.parsed["type"] == "CDS"]
            .sort_values("start")[["locus_tag", "translation"]]
            .set_index("locus_tag")
        )

        if processes < 1:
            processes = 1
        self.log(
            f"Running hmmscan on {len(analysis_df)} CDS with {processes} process(es)"
        )

        bounds = np.linspace(0, len(analysis_df), processes + 1, dtype=int)
        subsets = [
            analysis_df.iloc[bounds[i] : bounds[i + 1]] for i in range(processes)
        ]

        with tempfile.TemporaryDirectory(dir=self.tmp_dir()) as temp_main:
            hmmscan_processes = []
            output_files = []
            for i, subset in enumerate(subsets):
                input_path = Path(temp_main) / f"hmmscan_input_{i}.faa"
                output_path = Path(temp_main) / f"hmmscan_output_{i}.hmmout"
                with open(input_path, "w", encoding="utf-8") as faa_input:
                    for locus, row in subset.iterrows():
                        faa_input.write(f">{locus}\n{row['translation']}\n")
                cmd = [
                    "hmmscan",
                    "--cpu",
                    "1",
                    "--noali",
                    "--domtblout",
                    str(output_path),
                    self.data.config["hmmscan_db"],
                    str(input_path),
                ]
                self.debug(f"Spawning hmmscan[{i}]: {' '.join(cmd)}")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except OSError as e:
                    self.error(f"Failed to start hmmscan[{i}]: {e}")
                    continue
                hmmscan_processes.append((proc, output_path))
                output_files.append(output_path)

            all_dfs = []
            for i, (proc, out_path) in enumerate(hmmscan_processes):
                stdout, stderr = proc.communicate()
                self._log_external_output(f"hmmscan[{i}]", "stdout", stdout)
                self._log_external_output(f"hmmscan[{i}]", "stderr", stderr)
                ret = proc.returncode
                if ret != 0:
                    self.error(f"hmmscan[{i}] exited with code {ret}")
                    continue
                try:
                    df_part = read_hmmscan_output(str(out_path))
                    self.debug(f"hmmscan[{i}] parsed hits: {len(df_part)}")
                    all_dfs.append(df_part)
                except (OSError, ValueError) as e:
                    self.error(f"Failed to parse hmmscan output[{i}] ({out_path}): {e}")

            if all_dfs:
                full_df = pd.concat(all_dfs, ignore_index=True)
            else:
                full_df = pd.DataFrame()

            hits = (
                full_df.set_index("query_name")["hit"]
                .reindex(analysis_df.index)
                .replace(np.nan, None)
                .to_list()
            )

            self.vector = hits
            n_none = sum(1 for v in self.vector if v is None)
            self.log(
                f"HMMscan assignment completed: assigned={len(self.vector) - n_none}, "
                f"missing={n_none}, total={len(self.vector)}"
            )
            self.debug(
                f"assign_eggnog_annot elapsed: {time.perf_counter() - start_t:.3f}s"
            )
            return self

    def encode(
        self,
        normalize: bool = True,
        transform: bool = True,
        inplace: bool = True,
        return_mask: bool = False,
    ) -> Union["PLAST", np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Encode the plasmid query into an embedding vector.

        :param normalize: Whether to normalize the embedding vector.
        :type normalize: bool
        :param transform: Whether to apply weighting transformation.
        :type transform: bool
        :param inplace: If True, store the embedding in the object and return self.
        :type inplace: bool
        :param return_mask: If True, also return a mask of missing vectors.
        :type return_mask: bool
        :returns: If inplace is True, returns self. Otherwise, returns the embedding vector
            (and mask if return_mask is True).
        :rtype: PLAST or numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        :raises ValueError: If return_mask is True and inplace is True.
        :raises KeyError: On encoding failure.
        :raises AttributeError: On encoding failure.
        :raises ValueError: On encoding failure.
        :raises TypeError: On encoding failure.
        """
        self.debug(
            f"Encoding started (model='{self.model}', normalize={normalize}, "
            f"transform={transform}, inplace={inplace})"
        )
        start_t = time.perf_counter()
        try:
            model = self.data.get_model(self.model)
            model_config = self.data.config.get("models", {}).get(self.model, {})
            token_prefix = model_config.get("cluster_token_prefix", "")
            vocabulary = model.model.wv.key_to_index
            mask_missing = []
            n_fallback = 0
            vecs_w, weights = [], []
            for og in self.vector:
                raw_og = str(og) if og is not None else None
                token = (
                    self._cluster_model_token(raw_og, token_prefix)
                    if raw_og is not None
                    else None
                )
                if transform:
                    if token is not None and token in vocabulary:
                        w = model.idf.get(raw_og, model.idf.get(token, 1.0))
                        mask_missing.append(False)
                        vecs_w.append(model.model.wv.get_vector(token, norm=True) * w)
                        weights.append(w)
                    else:
                        mask_missing.append(True)
                        vecs_w.append(model.mean_embedding)
                        weights.append(1.0)
                        n_fallback += 1
                else:
                    if token is not None and token in vocabulary:
                        vecs_w.append(model.model.wv.get_vector(token, norm=True))
                        mask_missing.append(False)
                    else:
                        vecs_w.append(model.mean_embedding)
                        mask_missing.append(True)
                        n_fallback += 1
            if transform:
                emb = np.vstack(vecs_w).sum(0) / np.sum(weights)
            else:
                emb = np.array(vecs_w)
            if normalize:
                emb_norm = np.linalg.norm(emb)
                if emb_norm == 0:
                    self.error("Zero-norm embedding encountered during normalization")
                else:
                    emb /= emb_norm
            self.log(f"pLAST: {n_fallback} fallback vectors used for {self.name}")
            self.debug(
                f"Encoding finished in {time.perf_counter() - start_t:.3f}s; emb_shape={emb.shape}"
            )
            if inplace:
                if return_mask:
                    raise ValueError("Cannot return mask when inplace=True")
                self.embedding = emb.astype(np.float32)
                return self
            if return_mask:
                return emb.astype(np.float32), np.array(mask_missing)
            return emb.astype(np.float32)
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            self.error(f"Encoding failed: {e}")
            raise

    def _metadata_results_from_scores(
        self,
        scores_by_id: dict[str, float],
        extra_by_id: Union[dict[str, dict[str, Any]], None] = None,
    ) -> dict:
        result_columns = [
            "pLAST_distance",
            "accession",
            "length",
            "gc",
            "database",
            "topology",
            "taxid",
            "organism",
            "definition",
            "taxonomy",
            "rep_type(s)",
            "AMR",
            "relaxase_type(s)",
            "mpf_type",
            "orit_type(s)",
            "predicted_mobility",
            "primary_cluster_id",
        ]
        extra_by_id = extra_by_id or {}
        extra_columns = []
        for extra in extra_by_id.values():
            for column in extra:
                if column not in extra_columns:
                    extra_columns.append(column)

        ordered_ids = list(scores_by_id.keys())
        reduced_df = self.data.get_metadata_for_ids(ordered_ids, self.model)
        resolved_ids = [pid for pid in ordered_ids if pid in reduced_df.index]
        missing_ids = [pid for pid in ordered_ids if pid not in reduced_df.index]

        reduced_df = reduced_df.loc[resolved_ids].copy()
        if missing_ids:
            missing_df = pd.DataFrame(index=missing_ids)
            reduced_df = pd.concat([reduced_df, missing_df], axis=0)
            self.debug(
                f"Metadata missing for {len(missing_ids)} ranked results: "
                + ", ".join(missing_ids[:5])
            )
        if ordered_ids:
            reduced_df = reduced_df.loc[ordered_ids]

        reduced_df["pLAST_distance"] = [
            scores_by_id[pid] for pid in reduced_df.index
        ]
        for column in result_columns:
            if column not in reduced_df.columns:
                reduced_df[column] = None
        for column in extra_columns:
            reduced_df[column] = [
                extra_by_id.get(pid, {}).get(column) for pid in reduced_df.index
            ]

        return reduced_df[result_columns + extra_columns].to_dict(orient="index")

    def get_most_similar(self, maxret: int = 10, metric: str = "cosine") -> dict:
        """
        Compute most similar plasmids to the current embedding.

        :param maxret: Maximum number of results to return.
        :type maxret: int
        :param metric: Similarity metric ('cosine' or 'euclidean').
        :type metric: str
        :returns: Dictionary of most similar plasmids and their metadata.
        :rtype: dict
        :raises ValueError: If unknown metric is provided.
        """
        self.debug(f"get_most_similar called (maxret={maxret}, metric='{metric}')")
        start_t = time.perf_counter()
        if metric not in {"cosine", "euclidean"}:
            self.error(f"Unknown metric: {metric}")
            raise ValueError(f"Unknown metric: {metric}")

        search_index = self.data.get_search_index(self.model)
        candidate_count = len(search_index["ids"])
        self.debug(
            f"Similarity candidates for model '{self.model}': "
            f"used={candidate_count}, "
            f"skipped_by_metadata_index={search_index.get('skipped_by_metadata_index', 0)}"
        )

        if candidate_count == 0:
            reduced = {}
        else:
            query_embedding = np.asarray(self.embedding, dtype=np.float32)

            if metric == "cosine":
                query_norm = np.linalg.norm(query_embedding)
                if query_norm == 0:
                    scores = np.ones(candidate_count, dtype=np.float32)
                else:
                    scores = search_index["matrix_norm"] @ (
                        query_embedding / query_norm
                    )
                    zero_mask = search_index.get("zero_norm_mask")
                    if zero_mask is not None and np.any(zero_mask):
                        scores = scores.copy()
                        scores[zero_mask] = 1.0
                reverse = True
            else:
                query_sq_norm = float(np.dot(query_embedding, query_embedding))
                squared = (
                    search_index["matrix_sq_norms"]
                    + query_sq_norm
                    - 2.0 * (search_index["matrix"] @ query_embedding)
                )
                scores = np.sqrt(np.maximum(squared, 0.0))
                reverse = False

            limit = min(max(1, int(maxret)), candidate_count)
            if limit == candidate_count:
                sort_scores = -scores if reverse else scores
                selected = np.argsort(sort_scores, kind="stable")
            elif reverse:
                selected = np.argpartition(-scores, limit - 1)[:limit]
                selected = selected[np.argsort(-scores[selected], kind="stable")]
            else:
                selected = np.argpartition(scores, limit - 1)[:limit]
                selected = selected[np.argsort(scores[selected], kind="stable")]
            ids = search_index["ids"]
            reduced = {ids[int(i)]: float(scores[int(i)]) for i in selected}

        results = self._metadata_results_from_scores(reduced)
        self.results = results
        elapsed = time.perf_counter() - start_t
        self.log(
            f"get_most_similar finished in {elapsed:.3f}s, "
            f"returned {len(results)} results"
        )
        return results

    def module_search(
        self,
        maxret: int = 10,
        metric: str = "cosine",
        window_size: Union[int, None] = None,
        circular: bool = True,
        transform: bool = True,
        progress_callback: Union[Callable[[int, int], None], None] = None,
    ) -> dict:
        """
        Search for the current query as a local protein module in database plasmids.

        Each database plasmid is represented as ordered protein-cluster embeddings.
        The query is compared against query-sized sliding windows and the plasmid
        score is the maximum cosine similarity over its windows.
        """
        self.debug(
            f"module_search called (maxret={maxret}, metric='{metric}', "
            f"window_size={window_size}, circular={circular})"
        )
        start_t = time.perf_counter()
        if metric != "cosine":
            self.error(f"module_search supports only cosine metric, got: {metric}")
            raise ValueError("module_search supports only cosine metric.")
        if not self.vector:
            raise ValueError("module_search requires an ordered query vector.")

        query_window_size = int(window_size or len(self.vector))
        if query_window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if self.embedding is None:
            query_embedding = self.encode(
                normalize=True,
                transform=transform,
                inplace=False,
            )
        else:
            query_embedding = np.asarray(self.embedding, dtype=np.float32)
        query_embedding = self._normalise_one(query_embedding)

        search_index = self.data.get_search_index(self.model)
        candidate_ids = search_index["ids"]
        self.debug(
            f"Module search candidates for model '{self.model}': "
            f"{len(candidate_ids)}"
        )
        candidate_total = len(candidate_ids)
        if progress_callback is not None:
            progress_callback(0, candidate_total)
        progress_step = max(1, candidate_total // 100) if candidate_total else 1

        scored = []
        extra_by_id = {}
        skipped_no_vector = 0
        for processed, plasmid_id in enumerate(candidate_ids, start=1):
            target_vector = self._feature_vector_for_accession(plasmid_id)
            if not target_vector:
                skipped_no_vector += 1
                if (
                    progress_callback is not None
                    and (processed % progress_step == 0 or processed == candidate_total)
                ):
                    progress_callback(processed, candidate_total)
                continue

            target_vecs_w, target_weights, n_fallback = self._encoded_token_matrix(
                target_vector,
                transform=transform,
            )
            best = self._best_sliding_window_score(
                query_embedding,
                target_vecs_w,
                target_weights,
                query_window_size,
                circular=circular,
            )
            if best is None:
                skipped_no_vector += 1
                if (
                    progress_callback is not None
                    and (processed % progress_step == 0 or processed == candidate_total)
                ):
                    progress_callback(processed, candidate_total)
                continue

            score = float(best.pop("score"))
            scored.append((plasmid_id, score))
            best["module_fallback_vectors"] = int(n_fallback)
            extra_by_id[plasmid_id] = best
            if (
                progress_callback is not None
                and (processed % progress_step == 0 or processed == candidate_total)
            ):
                progress_callback(processed, candidate_total)

        limit = min(max(1, int(maxret)), len(scored)) if scored else 0
        selected = sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
        reduced = {plasmid_id: score for plasmid_id, score in selected}
        selected_extra = {plasmid_id: extra_by_id[plasmid_id] for plasmid_id in reduced}
        results = self._metadata_results_from_scores(reduced, selected_extra)
        self.results = results
        elapsed = time.perf_counter() - start_t
        self.log(
            f"module_search finished in {elapsed:.3f}s, "
            f"returned {len(results)} results, skipped_no_vector={skipped_no_vector}"
        )
        return results

    def draw_network(self, closest_num: int = 3) -> dict:
        """
        Compute network coordinates for the query and closest_num results.

        :param closest_num: Number of closest results to include.
        :type closest_num: int
        :returns: Dictionary with network coordinates and labels.
        :rtype: dict
        """
        self.debug(f"draw_network called (closest_num={closest_num})")
        model = self.data.get_model(self.model)
        result_ids = list(self.results.keys())
        if not result_ids:
            self.debug("draw_network called with no results")
            return {
                "x": [0],
                "y": [0],
                "ids": ["query"],
                "label": ["query"],
            }
        search_index = self.data.get_search_index(self.model)
        id_to_position = search_index["id_to_embedding_position"]
        plasmid_ids = [
            plasmid_id for plasmid_id in result_ids if plasmid_id in id_to_position
        ]
        indexes = [id_to_position[plasmid_id] for plasmid_id in plasmid_ids]
        pos_closest = [
            id_to_position[plasmid_id]
            for plasmid_id in result_ids[:closest_num]
            if plasmid_id in id_to_position
        ]
        umap_x = model.umap_mapper[:, 0][indexes].tolist()
        umap_y = model.umap_mapper[:, 1][indexes].tolist()

        if not pos_closest:
            self.debug("draw_network found no UMAP coordinates for closest results")
            return {
                "x": [0] + umap_x,
                "y": [0] + umap_y,
                "ids": ["query"] + plasmid_ids,
                "label": ["query"] + plasmid_ids,
            }

        closest_coords = zip(
            model.umap_mapper[:, 0][pos_closest],
            model.umap_mapper[:, 1][pos_closest],
        )
        query_x, query_y = [sum(x) / len(x) for x in zip(*closest_coords)]

        metadata_df = self.data.get_metadata_for_ids(plasmid_ids, self.model)
        if not metadata_df.empty and "plasmid_name" in metadata_df.columns:
            id_to_label = metadata_df["plasmid_name"].replace("-", "").to_dict()
        else:
            id_to_label = {}

        labels = [id_to_label.get(pid, pid) for pid in plasmid_ids]

        results = {
            "x": [query_x] + umap_x,
            "y": [query_y] + umap_y,
            "ids": ["query"] + plasmid_ids,
            "label": ["query"] + labels,
        }
        self.debug(f"draw_network produced {len(plasmid_ids)} nodes (excluding query)")
        return results
