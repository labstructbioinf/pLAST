"""
This module implements the core classes and methods for the PLAST (Plasmid Search Tool)
"""

import time
from pathlib import Path
from typing import Union
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
    cosine_similarity_numba,
    euclidean,
    warm_page_cache,
)


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
        self.file = None
        self.logger = logger
        self.disable_logs = False
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
        new_plast.file = self.file
        new_plast.disable_logs = self.disable_logs
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

    def tmp_dir(self) -> Path:
        tmp_dir = Path(self.data.config.get("tmp_dir", "tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    @staticmethod
    def get_by_accession(plast_id: str, data: PLASTData, model: str) -> "PLAST":
        """
        Get a PLAST object by its accession ID.

        :param plast_id: Accession ID.
        :type plast_id: str
        :param data: Reference PLASTData object.
        :type data: PLASTData
        :param model: Model name.
        :type model: str
        :returns: PLAST object for the accession.
        :rtype: PLAST
        :raises NotFoundError: If accession not found.
        """
        plast_id = plast_id.strip().split(".")[0]
        if plast_id not in data.gbfeatures:
            raise NotFoundError(f"Plasmid with accession {plast_id} not found.")
        plast = PLAST(data=data, model=model)
        plast.parsed = (
            pd.DataFrame.from_dict(data.gbfeatures[plast_id], orient="index")
            .rename_axis("locus_tag")
            .reset_index()
        )
        plast.name = plast_id
        plast.length = int(data.metadata.loc[plast_id, "length"])
        if data.config["models"][model]["type"] == "mmseqs2":
            plast.vector = [
                str(el) if el is not None else None
                for el in [
                    data.cluster_mapping.loc[(plast_id, prot)].item()
                    for prot in data.gbfeatures[plast_id].keys()
                ]
            ]
        else:
            plast.vector = [
                str(prot["eggnog"]) if prot["eggnog"] is not None else None
                for prot in data.gbfeatures[plast_id].values()
            ]
        plast.embedding = data.get_model(model).embeddings[plast_id]
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
                    f" -g 11 -p multi -q"
                )
                if self.length < 100000:
                    cmd += " -p meta"
                self.debug(f"Running Prodigal: {cmd}")
                try:
                    result = subprocess.run(
                        cmd, shell=True, check=True, capture_output=True, text=True
                    )
                    if result.stdout:
                        self.log(f"Prodigal stdout: {result.stdout}")
                    if result.stderr:
                        self.debug(f"Prodigal stderr: {result.stderr}")
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
        self.log(
            f"Running MMseqs2 easy-search on {len(analysis_df)} CDS "
            f"(threads={threads}, gpu={use_gpu and is_gpu_available()})"
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
                        (
                            self.data.config["mmseqs_db_padded"]
                            if use_gpu and is_gpu_available()
                            else self.data.config["mmseqs_db"]
                        ),
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
                    ] + (["--gpu", "1"] if (is_gpu_available() & use_gpu) else [])
                    self.debug(f"MMseqs2 command: {' '.join(cmd)}")
                    try:
                        sb = subprocess.run(
                            cmd,
                            capture_output=True,
                            check=True,
                        )
                        if sb.stdout:
                            self.debug(sb.stdout.decode("ASCII", errors="ignore"))
                        if sb.stderr:
                            # MMseqs often writes progress to stderr; log as debug
                            self.debug(sb.stderr.decode("ASCII", errors="ignore"))
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

                    self.vector = [
                        (
                            str(
                                self.data.cluster_mapping.loc[tuple(x.split("___"))][
                                    "cluster_id"
                                ]
                            )
                            if pd.notna(x)
                            else None
                        )
                        for x in read_m8(mmseqs_output.name, res_per_query=1).reindex(
                            analysis_df.index
                        )["target_id"]
                    ]
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
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except OSError as e:
                    self.error(f"Failed to start hmmscan[{i}]: {e}")
                    continue
                hmmscan_processes.append((proc, output_path))
                output_files.append(output_path)

            all_dfs = []
            for i, (proc, out_path) in enumerate(hmmscan_processes):
                ret = proc.wait()
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
            mask_missing = []
            n_fallback = 0
            vecs_w, weights = [], []
            for og in self.vector:
                if transform:
                    if isinstance(og, str) and og in model.model.wv.index_to_key:
                        w = model.idf.get(og, 1.0)
                        mask_missing.append(False)
                        vecs_w.append(model.model.wv.get_vector(og, norm=True) * w)
                        weights.append(w)
                    else:
                        mask_missing.append(True)
                        vecs_w.append(model.mean_embedding)
                        weights.append(1.0)
                        n_fallback += 1
                else:
                    if isinstance(og, str) and og in model.model.wv.index_to_key:
                        vecs_w.append(model.model.wv.get_vector(og, norm=True))
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
        model = self.data.get_model(self.model)
        dists = {}
        for pid, emb in model.embeddings.items():
            if metric == "cosine":
                dist = cosine_similarity_numba(self.embedding, emb)
            elif metric == "euclidean":
                dist = euclidean(self.embedding, emb)
            else:
                self.error(f"Unknown metric: {metric}")
                raise ValueError(f"Unknown metric: {metric}")
            dists[pid] = dist
        dists = {
            k: v
            for k, v in sorted(
                dists.items(),
                key=lambda item: item[1],
                reverse=(True if metric == "cosine" else False),
            )
        }
        reduced = dict(list(dists.items())[:maxret])
        reduced_df = self.data.metadata.loc[reduced.keys()]
        reduced_df["pLAST_distance"] = reduced.values()
        results = reduced_df[
            [
                "pLAST_distance",
                "length",
                "gc",
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
        ].to_dict(orient="index")
        self.results = results
        elapsed = time.perf_counter() - start_t
        self.log(
            f"get_most_similar finished in {elapsed:.3f}s, "
            f"returned {len(results)} results"
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
        ids = list(self.results.keys())
        closest = ids[:closest_num]
        ids = set(ids)
        pos_key = {}
        pos_closest = []
        for i, k in enumerate(model.embeddings):
            if k in ids:
                pos_key[i] = k
                if k in closest:
                    pos_closest.append(i)

        indexes = list(pos_key.keys())
        umap_x = model.umap_mapper[:, 0][indexes].tolist()
        umap_y = model.umap_mapper[:, 1][indexes].tolist()

        closest_coords = zip(
            model.umap_mapper[:, 0][pos_closest],
            model.umap_mapper[:, 1][pos_closest],
        )
        query_x, query_y = [sum(x) / len(x) for x in zip(*closest_coords)]

        plasmid_ids = list(pos_key.values())
        existing_ids = [pid for pid in plasmid_ids if pid in self.data.metadata.index]
        if existing_ids:
            plasmid_names = self.data.metadata.loc[
                existing_ids, "plasmid_name"
            ].replace("-", "")
            id_to_label = {pid: name for pid, name in zip(existing_ids, plasmid_names)}
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
