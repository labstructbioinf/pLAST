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
from sklearn.cluster import DBSCAN
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
    """Class representing a plasmid query for pLAST."""

    def __init__(self, data: PLASTData, model: Union[str, None] = None, logger=None):
        """Initialize the PLAST query object."""
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
        self.debugging = False
        self.disable_logs = False
        self.debug(f"Initialized PLAST(name='{self.name}', model='{self.model}')")

    def log(self, message: str, level: str = "info"):
        """Log a message via self.logger if available, else stdout."""
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

    def error(self, message: str):
        """Log an error message."""
        self.log(message, level="error")

    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, level="debug")

    def copy(self) -> "PLAST":
        """
        Create a copy of the PLAST object.
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
        new_plast.debugging = self.debugging
        new_plast.disable_logs = self.disable_logs
        return new_plast

    @staticmethod
    def get_by_accession(plast_id: str, data: PLASTData, model: str) -> "PLAST":
        """
        Get a PLAST object by its accession ID.
        """
        plast_id = plast_id.strip().split(".")[0]
        if plast_id not in data.gbfeatures:
            raise NotFoundError(f"Plasmid with accession {plast_id} not found.")
        plast = PLAST(data=data)
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

    def load_gbff(self, path_or_stream):
        """Load a GBFF file or stream and populate parsed, name, and length."""
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

    def load_nt_fasta(self, input_fasta):
        """
        Load a nucleotide FASTA, run Prodigal to predict ORFs, and store results in self.parsed.
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
            dir=self.data.config["tmp_dir"],
            mode="w+",
            delete=not self.debugging,
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
                dir=self.data.config["tmp_dir"],
                mode="w+",
                delete=not self.debugging,
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

    def encode(
        self, normalize: bool = True, transform: bool = True, inplace=True
    ) -> np.ndarray:
        """Encode the plasmid query into an embedding vector."""
        self.debug(
            f"Encoding started (model='{self.model}', normalize={normalize}, "
            f"transform={transform}, inplace={inplace})"
        )
        start_t = time.perf_counter()
        try:
            model = self.data.get_model(self.model)
            n_fallback = 0
            vecs_w, weights = [], []
            for og in self.vector:
                if transform:
                    if isinstance(og, str) and og in model.model.wv.index_to_key:
                        w = model.idf.get(og, 1.0)
                        vecs_w.append(model.model.wv.get_vector(og, norm=True) * w)
                        weights.append(w)
                    else:
                        vecs_w.append(model.mean_embedding)
                        weights.append(1.0)
                        n_fallback += 1
                else:
                    if isinstance(og, str) and og in model.model.wv.index_to_key:
                        vecs_w.append(model.model.wv.get_vector(og, norm=True))
                    else:
                        vecs_w.append(model.mean_embedding)
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
                self.embedding = emb.astype(np.float32)
                return self
            return emb.astype(np.float32)
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            self.error(f"Encoding failed: {e}")
            raise

    def to_dict(self):
        """
        Return the query as a dictionary.
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

    def to_json(self, filename: Union[str, None] = None):
        """
        Return the query as a JSON object.
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
        """Load the query from a JSON string or Path."""
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

    def assign_mmseqs_clusters(self, threads: int = 1, use_gpu: bool = True):
        """Assign clusters using MMseqs2."""
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
            mode="w", dir=self.data.config["tmp_dir"], prefix="mmseqs_search_"
        ) as faa_input:
            for locus, sequence in analysis_df.iterrows():
                faa_input.write(f">{locus}\n{sequence['translation']}\n")
            faa_input.flush()
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.data.config["tmp_dir"], prefix="mmseqs_output_"
            ) as mmseqs_output:
                with tempfile.TemporaryDirectory(
                    dir=self.data.config["tmp_dir"]
                ) as temp_dir:
                    cmd = [
                        "mmseqs",
                        "easy-search",
                        faa_input.name,
                        (
                            self.data.config["mmseqs_db_padded"]
                            if is_gpu_available()
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

    def assign_eggnog_annot(self, processes: int = 4):
        """
        Assign HMMscan annotations to the query plasmid by running multiple hmmscan
        processes in parallel (each using one CPU core) on subsets of the CDS translations.
        """
        db = Path(self.data.config["hmmscan_db"])
        files = [db] + [
            db.with_suffix(db.suffix + s) for s in (".h3m", ".h3i", ".h3f", ".h3p")
        ]
        warm_page_cache(
            files,
            threshold=0.9,
            state_dir=self.data.config.get("tmp_dir"),
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

        with tempfile.TemporaryDirectory(dir=self.data.config["tmp_dir"]) as temp_main:
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

    @staticmethod
    def cluster_orf_matches(emb1, emb2, eps=0.2, min_samples=2):
        """
        Cluster ORF embeddings from two plasmids using DBSCAN (cosine distance).

        Args:
          emb1: np.ndarray, shape (n1, d) — embeddings for plasmid1 (query)
          emb2: np.ndarray, shape (n2, d) — embeddings for plasmid2 (target)
          eps: float — DBSCAN eps parameter (cosine distance)
          min_samples: int — DBSCAN min_samples

        Returns:
          tuple:
            - results: dict mapping "cluster_<label>" -> {"query": [indices in emb1],
                                                        "target": [indices in emb2]}
              (includes noise cluster with label -1, represented as "cluster_-1")
            - labels: np.ndarray of cluster labels for the concatenated embeddings
        """
        # merge embeddings
        emb_all = np.vstack((emb1, emb2))
        n1 = emb1.shape[0]
        # run DBSCAN with cosine distance
        db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit(emb_all)
        labels = db.labels_

        results = {}
        for cluster_id in sorted(set(labels)):
            idxs = np.where(labels == cluster_id)[0]
            orfs1 = [int(i) for i in idxs if i < n1]
            orfs2 = [int(i - n1) for i in idxs if i >= n1]
            results["cluster_" + str(cluster_id)] = {"query": orfs1, "target": orfs2}
        return results, labels

    @staticmethod
    def rotate_parsed(pl: "PLAST", shift: int, reverse: bool = False) -> "PLAST":
        """
        Rotates the parsed CDS records (a DataFrame) by 'shift' positions (circularly)
        and reassigns the coordinates so that the first record in the rotated list starts at 1.

        If reverse=True:
          - reverse the order of elements first,
          - flip coordinates to reversed orientation:
                start' = L - end + 1
                end'   = L - start + 1
          - invert strand (if present): strand' = -strand
          - then apply the rotation by 'shift'
        """
        pl.debug(
            f"rotate_parsed called (shift={shift}, reverse={reverse}, "
            f"records={0 if pl.parsed is None else len(pl.parsed)}, "
            f"length={pl.length})"
        )
        if pl.parsed is None or (
            isinstance(pl.parsed, pd.DataFrame) and pl.parsed.empty
        ):
            return pl

        # Preserve original index (text labels) and its name
        orig_index = pl.parsed.index.tolist()
        orig_index_name = pl.parsed.index.name

        # Convert DataFrame to list of records for manipulation
        records = pl.parsed.to_dict(orient="records")
        n = len(records)
        if n == 0:
            return pl

        # Normalize shift (support negative and large shifts)
        shift = shift % n

        plen = int(pl.length)
        new_parsed = []

        if not reverse:
            # Rotate in the original orientation
            rotated = records[shift:] + records[:shift]
            rotated_index = orig_index[shift:] + orig_index[:shift]

            # Rebase so that the first record starts at 1
            offset = int(rotated[0]["start"]) - 1
            for new_index, record in enumerate(rotated):
                new_record = record.copy()
                new_record["index"] = new_index

                new_start = int(record["start"]) - offset
                new_end = int(record["end"]) - offset

                if new_start < 1:
                    new_start += plen
                    new_end += plen
                if new_start > plen:
                    new_start -= plen
                    new_end -= plen

                new_record["start"] = int(new_start)
                new_record["end"] = int(new_end)
                new_parsed.append(new_record)
        else:
            # 1) Reverse order and index first
            reversed_records = records[::-1]
            reversed_index = orig_index[::-1]

            # 2) Flip coordinates and strand into reversed orientation
            flipped_records = []
            for record in reversed_records:
                rec = record.copy()
                rev_start = plen - int(rec["end"]) + 1
                rev_end = plen - int(rec["start"]) + 1
                rec["start"] = int(rev_start)
                rec["end"] = int(rev_end)
                if "strand" in rec and pd.notna(rec["strand"]):
                    try:
                        rec["strand"] = -int(rec["strand"])
                    except (ValueError, TypeError):
                        pass
                flipped_records.append(rec)

            # 3) Apply rotation after reversing/flipping
            rotated = flipped_records[shift:] + flipped_records[:shift]
            rotated_index = reversed_index[shift:] + reversed_index[:shift]

            # 4) Rebase so the first record starts at 1
            offset = int(rotated[0]["start"]) - 1
            for new_index, record in enumerate(rotated):
                new_record = record.copy()
                new_record["index"] = new_index

                new_start = int(record["start"]) - offset
                new_end = int(record["end"]) - offset

                if new_start < 1:
                    new_start += plen
                    new_end += plen
                if new_start > plen:
                    new_start -= plen
                    new_end -= plen

                new_record["start"] = int(new_start)
                new_record["end"] = int(new_end)
                new_parsed.append(new_record)

        # Build DataFrame and restore the original text index in the rotated order
        new_df = pd.DataFrame(new_parsed)
        new_df.index = pd.Index(rotated_index, name=orig_index_name)
        pl.parsed = new_df
        pl.debug(
            f"rotate_parsed finished (shift={shift}, reverse={reverse}, records={len(pl.parsed)})"
        )
        return pl

    @staticmethod
    def rotate_plasmid(plast: "PLAST", shift: int, reverse: bool = False) -> "PLAST":
        """
        Returns a new PLAST object with the vector rotated by 'shift' positions.
        If reverse=True, the order of elements is reversed (and parsed is flipped accordingly).
        """
        plast.debug(f"rotate_plasmid called (shift={shift}, reverse={reverse})")
        rotated = plast.copy()

        rotated = PLAST.rotate_parsed(rotated, shift, reverse=reverse)

        if reverse:
            rotated.vector = rotated.vector[::-1]

        rotated.vector = rotated.vector[shift:] + rotated.vector[:shift]

        rotated.embedding = plast.embedding
        plast.debug("rotate_plasmid finished")
        return rotated

    @staticmethod
    def best_layout_min_crossings(arch1, arch2):
        """
        Compute the best layout (rotation and optional reversal) of arch2 to minimize
        crossings with arch1. Unknown tokens are ignored for matching.
        Returns a dict: {'reversed': bool, 'rotation': int, 'crossings': int, 'matches': int}.
        """

        def is_unknown_token(x):
            """Treat None, NaN, 'nan', 'na', '' as unknown."""
            if x is None:
                return True
            try:
                if isinstance(x, float) and x != x:  # NaN
                    return True
            except TypeError:
                pass
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ("", "nan", "na"):
                    return True
            return False

        def rotate_list(xs, k):
            """Rotate list xs left by k (circular)."""
            if not xs:
                return xs
            k = k % len(xs)
            return xs[k:] + xs[:k]

        def count_inversions(arr):
            """Count inversions via mergesort."""

            def mergesort(a):
                n = len(a)
                if n <= 1:
                    return a, 0
                m = n // 2
                left, li = mergesort(a[:m])
                right, ri = mergesort(a[m:])
                i = j = 0
                inv = li + ri
                out = []
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        out.append(left[i])
                        i += 1
                    else:
                        out.append(right[j])
                        j += 1
                        inv += len(left) - i
                out.extend(left[i:])
                out.extend(right[j:])
                return out, inv

            _, inv = mergesort(list(arr))
            return inv

        def mapping_and_crossings(top, bottom):
            """
            Build greedy k-th to k-th mapping (top_i -> bottom_j) and compute
            crossings for fixed bottom order. Unknown tokens are skipped.
            """
            positions = {}
            for j, val in enumerate(bottom):
                if is_unknown_token(val):
                    continue
                positions.setdefault(val, []).append(j)

            taken_idx = {}
            pairs = []
            seq_j = []

            for i, val in enumerate(top):
                if is_unknown_token(val):
                    continue
                js = positions.get(val)
                if not js:
                    continue
                k = taken_idx.get(val, 0)
                if k >= len(js):
                    continue
                j = js[k]
                taken_idx[val] = k + 1
                pairs.append((i, j))
                seq_j.append(j)

            crossings = count_inversions(seq_j)
            return crossings, pairs

        if not arch2:
            return {
                "reversed": False,
                "rotation": 0,
                "crossings": 0,
                "matches": 0,
            }

        best = None
        best_key = None

        for rev_flag in (False, True):
            base = arch2[::-1] if rev_flag else arch2
            m = len(base)
            for k in range(m):
                cand = rotate_list(base, k)
                crossings, pairs = mapping_and_crossings(arch1, cand)
                matches = len(pairs)
                # Sort by: fewer crossings, more matches, prefer non-reversed, smaller rotation
                key = (crossings, -matches, 0 if not rev_flag else 1, k)
                if best is None or key < best_key:
                    best = {
                        "reversed": rev_flag,
                        "rotation": k,
                        "crossings": crossings,
                        "matches": matches,
                    }
                    best_key = key
        return best

    def draw_network(self, closest_num=3):
        """Compute network coordinates for the query and closest_num results."""
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

    def get_most_similar(self, maxret=10, metric="cosine"):
        """Compute most similar plasmids to the current embedding."""
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
