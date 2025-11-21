"""
This module provides utility functions for parsing various bioinformatics file formats and writing
FASTA files. It includes functions to read and process FASTA, MMseqs m8, GenBank flat files (GBFF),
eggNOG-mapper hits, hmmscan domtblout outputs, and to write FASTA files from pandas DataFrames.
"""

import io
from typing import Union
import pandas as pd


def read_fasta(fasta: Union[str, io.BufferedReader]):
    """
    Parse a FASTA file handler or string into a dictionary.
    """
    if isinstance(fasta, str):
        fasta = fasta.split("\n")
    seqs = {}
    current_id = None
    current_seq = ""
    for line in fasta:
        if line.startswith(">"):
            if current_id:
                seqs[current_id] = current_seq
            current_id = line[1:].strip()
            current_seq = ""
        else:
            current_seq += line.strip()
    if current_id is not None:
        seqs[current_id] = current_seq
    return seqs


def read_m8(m8_filename: str, res_per_query: int = 1):
    """
    Parse MMseqs easy-search output file.
    :param m8_filename: Path to the m8 file.
    """
    results = pd.read_csv(
        m8_filename,
        sep="\t",
        header=None,
        names=[
            "query_id",
            "target_id",
            "pident",
            "alignment_length",
            "mismatches",
            "gap_opens",
            "query_start",
            "query_end",
            "target_start",
            "target_end",
            "evalue",
            "bitscore",
        ],
    )
    if res_per_query == 1:
        results = results.loc[
            results.groupby("query_id")["evalue"].idxmin()
        ]  # get results with the best e-value
    return results.set_index("query_id")


def read_gbff(stream: io.BufferedReader):
    """
    Parse a GenBank flat file (GBFF) from a file handler into a DataFrame
    containing CDS features and their attributes.
    """
    all_elements = []
    line = stream.readline().decode()
    accession = ""
    length = -1
    while not line.startswith("FEATURES"):
        if line.startswith("LOCUS"):
            accession = line.split()[1]
            length = int(line.split()[2])
        line = stream.readline().decode().strip()

    for line in stream:
        line = line.decode()
        element = {}
        if line.startswith("     CDS"):
            typ, coords = line.strip().split()
            element["type"] = typ
            element["coordinates"] = coords
            line = stream.readline().decode()
            current_attr = None
            current_val = None
            while line.startswith("                     "):
                line = line.strip()
                if line.startswith("/"):
                    if current_attr is not None:
                        element[current_attr] = current_val.strip('"')
                    attrtype = line[1:].split("=")
                    if len(attrtype) == 1:
                        current_attr = attrtype[0]
                        current_val = "1"
                    elif len(attrtype) == 2:
                        current_attr, current_val = attrtype
                else:
                    current_val += line
                line = stream.readline().decode()
            element[current_attr] = current_val.strip('"')
            all_elements.append(element)

    to_dataframe = {attr: [] for attr in {k for d in all_elements for k in d.keys()}}
    for element in all_elements:
        for attr in to_dataframe:
            if attr in element:
                to_dataframe[attr].append(element[attr])
            else:
                to_dataframe[attr].append(None)

    dataframe = pd.DataFrame(all_elements)
    tidy_cord = {"start": [], "end": [], "strand": [], "partial": []}
    for coord in dataframe["coordinates"].to_list():
        if coord.startswith("complement"):
            coord = coord[11:-1]
            strand = -1
        else:
            strand = 1
        if coord.startswith("join"):
            coord = coord[5:-1].split(",")
            if ".." in coord[0]:
                coord = coord[0]
            else:
                coord = coord[1]

        start, end = coord.split("..")
        partial = 0
        if "<" in start:
            start = start[1:]
            partial = 1
        if ">" in end:
            end = end[1:]
            partial = 1

        tidy_cord["partial"].append(partial)
        tidy_cord["start"].append(int(start))
        tidy_cord["end"].append(int(end))
        tidy_cord["strand"].append(int(strand))
    dataframe = pd.concat(
        [pd.DataFrame(tidy_cord), dataframe.reset_index(drop=True)], axis=1
    ).set_index(dataframe.index)
    dataframe.dropna(subset=["translation"], inplace=True)
    return dataframe.sort_values("start"), accession, length


def read_emapper_hits(emapper_hits_file: str, res_per_query: int = 1):
    """
    Parse eggnog-mapper output file into a DataFrame.
    """
    results = pd.read_csv(
        emapper_hits_file,
        sep="\t",
        comment="#",
        names=[
            "query_name",
            "hit",
            "evalue",
            "sum_score",
            "query_length",
            "hmmfrom",
            "hmmto",
            "seqfrom",
            "seqto",
            "query_coverage",
        ],
    )
    if res_per_query == 1:
        results = results.loc[results.groupby("query_name")["evalue"].idxmin()]
    return results.set_index("query_name")


def read_hmmscan_output(input_file_path: str) -> pd.DataFrame:
    """
    Parse hmmscan domtblout file into a DataFrame similar to eggnog-mapper output.
    """
    header = (
        "target_name\ttarget_acc\ttarget_len\tquery_name\tquery_acc\tquery_len\t"
        "eval_global\tscore_global\tbias_global\tthis_dom_num\ttotal_dom_num_per_seq\t"
        "c_eval\ti_eval\tscore\tbias\thmm_start\thmm_stop\taln_start\taln_stop\t"
        "env_start\tenv_stop\talign_reliable\tdescription\n"
    )

    with open(input_file_path, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
        processed_lines = [
            "\t".join(line.split()).strip() + "\n"
            for line in lines
            if not line.startswith("#")
        ]
    temp_file = io.StringIO("".join([header] + processed_lines))
    df = pd.read_csv(temp_file, sep="\t")

    rename_map = {
        "query_name": "query_name",
        "target_name": "hit",
        "eval_global": "evalue",
        "score": "sum_score",
        "query_len": "query_length",
        "hmm_start": "hmmfrom",
        "hmm_stop": "hmmto",
        "aln_start": "seqfrom",
        "aln_stop": "seqto",
    }
    df = df[list(rename_map.keys())].rename(columns=rename_map)
    df["query_coverage"] = (df["seqto"] - df["seqfrom"]) / df["query_length"]

    # Filter by evalue
    df = df[df["evalue"] <= 1e-3]

    # Choose the best hit for each query
    df = df.loc[df.groupby("query_name")["evalue"].idxmin()]
    df = df.sort_values(by="query_name").reset_index(drop=True)
    return df
