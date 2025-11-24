"""
This module provides functionality for aligning two sequences (architectures) with optional
similarity matrices, minimizing crossings and matching modules. It includes utilities for
handling unknown values, rotating sequences, counting inversions, extracting module pairs
from similarity matrices, and computing the best alignment layout.
"""

from typing import Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import label
from plast.plast import PLAST


class Alignment:
    """
    Represents an alignment with transformation and matching information.

    :param transformed: Transformed architecture (list after rotation/reversal).
    :type transformed: list
    :param is_reversed: Whether the architecture was reversed.
    :type is_reversed: bool
    :param rotation: Number of positions rotated.
    :type rotation: int
    :param crossings: Number of crossings (inversions) in the alignment.
    :type crossings: int
    :param mean_len: Mean absolute distance between matched indices.
    :type mean_len: float
    :param matches: Number of matched pairs.
    :type matches: int
    :param pairs: List of matched pairs (i, j).
    :type pairs: list[tuple[int, int]]
    """

    def __init__(
        self,
        transformed: list,
        is_reversed: bool,
        rotation: int,
        crossings: int,
        mean_len: float,
        matches: int,
        pairs: List[Tuple[int, int]],
    ):
        self.transformed = transformed
        self.is_reversed = is_reversed
        self.rotation = rotation
        self.crossings = crossings
        self.mean_len = mean_len
        self.matches = matches
        self.pairs = pairs


def is_unknown(x: Any) -> bool:
    """
    Determines whether a given value should be considered 'unknown'.

    A value is considered unknown if it is:
        - None
        - A float NaN (not-a-number)
        - A string that is empty, "nan", or "na" (case-insensitive, with whitespace ignored)

    :param x: The value to check.
    :type x: Any
    :returns: True if the value is considered unknown, False otherwise.
    :rtype: bool
    """
    if x is None:
        return True
    try:
        if isinstance(x, float) and x != x:
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(x, str):
        s = x.strip().lower()
        if s == "" or s == "nan" or s == "na":
            return True
    return False


def rotate(xs: List[Any], k: int) -> List[Any]:
    """
    Rotates the elements of a list by k positions.

    :param xs: The list to rotate.
    :type xs: list
    :param k: The number of positions to rotate the list by.
    :type k: int
    :returns: A new list with elements rotated by k positions.
    :rtype: list
    """
    if not xs:
        return xs
    k = k % len(xs)
    return xs[k:] + xs[:k]


def count_inversions(arr: List[int]) -> int:
    """
    Counts the number of inversions in the given array.

    An inversion is a pair of indices (i, j) such that i < j and arr[i] > arr[j].
    Uses a modified merge sort algorithm.

    :param arr: Iterable of comparable elements (e.g., list of integers).
    :type arr: list[int]
    :returns: The number of inversions in the array.
    :rtype: int
    """

    def mergesort(a: List[int]) -> Tuple[List[int], int]:
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


def extract_module_pairs(
    sim_matrix: np.ndarray, threshold: float = 0.99, min_size: int = 5
) -> List[Tuple[int, int]]:
    """
    Extracts pairs of indices (i, j) belonging to large modules in sim_matrix.

    :param sim_matrix: Similarity matrix.
    :type sim_matrix: np.ndarray
    :param threshold: Similarity threshold for module inclusion.
    :type threshold: float
    :param min_size: Minimum module size to consider.
    :type min_size: int
    :returns: List of (i, j) pairs in large modules.
    :rtype: list[tuple[int, int]]
    """
    mask = sim_matrix >= threshold
    labeled, num = label(mask, structure=np.ones((3, 3), dtype=int))
    pairs = [
        (i, j)
        for k in range(1, num + 1)
        for (i, j) in np.argwhere(labeled == k)
        if np.sum(labeled == k) >= min_size
    ]
    return pairs


def mapping_and_crossings(
    top: List[Any], bottom: List[Any]
) -> Tuple[int, List[Tuple[int, int]], float]:
    """
    Maps elements from 'top' to 'bottom', computes crossings (inversions),
    and mean distance between mapped indices.

    :param top: The first sequence to map from.
    :type top: list
    :param bottom: The second sequence to map to.
    :type bottom: list
    :returns: crossings, pairs, mean_len
    :rtype: tuple[int, list[tuple[int, int]], float]
    """
    positions = {}
    for j, val in enumerate(bottom):
        if is_unknown(val):
            continue
        positions.setdefault(val, []).append(j)
    taken_idx = {}
    pairs = []
    seq_j = []
    for i, val in enumerate(top):
        if is_unknown(val):
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
    lengths = [abs(i - j) for (i, j) in pairs]
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    return crossings, pairs, mean_len


def best_layout_min_crossings(
    arch1: List[Any],
    arch2: List[Any],
    sim_matrix: Optional[np.ndarray] = None,
    min_module_size: int = 5,
    threshold: float = 1.0,
) -> Alignment:
    """
    Finds the best transformation (rotation and reversal) of `arch2` to align with `arch1`
    such that the number of crossings between matched modules is minimized.

    Supports:
      1. Simple matching when `min_module_size == 1` and `sim_matrix is None`.
      2. Module-based matching using a similarity matrix, threshold, and minimum module size.

    :param arch1: Reference architecture (sequence of modules).
    :type arch1: list
    :param arch2: Architecture to be transformed and aligned.
    :type arch2: list
    :param sim_matrix: Precomputed similarity matrix between modules.
    :type sim_matrix: np.ndarray, optional
    :param min_module_size: Minimum size of modules to consider for matching.
    :type min_module_size: int
    :param threshold: Similarity threshold for considering a module pair as a match.
    :type threshold: float
    :returns: Alignment object with best transformation and matching info.
    :rtype: Alignment
    """
    m = len(arch2)

    if min_module_size == 1 and sim_matrix is None:
        if m == 0:
            return Alignment(
                transformed=arch2,
                is_reversed=False,
                rotation=0,
                crossings=0,
                mean_len=0.0,
                matches=0,
                pairs=[],
            )

        best = None
        best_key = None
        for rev_flag in (False, True):
            base = arch2[::-1] if rev_flag else arch2
            for k in range(m):
                cand = rotate(base, k)
                crossings, pairs, mean_len = mapping_and_crossings(arch1, cand)
                matches = len(pairs)
                key = (crossings, -matches, rev_flag, k)
                if best is None or key < best_key:
                    best = Alignment(
                        transformed=cand,
                        is_reversed=rev_flag,
                        rotation=k,
                        crossings=crossings,
                        mean_len=mean_len,
                        matches=matches,
                        pairs=pairs,
                    )
                    best_key = key
        return best

    if sim_matrix is None:
        sim_matrix = (np.array(arch1)[:, None] == np.array(arch2)[None, :]).astype(int)
    pairs = list(extract_module_pairs(sim_matrix, threshold, min_module_size))
    matches = len(pairs)

    if m == 0 or matches == 0:
        return Alignment(
            transformed=arch2,
            is_reversed=False,
            rotation=0,
            crossings=0,
            mean_len=0.0,
            matches=matches,
            pairs=pairs,
        )

    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    best = None
    best_key = None

    for rev_flag in (False, True):
        base = arch2[::-1] if rev_flag else arch2

        for k in range(m):
            transformed_pairs = []
            seq_j = []
            lengths = []

            for i, j in pairs_sorted:
                j_base = (m - 1 - j) if rev_flag else j
                j_new = (j_base - k) % m
                transformed_pairs.append((i, j_new))
                seq_j.append(j_new)
                lengths.append(abs(i - j_new))

            crossings = count_inversions(seq_j)
            mean_len = float(np.mean(lengths)) if lengths else 0.0

            cand = rotate(base, k)

            key = (mean_len, crossings, rev_flag, k)

            if best is None or key < best_key:
                best = Alignment(
                    transformed=cand,
                    is_reversed=rev_flag,
                    rotation=k,
                    crossings=crossings,
                    mean_len=mean_len,
                    matches=matches,
                    pairs=transformed_pairs,
                )
                best_key = key

    return best


def rotate_parsed(pl: PLAST, shift: int, reverse: bool = False) -> PLAST:
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

    :param pl: PLAST object to rotate.
    :type pl: PLAST
    :param shift: Number of positions to rotate.
    :type shift: int
    :param reverse: Whether to reverse the order and flip coordinates.
    :type reverse: bool
    :returns: Rotated PLAST object.
    :rtype: PLAST
    """
    pl.debug(
        f"rotate_parsed called (shift={shift}, reverse={reverse}, "
        f"records={0 if pl.parsed is None else len(pl.parsed)}, "
        f"length={pl.length})"
    )
    if pl.parsed is None or (isinstance(pl.parsed, pd.DataFrame) and pl.parsed.empty):
        return pl

    orig_index = pl.parsed.index.tolist()
    orig_index_name = pl.parsed.index.name

    records = pl.parsed.to_dict(orient="records")
    n = len(records)
    if n == 0:
        return pl

    shift = shift % n
    plen = int(pl.length)
    new_parsed = []

    if not reverse:
        rotated = records[shift:] + records[:shift]
        rotated_index = orig_index[shift:] + orig_index[:shift]
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
        reversed_records = records[::-1]
        reversed_index = orig_index[::-1]
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
        rotated = flipped_records[shift:] + flipped_records[:shift]
        rotated_index = reversed_index[shift:] + reversed_index[:shift]
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

    new_df = pd.DataFrame(new_parsed)
    new_df.index = pd.Index(rotated_index, name=orig_index_name)
    pl.parsed = new_df
    pl.debug(
        f"rotate_parsed finished (shift={shift}, reverse={reverse}, records={len(pl.parsed)})"
    )
    return pl


def rotate_plasmid(plast: PLAST, shift: int, reverse: bool = False) -> PLAST:
    """
    Returns a new PLAST object with the vector rotated by 'shift' positions.
    If reverse=True, the order of elements is reversed (and parsed is flipped accordingly).

    :param plast: PLAST object to rotate.
    :type plast: PLAST
    :param shift: Number of positions to rotate.
    :type shift: int
    :param reverse: Whether to reverse the order and flip coordinates.
    :type reverse: bool
    :returns: Rotated PLAST object.
    :rtype: PLAST
    """
    plast.debug(f"rotate_plasmid called (shift={shift}, reverse={reverse})")
    rotated = plast.copy()
    rotated = rotate_parsed(rotated, shift, reverse=reverse)
    if reverse:
        rotated.vector = rotated.vector[::-1]
    rotated.vector = rotated.vector[shift:] + rotated.vector[:shift]
    rotated.embedding = plast.embedding
    plast.debug("rotate_plasmid finished")
    return rotated
