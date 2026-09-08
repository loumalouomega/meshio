"""Dataset health: the server-side producer of the per-entry / per-manifest
summary the browser dataset manager otherwise computes from its own WASM
scans (doc/dashboard.md).

Pure stdlib + meshioplusplus, Python 3.8 -- tested in the default matrix with
no optional extra. The shape is a superset of the browser's ``EntryScan`` /
``ManifestHealth`` (``src/viewer/src/dataset/types.ts``), snake_cased, so the
page's ``health.ts:fromServerHealth`` fills the same structure and the card
renderer takes either producer.

One rule is load-bearing and shared with the browser: a ``quality:*``
array's NaN means "metric N/A for this cell type" by design
(``compute_quality``'s own convention), so quality arrays are **excluded**
from the NaN/Inf counts. Meshes are read one at a time (the sequence
streaming invariant); a failing entry is reported, never fatal.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional

from .. import compute_quality, data_info
from .._dataset import check_pairing

QUALITY_PREFIX = "quality:"
_SCALED_JACOBIAN_KEYS = ("quality:scaled_jacobian", "scaled_jacobian")


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def empty_scan() -> dict:
    """The scan recorded for an entry that could not be read (``steps`` 0)."""
    return {
        "steps": 0,
        "num_nan": 0,
        "num_inf": 0,
        "num_inverted": 0,
        "num_degenerate": 0,
        "min_scaled_jacobian": None,
        "arrays": [],
        "target_steps": None,
        "pairing_error": None,
    }


def mesh_scan(mesh, *, quality: bool = True) -> dict:
    """One mesh's contribution: NaN/Inf over its DATA arrays, the quality
    lane, and the arrays present as ``<location>:<name>``."""
    rows = [r for r in data_info(mesh) if not str(r["name"]).startswith(QUALITY_PREFIX)]
    scan = {
        "num_nan": int(sum(int(r["num_nan"]) for r in rows)),
        "num_inf": int(sum(int(r["num_inf"]) for r in rows)),
        "num_inverted": 0,
        "num_degenerate": 0,
        "min_scaled_jacobian": None,
        "arrays": sorted(f"{r['location']}:{r['name']}" for r in rows),
    }
    if quality:
        try:
            report = compute_quality(mesh)
        except Exception:  # noqa: BLE001 - quality is best-effort, like the browser's
            return scan
        scan["num_inverted"] = int(report.get("num_inverted", 0))
        scan["num_degenerate"] = int(report.get("num_degenerate", 0))
        metrics = report.get("metrics", {})
        for key in _SCALED_JACOBIAN_KEYS:
            if key in metrics:
                scan["min_scaled_jacobian"] = _finite(metrics[key].get("min"))
                break
    return scan


def _merge(total: dict, part: dict) -> None:
    for key in ("num_nan", "num_inf", "num_inverted", "num_degenerate"):
        total[key] += part[key]
    sj = part.get("min_scaled_jacobian")
    if sj is not None:
        current = total.get("min_scaled_jacobian")
        total["min_scaled_jacobian"] = sj if current is None else min(current, sj)


def entry_health(
    entry,
    *,
    quality: bool = True,
    all_steps: bool = False,
    read_kwargs=None,
    guard=None,
) -> dict:
    """One manifest entry's scan over step 0 (or every step).

    A ``guard`` (a :class:`~meshioplusplus.GeometryGuard`) additionally scores
    the entry's shape against the distribution it was fitted on. That is
    advisory and does NOT make an entry bad: an unusual part is a fact about
    the dataset, not a defect in the mesh.
    """
    try:
        plan = entry.entries()
        series = entry.time_series(**(read_kwargs or {}))
        steps = len(plan)
        scan = empty_scan()
        scan["steps"] = steps
        if entry.target:
            # A paired entry's Target is checked separately from the mesh scan:
            # a mismatched pair is a manifest problem, not a bad mesh, and it
            # must not make the entry look unreadable.
            try:
                scan["target_steps"] = check_pairing(entry)
            except Exception as e:  # noqa: BLE001
                scan["pairing_error"] = str(e)
        arrays = set()
        for index in range(steps) if all_steps else range(min(steps, 1)):
            _, mesh = series[index]  # one mesh alive at a time
            part = mesh_scan(mesh, quality=quality)
            _merge(scan, part)
            arrays.update(part["arrays"])
            if guard is not None and index == 0:
                try:
                    scan["guard"] = guard.check(mesh)
                except Exception as e:  # noqa: BLE001 -- advisory
                    scan["guard"] = {"error": str(e)}
            del mesh
        scan["arrays"] = sorted(arrays)
        return scan
    except Exception as e:  # noqa: BLE001 - a bad entry is reported, never fatal
        failed = empty_scan()
        failed["error"] = str(e)
        return failed


def scan_is_bad(scan: dict) -> bool:
    sj = scan.get("min_scaled_jacobian")
    return (
        scan["steps"] == 0
        or scan["num_nan"] + scan["num_inf"] > 0
        or scan["num_inverted"] > 0
        or scan["num_degenerate"] > 0
        or (sj is not None and sj < 0)
        # A pair whose two sides disagree trains the model on mismatched steps,
        # which is a silent wrong answer rather than a crash -- so it is bad.
        or scan.get("pairing_error") is not None
    )


def split_balance(splits: Dict[Optional[str], int], total: int) -> List[dict]:
    """``{split: count}`` -> ordered rows, unassigned (``''``) last."""
    rows = [
        {
            "split": "" if k is None else str(k),
            "count": int(v),
            "fraction": (v / total) if total else 0.0,
        }
        for k, v in splits.items()
    ]
    rows.sort(key=lambda r: (r["split"] == "", r["split"]))
    return rows


def manifest_health(
    manifest,
    entries: Iterable = None,
    *,
    quality: bool = True,
    all_steps: bool = False,
    read_kwargs=None,
    before_entry: Optional[Callable] = None,
    guard=None,
) -> dict:
    """Aggregate the scans of ``entries`` (default: every entry) of a
    :class:`~meshioplusplus.DatasetManifest`. ``before_entry(entry)`` runs
    ahead of each scan (the MCP tool sandboxes the entry's resolved paths
    there); its exception is recorded as that entry's error."""
    selected = list(manifest) if entries is None else list(entries)
    total = len(manifest)
    scans: Dict[str, dict] = {}
    for entry in selected:
        if before_entry is not None:
            try:
                before_entry(entry)
            except Exception as e:  # noqa: BLE001
                failed = empty_scan()
                failed["error"] = str(e)
                scans[entry.id] = failed
                continue
        scans[entry.id] = entry_health(
            entry,
            quality=quality,
            all_steps=all_steps,
            read_kwargs=read_kwargs,
            guard=guard,
        )
    totals = {
        "num_nan": 0,
        "num_inf": 0,
        "num_inverted": 0,
        "num_degenerate": 0,
        "min_scaled_jacobian": None,
    }
    union = set()
    readable = []
    bad = []
    for entry_id, scan in scans.items():
        _merge(totals, scan)
        if scan_is_bad(scan):
            bad.append(entry_id)
        if scan["steps"] > 0:
            readable.append(entry_id)
            union.update(scan["arrays"])
    fields_missing = {}
    for entry_id in readable:
        have = set(scans[entry_id]["arrays"])
        missing = sorted(union - have)
        if missing:
            fields_missing[entry_id] = missing
    out_of_distribution = [
        entry_id
        for entry_id, scan in scans.items()
        if isinstance(scan.get("guard"), dict) and scan["guard"].get("verdict") == "out"
    ]
    splits = manifest.splits()
    return {
        "producer": "server",
        "name": manifest.name,
        "num_entries": total,
        "scanned": len(scans),
        "splits": {("" if k is None else str(k)): int(v) for k, v in splits.items()},
        "split_balance": split_balance(splits, total),
        "entries": scans,
        "fields_missing": fields_missing,
        "totals": totals,
        "bad_entries": bad,
        # Advisory and separate from bad_entries: an unusual shape is a fact
        # about the dataset, not a defect in the mesh.
        "out_of_distribution": out_of_distribution,
    }
