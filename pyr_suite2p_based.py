import os
import re
import glob
import pickle
import copy
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

# Reuse the same spike/event conventions from Pyr_event_cal2
import pyr_event_cal2_pipeline as pe
try:
    import pyr_event_cal3_pipeline as pe3
except Exception:
    pe3 = None

FS_VOL_HZ_DEFAULT = 500.0
FS_CA_HZ_FALLBACK = 30.0
REAL_FR_SMOOTH_SIGMA_S_DEFAULT = 0.10
PRED_FR_SMOOTH_SIGMA_S_DEFAULT = None  # None -> use REAL_FR_SMOOTH_SIGMA_S_DEFAULT
RESID_Z_K_DEFAULT = 3.0
MIN_SEG_DUR_S_DEFAULT = 0.30
EVENT_ISI_MS_DEFAULT = 30.0
DB_PATH = getattr(pe, "DB_PATH", None)


def set_db_path(db_path):
    """Set module-level DB path used when db_path is not passed explicitly."""
    global DB_PATH
    DB_PATH = str(db_path) if db_path is not None else None


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_to_csv(df, out_csv_path, index=False):
    """
    Write CSV; if destination is locked (PermissionError), save a timestamped fallback.
    Returns the actual path written.
    """
    out_csv_path = str(out_csv_path)
    try:
        df.to_csv(out_csv_path, index=index)
        return out_csv_path
    except PermissionError:
        root, ext = os.path.splitext(out_csv_path)
        if not ext:
            ext = ".csv"
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        alt_csv = f"{root}__{ts}{ext}"
        df.to_csv(alt_csv, index=index)
        return alt_csv


def _safe_pickle_load(fp):
    """
    Load pickle robustly across NumPy private-module path changes.
    """
    try:
        return pickle.load(fp)
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or "")
        # Cross-version compatibility: some pickles reference `numpy._core.*`
        # while older environments expose `numpy.core.*`.
        if missing.startswith("numpy._core"):
            import sys
            import numpy as _np

            if "numpy._core" not in sys.modules:
                sys.modules["numpy._core"] = _np.core
            if "numpy._core.multiarray" not in sys.modules:
                sys.modules["numpy._core.multiarray"] = _np.core.multiarray
            if "numpy._core._multiarray_umath" not in sys.modules:
                try:
                    sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath
                except Exception:
                    pass
            fp.seek(0)
            return pickle.load(fp)
        raise


def _safe_cal_sr_from_db(cell_folder, db_path=None, fallback=FS_CA_HZ_FALLBACK):
    if db_path is None:
        db_path = DB_PATH
    if (db_path is None) or (not os.path.isfile(str(db_path))):
        return float(fallback)
    try:
        db = pd.read_csv(db_path)
        if "Link" not in db.columns:
            return float(fallback)
        hit = db.loc[db["Link"].astype(str).str.lower() == str(cell_folder).lower()]
        if len(hit) == 0:
            return float(fallback)
        return float(pe._safe_cal_sr(hit.iloc[0].get("CALsr", fallback), default=fallback))
    except Exception:
        return float(fallback)


def _robust_z(x):
    x = np.asarray(x, dtype=float).ravel()
    med = float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan
    mad = float(np.nanmedian(np.abs(x - med))) if np.isfinite(med) else np.nan
    scale = 1.4826 * mad if np.isfinite(mad) else np.nan
    if (not np.isfinite(scale)) or (scale <= 0):
        scale = float(np.nanstd(x))
    if (not np.isfinite(scale)) or (scale <= 0):
        scale = 1.0
    z = (x - med) / scale
    return z, med, scale


def _fit_nonneg_scale(pred_raw, real_ref, clip_q=99.5):
    """Fit non-negative scale alpha for pred_cal = alpha * pred_raw."""
    p = np.asarray(pred_raw, dtype=float).ravel()
    r = np.asarray(real_ref, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(r) & (p >= 0) & (r >= 0)
    if np.sum(m) < 10:
        return 1.0
    p = p[m]
    r = r[m]
    if p.size < 10 or np.nanstd(p) <= 0:
        return 1.0
    try:
        q = float(clip_q)
    except Exception:
        q = 99.5
    q = min(100.0, max(90.0, q))
    p_hi = float(np.nanpercentile(p, q))
    r_hi = float(np.nanpercentile(r, q))
    keep = (p <= p_hi) & (r <= r_hi)
    if np.sum(keep) >= 10:
        p_fit = p[keep]
        r_fit = r[keep]
    else:
        p_fit = p
        r_fit = r
    den = float(np.dot(p_fit, p_fit))
    if (not np.isfinite(den)) or den <= 0:
        return 1.0
    alpha = float(np.dot(p_fit, r_fit) / den)
    if (not np.isfinite(alpha)) or alpha <= 0:
        return 1.0
    return float(alpha)


def _fit_nonneg_scale_active(pred_raw, real_ref, active_q=80.0, clip_q=99.5):
    """Fit non-negative alpha using only active real-FR frames."""
    p = np.asarray(pred_raw, dtype=float).ravel()
    r = np.asarray(real_ref, dtype=float).ravel()
    m = np.isfinite(p) & np.isfinite(r) & (p >= 0) & (r >= 0)
    if np.sum(m) < 10:
        return 1.0
    p = p[m]
    r = r[m]
    if p.size < 10 or np.nanstd(p) <= 0 or np.nanstd(r) <= 0:
        return 1.0
    try:
        aq = float(active_q)
    except Exception:
        aq = 80.0
    aq = min(99.0, max(50.0, aq))
    r_thr = float(np.nanpercentile(r, aq))
    active = r >= r_thr
    if np.sum(active) < 10:
        active = r >= float(np.nanpercentile(r, 70.0))
    if np.sum(active) < 10:
        active = np.ones_like(r, dtype=bool)
    return _fit_nonneg_scale(p[active], r[active], clip_q=clip_q)


def _mask_to_segments(mask, min_len=1):
    m = np.asarray(mask, dtype=bool).ravel()
    segs = []
    if m.size == 0:
        return segs
    i = 0
    while i < m.size:
        if not m[i]:
            i += 1
            continue
        s = int(i)
        while i < m.size and m[i]:
            i += 1
        e = int(i - 1)
        if (e - s + 1) >= int(max(1, min_len)):
            segs.append((s, e))
    return segs


def _in_any_segment(frame_idx, segments):
    fi = int(frame_idx)
    for s, e in segments:
        if s <= fi <= e:
            return True
    return False


def _suffix_from_pkl_name(path):
    name = os.path.basename(str(path))
    patterns = [
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$",
        r"^event_spike_overlay__plus_plateau(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$",
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_after_peak\.pkl$",
        r"^spike_detection_refined_new_plus_plateau(?P<suf>[mr]\d+)?\.pkl$",
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?\.pkl$",
        r"^final_correct_spike_detection(?P<suf>[mr]\d+)?\.pkl$",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            suf = m.group("suf")
            return "main" if not suf else str(suf).lower()
    return "main"


def _is_rm_complex_highplateau_pkl(path):
    """Return True for corrected spike PKLs with high-plateau removal."""
    if not isinstance(path, str) or path.strip() == "":
        return False
    name = os.path.basename(str(path))
    return bool(
        re.search(
            r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$",
            name,
            flags=re.IGNORECASE,
        )
    )


def _preferred_spike_pkl_priority(path):
    """
    Return (priority, suffix) for spike PKLs used to build real FR.

    Lower priority wins. The corrected high-plateau-removal PKL is preferred when
    available because it contains the post-removal `vm_all_spikes` used for FR.
    """
    name = os.path.basename(str(path))
    patterns = [
        (0, r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$"),
        (1, r"^event_spike_overlay__plus_plateau(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$"),
        (2, r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_after_peak\.pkl$"),
        (3, r"^spike_detection_refined_new_plus_plateau(?P<suf>[mr]\d+)?\.pkl$"),
        (4, r"^final_correct_spike_detection(?P<suf>[mr]\d+)?\.pkl$"),
        (5, r"^spike_detection_refined_new(?P<suf>[mr]\d+)?\.pkl$"),
    ]
    for priority, pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            suf = m.group("suf")
            return int(priority), ("main" if not suf else str(suf).lower())
    return 99, _suffix_from_pkl_name(path)


def _find_preferred_spike_pkls(cell_folder):
    """
    Find one best spike-detection PKL per suffix for suite2p/CASCADE FR evaluation.

    Priority:
      1. corrected high-plateau removal PKL
      2. older after-peak removal PKL
      3. plus-plateau PKL
      4. final_correct_spike_detection PKL
      5. raw spike_detection_refined_new PKL
    """
    candidates = []
    for pat in (
        "spike_detection_refined_new*.pkl",
        "event_spike_overlay__plus_plateau*_rm_complex_highplateau.pkl",
        "final_correct_spike_detection*.pkl",
    ):
        candidates.extend(glob.glob(os.path.join(cell_folder, pat)))

    best_by_suffix = {}
    for p in sorted(set(candidates), key=lambda x: os.path.basename(str(x)).lower()):
        if not os.path.isfile(p):
            continue
        priority, suffix = _preferred_spike_pkl_priority(p)
        old = best_by_suffix.get(suffix)
        if old is None or priority < old[0]:
            best_by_suffix[suffix] = (priority, p)

    suffix_rank = {"main": -1, "m0": 0, "r0": 1, "m1": 2, "r1": 3}
    return [
        p
        for suffix, (_priority, p) in sorted(
            best_by_suffix.items(),
            key=lambda kv: (suffix_rank.get(kv[0], 99), kv[0], os.path.basename(kv[1][1]).lower()),
        )
    ]


def _pick_pkl(cell_folder, pkl_path=None):
    if pkl_path is not None and os.path.isfile(str(pkl_path)):
        return str(pkl_path)
    pkl_list = _find_preferred_spike_pkls(cell_folder)
    if len(pkl_list) == 0:
        raise FileNotFoundError(f"No spike detection pkl found in: {cell_folder}")
    for p in pkl_list:
        if _suffix_from_pkl_name(p) == "main":
            return p
    return pkl_list[0]


def _find_col_case_insensitive(columns, candidates):
    cols = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols:
            return cols[key]
    return None


def _read_csv_1d(path):
    arr = pd.read_csv(path).to_numpy(dtype=float).ravel()
    return np.asarray(arr, dtype=float)


def _align_1d_len(x, target_len, fill_value=np.nan):
    a = np.asarray(x, dtype=float).ravel()
    n = int(max(0, target_len))
    if a.size == n:
        return a
    if a.size > n:
        return a[:n]
    out = np.full(n, fill_value, dtype=float)
    out[: a.size] = a
    return out


def _resolve_suite2p_roi_idx_for_cell(cell_folder, cal_nb_trace, cal_raw_trace=None):
    suite2p_dir = os.path.join(os.path.dirname(cell_folder), "Sync", "cal", "suite2p", "plane0")
    f_path = os.path.join(suite2p_dir, "F.npy")
    spks_path = os.path.join(suite2p_dir, "spks.npy")

    if not os.path.isfile(f_path):
        return {
            "roi_idx": None,
            "match_corr": np.nan,
            "matched_reference": None,
            "suite2p_dir": suite2p_dir,
            "f_path": f_path,
            "spks_path": spks_path,
            "found": False,
        }

    candidates = [("calTraceNB", cal_nb_trace), ("calTrace", cal_raw_trace)]
    best_idx = None
    best_corr = -np.inf
    best_ref = None

    for ref_name, ref_trace in candidates:
        if ref_trace is None:
            continue
        rr = np.asarray(ref_trace, dtype=float).ravel()
        if rr.size < 3:
            continue
        try:
            idx, corr = pe._resolve_suite2p_row_idx(cell_folder, rr, suite2p_dir)
        except Exception:
            idx, corr = (None, np.nan)

        corr_v = float(corr) if np.isfinite(corr) else -np.inf
        if idx is not None and corr_v > best_corr:
            best_idx = int(idx)
            best_corr = corr_v
            best_ref = ref_name

    if best_idx is None:
        try:
            fallback_idx = pe._cell_idx_from_folder(cell_folder)
        except Exception:
            fallback_idx = None
        if fallback_idx is not None:
            best_idx = int(fallback_idx)
            best_corr = np.nan
            best_ref = "cell_folder_index"

    return {
        "roi_idx": best_idx,
        "match_corr": float(best_corr) if np.isfinite(best_corr) else np.nan,
        "matched_reference": best_ref,
        "suite2p_dir": suite2p_dir,
        "f_path": f_path,
        "spks_path": spks_path,
        "found": True,
    }


def _load_suite2p_spks_for_cell(cell_folder, cal_nb_trace, cal_raw_trace=None):
    meta = _resolve_suite2p_roi_idx_for_cell(
        cell_folder=cell_folder,
        cal_nb_trace=cal_nb_trace,
        cal_raw_trace=cal_raw_trace,
    )
    spks_path = meta.get("spks_path", None)
    roi_idx = meta.get("roi_idx", None)
    if (spks_path is None) or (not os.path.isfile(spks_path)):
        meta["spks_loaded"] = False
        meta["spks_reason"] = f"Missing suite2p spks.npy: {spks_path}"
        return None, meta

    try:
        spks = np.asarray(np.load(spks_path, mmap_mode="r"), dtype=float)
    except Exception as e:
        meta["spks_loaded"] = False
        meta["spks_reason"] = f"Failed loading spks.npy: {e}"
        return None, meta

    if spks.ndim == 1:
        spks = spks.reshape(1, -1)
    if spks.ndim < 2 or spks.shape[0] == 0:
        meta["spks_loaded"] = False
        meta["spks_reason"] = f"Invalid spks.npy shape: {spks.shape}"
        return None, meta

    if roi_idx is None or int(roi_idx) < 0 or int(roi_idx) >= int(spks.shape[0]):
        roi_idx = 0
        meta["roi_idx"] = 0
        meta["matched_reference"] = "fallback_roi0"
        if not np.isfinite(meta.get("match_corr", np.nan)):
            meta["match_corr"] = np.nan

    roi_trace = np.asarray(spks[int(roi_idx)], dtype=float).ravel()
    meta["spks_loaded"] = True
    meta["spks_shape"] = tuple(int(v) for v in spks.shape)
    return roi_trace, meta


def _read_mask_csv(mask_path, expected_len):
    if (mask_path is None) or (not os.path.isfile(mask_path)):
        return np.ones(int(max(0, expected_len)), dtype=bool), False

    try:
        raw = pd.read_csv(mask_path).to_numpy()
    except Exception:
        return np.ones(int(max(0, expected_len)), dtype=bool), False

    if raw.size == 0:
        return np.ones(int(max(0, expected_len)), dtype=bool), False

    m = np.asarray(raw, dtype=float).ravel()
    good = np.isfinite(m)
    if not np.any(good):
        return np.ones(int(max(0, expected_len)), dtype=bool), False

    m = m[good] > 0
    n = int(max(0, expected_len))
    if m.size == n:
        return m.astype(bool), True

    if m.size > n:
        return m[:n].astype(bool), True

    out = np.ones(n, dtype=bool)
    out[: m.size] = m.astype(bool)
    return out, True


def _dff_from_percentile(trace, percentile):
    x = np.asarray(trace, dtype=float).ravel()
    if x.size == 0:
        return x.copy(), np.nan
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.full_like(x, np.nan), np.nan

    p = float(np.nanpercentile(x[finite], float(percentile)))
    denom = p
    if (not np.isfinite(denom)) or (abs(denom) < 1e-12):
        denom = 1e-12 if ((not np.isfinite(denom)) or (denom >= 0)) else -1e-12
    return (x - p) / denom, float(p)


def _state_suffix_order(n_segments):
    out = []
    for i in range(int(max(0, n_segments))):
        if i % 2 == 0:
            out.append(f"m{i // 2}")
        else:
            out.append(f"r{i // 2}")
    return out


def _segment_bounds_from_changepoints(n_vol, changepoints):
    n = int(max(0, n_vol))
    if n <= 0:
        return []

    cp = []
    for v in np.asarray(changepoints, dtype=float).ravel():
        if np.isfinite(v):
            cp.append(int(v))
    cp = sorted(set(cp))
    cp = [int(min(n - 1, max(0, c))) for c in cp]

    bounds = []
    start = 0
    for c in cp:
        if c < start:
            continue
        bounds.append((int(start), int(c)))
        start = int(c + 1)
    if start <= n - 1:
        bounds.append((int(start), int(n - 1)))
    return bounds


def _vol_bounds_to_cal_bounds(v_start, v_end, cal_len, vol_sr_hz, cal_sr_hz):
    c_len = int(max(0, cal_len))
    if c_len <= 0:
        return (0, -1)

    vs = int(max(0, v_start))
    ve = int(max(vs, v_end))
    c0 = int(round(float(vs) * float(cal_sr_hz) / float(vol_sr_hz)))
    c1 = int(round(float(ve + 1) * float(cal_sr_hz) / float(vol_sr_hz))) - 1
    c0 = int(max(0, min(c_len - 1, c0)))
    c1 = int(max(c0, min(c_len - 1, c1)))
    return (c0, c1)


def _extract_changepoints(cell_folder):
    cp_path = os.path.join(cell_folder, "changepoint.csv")
    if not os.path.isfile(cp_path):
        return [], None
    try:
        cp_df = pd.read_csv(cp_path)
        if cp_df.shape[1] == 0:
            return [], cp_path
        vals = cp_df.iloc[:, 0].to_numpy(dtype=float).ravel()
        vals = [int(v) for v in vals if np.isfinite(v)]
        return vals, cp_path
    except Exception:
        return [], cp_path


def _collect_cell_specs(db_path=None, cell_paths=None, default_cal_sr_hz=FS_CA_HZ_FALLBACK):
    out = []

    if db_path is not None:
        db = pd.read_csv(db_path)
        link_col = _find_col_case_insensitive(db.columns, ("Link", "link"))
        cal_col = _find_col_case_insensitive(db.columns, ("CalSr", "CALsr", "calSr", "calsr"))
        brain_col = _find_col_case_insensitive(db.columns, ("brainState", "brain state", "motor", "state"))
        if link_col is None:
            raise ValueError(f"Could not find 'Link' column in database: {db_path}")

        for _, row in db.iterrows():
            cell_path = str(row.get(link_col, "")).strip()
            if cell_path == "":
                continue
            cal_sr = _safe_float(
                row.get(cal_col, default_cal_sr_hz) if cal_col is not None else default_cal_sr_hz,
                default=default_cal_sr_hz,
            )
            brain_state = str(row.get(brain_col, "")) if brain_col is not None else ""
            out.append(
                {
                    "cell_path": cell_path,
                    "cal_sr_hz": float(cal_sr),
                    "brain_state": brain_state,
                    "from_db": True,
                }
            )

    if cell_paths is not None:
        if isinstance(cell_paths, (str, os.PathLike)):
            paths = [str(cell_paths)]
        else:
            paths = [str(p) for p in cell_paths]
        for p in paths:
            s = str(p).strip()
            if s == "":
                continue
            out.append(
                {
                    "cell_path": s,
                    "cal_sr_hz": float(default_cal_sr_hz),
                    "brain_state": "",
                    "from_db": False,
                }
            )

    # Keep first occurrence per path (case-insensitive)
    uniq = []
    seen = set()
    for spec in out:
        key = str(spec["cell_path"]).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(spec)
    return uniq


def load_cells_with_spike_pkls(
    db_path=None,
    cell_paths=None,
    default_cal_sr_hz=FS_CA_HZ_FALLBACK,
    vol_sr_hz=FS_VOL_HZ_DEFAULT,
    require_existing_dirs=True,
):
    """
    Load per-cell voltage/calcium traces + spike PKLs for suite2p-based workflows.

    Supports two entry modes:
    1) database mode via `db_path` (reads Link + CalSr columns, case-insensitive), and
    2) direct path mode via `cell_paths`.

    Important loading behavior:
      - `main` state uses the full cell trace and the preferred spike PKL.
      - corrected high-plateau-removal PKLs
        (`spike_detection_refined_new{suffix}_rm_complex_highplateau.pkl`) are preferred
        when available, then older corrected/plus-plateau/raw PKLs are used as fallback.
      - motor/rest states (`m0`, `r0`, `m1`, `r1`, ...) are loaded when split traces exist.
      - calcium masks and voltage masks are attached to each state and applied later by the
        evaluation functions.

    Returns a list of dictionaries, one per cell:
      - full traces (`full`)
      - per-state entries (`states`) where state is `main` or motor-split (`m0`, `r0`, ...)
      - loaded spike-pkl payload per state (`spike_pkl_data`)
    """
    specs = _collect_cell_specs(
        db_path=db_path,
        cell_paths=cell_paths,
        default_cal_sr_hz=default_cal_sr_hz,
    )
    if len(specs) == 0:
        raise ValueError("No cell paths were provided/found. Use db_path and/or cell_paths.")

    all_cells = []
    for spec in specs:
        cell_folder = str(spec["cell_path"])
        if require_existing_dirs and (not os.path.isdir(cell_folder)):
            all_cells.append(
                {
                    "cell_path": cell_folder,
                    "loaded": False,
                    "error": f"Cell folder does not exist: {cell_folder}",
                }
            )
            continue

        # Calcium loading:
        # - prefer calTraceNBdf (already dF/F)
        # - else calTraceNB and compute dF/F with p8
        # - calRaw is always calTraceNB when present
        cal_nb_path = os.path.join(cell_folder, "calTraceNB.csv")
        cal_nbdf_path = os.path.join(cell_folder, "calTraceNBdf.csv")
        if not os.path.isfile(cal_nbdf_path):
            cal_nbdf_alt = os.path.join(cell_folder, "calTraceNBdf..csv")
            if os.path.isfile(cal_nbdf_alt):
                cal_nbdf_path = cal_nbdf_alt

        cal_nb = None
        cal_raw_trace_for_roi = None
        cal_raw_source = None
        cal_trace_path = os.path.join(cell_folder, "calTrace.csv")
        if os.path.isfile(cal_trace_path):
            try:
                cal_raw_trace_for_roi = _read_csv_1d(cal_trace_path).ravel()
            except Exception:
                cal_raw_trace_for_roi = None

        if os.path.isfile(cal_nb_path):
            cal_nb = _read_csv_1d(cal_nb_path).ravel()
            cal_raw_source = "calTraceNB.csv"
        else:
            if os.path.isfile(cal_trace_path):
                cal_nb = _read_csv_1d(cal_trace_path).ravel()
                cal_raw_source = "calTrace.csv"

        if cal_nb is None:
            all_cells.append(
                {
                    "cell_path": cell_folder,
                    "loaded": False,
                    "error": "Missing calcium trace (need calTraceNB.csv or calTrace.csv)",
                }
            )
            continue

        if os.path.isfile(cal_nbdf_path):
            cal_dff = _read_csv_1d(cal_nbdf_path).ravel()
            if cal_dff.size != cal_nb.size:
                n = min(cal_dff.size, cal_nb.size)
                cal_dff = cal_dff[:n]
                cal_nb = cal_nb[:n]
            cal_dff_source = os.path.basename(cal_nbdf_path)
            cal_f0 = np.nan
        else:
            cal_dff, cal_f0 = _dff_from_percentile(cal_nb, percentile=8.0)
            cal_dff_source = "computed_from_calTraceNB_p8"

        # Voltage loading:
        # - load volTrace.csv and compute dF/F with p10
        vol_path = os.path.join(cell_folder, "volTrace.csv")
        if not os.path.isfile(vol_path):
            vol_fallback = os.path.join(cell_folder, "volTraceDF.csv")
            if os.path.isfile(vol_fallback):
                vol_path = vol_fallback
            else:
                all_cells.append(
                    {
                        "cell_path": cell_folder,
                        "loaded": False,
                        "error": "Missing voltage trace (need volTrace.csv)",
                    }
                )
                continue

        vol_raw = _read_csv_1d(vol_path).ravel()
        vol_dff, vol_f0 = _dff_from_percentile(vol_raw, percentile=10.0)

        # Masks
        cal_mask_path = os.path.join(cell_folder, "calMask.csv")
        vol_mask_path = os.path.join(cell_folder, "volMask.csv")
        cal_mask, cal_mask_found = _read_mask_csv(cal_mask_path, expected_len=cal_nb.size)
        vol_mask, vol_mask_found = _read_mask_csv(vol_mask_path, expected_len=vol_raw.size)

        # Keep original frame indexing: masks are represented as NaNs, not frame removal.
        cal_raw_masked = np.asarray(cal_nb, dtype=float).copy()
        cal_raw_masked[~cal_mask] = np.nan
        cal_dff_masked = np.asarray(cal_dff, dtype=float).copy()
        cal_dff_masked[~cal_mask] = np.nan
        vol_raw_masked = np.asarray(vol_raw, dtype=float).copy()
        vol_raw_masked[~vol_mask] = np.nan
        vol_dff_masked = np.asarray(vol_dff, dtype=float).copy()
        vol_dff_masked[~vol_mask] = np.nan

        # suite2p spks for this cell ROI (ROI chosen by best corr to suite2p F)
        spks_roi, suite2p_meta = _load_suite2p_spks_for_cell(
            cell_folder=cell_folder,
            cal_nb_trace=cal_nb,
            cal_raw_trace=cal_raw_trace_for_roi,
        )
        if spks_roi is None:
            spks_roi_aligned = np.full(cal_nb.size, np.nan, dtype=float)
        else:
            spks_roi_aligned = _align_1d_len(spks_roi, cal_nb.size, fill_value=np.nan)
        spks_roi_masked = np.asarray(spks_roi_aligned, dtype=float).copy()
        spks_roi_masked[~cal_mask] = np.nan

        # PKLs + motor state handling
        pkl_paths = _find_preferred_spike_pkls(cell_folder)
        pkl_suffix_to_path = {_suffix_from_pkl_name(p).lower(): p for p in pkl_paths}
        has_motor_suffix_pkls = any(re.fullmatch(r"[mr]\d+", k) for k in pkl_suffix_to_path.keys())
        brain_state = str(spec.get("brain_state", "")).strip().lower()
        is_motor = ("motor" in brain_state) or has_motor_suffix_pkls

        cp_vals, cp_path = _extract_changepoints(cell_folder)
        vol_bounds = _segment_bounds_from_changepoints(vol_raw.size, cp_vals) if is_motor else []
        state_order = _state_suffix_order(len(vol_bounds))

        states = []
        if is_motor and len(vol_bounds) > 0:
            for (v0, v1), suf in zip(vol_bounds, state_order):
                c0, c1 = _vol_bounds_to_cal_bounds(v0, v1, cal_len=cal_nb.size, vol_sr_hz=vol_sr_hz, cal_sr_hz=spec["cal_sr_hz"])

                vol_seg = vol_raw[v0 : v1 + 1]
                vol_dff_seg = vol_dff[v0 : v1 + 1]
                vol_m_seg = vol_mask[v0 : v1 + 1]

                cal_seg = cal_nb[c0 : c1 + 1]
                cal_dff_seg = cal_dff[c0 : c1 + 1]
                cal_m_seg = cal_mask[c0 : c1 + 1]
                spks_seg = spks_roi_aligned[c0 : c1 + 1]

                pkl_path = pkl_suffix_to_path.get(suf.lower(), None)
                pkl_data = None
                if pkl_path is not None and os.path.isfile(pkl_path):
                    with open(pkl_path, "rb") as f:
                        pkl_data = _safe_pickle_load(f)

                states.append(
                    {
                        "state": suf,
                        "pkl_path": pkl_path,
                        "spike_pkl_data": pkl_data,
                        "vol_idx_bounds": (int(v0), int(v1)),
                        "cal_idx_bounds": (int(c0), int(c1)),
                        "volRaw": np.asarray(vol_seg, dtype=float),
                        "volDff": np.asarray(vol_dff_seg, dtype=float),
                        "volMask": np.asarray(vol_m_seg, dtype=bool),
                        "volRawMasked": np.where(np.asarray(vol_m_seg, dtype=bool), np.asarray(vol_seg, dtype=float), np.nan),
                        "volDffMasked": np.where(np.asarray(vol_m_seg, dtype=bool), np.asarray(vol_dff_seg, dtype=float), np.nan),
                        "calRaw": np.asarray(cal_seg, dtype=float),
                        "calDff": np.asarray(cal_dff_seg, dtype=float),
                        "calMask": np.asarray(cal_m_seg, dtype=bool),
                        "calRawMasked": np.where(np.asarray(cal_m_seg, dtype=bool), np.asarray(cal_seg, dtype=float), np.nan),
                        "calDffMasked": np.where(np.asarray(cal_m_seg, dtype=bool), np.asarray(cal_dff_seg, dtype=float), np.nan),
                        "suite2pSpks": np.asarray(spks_seg, dtype=float),
                        "suite2pSpksMasked": np.where(np.asarray(cal_m_seg, dtype=bool), np.asarray(spks_seg, dtype=float), np.nan),
                    }
                )
        else:
            main_pkl = _pick_pkl(cell_folder, pkl_path=None) if len(pkl_paths) > 0 else None
            main_data = None
            if main_pkl is not None and os.path.isfile(main_pkl):
                with open(main_pkl, "rb") as f:
                    main_data = _safe_pickle_load(f)

            states.append(
                {
                    "state": "main",
                    "pkl_path": main_pkl,
                    "spike_pkl_data": main_data,
                    "vol_idx_bounds": (0, int(max(0, vol_raw.size - 1))),
                    "cal_idx_bounds": (0, int(max(0, cal_nb.size - 1))),
                    "volRaw": np.asarray(vol_raw, dtype=float),
                    "volDff": np.asarray(vol_dff, dtype=float),
                    "volMask": np.asarray(vol_mask, dtype=bool),
                    "volRawMasked": np.asarray(vol_raw_masked, dtype=float),
                    "volDffMasked": np.asarray(vol_dff_masked, dtype=float),
                    "calRaw": np.asarray(cal_nb, dtype=float),
                    "calDff": np.asarray(cal_dff, dtype=float),
                    "calMask": np.asarray(cal_mask, dtype=bool),
                    "calRawMasked": np.asarray(cal_raw_masked, dtype=float),
                    "calDffMasked": np.asarray(cal_dff_masked, dtype=float),
                    "suite2pSpks": np.asarray(spks_roi_aligned, dtype=float),
                    "suite2pSpksMasked": np.asarray(spks_roi_masked, dtype=float),
                }
            )

        all_cells.append(
            {
                "cell_path": cell_folder,
                "loaded": True,
                "from_db": bool(spec.get("from_db", False)),
                "brain_state": spec.get("brain_state", ""),
                "is_motor": bool(is_motor),
                "cal_sr_hz": float(spec["cal_sr_hz"]),
                "vol_sr_hz": float(vol_sr_hz),
                "sources": {
                    "cal_nb": cal_raw_source,
                    "cal_dff": cal_dff_source,
                    "vol_raw": os.path.basename(vol_path),
                    "cal_mask_found": bool(cal_mask_found),
                    "vol_mask_found": bool(vol_mask_found),
                    "changepoint": cp_path,
                    "suite2p_f_path": suite2p_meta.get("f_path", None),
                    "suite2p_spks_path": suite2p_meta.get("spks_path", None),
                },
                "suite2p": {
                    "roi_idx": suite2p_meta.get("roi_idx", None),
                    "roi_match_corr": suite2p_meta.get("match_corr", np.nan),
                    "roi_reference": suite2p_meta.get("matched_reference", None),
                    "spks_loaded": bool(suite2p_meta.get("spks_loaded", False)),
                    "spks_reason": suite2p_meta.get("spks_reason", None),
                    "spks_shape": suite2p_meta.get("spks_shape", None),
                },
                "f0": {
                    "cal_p8": float(cal_f0) if np.isfinite(cal_f0) else np.nan,
                    "vol_p10": float(vol_f0) if np.isfinite(vol_f0) else np.nan,
                },
                "full": {
                    "calRaw": np.asarray(cal_nb, dtype=float),
                    "calDff": np.asarray(cal_dff, dtype=float),
                    "calMask": np.asarray(cal_mask, dtype=bool),
                    "calRawMasked": np.asarray(cal_raw_masked, dtype=float),
                    "calDffMasked": np.asarray(cal_dff_masked, dtype=float),
                    "suite2pSpks": np.asarray(spks_roi_aligned, dtype=float),
                    "suite2pSpksMasked": np.asarray(spks_roi_masked, dtype=float),
                    "volRaw": np.asarray(vol_raw, dtype=float),
                    "volDff": np.asarray(vol_dff, dtype=float),
                    "volMask": np.asarray(vol_mask, dtype=bool),
                    "volRawMasked": np.asarray(vol_raw_masked, dtype=float),
                    "volDffMasked": np.asarray(vol_dff_masked, dtype=float),
                },
                "states": states,
            }
        )

    return all_cells


def _pearson_corr_valid(x, y):
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = min(xv.size, yv.size)
    if n < 3:
        return np.nan
    xv = xv[:n]
    yv = yv[:n]
    m = np.isfinite(xv) & np.isfinite(yv)
    if np.sum(m) < 3:
        return np.nan
    xv = xv[m]
    yv = yv[m]
    if float(np.nanstd(xv)) <= 0 or float(np.nanstd(yv)) <= 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def _best_lag_corr_spks_after_fr(fr_smooth, spks_smooth, max_lag_frames):
    fr = np.asarray(fr_smooth, dtype=float).ravel()
    sp = np.asarray(spks_smooth, dtype=float).ravel()
    n = min(fr.size, sp.size)
    if n < 3:
        return np.nan, 0
    fr = fr[:n]
    sp = sp[:n]

    max_lag = int(max(0, max_lag_frames))
    max_lag = min(max_lag, n - 2)
    best_corr = np.nan
    best_lag = 0
    for lag in range(max_lag + 1):
        if lag == 0:
            c = _pearson_corr_valid(fr, sp)
        else:
            # Calcium (spks) should come after voltage-derived FR.
            c = _pearson_corr_valid(fr[:-lag], sp[lag:])
        if np.isnan(c):
            continue
        if np.isnan(best_corr) or (c > best_corr):
            best_corr = float(c)
            best_lag = int(lag)
    return best_corr, best_lag


def _predict_cascade_for_single_trace(
    cal_trace_dff,
    model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    model_folder=None,
    threshold=0,
    verbosity=0,
):
    """
    Predict spike-rate-like activity from a single dF/F calcium trace using CASCADE.
    """
    x = np.asarray(cal_trace_dff, dtype=float).ravel()
    if x.size == 0:
        return np.asarray([], dtype=float)

    traces = x.reshape(1, -1).astype(float)  # neurons x time

    try:
        from cascade2p import cascade as _cascade_mod
    except Exception as exc:
        raise ImportError(
            "CASCADE is not available in this environment. "
            "Install the CASCADE package/repo so `from cascade2p import cascade` works."
        ) from exc

    # Use a stable default model directory (module-local) instead of cwd-relative path.
    if model_folder is None:
        model_folder_use = os.path.abspath(os.path.join(os.path.dirname(__file__), "Pretrained_models"))
    else:
        model_folder_use = os.path.abspath(str(model_folder))
    os.makedirs(model_folder_use, exist_ok=True)

    def _download_model_safe(name):
        if not hasattr(_cascade_mod, "download_model"):
            return None
        try:
            return _cascade_mod.download_model(str(name), model_folder=str(model_folder_use), verbose=max(0, int(verbosity)))
        except TypeError:
            try:
                return _cascade_mod.download_model(str(name), model_folder=str(model_folder_use))
            except TypeError:
                try:
                    return _cascade_mod.download_model(str(name), verbose=max(0, int(verbosity)))
                except TypeError:
                    return _cascade_mod.download_model(str(name))

    model_cfg = os.path.join(model_folder_use, str(model_name), "config.yaml")
    dl_err = None
    if not os.path.isfile(model_cfg):
        # Refresh model index first (CASCADE convention), then fetch requested model.
        try:
            _download_model_safe("update_models")
        except Exception:
            pass
        try:
            _download_model_safe(model_name)
        except Exception as exc:
            dl_err = exc

    try:
        y = _cascade_mod.predict(
            str(model_name),
            traces,
            model_folder=str(model_folder_use),
            threshold=threshold,
            verbosity=max(0, int(verbosity)),
        )
    except TypeError:
        try:
            y = _cascade_mod.predict(str(model_name), traces, str(model_folder_use))
        except Exception:
            y = _cascade_mod.predict(str(model_name), traces)
    except Exception as exc:
        if (not os.path.isfile(model_cfg)) and (dl_err is not None):
            raise RuntimeError(
                f"CASCADE model '{model_name}' is missing in '{model_folder_use}' "
                f"and auto-download failed: {dl_err}"
            ) from exc
        raise

    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        y = y[0]
    return np.asarray(y, dtype=float).ravel()


def _parse_cascade_model_smoothing_s(model_name):
    """
    Parse smoothing from CASCADE model name and return seconds.
    Supports patterns like:
      - ...smoothing50ms
      - ...smoothing_25ms
      - ...smoothing-25ms
    """
    s = str(model_name)
    m = re.search(r"smoothing[_-]?(\d+(?:\.\d+)?)ms", s, flags=re.IGNORECASE)
    if not m:
        return np.nan
    try:
        ms = float(m.group(1))
    except Exception:
        return np.nan
    if (not np.isfinite(ms)) or (ms <= 0):
        return np.nan
    return float(ms / 1000.0)


def _linear_fit_scale_offset(x, y):
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(xv) & np.isfinite(yv)
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.full_like(xv, np.nan, dtype=float)
    X = xv[valid].reshape(-1, 1)
    Y = yv[valid]

    scale = np.nan
    offset = np.nan
    try:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression(fit_intercept=True)
        model.fit(X, Y)
        scale = float(model.coef_[0])
        offset = float(model.intercept_)
    except Exception:
        # Fallback without sklearn
        A = np.column_stack([X[:, 0], np.ones(X.shape[0], dtype=float)])
        beta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        scale = float(beta[0])
        offset = float(beta[1])

    pred = scale * xv + offset
    return scale, offset, pred


def _spike_idx_to_fr_on_cal(spike_idx_vol, n_cal, vol_sr_hz, cal_sr_hz):
    n_cal = int(max(0, n_cal))
    if n_cal <= 0:
        return np.array([], dtype=float)
    sp = pe._as_sorted_unique_int(spike_idx_vol)
    if sp.size == 0:
        return np.zeros(n_cal, dtype=float)
    t_sp = np.asarray(sp, dtype=float) / float(vol_sr_hz)
    edges = np.arange(n_cal + 1, dtype=float) / float(cal_sr_hz)
    counts, _ = np.histogram(t_sp, bins=edges)
    return counts.astype(float) * float(cal_sr_hz)


def _robust_error_z(error):
    e = np.asarray(error, dtype=float).ravel()
    out = np.full_like(e, np.nan, dtype=float)
    valid = np.isfinite(e)
    if np.sum(valid) < 3:
        return out, np.nan, np.nan
    ev = e[valid]
    center = float(np.median(ev))
    sigma = float(1.4826 * np.median(np.abs(ev - center)))
    if (not np.isfinite(sigma)) or (sigma <= 0):
        sigma = float(np.nanstd(ev))
    if (not np.isfinite(sigma)) or (sigma <= 0):
        return out, center, np.nan
    out[valid] = (ev - center) / sigma
    return out, center, sigma


def _standard_error_z(error):
    e = np.asarray(error, dtype=float).ravel()
    out = np.full_like(e, np.nan, dtype=float)
    valid = np.isfinite(e)
    if np.sum(valid) < 3:
        return out, np.nan, np.nan
    ev = e[valid]
    center = float(np.mean(ev))
    sigma = float(np.std(ev, ddof=0))
    if (not np.isfinite(sigma)) or (sigma <= 0):
        return out, center, np.nan
    out[valid] = (ev - center) / sigma
    return out, center, sigma


def _fr_bin_matched_error_z(error, real_fr, bin_edges_hz=(0.0, 5.0, 10.0, 15.0, 20.0, np.inf)):
    """
    z_bias[t] = err[t] / sigma(err in the real-FR bin of frame t)
    Bins are defined over real FR in Hz, e.g. 0-5, 5-10, 10-15, 15-20, 20+.
    """
    e = np.asarray(error, dtype=float).ravel()
    r = np.asarray(real_fr, dtype=float).ravel()
    n = min(e.size, r.size)
    e = e[:n]
    r = r[:n]
    out = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(e) & np.isfinite(r)
    if np.sum(valid) < 3:
        return out, 0.0, np.nan, np.asarray(bin_edges_hz, dtype=float), np.full(max(1, len(bin_edges_hz) - 1), np.nan, dtype=float)

    edges = np.asarray(bin_edges_hz, dtype=float).ravel()
    if edges.size < 2:
        edges = np.array([0.0, np.inf], dtype=float)

    # Assign each valid frame to FR bin using real FR (negative values clipped to 0).
    rv = np.maximum(r[valid], 0.0)
    ev = e[valid]
    # Bin index in [0, n_bins-1].
    b = np.digitize(rv, edges[1:-1], right=False)
    n_bins = int(edges.size - 1)
    sigmas = np.full(n_bins, np.nan, dtype=float)

    global_sigma = float(np.nanstd(ev))
    if (not np.isfinite(global_sigma)) or (global_sigma <= 0):
        global_sigma = np.nan

    zv = np.full(ev.shape, np.nan, dtype=float)
    for bi in range(n_bins):
        m = b == bi
        if np.sum(m) < 3:
            sigma_b = np.nan
        else:
            sigma_b = float(np.nanstd(ev[m], ddof=0))
        if (not np.isfinite(sigma_b)) or (sigma_b <= 0):
            sigma_b = global_sigma
        sigmas[bi] = sigma_b
        if np.isfinite(sigma_b) and sigma_b > 0 and np.any(m):
            zv[m] = ev[m] / sigma_b

    out[valid] = zv
    # For compatibility with existing summary fields:
    # center is fixed at 0 by definition, sigma is median bin sigma.
    sigma_eff = float(np.nanmedian(sigmas)) if np.any(np.isfinite(sigmas)) else np.nan
    return out, 0.0, sigma_eff, edges, sigmas


def _windows_to_mask(windows, n):
    m = np.zeros(int(max(0, n)), dtype=bool)
    if m.size == 0:
        return m
    for s, e in windows:
        ss = int(max(0, s))
        ee = int(min(m.size - 1, e))
        if ee >= ss:
            m[ss : ee + 1] = True
    return m


def _merge_close_windows(windows, max_gap=0):
    if not windows:
        return []
    gap = int(max(0, max_gap))
    ww = sorted([(int(s), int(e)) for s, e in windows], key=lambda z: (z[0], z[1]))
    out = [ww[0]]
    for s, e in ww[1:]:
        ps, pe = out[-1]
        if s <= (pe + 1 + gap):
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _rolling_mean_with_min_valid(x, win_frames, min_valid_frac=0.80):
    a = np.asarray(x, dtype=float).ravel()
    n = int(a.size)
    if n == 0:
        return np.array([], dtype=float)
    w = int(max(1, win_frames))
    k = np.ones(w, dtype=float)
    valid = np.isfinite(a).astype(float)
    ax = np.where(np.isfinite(a), a, 0.0)
    num = np.convolve(ax, k, mode="same")
    den = np.convolve(valid, k, mode="same")
    need = int(max(1, np.ceil(float(min_valid_frac) * float(w))))
    out = np.full(n, np.nan, dtype=float)
    ok = den >= need
    out[ok] = num[ok] / den[ok]
    return out


def detect_under_over_shift_null_multiscale(
    error_pred_minus_real,
    cal_sr_hz,
    window_sizes_s=(0.25, 0.50, 1.00),
    z_thresh=2.0,
    n_shifts=200,
    min_valid_frac=0.80,
    min_supported_scales=2,
    merge_gap_frames=0,
    random_seed=0,
):
    """
    Multiscale under/over detection using a circular-shift null.

    Steps:
    1) error = predicted FR - real FR
    2) For each window size, compare sliding-window mean error vs circular-shift null
    3) Convert to z-like score: (obs - null_mean) / null_std
    4) Mark significant windows with z > z_thresh (over) or z < -z_thresh (under)
    5) Keep episodes supported by at least `min_supported_scales` window sizes

    Sign convention:
      - input must be predicted FR - real FR
      - positive episodes are overestimation
      - negative episodes are underestimation

    Returns masks/windows in calcium-frame coordinates.
    """
    e = np.asarray(error_pred_minus_real, dtype=float).ravel()
    n = int(e.size)
    out_empty = {
        "over_mask": np.zeros(n, dtype=bool),
        "under_mask": np.zeros(n, dtype=bool),
        "missed_mask": np.zeros(n, dtype=bool),
        "over_windows": [],
        "under_windows": [],
        "missed_windows": [],
        "z_score_trace": np.full(n, np.nan, dtype=float),
        "support_over": np.zeros(n, dtype=int),
        "support_under": np.zeros(n, dtype=int),
        "window_sizes_frames": [],
    }
    if n < 5 or (not np.any(np.isfinite(e))):
        return out_empty

    try:
        fs = float(cal_sr_hz)
    except Exception:
        fs = np.nan
    if (not np.isfinite(fs)) or fs <= 0:
        return out_empty

    # Normalize/validate window sizes
    w_frames = []
    for ws in window_sizes_s:
        try:
            wf = int(max(1, round(float(ws) * fs)))
        except Exception:
            continue
        if wf not in w_frames:
            w_frames.append(wf)
    if len(w_frames) == 0:
        return out_empty

    # Shifts for null (exclude shift=0).
    all_shifts = np.arange(1, n, dtype=int)
    if all_shifts.size == 0:
        return out_empty
    k = int(min(max(1, int(n_shifts)), int(all_shifts.size)))
    rng = np.random.default_rng(int(random_seed))
    if k >= all_shifts.size:
        shifts = all_shifts
    else:
        shifts = np.sort(rng.choice(all_shifts, size=k, replace=False))

    support_over = np.zeros(n, dtype=int)
    support_under = np.zeros(n, dtype=int)
    z_sum = np.zeros(n, dtype=float)
    z_cnt = np.zeros(n, dtype=int)

    for wf in w_frames:
        obs = _rolling_mean_with_min_valid(e, wf, min_valid_frac=min_valid_frac)
        null_stack = np.full((int(shifts.size), n), np.nan, dtype=float)
        for i, sh in enumerate(shifts.tolist()):
            es = np.roll(e, int(sh))
            null_stack[i, :] = _rolling_mean_with_min_valid(es, wf, min_valid_frac=min_valid_frac)
        mu = np.nanmean(null_stack, axis=0)
        sd = np.nanstd(null_stack, axis=0, ddof=0)
        z = np.full(n, np.nan, dtype=float)
        ok = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        z[ok] = (obs[ok] - mu[ok]) / sd[ok]

        zf = np.isfinite(z)
        z_sum[zf] += z[zf]
        z_cnt[zf] += 1

        pos = z > float(z_thresh)    # over-estimation: predicted > real
        neg = z < -float(z_thresh)   # under-estimation: predicted < real

        if int(merge_gap_frames) > 0:
            pos = _windows_to_mask(_merge_close_windows(_true_windows(pos), max_gap=int(merge_gap_frames)), n)
            neg = _windows_to_mask(_merge_close_windows(_true_windows(neg), max_gap=int(merge_gap_frames)), n)

        support_over += pos.astype(int)
        support_under += neg.astype(int)

    min_sup = int(max(1, min(int(min_supported_scales), len(w_frames))))
    over_mask = support_over >= min_sup
    under_mask = support_under >= min_sup
    missed_mask = over_mask | under_mask

    over_windows = _merge_close_windows(_true_windows(over_mask), max_gap=int(max(0, merge_gap_frames)))
    under_windows = _merge_close_windows(_true_windows(under_mask), max_gap=int(max(0, merge_gap_frames)))
    missed_windows = _merge_close_windows(_true_windows(missed_mask), max_gap=int(max(0, merge_gap_frames)))

    over_mask = _windows_to_mask(over_windows, n)
    under_mask = _windows_to_mask(under_windows, n)
    missed_mask = _windows_to_mask(missed_windows, n)

    z_score_trace = np.full(n, np.nan, dtype=float)
    z_ok = z_cnt > 0
    z_score_trace[z_ok] = z_sum[z_ok] / z_cnt[z_ok]

    return {
        "over_mask": over_mask,
        "under_mask": under_mask,
        "missed_mask": missed_mask,
        "over_windows": over_windows,
        "under_windows": under_windows,
        "missed_windows": missed_windows,
        "z_score_trace": z_score_trace,
        "support_over": support_over,
        "support_under": support_under,
        "window_sizes_frames": w_frames,
    }


def _true_windows(mask_bool):
    m = np.asarray(mask_bool, dtype=bool).ravel()
    windows = []
    if m.size == 0:
        return windows
    i = 0
    n = int(m.size)
    while i < n:
        if not m[i]:
            i += 1
            continue
        s = int(i)
        while i < n and m[i]:
            i += 1
        e = int(i - 1)
        windows.append((s, e))
    return windows


def _event_class_and_size(ev):
    et = str(ev.get("event_type", "simple")).strip().lower()
    is_plateau = bool(ev.get("is_plateau_event", False)) or (et == "plateau")
    is_complex = bool(ev.get("is_complex_event", False)) or (et == "complex")
    if is_plateau:
        cls = "plateau"
    elif is_complex:
        cls = "complex"
    else:
        cls = "simple"
    sp = pe._as_sorted_unique_int(ev.get("spikes", []))
    if sp.size > 0:
        n_sp = int(sp.size)
    else:
        try:
            n_sp = int(ev.get("n_spikes", 1))
        except Exception:
            n_sp = 1
    n_sp = int(max(1, n_sp))
    return cls, n_sp


def _first_spike_or_start_v(ev):
    sp = pe._as_sorted_unique_int(ev.get("spikes", []))
    if sp.size > 0:
        return int(sp[0])
    try:
        return int(ev.get("start_frame", 0))
    except Exception:
        return 0


def _vol_idx_to_cal_idx(v_idx, vol_sr, cal_sr, cal_len):
    n = int(max(0, cal_len))
    if n <= 0:
        return 0
    t = float(v_idx) / float(vol_sr)
    c = int(round(t * float(cal_sr)))
    return int(max(0, min(n - 1, c)))


def _normalize_bool_mask(mask_like, target_len):
    n = int(max(0, target_len))
    if n <= 0:
        return np.array([], dtype=bool)
    m = np.asarray(mask_like, dtype=bool).ravel()
    if m.size == n:
        return m
    out = np.ones(n, dtype=bool)
    out[: min(n, m.size)] = m[: min(n, m.size)]
    return out


def _estimate_compact_index_offset_from_trace(pkl_data, vol_raw_state, vol_mask_state, max_shift_frames=8):
    """
    Estimate a small compact-index offset between PKL trace_vol and state compact voltage.

    Returns `off` in compact-index samples such that:
      compact_idx_state ~= compact_idx_pkl + off
    """
    if (not isinstance(pkl_data, dict)) or (vol_raw_state is None):
        return 0
    try:
        tv = np.asarray(pkl_data.get("trace_vol", []), dtype=float).ravel()
    except Exception:
        return 0
    v = np.asarray(vol_raw_state, dtype=float).ravel()
    if tv.size < 20 or v.size < 20:
        return 0
    vmask = _normalize_bool_mask(vol_mask_state, v.size)
    vc = np.asarray(v[vmask], dtype=float).ravel()
    if vc.size < 20:
        return 0

    def _corr(a, b):
        n = min(a.size, b.size)
        if n < 20:
            return np.nan
        a = np.asarray(a[:n], dtype=float)
        b = np.asarray(b[:n], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if np.sum(ok) < 20:
            return np.nan
        a = a[ok]
        b = b[ok]
        if (a.size < 20) or (np.nanstd(a) <= 0) or (np.nanstd(b) <= 0):
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    max_s = int(max(0, max_shift_frames))
    base_r = _corr(tv, vc)
    best_r = np.nan
    best_off = 0
    for off in range(-max_s, max_s + 1):
        if off >= 0:
            a = tv
            b = vc[off:]
        else:
            k = -off
            a = tv[k:]
            b = vc
        r = _corr(a, b)
        if not np.isfinite(r):
            continue
        if (not np.isfinite(best_r)) or (r > best_r):
            best_r = float(r)
            best_off = int(off)

    if np.isfinite(base_r) and np.isfinite(best_r) and (best_r - base_r) < 0.01:
        return 0
    return int(best_off)


def _spike_idx_from_state_pkl_aligned(
    pkl_data,
    vol_len,
    vol_mask_state=None,
    vol_raw_state=None,
    pkl_path=None,
):
    """
    Return vm_all_spikes aligned to the state's local full-length voltage index space.

    Handles two common PKL index conventions:
      1) full local indices (0..vol_len-1)
      2) mask-compacted indices (0..n_true(vol_mask)-1), typically when spike detection
         was run on TraceV[VolMask] and indices were not remapped back.
    """
    if not isinstance(pkl_data, dict):
        return np.array([], dtype=int)

    sp = pe._as_sorted_unique_int(pkl_data.get("vm_all_spikes", []))
    n_v = int(max(0, vol_len))
    if sp.size == 0 or n_v <= 0:
        return np.array([], dtype=int)

    # Corrected high-plateau PKLs are expected to already be in full local index space.
    # Bypass compact-index remap and offset estimation for them.
    if _is_rm_complex_highplateau_pkl(pkl_path):
        return sp[(sp >= 0) & (sp < n_v)]

    info = _detect_state_pkl_compact_index_mode(
        pkl_data=pkl_data,
        vol_len=n_v,
        vol_mask_state=vol_mask_state,
    )
    vmask = info["vol_mask"]
    needs_remap = bool(info["needs_remap"])

    if needs_remap:
        keep_idx = np.flatnonzero(vmask)
        if keep_idx.size <= 0:
            return np.array([], dtype=int)
        compact_off = _estimate_compact_index_offset_from_trace(
            pkl_data=pkl_data,
            vol_raw_state=vol_raw_state,
            vol_mask_state=vmask,
            max_shift_frames=8,
        )
        sp_adj = np.asarray(sp, dtype=np.int64) + int(compact_off)
        good = (sp_adj >= 0) & (sp_adj < keep_idx.size)
        sp_map = keep_idx[sp_adj[good]]
        sp_map = pe._as_sorted_unique_int(sp_map)
        return sp_map[(sp_map >= 0) & (sp_map < n_v)]

    # Already in full local index space.
    return sp[(sp >= 0) & (sp < n_v)]


def _detect_state_pkl_compact_index_mode(pkl_data, vol_len, vol_mask_state=None):
    """
    Detect whether PKL spike indices are in mask-compacted voltage-index space.

    Returns a dict with:
      - needs_remap: bool
      - tv_len: length of pkl trace_vol (0 if missing)
      - n_true: number of True in state volMask
      - sp_max: max vm_all_spikes index (-1 if empty)
      - vol_mask: normalized bool mask (length vol_len)
    """
    n_v = int(max(0, vol_len))
    vmask = _normalize_bool_mask(
        np.ones(n_v, dtype=bool) if vol_mask_state is None else vol_mask_state,
        n_v,
    )
    n_true = int(np.sum(vmask))

    sp = pe._as_sorted_unique_int(pkl_data.get("vm_all_spikes", [])) if isinstance(pkl_data, dict) else np.array([], dtype=int)
    sp_max = int(np.max(sp)) if sp.size > 0 else -1

    try:
        tv = np.asarray(pkl_data.get("trace_vol", []), dtype=float).ravel() if isinstance(pkl_data, dict) else np.array([], dtype=float)
        tv_len = int(tv.size)
    except Exception:
        tv_len = 0

    needs_remap = False
    if (tv_len > 0) and (tv_len < n_v) and (abs(tv_len - n_true) <= 2) and (sp_max < max(tv_len, 1)):
        needs_remap = True
    elif (sp_max >= n_v) and (sp_max < max(n_true, 1)):
        needs_remap = True

    return {
        "needs_remap": bool(needs_remap),
        "tv_len": int(tv_len),
        "n_true": int(n_true),
        "sp_max": int(sp_max),
        "vol_mask": np.asarray(vmask, dtype=bool),
    }


def _map_idx_array_compact_to_full(idx_like, keep_idx, vol_len):
    vals = np.asarray(idx_like, dtype=float).ravel()
    if vals.size == 0:
        return np.array([], dtype=np.int64), 0
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=np.int64), 0
    idx = np.asarray(np.round(vals), dtype=np.int64)
    good = (idx >= 0) & (idx < int(keep_idx.size))
    n_drop = int(np.sum(~good))
    idx = idx[good]
    if idx.size == 0:
        return np.array([], dtype=np.int64), n_drop
    mapped = np.asarray(keep_idx[idx], dtype=np.int64)
    good2 = (mapped >= 0) & (mapped < int(max(0, vol_len)))
    n_drop += int(np.sum(~good2))
    return mapped[good2], n_drop


def _rewrite_one_state_pkl_indices(pkl_data, vol_len, vol_mask_state=None, vol_raw_state=None):
    """
    Rewrite relevant index fields from compact-index to full-index space for one state PKL dict.
    """
    if not isinstance(pkl_data, dict):
        return pkl_data, {"changed": False, "reason": "pkl_data_not_dict", "changed_keys": []}

    det = _detect_state_pkl_compact_index_mode(
        pkl_data=pkl_data,
        vol_len=vol_len,
        vol_mask_state=vol_mask_state,
    )
    if not bool(det["needs_remap"]):
        return pkl_data, {
            "changed": False,
            "reason": "already_full_index",
            "changed_keys": [],
            "tv_len": int(det["tv_len"]),
            "n_true": int(det["n_true"]),
            "sp_max": int(det["sp_max"]),
        }

    d2 = copy.deepcopy(pkl_data)
    keep_idx = np.flatnonzero(np.asarray(det["vol_mask"], dtype=bool))
    compact_off = _estimate_compact_index_offset_from_trace(
        pkl_data=pkl_data,
        vol_raw_state=vol_raw_state,
        vol_mask_state=np.asarray(det["vol_mask"], dtype=bool),
        max_shift_frames=8,
    )
    n_v = int(max(0, vol_len))
    changed_keys = []
    n_dropped_total = 0

    index_keys = [
        "vm_all_spikes",
        "vm_simple_spikes",
        "vm_complex_spikes",
        "vm_simple_spike",
        "vm_complex_spike",
        "all_spikes",
        "spike_indices",
        "spikes",
        "spike_idx",
        "spikeID",
        "vm_simple_spikes_before_raw_threshold",
        "vm_simple_spikes_before_spikeclicy",
        "vm_complex_spikes_before_spikeclicy",
    ]

    for k in index_keys:
        if k not in d2:
            continue
        vals = np.asarray(d2.get(k, []), dtype=float).ravel()
        if vals.size > 0:
            vals = vals + float(compact_off)
        mapped, n_drop = _map_idx_array_compact_to_full(vals, keep_idx=keep_idx, vol_len=n_v)
        d2[k] = pe._as_sorted_unique_int(mapped)
        changed_keys.append(k)
        n_dropped_total += int(n_drop)

    pdct = d2.get("vm_plateaus_dict", None)
    if isinstance(pdct, dict):
        for k in ("starts", "ends", "locs"):
            if k in pdct:
                vals = np.asarray(pdct.get(k, []), dtype=float).ravel()
                if vals.size > 0:
                    vals = vals + float(compact_off)
                mapped, n_drop = _map_idx_array_compact_to_full(vals, keep_idx=keep_idx, vol_len=n_v)
                pdct[k] = np.asarray(mapped, dtype=np.int64)
                changed_keys.append(f"vm_plateaus_dict.{k}")
                n_dropped_total += int(n_drop)

        if isinstance(pdct.get("spike_indices", None), list):
            out_si = []
            for row in pdct.get("spike_indices", []):
                vals = np.asarray(row, dtype=float).ravel()
                if vals.size > 0:
                    vals = vals + float(compact_off)
                mapped, n_drop = _map_idx_array_compact_to_full(vals, keep_idx=keep_idx, vol_len=n_v)
                out_si.append(np.asarray(pe._as_sorted_unique_int(mapped), dtype=np.int64))
                n_dropped_total += int(n_drop)
            pdct["spike_indices"] = out_si
            changed_keys.append("vm_plateaus_dict.spike_indices")
        d2["vm_plateaus_dict"] = pdct

    return d2, {
        "changed": True,
        "reason": "remapped_compact_to_full",
        "changed_keys": sorted(list(set(changed_keys))),
        "n_dropped_indices": int(n_dropped_total),
        "compact_offset_frames": int(compact_off),
        "tv_len": int(det["tv_len"]),
        "n_true": int(det["n_true"]),
        "sp_max": int(det["sp_max"]),
    }


def rewrite_spike_pkls_with_aligned_indices(
    db_path=None,
    cell_paths=None,
    default_cal_sr_hz=FS_CA_HZ_FALLBACK,
    vol_sr_hz=FS_VOL_HZ_DEFAULT,
    require_existing_dirs=True,
    backup_suffix=".pre_indexfix.bak",
    dry_run=False,
    verbose=True,
    report_csv_path=None,
):
    """
    Rewrite spike PKLs in-place so index arrays are in full state-local voltage index space.

    Uses current mask/changepoint loading to detect compact-index PKLs and remap them.
    A backup copy is created once per PKL path (unless dry_run=True).
    """
    cells = load_cells_with_spike_pkls(
        db_path=db_path,
        cell_paths=cell_paths,
        default_cal_sr_hz=default_cal_sr_hz,
        vol_sr_hz=vol_sr_hz,
        require_existing_dirs=require_existing_dirs,
    )

    rows = []
    seen_paths = set()
    for cell in cells:
        if not bool(cell.get("loaded", False)):
            rows.append(
                {
                    "cell_path": str(cell.get("cell_path", "")),
                    "state": "",
                    "pkl_path": "",
                    "status": "skip_not_loaded",
                    "reason": str(cell.get("error", "")),
                }
            )
            continue

        cpath = str(cell.get("cell_path", ""))
        for st in cell.get("states", []):
            pkl_path = st.get("pkl_path", None)
            pkl_data = st.get("spike_pkl_data", None)
            state = str(st.get("state", ""))
            vol_raw = np.asarray(st.get("volRaw", []), dtype=float).ravel()
            vol_mask = np.asarray(st.get("volMask", np.ones(vol_raw.size, dtype=bool)), dtype=bool)
            vol_len = int(vol_raw.size)

            if (not isinstance(pkl_path, str)) or (pkl_path.strip() == ""):
                rows.append(
                    {
                        "cell_path": cpath,
                        "state": state,
                        "pkl_path": "",
                        "status": "skip_no_pkl",
                        "reason": "",
                    }
                )
                continue

            pkl_key = os.path.normcase(os.path.abspath(pkl_path))
            if pkl_key in seen_paths:
                rows.append(
                    {
                        "cell_path": cpath,
                        "state": state,
                        "pkl_path": pkl_path,
                        "status": "skip_duplicate_pkl_in_states",
                        "reason": "",
                    }
                )
                continue
            seen_paths.add(pkl_key)

            if not isinstance(pkl_data, dict):
                rows.append(
                    {
                        "cell_path": cpath,
                        "state": state,
                        "pkl_path": pkl_path,
                        "status": "skip_bad_pkl_data",
                        "reason": "spike_pkl_data is not dict",
                    }
                )
                continue

            d_new, info = _rewrite_one_state_pkl_indices(
                pkl_data=pkl_data,
                vol_len=vol_len,
                vol_mask_state=vol_mask,
                vol_raw_state=vol_raw,
            )

            if not bool(info.get("changed", False)):
                rows.append(
                    {
                        "cell_path": cpath,
                        "state": state,
                        "pkl_path": pkl_path,
                        "status": "no_change",
                        "reason": str(info.get("reason", "")),
                        "tv_len": info.get("tv_len", np.nan),
                        "n_true": info.get("n_true", np.nan),
                        "sp_max": info.get("sp_max", np.nan),
                        "compact_offset_frames": info.get("compact_offset_frames", 0),
                    }
                )
                continue

            status = "dry_run_would_rewrite" if bool(dry_run) else "rewritten"
            err_msg = ""
            if not bool(dry_run):
                try:
                    bkp = str(pkl_path) + str(backup_suffix)
                    if (backup_suffix is not None) and (str(backup_suffix).strip() != "") and (not os.path.isfile(bkp)):
                        shutil.copy2(pkl_path, bkp)
                    with open(pkl_path, "wb") as f:
                        pickle.dump(d_new, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as ex:
                    status = "rewrite_failed"
                    err_msg = str(ex)

            rows.append(
                {
                    "cell_path": cpath,
                    "state": state,
                    "pkl_path": pkl_path,
                    "status": status,
                    "reason": str(info.get("reason", "")) if err_msg == "" else err_msg,
                    "tv_len": info.get("tv_len", np.nan),
                    "n_true": info.get("n_true", np.nan),
                    "sp_max": info.get("sp_max", np.nan),
                    "n_dropped_indices": info.get("n_dropped_indices", 0),
                    "compact_offset_frames": info.get("compact_offset_frames", 0),
                    "changed_keys": ";".join(info.get("changed_keys", [])),
                }
            )

    rep = pd.DataFrame(rows)
    if isinstance(report_csv_path, str) and report_csv_path.strip() != "":
        try:
            os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)
        except Exception:
            pass
        try:
            rep.to_csv(report_csv_path, index=False)
        except Exception:
            pass

    if bool(verbose):
        try:
            n_rw = int(np.sum(rep["status"] == "rewritten")) if "status" in rep.columns else 0
            n_dr = int(np.sum(rep["status"] == "dry_run_would_rewrite")) if "status" in rep.columns else 0
            n_nc = int(np.sum(rep["status"] == "no_change")) if "status" in rep.columns else 0
            n_er = int(np.sum(rep["status"] == "rewrite_failed")) if "status" in rep.columns else 0
            print(f"[rewrite_spike_pkls_with_aligned_indices] rewritten={n_rw} dry_run={n_dr} no_change={n_nc} failed={n_er}")
        except Exception:
            pass
    return rep


def _plot_under_over_stacked_bars(event_df, title, save_html, save_svg=None, total_counts_by_class=None):
    class_order = ("simple", "complex", "plateau")
    color_map = {"simple": "black", "complex": "red", "plateau": "purple"}

    if event_df is None:
        event_df = pd.DataFrame(columns=["category", "event_class", "n_spikes"])
    else:
        event_df = event_df.copy()
    if "category" not in event_df.columns:
        event_df["category"] = ""
    if "event_class" not in event_df.columns:
        event_df["event_class"] = "simple"
    if "n_spikes" not in event_df.columns:
        event_df["n_spikes"] = 1
    event_df["category"] = event_df["category"].astype(str).str.lower()
    event_df["event_class"] = event_df["event_class"].astype(str).str.lower()

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
        ],
        subplot_titles=(
            "Underestimate windows (pred < real)",
            "Overestimate windows (pred > real)",
            "All flagged windows (under + over)",
            "Underestimate: simple / complex / plateau (%)",
            "Overestimate: simple / complex / plateau (%)",
            "All flagged: simple / complex / plateau (%)",
            "Complex events: under / over / other (from total complex)",
            "Simple events: under / over / other (from total simple)",
            "Plateau events: under / over / other (from total plateau)",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
    )

    for col_idx, cat in ((1, "under"), (2, "over"), (3, "all")):
        if cat == "all":
            sub = event_df[event_df["category"].isin(["under", "over"])].copy()
        else:
            sub = event_df[event_df["category"] == cat].copy()
        if len(sub) == 0:
            xref = "x domain" if int(col_idx) == 1 else f"x{int(col_idx)} domain"
            yref = "y domain" if int(col_idx) == 1 else f"y{int(col_idx)} domain"
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=xref,
                yref=yref,
                text="No events",
                showarrow=False,
                font=dict(size=13, color="#555"),
            )
            continue
        grp = (
            sub.groupby(["n_spikes", "event_class"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["n_spikes", "event_class"])
        )
        for evc in class_order:
            ss = grp[grp["event_class"] == evc].copy()
            if len(ss) == 0:
                continue
            fig.add_trace(
                go.Bar(
                    x=ss["n_spikes"].astype(int),
                    y=ss["count"].astype(int),
                    name=evc,
                    marker_color=color_map.get(evc, "#777777"),
                    legendgroup=evc,
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="event size (# spikes)", row=1, col=col_idx)
        fig.update_yaxes(title_text="# events", row=1, col=col_idx)
        vals = [int(np.sum(sub["event_class"] == evc)) for evc in class_order]
        if np.sum(vals) <= 0:
            vals = [1, 1, 1]
            labels = [f"{evc} (0)" for evc in class_order]
        else:
            labels = list(class_order)
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=vals,
                marker=dict(colors=[color_map[k] for k in class_order]),
                textinfo="label+percent+value",
                sort=False,
                showlegend=False,
            ),
            row=2,
            col=col_idx,
        )

    # Class-centered pies using totals (denominator = all events of each class)
    if total_counts_by_class is None:
        total_counts_by_class = {}
    class_colors = ["#F2CF1D", "#2ca02c", "#BDBDBD"]  # under, over, other
    for col_idx, evc in enumerate(("complex", "simple", "plateau"), start=1):
        total_n = int(max(0, total_counts_by_class.get(evc, 0)))
        under_n = int(np.sum((event_df["category"] == "under") & (event_df["event_class"] == evc))) if len(event_df) else 0
        over_n = int(np.sum((event_df["category"] == "over") & (event_df["event_class"] == evc))) if len(event_df) else 0
        other_n = max(0, total_n - under_n - over_n)
        vals = [under_n, over_n, other_n]
        if np.sum(vals) <= 0:
            vals = [1, 1, 1]
            labels = ["under (0)", "over (0)", "other (0)"]
        else:
            labels = ["under", "over", "other"]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=vals,
                marker=dict(colors=class_colors),
                textinfo="label+percent+value",
                sort=False,
                showlegend=False,
            ),
            row=3,
            col=col_idx,
        )

    fig.update_layout(
        template="simple_white",
        width=1900,
        height=1450,
        barmode="stack",
        title=title,
        legend=dict(orientation="h"),
    )
    fig.write_html(save_html)
    if save_svg is not None:
        try:
            fig.write_image(save_svg)
        except Exception:
            pass


def evaluate_suite2p_spks_as_continuous_activity(
    loaded_cells,
    out_subdir="suite2p_spks_eval_continuous",
    smooth_sigma_s=0.2,
    max_lag_s=0.5,
    save_svg=False,
    z_thresh=2.0,
    error_z_mode="standard",
    shift_null_window_sizes_s=(0.25, 0.50, 1.00),
    shift_null_n_shifts=200,
    shift_null_min_supported_scales=2,
    shift_null_min_valid_frac=0.80,
    shift_null_merge_gap_s=0.0,
    shift_null_seed=0,
    corrlation_thr=None,
    correlation_thr=None,
    apply_linear_scaling=True,
    use_cal_mask=True,
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\suite2p_vs_realfr\countineous_aproch",
    spike_idx_offset_frames=0,
):
    """
    Evaluate suite2p deconvolved `spks` as continuous activity vs real FR.

    Per cell/state:
    1) Build FR on calcium bins from voltage spikes (`vm_all_spikes` in pkl), then Gaussian smooth.
    2) Gaussian smooth suite2p `spks`.
    3) Pearson correlation at zero lag and best lag (spks lagging FR by 0..max_lag_s).
    4) Optionally fit linear scaling (with intercept) from smoothed spks to smoothed FR.
    5) Save two plots:
       - scatter + fitted line
       - FR(real) vs FR(pred from spks) over time

    Error/episode modes:
      - `error_z_mode="standard"`: z-score uses one global error sigma per cell/state.
      - `error_z_mode="fr_bin"`: z-score uses the sigma from the matching real-FR bin
        (0-5, 5-10, 10-15, 15-20, 20+ Hz).
      - `error_z_mode="shift_null"`: over/under windows are detected from multiscale
        circular-shift null tests instead of pointwise z-thresholding.

    Correlation filtering:
      - `corrlation_thr` is kept for backward compatibility with existing notebooks.
      - `correlation_thr` is the correctly spelled alias.
      - Either threshold filters low-correlation cells from pooled stacked-bar summaries;
        per-cell outputs are still generated.
    """
    rows = []
    all_series_for_summary = []
    all_event_rows = []
    all_event_rows_closest_pre500 = []
    all_total_counts_by_class = {"simple": 0, "complex": 0, "plateau": 0}
    z_mode_requested = str(error_z_mode).strip().lower()
    z_mode_file_tag = re.sub(r"[^a-z0-9]+", "_", z_mode_requested).strip("_")
    if z_mode_file_tag == "":
        z_mode_file_tag = "mode"
    shift_null_aliases = {
        "shift_null",
        "shiftnull",
        "circular_shift",
        "multiscale_shift_null",
        "window_shift_null",
    }
    z_mode_title_tag = z_mode_requested
    try:
        _ws_in = [float(v) for v in np.asarray(shift_null_window_sizes_s, dtype=float).ravel().tolist() if np.isfinite(v) and float(v) > 0]
    except Exception:
        _ws_in = []
    if (z_mode_requested in shift_null_aliases) and (len(_ws_in) == 1):
        _w = float(_ws_in[0])
        _w_txt = f"{_w:.3f}".rstrip("0").rstrip(".")
        z_mode_file_tag = f"{z_mode_file_tag}_singlew_{_w_txt.replace('.', 'p')}s"
        z_mode_title_tag = f"{z_mode_requested} | single_window={_w_txt}s"

    def _lag_correct_spks(spks_smooth_in, lag_frames):
        x = np.asarray(spks_smooth_in, dtype=float).ravel()
        n = x.size
        lag = int(max(0, lag_frames))
        if lag <= 0:
            return x.copy()
        out = np.full(n, np.nan, dtype=float)
        if lag < n:
            # Calcium should come after voltage, so advance calcium by lag.
            out[: n - lag] = x[lag:]
        return out

    for cell in loaded_cells:
        if not bool(cell.get("loaded", False)):
            continue
        cell_path = str(cell.get("cell_path", ""))
        cell_name = os.path.basename(cell_path)
        cal_sr = float(cell.get("cal_sr_hz", FS_CA_HZ_FALLBACK))
        vol_sr = float(cell.get("vol_sr_hz", FS_VOL_HZ_DEFAULT))
        sigma_frames = float(smooth_sigma_s) * float(cal_sr)
        max_lag_frames = int(round(float(max_lag_s) * float(cal_sr)))

        out_dir = os.path.join(cell_path, str(out_subdir))
        os.makedirs(out_dir, exist_ok=True)
        cell_event_rows = []
        cell_event_rows_closest_pre500 = []
        cell_total_counts_by_class = {"simple": 0, "complex": 0, "plateau": 0}

        for st in cell.get("states", []):
            state = str(st.get("state", "main"))
            pkl_data = st.get("spike_pkl_data", None)
            if not isinstance(pkl_data, dict):
                rows.append(
                    {
                        "cell_path": cell_path,
                        "state": state,
                        "ok": False,
                        "reason": "Missing spike_pkl_data",
                    }
                )
                continue

            spks = np.asarray(st.get("suite2pSpks", []), dtype=float).ravel()
            n_cal = int(spks.size)
            if n_cal <= 2:
                rows.append(
                    {
                        "cell_path": cell_path,
                        "state": state,
                        "ok": False,
                        "reason": "Empty suite2pSpks for state",
                    }
                )
                continue

            vol_len = int(np.asarray(st.get("volRaw", []), dtype=float).size)
            if bool(use_cal_mask):
                cal_mask_state = _normalize_bool_mask(st.get("calMask", np.ones(n_cal, dtype=bool)), n_cal)
            else:
                cal_mask_state = np.ones(n_cal, dtype=bool)
            vol_mask_state = _normalize_bool_mask(st.get("volMask", np.ones(vol_len, dtype=bool)), vol_len)
            spk_idx = _spike_idx_from_state_pkl_aligned(
                pkl_data=pkl_data,
                vol_len=vol_len,
                vol_mask_state=vol_mask_state,
                vol_raw_state=np.asarray(st.get("volRaw", []), dtype=float).ravel(),
                pkl_path=st.get("pkl_path", None),
            )

            # Optional tiny manual nudge in voltage frames for residual alignment.
            try:
                _sp_off = int(round(float(spike_idx_offset_frames)))
            except Exception:
                _sp_off = 0
            if _sp_off != 0 and spk_idx.size > 0:
                spk_idx = pe._as_sorted_unique_int(np.asarray(spk_idx, dtype=np.int64) + int(_sp_off))
                spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < vol_len)]

            # Always enforce masks for spike indices:
            # keep only spikes in volMask=True and mapped calMask=True frames.
            if spk_idx.size > 0:
                spk_idx = spk_idx[vol_mask_state[spk_idx]]
            if spk_idx.size > 0:
                cidx_sp = np.array(
                    [_vol_idx_to_cal_idx(vv, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal) for vv in spk_idx],
                    dtype=int,
                )
                keep_sp = cal_mask_state[cidx_sp]
                spk_idx = spk_idx[keep_sp]

            fr_on_cal = _spike_idx_to_fr_on_cal(
                spike_idx_vol=spk_idx,
                n_cal=n_cal,
                vol_sr_hz=vol_sr,
                cal_sr_hz=cal_sr,
            )
            fr_smooth = gaussian_filter1d(fr_on_cal.astype(float), sigma=sigma_frames, mode="nearest")
            spks_smooth = gaussian_filter1d(spks.astype(float), sigma=sigma_frames, mode="nearest")

            # Always mark masked calcium frames as NaN (do not drop), so analysis
            # uses masked traces only and excludes False-mask regions.
            invalid = (~cal_mask_state) if bool(use_cal_mask) else np.zeros(n_cal, dtype=bool)
            fr_on_cal = np.asarray(fr_on_cal, dtype=float)
            fr_smooth = np.asarray(fr_smooth, dtype=float)
            spks = np.asarray(spks, dtype=float)
            spks_smooth = np.asarray(spks_smooth, dtype=float)
            fr_on_cal[invalid] = np.nan
            fr_smooth[invalid] = np.nan
            spks[invalid] = np.nan
            spks_smooth[invalid] = np.nan

            corr_0lag = _pearson_corr_valid(spks_smooth, fr_smooth)
            corr_best_lag, lag_frames_best = _best_lag_corr_spks_after_fr(
                fr_smooth=fr_smooth,
                spks_smooth=spks_smooth,
                max_lag_frames=max_lag_frames,
            )
            lag_s_best = float(lag_frames_best) / float(cal_sr)

            spks_smooth_lagcorr = _lag_correct_spks(spks_smooth, lag_frames_best)
            if bool(apply_linear_scaling):
                scale, offset, fr_pred = _linear_fit_scale_offset(spks_smooth, fr_smooth)
                scale_lag, offset_lag, fr_pred_lag = _linear_fit_scale_offset(spks_smooth_lagcorr, fr_smooth)
            else:
                scale, offset = 1.0, 0.0
                scale_lag, offset_lag = 1.0, 0.0
                fr_pred = np.asarray(spks_smooth, dtype=float)
                fr_pred_lag = np.asarray(spks_smooth_lagcorr, dtype=float)
            corr_pred_vs_real = _pearson_corr_valid(fr_pred, fr_smooth)
            corr_pred_lag_vs_real = _pearson_corr_valid(fr_pred_lag, fr_smooth)
            # Legacy convention (kept for z-based modes): real - predicted
            err_lag = np.asarray(fr_smooth, dtype=float) - np.asarray(fr_pred_lag, dtype=float)
            # New shift-null method convention: predicted - real
            err_pred_minus_real = np.asarray(fr_pred_lag, dtype=float) - np.asarray(fr_smooth, dtype=float)
            err_bin_edges = np.array([0.0, 5.0, 10.0, 15.0, 20.0, np.inf], dtype=float)
            err_bin_sigmas = np.full(err_bin_edges.size - 1, np.nan, dtype=float)
            shift_null_scales_frames = []
            z_mode = str(error_z_mode).strip().lower()
            err_for_plot = err_lag
            err_plot_label = "error (real - pred)"
            z_panel_title = "Error z-score"
            if z_mode in ("standard", "std", "z"):
                err_z, err_center, err_sigma = _standard_error_z(err_lag)
                z_mode_used = "standard"
            elif z_mode in ("fr_bin", "fr_bin_matched", "z_bias", "binned"):
                err_z, err_center, err_sigma, err_bin_edges, err_bin_sigmas = _fr_bin_matched_error_z(
                    err_lag,
                    fr_smooth,
                    bin_edges_hz=err_bin_edges,
                )
                z_mode_used = "fr_bin_matched"
            elif z_mode in ("shift_null", "shiftnull", "circular_shift", "multiscale_shift_null", "window_shift_null"):
                merge_gap_frames = int(max(0, round(float(shift_null_merge_gap_s) * float(cal_sr))))
                det = detect_under_over_shift_null_multiscale(
                    error_pred_minus_real=err_pred_minus_real,
                    cal_sr_hz=cal_sr,
                    window_sizes_s=shift_null_window_sizes_s,
                    z_thresh=float(z_thresh),
                    n_shifts=int(shift_null_n_shifts),
                    min_valid_frac=float(shift_null_min_valid_frac),
                    min_supported_scales=int(shift_null_min_supported_scales),
                    merge_gap_frames=merge_gap_frames,
                    random_seed=int(shift_null_seed),
                )
                over_mask = np.asarray(det["over_mask"], dtype=bool)
                under_mask = np.asarray(det["under_mask"], dtype=bool)
                missed_mask = np.asarray(det["missed_mask"], dtype=bool)
                over_windows = list(det["over_windows"])
                under_windows = list(det["under_windows"])
                missed_windows = list(det["missed_windows"])
                err_z = np.asarray(det["z_score_trace"], dtype=float)
                shift_null_scales_frames = [int(v) for v in det.get("window_sizes_frames", [])]
                err_center = float(np.nanmean(err_pred_minus_real)) if np.any(np.isfinite(err_pred_minus_real)) else np.nan
                err_sigma = float(np.nanstd(err_pred_minus_real)) if np.any(np.isfinite(err_pred_minus_real)) else np.nan
                err_for_plot = err_pred_minus_real
                err_plot_label = "error (pred - real)"
                z_panel_title = "Shift-null multiscale z-score"
                z_mode_used = "shift_null_multiscale"
            else:
                err_z, err_center, err_sigma = _robust_error_z(err_lag)
                z_mode_used = "robust"
            if z_mode_used != "shift_null_multiscale":
                over_mask = np.isfinite(err_z) & (err_z < -z_thresh)   # predicted > real (because err=real-pred)
                under_mask = np.isfinite(err_z) & (err_z > z_thresh)   # predicted < real
                missed_mask = np.isfinite(err_z) & (np.abs(err_z) > z_thresh)  # either under or over
                over_windows = _true_windows(over_mask)
                under_windows = _true_windows(under_mask)
                missed_windows = _true_windows(missed_mask)

            # Event typing inside under/over windows (based on first spike/start time)
            try:
                ev_list = _events_from_pkl_same_as_pyr_event_cal2(
                    pkl_data,
                    trace_len_v=int(np.asarray(st.get("volRaw", []), dtype=float).size),
                    fs_v_hz=vol_sr,
                    isi_ms=EVENT_ISI_MS_DEFAULT,
                )
            except Exception:
                ev_list = []
            state_total_counts_by_class = {"simple": 0, "complex": 0, "plateau": 0}
            allowed_spikes_set = set(int(v) for v in np.asarray(spk_idx, dtype=int).ravel().tolist())
            ev_entries = []
            for ev in ev_list:
                ev_sp = pe._as_sorted_unique_int(ev.get("spikes", []))
                if ev_sp.size == 0:
                    v0 = _first_spike_or_start_v(ev)
                    ev_sp = np.array([int(v0)], dtype=int)
                ev_sp = ev_sp[(ev_sp >= 0) & (ev_sp < vol_len)]
                if ev_sp.size == 0:
                    continue
                # Keep only spikes that survived mask filtering.
                ev_sp = np.array([int(vv) for vv in ev_sp if int(vv) in allowed_spikes_set], dtype=int)
                if ev_sp.size == 0:
                    continue

                v_idx = int(ev_sp[0])
                c_idx = _vol_idx_to_cal_idx(v_idx, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
                cls, _nsp0 = _event_class_and_size(ev)
                nsp = int(ev_sp.size)
                state_total_counts_by_class[cls] = int(state_total_counts_by_class.get(cls, 0)) + 1
                cell_total_counts_by_class[cls] = int(cell_total_counts_by_class.get(cls, 0)) + 1
                all_total_counts_by_class[cls] = int(all_total_counts_by_class.get(cls, 0)) + 1
                ev_entries.append(
                    {
                        "event_class": cls,
                        "n_spikes": int(nsp),
                        "cal_idx": int(c_idx),
                        "vol_idx": int(v_idx),
                    }
                )
                if 0 <= int(c_idx) < n_cal and bool(under_mask[int(c_idx)]):
                    row_ev = {
                        "cell_path": cell_path,
                        "cell_name": cell_name,
                        "state": state,
                        "fit_corr_pred_lag_vs_real": float(corr_pred_lag_vs_real) if np.isfinite(corr_pred_lag_vs_real) else np.nan,
                        "category": "under",
                        "event_class": cls,
                        "n_spikes": int(nsp),
                        "cal_idx": int(c_idx),
                        "vol_idx": int(v_idx),
                    }
                    cell_event_rows.append(row_ev)
                    all_event_rows.append(row_ev.copy())
                if 0 <= int(c_idx) < n_cal and bool(over_mask[int(c_idx)]):
                    row_ev = {
                        "cell_path": cell_path,
                        "cell_name": cell_name,
                        "state": state,
                        "fit_corr_pred_lag_vs_real": float(corr_pred_lag_vs_real) if np.isfinite(corr_pred_lag_vs_real) else np.nan,
                        "category": "over",
                        "event_class": cls,
                        "n_spikes": int(nsp),
                        "cal_idx": int(c_idx),
                        "vol_idx": int(v_idx),
                    }
                    cell_event_rows.append(row_ev)
                    all_event_rows.append(row_ev.copy())

            # Fallback assignment for empty mismatch windows:
            # if no event falls inside an under/over window, use the closest event
            # within 500 ms before window onset.
            pre_frames = int(max(1, round(0.5 * float(cal_sr))))
            for _cat, _wins in (("under", under_windows), ("over", over_windows)):
                for (_ws, _we) in _wins:
                    has_inside = any((_ws <= int(evx["cal_idx"]) <= _we) for evx in ev_entries)
                    if has_inside:
                        continue
                    cands = [
                        evx
                        for evx in ev_entries
                        if (int(evx["cal_idx"]) < int(_ws)) and (int(evx["cal_idx"]) >= int(_ws) - pre_frames)
                    ]
                    if len(cands) == 0:
                        continue
                    best = max(cands, key=lambda d: int(d["cal_idx"]))
                    row_fb = {
                        "cell_path": cell_path,
                        "cell_name": cell_name,
                        "state": state,
                        "fit_corr_pred_lag_vs_real": float(corr_pred_lag_vs_real) if np.isfinite(corr_pred_lag_vs_real) else np.nan,
                        "category": str(_cat),
                        "event_class": str(best["event_class"]),
                        "n_spikes": int(best["n_spikes"]),
                        "cal_idx": int(best["cal_idx"]),
                        "vol_idx": int(best["vol_idx"]),
                        "match_mode": "closest_pre500ms_empty_window",
                        "window_start_idx": int(_ws),
                        "window_end_idx": int(_we),
                    }
                    cell_event_rows_closest_pre500.append(row_fb)
                    all_event_rows_closest_pre500.append(row_fb.copy())

            # Plot 1: scatter + black linear fit
            valid_xy = np.isfinite(spks_smooth) & np.isfinite(fr_smooth)
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=spks_smooth[valid_xy],
                    y=fr_smooth[valid_xy],
                    mode="markers",
                    marker=dict(size=4, color="rgba(31,119,180,0.45)"),
                    name="time points",
                )
            )
            if np.isfinite(scale) and np.isfinite(offset) and np.any(valid_xy):
                xline = np.linspace(
                    float(np.nanmin(spks_smooth[valid_xy])),
                    float(np.nanmax(spks_smooth[valid_xy])),
                    200,
                )
                yline = scale * xline + offset
                fit_name = "linear fit" if bool(apply_linear_scaling) else "identity (no scaling)"
                fig1.add_trace(
                    go.Scatter(
                        x=xline,
                        y=yline,
                        mode="lines",
                        line=dict(color="black", width=2),
                        name=fit_name,
                    )
                )
            fig1.update_layout(
                template="simple_white",
                width=900,
                height=650,
                title=(
                    f"{os.path.basename(cell_path)} | {state} | spks_smooth vs fr_smooth "
                    f"(r={corr_0lag:.3f}, best_lag_r={corr_best_lag:.3f}, lag={lag_s_best:.3f}s, z_mode={z_mode_used})"
                ),
                xaxis_title="suite2p spks (smoothed)",
                yaxis_title="real FR on calcium bins (smoothed, Hz)",
            )
            fig1_html = os.path.join(out_dir, f"{state}_linear_fit_scatter_{z_mode_file_tag}.html")
            fig1.write_html(fig1_html)
            fig1_svg = None
            if bool(save_svg):
                fig1_svg = os.path.join(out_dir, f"{state}_linear_fit_scatter_{z_mode_file_tag}.svg")
                try:
                    fig1.write_image(fig1_svg)
                except Exception:
                    fig1_svg = None

            # Plot 2: lag-corrected predicted FR from spks vs real FR + error subplot
            t = np.arange(n_cal, dtype=float) / float(cal_sr)
            is_shift_null_plot = (z_mode_used == "shift_null_multiscale")
            has_overlay_subplot = True
            fig2_rows = 4 if has_overlay_subplot else 3
            fig2_subtitles = (
                "Real FR vs lag-corrected fitted FR from spks",
                f"Error ({err_plot_label.split('(')[-1].rstrip(')')})",
                z_panel_title,
            )
            if has_overlay_subplot:
                fig2_subtitles = fig2_subtitles + ("Full voltage+calcium overlay (spikes colored by type)",)
            fig2_height = 1280 if has_overlay_subplot else 980
            fig2_specs = [[{}], [{}], [{}], [{"secondary_y": True}]] if has_overlay_subplot else None
            fig2 = make_subplots(
                rows=fig2_rows,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.07,
                subplot_titles=fig2_subtitles,
                specs=fig2_specs,
            )
            fig2.add_trace(
                go.Scatter(
                    x=t,
                    y=fr_smooth,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="real FR (smooth)",
                ),
                row=1,
                col=1,
            )
            fig2.add_trace(
                go.Scatter(
                    x=t,
                    y=fr_pred_lag,
                    mode="lines",
                    line=dict(color="firebrick", width=2),
                    name=f"fitted FR from spks (lag-corrected, r={corr_pred_lag_vs_real:.3f})",
                ),
                row=1,
                col=1,
            )
            fig2.add_trace(
                go.Scatter(
                    x=t,
                    y=err_for_plot,
                    mode="lines",
                    line=dict(color="royalblue", width=1.6),
                    name=err_plot_label,
                ),
                row=2,
                col=1,
            )
            fig2.add_trace(
                go.Scatter(
                    x=t,
                    y=err_z,
                    mode="lines",
                    line=dict(color="#7F3C8D", width=1.4),
                    name="error z",
                ),
                row=3,
                col=1,
            )
            fig2.update_layout(
                template="simple_white",
                width=1200,
                height=fig2_height,
                title=(
                    f"{os.path.basename(cell_path)} | {state} | lag-corrected fit "
                    f"(lag={lag_s_best*1000.0:.1f} ms, fit corr={corr_pred_lag_vs_real:.3f}, "
                    f"err mean={float(np.nanmean(err_for_plot)):.3f} Hz, err sigma={err_sigma:.3f} Hz, z={z_mode_used})"
                ),
            )
            if has_overlay_subplot:
                # Use state-sliced traces so motor/rest overlays stay aligned to the
                # analyzed state. PKL trace fields can be full-trace or differently masked.
                trace_vol_full = np.asarray(st.get("volRaw", []), dtype=float).ravel()
                if trace_vol_full.size == 0:
                    trace_vol_full = np.asarray(pkl_data.get("trace_vol", []), dtype=float).ravel()
                trace_cal_full = np.asarray(st.get("calRaw", []), dtype=float).ravel()
                if trace_cal_full.size == 0:
                    trace_cal_full = np.asarray(pkl_data.get("trace_cal", []), dtype=float).ravel()

                if trace_vol_full.size > 0:
                    tv_full = np.arange(trace_vol_full.size, dtype=float) / float(vol_sr)
                    fig2.add_trace(
                        go.Scatter(
                            x=tv_full,
                            y=trace_vol_full,
                            mode="lines",
                            line=dict(color="#d62728", width=1.2),
                            name="voltage",
                        ),
                        row=4,
                        col=1,
                        secondary_y=False,
                    )
                if trace_cal_full.size > 0:
                    tc_full = np.arange(trace_cal_full.size, dtype=float) / float(cal_sr)
                    fig2.add_trace(
                        go.Scatter(
                            x=tc_full,
                            y=trace_cal_full,
                            mode="lines",
                            line=dict(color="mediumseagreen", width=1.2),
                            name="calcium",
                        ),
                        row=4,
                        col=1,
                        secondary_y=True,
                    )
                # Color spikes by event type while keeping all detected spikes visible.
                if trace_vol_full.size > 0:
                    detected_sp = pe._as_sorted_unique_int(np.asarray(spk_idx, dtype=int).ravel())
                    detected_sp = detected_sp[(detected_sp >= 0) & (detected_sp < trace_vol_full.size)]

                    # Default all detected spikes to "other" so spikes are never lost in this panel.
                    spk_color = {int(v): "#7f7f7f" for v in detected_sp.tolist()}
                    spk_prio = {int(v): 0 for v in detected_sp.tolist()}

                    for ev in ev_list:
                        ev_sp = pe._as_sorted_unique_int(ev.get("spikes", []))
                        ev_sp = ev_sp[(ev_sp >= 0) & (ev_sp < trace_vol_full.size)]
                        if ev_sp.size == 0:
                            continue
                        et = str(ev.get("event_type", "simple")).strip().lower()

                        # Requested event colors: blue / green / pink; no dedicated plateau color.
                        if et == "complex":
                            col = "#ff69b4"   # pink
                            pri = 3
                        elif et == "plateau":
                            # Keep plateau spikes visible as "other" (no dedicated plateau color).
                            col = "#7f7f7f"
                            pri = 1
                        else:
                            if int(ev_sp.size) <= 1:
                                col = "#1f77b4"   # blue
                            else:
                                col = "#2ca02c"   # green
                            pri = 2

                        for sv in ev_sp.tolist():
                            sv = int(sv)
                            if sv not in spk_color:
                                continue
                            oldp = spk_prio.get(sv, -1)
                            if pri >= oldp:
                                spk_prio[sv] = int(pri)
                                spk_color[sv] = col

                    if len(spk_color) > 0:
                        # Draw in deterministic color groups.
                        for cname, col in (
                            ("single", "#1f77b4"),
                            ("simple_burst", "#2ca02c"),
                            ("complex", "#ff69b4"),
                            ("other", "#7f7f7f"),
                        ):
                            idxs = np.array([k for k, c in spk_color.items() if c == col], dtype=int)
                            if idxs.size == 0:
                                continue
                            idxs = np.sort(idxs)
                            fig2.add_trace(
                                go.Scatter(
                                    x=tv_full[idxs],
                                    y=trace_vol_full[idxs],
                                    mode="markers",
                                    marker=dict(color=col, size=5),
                                    name=f"spikes {cname}",
                                ),
                                row=4,
                                col=1,
                                secondary_y=False,
                            )
            # Shade robust-error windows on the main FR panel:
            # yellow = underestimate (real > predicted), green = overestimate (predicted > real)
            for (ws, we) in under_windows:
                x0 = float(ws) / float(cal_sr)
                x1 = float(we + 1) / float(cal_sr)
                fig2.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="yellow",
                    opacity=0.16,
                    line_width=0,
                    layer="below",
                    row=1,
                    col=1,
                )
                if has_overlay_subplot:
                    fig2.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor="yellow",
                        opacity=0.16,
                        line_width=0,
                        layer="below",
                        row=4,
                        col=1,
                    )
            for (ws, we) in over_windows:
                x0 = float(ws) / float(cal_sr)
                x1 = float(we + 1) / float(cal_sr)
                fig2.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor="green",
                    opacity=0.14,
                    line_width=0,
                    layer="below",
                    row=1,
                    col=1,
                )
                if has_overlay_subplot:
                    fig2.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor="green",
                        opacity=0.14,
                        line_width=0,
                        layer="below",
                        row=4,
                        col=1,
                    )
            if has_overlay_subplot:
                # Legend entries for window colors (shapes do not appear in legend by default).
                fig2.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(symbol="square", size=10, color="yellow"),
                        name="underestimate window",
                    ),
                    row=4,
                    col=1,
                    secondary_y=False,
                )
                fig2.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(symbol="square", size=10, color="green"),
                        name="overestimate window",
                    ),
                    row=4,
                    col=1,
                    secondary_y=False,
                )
            fig2.update_xaxes(title_text="time (s)", row=2, col=1)
            fig2.update_xaxes(title_text="time (s)", row=3, col=1)
            if has_overlay_subplot:
                fig2.update_xaxes(title_text="time (s)", row=4, col=1)
            fig2.update_yaxes(title_text="FR (Hz)", row=1, col=1)
            fig2.update_yaxes(title_text="error (Hz)", row=2, col=1)
            fig2.update_yaxes(title_text="z", row=3, col=1)
            if has_overlay_subplot:
                fig2.update_yaxes(title_text="Voltage (a.u.)", row=4, col=1, secondary_y=False)
                fig2.update_yaxes(title_text="Calcium (a.u.)", row=4, col=1, secondary_y=True)
            fig2.add_hline(y=0.0, line_width=1.0, line_dash="dash", line_color="#777777", row=3, col=1)
            fig2.add_hline(y=float(z_thresh), line_width=1.0, line_dash="dot", line_color="#F2CF1D", row=3, col=1)
            fig2.add_hline(y=-float(z_thresh), line_width=1.0, line_dash="dot", line_color="#2ca02c", row=3, col=1)
            fig2_html = os.path.join(out_dir, f"{state}_fr_real_vs_fitted_{z_mode_file_tag}.html")
            fig2.write_html(fig2_html)
            fig2_svg = None
            if bool(save_svg):
                fig2_svg = os.path.join(out_dir, f"{state}_fr_real_vs_fitted_{z_mode_file_tag}.svg")
                try:
                    fig2.write_image(fig2_svg)
                except Exception:
                    fig2_svg = None

            # Save robust error-z windows table per cell/state
            win_rows = []
            for lbl, wins, mask_arr in (
                ("over_estimation", over_windows, over_mask),
                ("under_estimation", under_windows, under_mask),
                ("missed", missed_windows, missed_mask),
            ):
                for (ws, we) in wins:
                    seg_e = err_lag[ws : we + 1]
                    seg_z = err_z[ws : we + 1]
                    win_rows.append(
                        {
                            "type": lbl,
                            "start_idx": int(ws),
                            "end_idx": int(we),
                            "start_s": float(ws / cal_sr),
                            "end_s": float(we / cal_sr),
                            "duration_s": float((we - ws + 1) / cal_sr),
                            "n_frames": int(we - ws + 1),
                            "mean_error_hz": float(np.nanmean(seg_e)) if np.any(np.isfinite(seg_e)) else np.nan,
                            "mean_error_z": float(np.nanmean(seg_z)) if np.any(np.isfinite(seg_z)) else np.nan,
                            "max_abs_error_z": float(np.nanmax(np.abs(seg_z))) if np.any(np.isfinite(seg_z)) else np.nan,
                        }
                    )
            win_df = pd.DataFrame(win_rows)
            win_csv = os.path.join(out_dir, f"{state}_error_z_windows_{z_mode_file_tag}.csv")
            win_csv = _safe_to_csv(win_df, win_csv, index=False)
            duration_s_state = float(n_cal) / float(cal_sr) if (np.isfinite(cal_sr) and cal_sr > 0) else np.nan
            n_non_simple_events = int(state_total_counts_by_class.get("complex", 0)) + int(
                state_total_counts_by_class.get("plateau", 0)
            )
            n_simple_single_events = int(
                np.sum(
                    [
                        (str(evx.get("event_class", "")) == "simple") and (int(evx.get("n_spikes", 0)) == 1)
                        for evx in ev_entries
                    ]
                )
            )
            non_simple_event_freq_hz = (
                float(n_non_simple_events) / float(duration_s_state)
                if (np.isfinite(duration_s_state) and duration_s_state > 0)
                else np.nan
            )
            simple_single_event_freq_hz = (
                float(n_simple_single_events) / float(duration_s_state)
                if (np.isfinite(duration_s_state) and duration_s_state > 0)
                else np.nan
            )

            rows.append(
                {
                    "cell_path": cell_path,
                    "state": state,
                    "ok": True,
                    "cal_sr_hz": cal_sr,
                    "vol_sr_hz": vol_sr,
                    "smooth_sigma_s": float(smooth_sigma_s),
                    "smooth_sigma_frames": float(sigma_frames),
                    "pearson_r_0lag": corr_0lag,
                    "best_lag_pearson_r": corr_best_lag,
                    "best_lag_frames": int(lag_frames_best),
                    "best_lag_s": lag_s_best,
                    "scale": scale,
                    "offset": offset,
                    "fit_corr_pred_vs_real": corr_pred_vs_real,
                    "scale_lag": scale_lag,
                    "offset_lag": offset_lag,
                    "fit_corr_pred_lag_vs_real": corr_pred_lag_vs_real,
                    "mean_real_fr_hz": float(np.nanmean(fr_smooth)) if np.any(np.isfinite(fr_smooth)) else np.nan,
                    "mean_pred_fr_hz": float(np.nanmean(fr_pred)) if np.any(np.isfinite(fr_pred)) else np.nan,
                    "mean_pred_lag_fr_hz": float(np.nanmean(fr_pred_lag)) if np.any(np.isfinite(fr_pred_lag)) else np.nan,
                    "error_z_mode": z_mode_used,
                    "error_center_median_hz": err_center,
                    "error_sigma_mad_hz": err_sigma,
                    "error_bin_edges_hz": ";".join("inf" if np.isinf(v) else f"{float(v):g}" for v in np.asarray(err_bin_edges, dtype=float)),
                    "error_bin_sigmas_hz": ";".join("" if (not np.isfinite(v)) else f"{float(v):.6g}" for v in np.asarray(err_bin_sigmas, dtype=float)),
                    "shift_null_scales_frames": ";".join(str(int(v)) for v in shift_null_scales_frames),
                    "error_mean_hz": float(np.nanmean(err_for_plot)) if np.any(np.isfinite(err_for_plot)) else np.nan,
                    "error_rmse_hz": float(np.sqrt(np.nanmean(err_for_plot**2))) if np.any(np.isfinite(err_for_plot)) else np.nan,
                    "n_missed_frames_abs_z_gt_2": int(np.sum(missed_mask)),
                    "n_under_frames_z_gt_2": int(np.sum(under_mask)),
                    "n_over_frames_z_lt_minus2": int(np.sum(over_mask)),
                    "n_missed_windows_abs_z_gt_2": int(len(missed_windows)),
                    "n_under_windows_z_gt_2": int(len(under_windows)),
                    "n_over_windows_z_lt_minus2": int(len(over_windows)),
                    "frac_missed_frames_abs_z_gt_2": float(np.mean(missed_mask)) if missed_mask.size > 0 else np.nan,
                    "frac_under_frames_z_gt_2": float(np.mean(under_mask)) if under_mask.size > 0 else np.nan,
                    "frac_over_frames_z_lt_minus2": float(np.mean(over_mask)) if over_mask.size > 0 else np.nan,
                    "n_timepoints": int(n_cal),
                    "n_voltage_spikes": int(spk_idx.size),
                    "n_simple_events_total": int(state_total_counts_by_class.get("simple", 0)),
                    "n_complex_events_total": int(state_total_counts_by_class.get("complex", 0)),
                    "n_plateau_events_total": int(state_total_counts_by_class.get("plateau", 0)),
                    "n_non_simple_events_total": int(n_non_simple_events),
                    "n_simple_single_events_total": int(n_simple_single_events),
                    "duration_s_state": float(duration_s_state) if np.isfinite(duration_s_state) else np.nan,
                    "non_simple_event_freq_hz": float(non_simple_event_freq_hz)
                    if np.isfinite(non_simple_event_freq_hz)
                    else np.nan,
                    "simple_single_event_freq_hz": float(simple_single_event_freq_hz)
                    if np.isfinite(simple_single_event_freq_hz)
                    else np.nan,
                    "plot_linear_fit_html": fig1_html,
                    "plot_linear_fit_svg": fig1_svg,
                    "plot_fr_compare_html": fig2_html,
                    "plot_fr_compare_svg": fig2_svg,
                    "error_windows_csv": win_csv,
                }
            )
            all_series_for_summary.append(
                {
                    "label": f"{os.path.basename(cell_path)} | {state}",
                    "t": t,
                    "fr_real": np.asarray(fr_smooth, dtype=float),
                    "fr_pred_lag": np.asarray(fr_pred_lag, dtype=float),
                    "err_lag": np.asarray(err_lag, dtype=float),
                    "fit_corr": float(corr_pred_lag_vs_real) if np.isfinite(corr_pred_lag_vs_real) else np.nan,
                }
            )

        # Per-cell stacked bars: under vs over by event type/size (across all states in this cell)
        cell_evt_df = pd.DataFrame(cell_event_rows)
        cell_evt_csv = os.path.join(out_dir, "event_under_over_by_type_size_all_states.csv")
        if len(cell_evt_df) > 0:
            cell_evt_csv = _safe_to_csv(cell_evt_df, cell_evt_csv, index=False)
        else:
            cell_evt_csv = _safe_to_csv(pd.DataFrame(
                columns=["cell_path", "cell_name", "state", "category", "event_class", "n_spikes", "cal_idx", "vol_idx"]
            ), cell_evt_csv, index=False)
        cell_evt_html = os.path.join(out_dir, f"event_under_over_stacked_bars_all_states_{z_mode_file_tag}.html")
        cell_evt_svg = os.path.join(out_dir, f"event_under_over_stacked_bars_all_states_{z_mode_file_tag}.svg") if bool(save_svg) else None
        _plot_under_over_stacked_bars(
            event_df=cell_evt_df,
            title=f"{cell_name} | events in under/over windows (all states) | z_mode={z_mode_title_tag}",
            save_html=cell_evt_html,
            save_svg=cell_evt_svg,
            total_counts_by_class=cell_total_counts_by_class,
        )

        # Per-cell fallback plot: closest event in 500 ms before empty under/over window onset.
        cell_evt_fb_df = pd.DataFrame(cell_event_rows_closest_pre500)
        cell_evt_fb_csv = os.path.join(out_dir, f"event_under_over_closest_pre500ms_by_type_size_all_states_{z_mode_file_tag}.csv")
        if len(cell_evt_fb_df) > 0:
            cell_evt_fb_csv = _safe_to_csv(cell_evt_fb_df, cell_evt_fb_csv, index=False)
        else:
            cell_evt_fb_csv = _safe_to_csv(pd.DataFrame(
                columns=["cell_path", "cell_name", "state", "category", "event_class", "n_spikes", "cal_idx", "vol_idx", "match_mode", "window_start_idx", "window_end_idx"]
            ), cell_evt_fb_csv, index=False)
        cell_evt_fb_html = os.path.join(out_dir, f"event_under_over_closest_pre500ms_stacked_bars_all_states_{z_mode_file_tag}.html")
        cell_evt_fb_svg = os.path.join(out_dir, f"event_under_over_closest_pre500ms_stacked_bars_all_states_{z_mode_file_tag}.svg") if bool(save_svg) else None
        _plot_under_over_stacked_bars(
            event_df=cell_evt_fb_df,
            title=f"{cell_name} | closest event to underestimate/over estimate empty window (all states) | z_mode={z_mode_title_tag}",
            save_html=cell_evt_fb_html,
            save_svg=cell_evt_fb_svg,
            total_counts_by_class=cell_total_counts_by_class,
        )

    res_df = pd.DataFrame(rows)

    # Population summaries
    valid = res_df.loc[res_df["ok"] == True].copy() if ("ok" in res_df.columns) else pd.DataFrame()
    if summary_out_dir is not None:
        summary_out_dir = str(summary_out_dir)
        os.makedirs(summary_out_dir, exist_ok=True)

        if len(valid) > 0:
            # 0) Save lag/correlation/error summary tables
            valid_csv = valid.copy()
            for _col in (
                "pearson_r_0lag",
                "best_lag_pearson_r",
                "fit_corr_pred_vs_real",
                "fit_corr_pred_lag_vs_real",
                "best_lag_s",
                "mean_real_fr_hz",
                "mean_pred_fr_hz",
                "mean_pred_lag_fr_hz",
                "error_mean_hz",
                "error_sigma_mad_hz",
            ):
                if _col in valid_csv.columns:
                    valid_csv[_col] = pd.to_numeric(valid_csv[_col], errors="coerce")

            if "best_lag_s" in valid_csv.columns:
                valid_csv["best_lag_ms"] = 1000.0 * pd.to_numeric(valid_csv["best_lag_s"], errors="coerce")
            else:
                valid_csv["best_lag_ms"] = np.nan

            per_state_cols = [
                "cell_path",
                "state",
                "pearson_r_0lag",
                "best_lag_pearson_r",
                "fit_corr_pred_vs_real",
                "fit_corr_pred_lag_vs_real",
                "best_lag_s",
                "best_lag_ms",
                "mean_real_fr_hz",
                "mean_pred_fr_hz",
                "mean_pred_lag_fr_hz",
                "error_mean_hz",
                "error_sigma_mad_hz",
            ]
            per_state_cols = [c for c in per_state_cols if c in valid_csv.columns]
            lag_corr_state_csv = os.path.join(summary_out_dir, f"lag_corr_error_summary_by_state_{z_mode_file_tag}.csv")
            lag_corr_state_csv = _safe_to_csv(valid_csv.loc[:, per_state_cols], lag_corr_state_csv, index=False)

            agg_map = {
                "state": ("state", lambda s: ";".join(sorted(set(str(v) for v in s)))),
                "n_states_valid": ("state", "nunique"),
                "corr_before_lag_mean": ("pearson_r_0lag", "mean"),
                "corr_after_lag_mean": ("best_lag_pearson_r", "mean"),
                "fit_corr_before_lag_mean": ("fit_corr_pred_vs_real", "mean"),
                "fit_corr_after_lag_mean": ("fit_corr_pred_lag_vs_real", "mean"),
                "optimal_lag_s_mean": ("best_lag_s", "mean"),
                "optimal_lag_ms_mean": ("best_lag_ms", "mean"),
                "mean_real_fr_hz_mean": ("mean_real_fr_hz", "mean"),
                "mean_pred_fr_hz_mean": ("mean_pred_fr_hz", "mean"),
                "mean_pred_lag_fr_hz_mean": ("mean_pred_lag_fr_hz", "mean"),
                "mean_error_hz_mean": ("error_mean_hz", "mean"),
                "error_sigma_hz_mean": ("error_sigma_mad_hz", "mean"),
            }
            # Keep only aggregations whose source column exists.
            agg_map_use = {}
            for out_name, (src_col, fn) in agg_map.items():
                if src_col in valid_csv.columns:
                    agg_map_use[out_name] = (src_col, fn)
            lag_corr_cell_csv = os.path.join(summary_out_dir, f"lag_corr_error_summary_by_cell_path_{z_mode_file_tag}.csv")
            lag_corr_cell_df = (
                valid_csv.groupby("cell_path", as_index=False)
                .agg(**agg_map_use)
                .rename(columns={"state": "states"})
            )
            lag_corr_cell_df["row_kind"] = "cell_path_mean"
            lag_corr_cell_df["state"] = "all"

            # Add explicit motor-state rows (m0/r0/...) with full per-state metrics
            # into the same by-cell-path summary file.
            if "state" in valid_csv.columns:
                per_state_plus = valid_csv.loc[:, per_state_cols].copy()
                st_norm = per_state_plus["state"].astype(str).str.strip().str.lower()
                motor_mask = st_norm.str.fullmatch(r"[mr]\d+")
                motor_rows = per_state_plus.loc[motor_mask].copy()
                if len(motor_rows) > 0:
                    motor_rows_out = pd.DataFrame(
                        {
                            "cell_path": motor_rows["cell_path"].astype(str),
                            "states": motor_rows["state"].astype(str),
                            "n_states_valid": 1,
                            "corr_before_lag_mean": pd.to_numeric(motor_rows.get("pearson_r_0lag", np.nan), errors="coerce"),
                            "corr_after_lag_mean": pd.to_numeric(motor_rows.get("best_lag_pearson_r", np.nan), errors="coerce"),
                            "fit_corr_before_lag_mean": pd.to_numeric(motor_rows.get("fit_corr_pred_vs_real", np.nan), errors="coerce"),
                            "fit_corr_after_lag_mean": pd.to_numeric(motor_rows.get("fit_corr_pred_lag_vs_real", np.nan), errors="coerce"),
                            "optimal_lag_s_mean": pd.to_numeric(motor_rows.get("best_lag_s", np.nan), errors="coerce"),
                            "optimal_lag_ms_mean": pd.to_numeric(motor_rows.get("best_lag_ms", np.nan), errors="coerce"),
                            "mean_real_fr_hz_mean": pd.to_numeric(motor_rows.get("mean_real_fr_hz", np.nan), errors="coerce"),
                            "mean_pred_fr_hz_mean": pd.to_numeric(motor_rows.get("mean_pred_fr_hz", np.nan), errors="coerce"),
                            "mean_pred_lag_fr_hz_mean": pd.to_numeric(motor_rows.get("mean_pred_lag_fr_hz", np.nan), errors="coerce"),
                            "mean_error_hz_mean": pd.to_numeric(motor_rows.get("error_mean_hz", np.nan), errors="coerce"),
                            "error_sigma_hz_mean": pd.to_numeric(motor_rows.get("error_sigma_mad_hz", np.nan), errors="coerce"),
                            "row_kind": "motor_state",
                            "state": motor_rows["state"].astype(str),
                        }
                    )
                    lag_corr_cell_df = pd.concat([lag_corr_cell_df, motor_rows_out], axis=0, ignore_index=True)
            lag_corr_cell_csv = _safe_to_csv(lag_corr_cell_df, lag_corr_cell_csv, index=False)
            res_df.attrs["summary_lag_corr_state_csv"] = lag_corr_state_csv
            res_df.attrs["summary_lag_corr_cell_csv"] = lag_corr_cell_csv

            # 0b) Explained-variance analysis for lag-corrected fit correlation.
            # Response: Fisher-z transformed fit_corr_pred_lag_vs_real.
            # Predictors: mean FR, non-simple event frequency, simple-single event frequency, state.
            def _fisher_z_local(r_like):
                rr = np.asarray(r_like, dtype=float).ravel()
                rr = np.clip(rr, -0.999999, 0.999999)
                return np.arctanh(rr)

            def _ols_r2_local(y, X):
                y = np.asarray(y, dtype=float).ravel()
                X = np.asarray(X, dtype=float)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                if y.size == 0:
                    return np.nan, 0
                if X.size == 0:
                    X = np.empty((y.size, 0), dtype=float)
                if X.shape[0] != y.size:
                    return np.nan, 0
                ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                yv = y[ok]
                Xv = X[ok]
                n = int(yv.size)
                if n < max(8, Xv.shape[1] + 3):
                    return np.nan, n
                Xd = np.column_stack([np.ones(n, dtype=float), Xv])
                try:
                    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
                except Exception:
                    return np.nan, n
                yhat = Xd @ beta
                ss_res = float(np.nansum((yv - yhat) ** 2))
                ymu = float(np.nanmean(yv))
                ss_tot = float(np.nansum((yv - ymu) ** 2))
                if (not np.isfinite(ss_tot)) or ss_tot <= 0:
                    return np.nan, n
                return float(1.0 - ss_res / ss_tot), n

            ev_csv = os.path.join(summary_out_dir, f"corr_explained_variance_{z_mode_file_tag}.csv")
            ev_html = os.path.join(summary_out_dir, f"corr_explained_variance_{z_mode_file_tag}.html")
            ev_svg = os.path.join(summary_out_dir, f"corr_explained_variance_{z_mode_file_tag}.svg") if bool(save_svg) else None

            y_corr = pd.to_numeric(valid_csv.get("fit_corr_pred_lag_vs_real", np.nan), errors="coerce").to_numpy(dtype=float)
            y_z = _fisher_z_local(y_corr)
            x_mean_fr = pd.to_numeric(valid_csv.get("mean_real_fr_hz", np.nan), errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            x_non_simple = pd.to_numeric(valid_csv.get("non_simple_event_freq_hz", np.nan), errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            x_simple_single = pd.to_numeric(valid_csv.get("simple_single_event_freq_hz", np.nan), errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
            if "state" in valid_csv.columns:
                state_dum = pd.get_dummies(valid_csv["state"].astype(str), prefix="state", drop_first=True)
                x_state = state_dum.to_numpy(dtype=float) if state_dum.shape[1] > 0 else np.empty((len(valid_csv), 0), dtype=float)
            else:
                x_state = np.empty((len(valid_csv), 0), dtype=float)

            groups = {
                "mean_FR": x_mean_fr,
                "non_simple_freq": x_non_simple,
                "simple_single_freq": x_simple_single,
                "state": x_state,
            }
            group_order = ["mean_FR", "non_simple_freq", "simple_single_freq", "state"]
            parts_full = [groups[g] for g in group_order if groups[g].size > 0]
            X_full = np.column_stack(parts_full) if len(parts_full) > 0 else np.empty((len(valid_csv), 0), dtype=float)
            full_r2, n_used = _ols_r2_local(y_z, X_full)
            ev_rows = []
            for g in group_order:
                if groups[g].size <= 0:
                    continue
                red_parts = [groups[k] for k in group_order if (k != g and groups[k].size > 0)]
                X_red = np.column_stack(red_parts) if len(red_parts) > 0 else np.empty((len(valid_csv), 0), dtype=float)
                red_r2, _ = _ols_r2_local(y_z, X_red)
                delta = np.nan
                share = np.nan
                if np.isfinite(full_r2) and np.isfinite(red_r2):
                    delta = float(full_r2 - red_r2)
                    if abs(full_r2) > 1e-12:
                        share = float(delta / full_r2)
                ev_rows.append(
                    {
                        "target": "fit_corr_pred_lag_vs_real",
                        "response_transform": "fisher_z",
                        "predictor_group": str(g),
                        "n_used": int(n_used),
                        "full_r2": float(full_r2) if np.isfinite(full_r2) else np.nan,
                        "reduced_r2": float(red_r2) if np.isfinite(red_r2) else np.nan,
                        "delta_r2": float(delta) if np.isfinite(delta) else np.nan,
                        "delta_r2_share": float(share) if np.isfinite(share) else np.nan,
                    }
                )
            ev_df = pd.DataFrame(ev_rows)
            ev_csv = _safe_to_csv(ev_df, ev_csv, index=False)
            res_df.attrs["summary_explained_csv"] = ev_csv

            if len(ev_df) > 0:
                fig_ev = go.Figure()
                fig_ev.add_trace(
                    go.Bar(
                        x=ev_df["predictor_group"].astype(str).tolist(),
                        y=pd.to_numeric(ev_df["delta_r2"], errors="coerce").to_numpy(dtype=float),
                        marker=dict(color="#4C78A8"),
                        text=[f"{v:.3g}" if np.isfinite(v) else "nan" for v in pd.to_numeric(ev_df["delta_r2"], errors="coerce").to_numpy(dtype=float)],
                        textposition="outside",
                        name="delta R2",
                    )
                )
                fig_ev.update_layout(
                    template="simple_white",
                    width=980,
                    height=560,
                    title=f"Explained variance of fit correlation (drop-one partial R2) | z_mode={z_mode_title_tag}",
                    xaxis_title="Predictor group removed from full model",
                    yaxis_title="Drop in R2 (full - reduced)",
                    showlegend=False,
                )
                fig_ev.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.14)", zeroline=False)
                fig_ev.write_html(ev_html)
                if ev_svg is not None:
                    try:
                        fig_ev.write_image(ev_svg)
                    except Exception:
                        ev_svg = None
                res_df.attrs["summary_explained_html"] = ev_html
                res_df.attrs["summary_explained_svg"] = ev_svg

            # 1) Histograms:
            #    - direct spks-vs-FR correlation (no-lag, lag-corrected)
            #    - chosen lag (ms)
            #    - fitted/scaled prediction-vs-real correlation (no-lag, lag-corrected)
            #    - error mean and error std/sigma across all cells/states
            fig_hist = make_subplots(
                rows=2,
                cols=5,
                subplot_titles=(
                    "Direct Pearson r (no lag correction)",
                    "Direct Pearson r (lag corrected)",
                    "Chosen lag (ms)",
                    "Fitted Pearson r (no lag correction)",
                    "Fitted Pearson r (lag corrected)",
                    "Error mean (Hz)",
                    "Error std/sigma (Hz)",
                    "Fitted r (lag corrected) vs non-simple event frequency",
                    "Fitted r (lag corrected) vs mean FR",
                    "Fitted r (lag corrected) vs simple single-spike event frequency",
                ),
            )
            r0 = pd.to_numeric(valid.get("pearson_r_0lag", np.nan), errors="coerce").to_numpy(dtype=float)
            rl = pd.to_numeric(valid.get("best_lag_pearson_r", np.nan), errors="coerce").to_numpy(dtype=float)
            lag_ms = 1000.0 * pd.to_numeric(valid.get("best_lag_s", np.nan), errors="coerce").to_numpy(dtype=float)
            fit_r0 = pd.to_numeric(valid.get("fit_corr_pred_vs_real", np.nan), errors="coerce").to_numpy(dtype=float)
            fit_rl = pd.to_numeric(valid.get("fit_corr_pred_lag_vs_real", np.nan), errors="coerce").to_numpy(dtype=float)
            emean = pd.to_numeric(valid.get("error_mean_hz", np.nan), errors="coerce").to_numpy(dtype=float)
            estd = pd.to_numeric(valid.get("error_sigma_mad_hz", np.nan), errors="coerce").to_numpy(dtype=float)
            freq_non_simple = pd.to_numeric(valid.get("non_simple_event_freq_hz", np.nan), errors="coerce").to_numpy(dtype=float)
            mean_fr = pd.to_numeric(valid.get("mean_real_fr_hz", np.nan), errors="coerce").to_numpy(dtype=float)
            freq_simple_single = pd.to_numeric(valid.get("simple_single_event_freq_hz", np.nan), errors="coerce").to_numpy(dtype=float)
            lbl = (
                valid.get("cell_path", pd.Series([""] * len(valid))).astype(str)
                + " | "
                + valid.get("state", pd.Series([""] * len(valid))).astype(str)
            ).to_numpy()

            hist_style = dict(color="rgba(0,0,0,0.45)", line=dict(color="rgba(0,0,0,0.95)", width=1.0))
            fig_hist.add_trace(go.Histogram(x=r0[np.isfinite(r0)], marker=hist_style, name="no lag"), row=1, col=1)
            fig_hist.add_trace(go.Histogram(x=rl[np.isfinite(rl)], marker=hist_style, name="lag corrected"), row=1, col=2)
            fig_hist.add_trace(go.Histogram(x=lag_ms[np.isfinite(lag_ms)], marker=hist_style, name="lag ms"), row=1, col=3)
            fig_hist.add_trace(go.Histogram(x=fit_r0[np.isfinite(fit_r0)], marker=hist_style, name="fit no lag"), row=1, col=4)
            fig_hist.add_trace(go.Histogram(x=fit_rl[np.isfinite(fit_rl)], marker=hist_style, name="fit lag corrected"), row=1, col=5)
            fig_hist.add_trace(go.Histogram(x=emean[np.isfinite(emean)], marker=hist_style, name="error mean"), row=2, col=1)
            fig_hist.add_trace(go.Histogram(x=estd[np.isfinite(estd)], marker=hist_style, name="error sigma"), row=2, col=2)
            def _add_fit_line_with_text(_fig, _x, _y, row, col, _color="rgba(0,0,0,0.8)"):
                xx = np.asarray(_x, dtype=float).ravel()
                yy = np.asarray(_y, dtype=float).ravel()
                ok = np.isfinite(xx) & np.isfinite(yy)
                if np.sum(ok) < 3:
                    return
                xx = xx[ok]
                yy = yy[ok]
                if (not np.isfinite(np.nanstd(xx))) or (float(np.nanstd(xx)) <= 0):
                    return
                try:
                    m, b = np.polyfit(xx, yy, 1)
                except Exception:
                    return
                if (not np.isfinite(m)) or (not np.isfinite(b)):
                    return
                x0 = float(np.nanmin(xx))
                x1 = float(np.nanmax(xx))
                if not (np.isfinite(x0) and np.isfinite(x1)):
                    return
                if x1 <= x0:
                    return
                xline = np.linspace(x0, x1, 120)
                yline = m * xline + b
                _fig.add_trace(
                    go.Scatter(
                        x=xline,
                        y=yline,
                        mode="lines",
                        line=dict(color=_color, width=2),
                        showlegend=False,
                        hovertemplate="linear fit<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                r = _pearson_corr_valid(xx, yy)
                y0 = float(np.nanmin(yy))
                y1 = float(np.nanmax(yy))
                xt = x0 + 0.98 * (x1 - x0)
                yt = y0 + 0.04 * (y1 - y0)
                txt = f"r={r:.3f}<br>m={m:.3g}"
                _fig.add_trace(
                    go.Scatter(
                        x=[xt],
                        y=[yt],
                        mode="text",
                        text=[txt],
                        textposition="bottom right",
                        textfont=dict(size=11, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            m_sc1 = np.isfinite(freq_non_simple) & np.isfinite(fit_rl)
            fig_hist.add_trace(
                go.Scatter(
                    x=freq_non_simple[m_sc1],
                    y=fit_rl[m_sc1],
                    mode="markers",
                    marker=dict(color="black", size=6, opacity=0.75),
                    text=lbl[m_sc1],
                    hovertemplate="%{text}<br>non-simple freq=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=3,
            )
            m_sc2 = np.isfinite(mean_fr) & np.isfinite(fit_rl)
            fig_hist.add_trace(
                go.Scatter(
                    x=mean_fr[m_sc2],
                    y=fit_rl[m_sc2],
                    mode="markers",
                    marker=dict(color="black", size=6, opacity=0.75),
                    text=lbl[m_sc2],
                    hovertemplate="%{text}<br>mean FR=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=4,
            )
            m_sc3 = np.isfinite(freq_simple_single) & np.isfinite(fit_rl)
            fig_hist.add_trace(
                go.Scatter(
                    x=freq_simple_single[m_sc3],
                    y=fit_rl[m_sc3],
                    mode="markers",
                    marker=dict(color="black", size=6, opacity=0.75),
                    text=lbl[m_sc3],
                    hovertemplate="%{text}<br>simple-single freq=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=5,
            )
            _add_fit_line_with_text(fig_hist, freq_non_simple[m_sc1], fit_rl[m_sc1], row=2, col=3)
            _add_fit_line_with_text(fig_hist, mean_fr[m_sc2], fit_rl[m_sc2], row=2, col=4)
            _add_fit_line_with_text(fig_hist, freq_simple_single[m_sc3], fit_rl[m_sc3], row=2, col=5)
            fig_hist.update_layout(
                template="simple_white",
                width=3000,
                height=980,
                title=f"Suite2p spks vs real FR | population histograms + frequency scatter | z_mode={z_mode_title_tag}",
                showlegend=False,
            )
            fig_hist.update_xaxes(title_text="r", row=1, col=1)
            fig_hist.update_xaxes(title_text="r", row=1, col=2)
            fig_hist.update_xaxes(title_text="lag (ms)", row=1, col=3)
            fig_hist.update_xaxes(title_text="r", row=1, col=4)
            fig_hist.update_xaxes(title_text="r", row=1, col=5)
            fig_hist.update_xaxes(title_text="Hz", row=2, col=1)
            fig_hist.update_xaxes(title_text="Hz", row=2, col=2)
            fig_hist.update_xaxes(title_text="non-simple events/s", row=2, col=3)
            fig_hist.update_xaxes(title_text="mean FR (Hz)", row=2, col=4)
            fig_hist.update_xaxes(title_text="single-simple events/s", row=2, col=5)
            fig_hist.update_yaxes(title_text="count", row=1, col=1)
            fig_hist.update_yaxes(title_text="count", row=1, col=2)
            fig_hist.update_yaxes(title_text="count", row=1, col=3)
            fig_hist.update_yaxes(title_text="count", row=1, col=4)
            fig_hist.update_yaxes(title_text="count", row=1, col=5)
            fig_hist.update_yaxes(title_text="count", row=2, col=1)
            fig_hist.update_yaxes(title_text="count", row=2, col=2)
            fig_hist.update_yaxes(title_text="fit r (lag corrected)", row=2, col=3)
            fig_hist.update_yaxes(title_text="fit r (lag corrected)", row=2, col=4)
            fig_hist.update_yaxes(title_text="fit r (lag corrected)", row=2, col=5)

            hist_html = os.path.join(summary_out_dir, f"hist_corr_and_lag_{z_mode_file_tag}.html")
            fig_hist.write_html(hist_html)
            hist_svg = None
            if bool(save_svg):
                hist_svg = os.path.join(summary_out_dir, f"hist_corr_and_lag_{z_mode_file_tag}.svg")
                try:
                    fig_hist.write_image(hist_svg)
                except Exception:
                    hist_svg = None
            res_df.attrs["summary_hist_html"] = hist_html
            res_df.attrs["summary_hist_svg"] = hist_svg

            # Extra summary figure: the 3 scatter panels only
            fig_sc3 = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Fitted r (lag corrected) vs non-simple event frequency",
                    "Fitted r (lag corrected) vs mean FR",
                    "Fitted r (lag corrected) vs simple single-spike event frequency",
                ),
            )
            fig_sc3.add_trace(
                go.Scatter(
                    x=freq_non_simple[m_sc1],
                    y=fit_rl[m_sc1],
                    mode="markers",
                    marker=dict(color="black", size=7, opacity=0.78),
                    text=lbl[m_sc1],
                    hovertemplate="%{text}<br>non-simple freq=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig_sc3.add_trace(
                go.Scatter(
                    x=mean_fr[m_sc2],
                    y=fit_rl[m_sc2],
                    mode="markers",
                    marker=dict(color="black", size=7, opacity=0.78),
                    text=lbl[m_sc2],
                    hovertemplate="%{text}<br>mean FR=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig_sc3.add_trace(
                go.Scatter(
                    x=freq_simple_single[m_sc3],
                    y=fit_rl[m_sc3],
                    mode="markers",
                    marker=dict(color="black", size=7, opacity=0.78),
                    text=lbl[m_sc3],
                    hovertemplate="%{text}<br>simple-single freq=%{x:.4g} Hz<br>fit r=%{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
            _add_fit_line_with_text(fig_sc3, freq_non_simple[m_sc1], fit_rl[m_sc1], row=1, col=1)
            _add_fit_line_with_text(fig_sc3, mean_fr[m_sc2], fit_rl[m_sc2], row=1, col=2)
            _add_fit_line_with_text(fig_sc3, freq_simple_single[m_sc3], fit_rl[m_sc3], row=1, col=3)
            fig_sc3.update_layout(
                template="simple_white",
                width=1950,
                height=680,
                title=f"Suite2p spks vs real FR | frequency scatter summary | z_mode={z_mode_title_tag}",
                showlegend=False,
            )
            fig_sc3.update_xaxes(title_text="non-simple events/s", row=1, col=1)
            fig_sc3.update_xaxes(title_text="mean FR (Hz)", row=1, col=2)
            fig_sc3.update_xaxes(title_text="single-simple events/s", row=1, col=3)
            fig_sc3.update_yaxes(title_text="fit r (lag corrected)", row=1, col=1)
            fig_sc3.update_yaxes(title_text="fit r (lag corrected)", row=1, col=2)
            fig_sc3.update_yaxes(title_text="fit r (lag corrected)", row=1, col=3)
            hist_sc3_html = os.path.join(summary_out_dir, f"hist_corr_and_lag_scatter3_{z_mode_file_tag}.html")
            fig_sc3.write_html(hist_sc3_html)
            hist_sc3_svg = None
            if bool(save_svg):
                hist_sc3_svg = os.path.join(summary_out_dir, f"hist_corr_and_lag_scatter3_{z_mode_file_tag}.svg")
                try:
                    fig_sc3.write_image(hist_sc3_svg)
                except Exception:
                    hist_sc3_svg = None
            res_df.attrs["summary_hist_scatter3_html"] = hist_sc3_html
            res_df.attrs["summary_hist_scatter3_svg"] = hist_sc3_svg

        if len(all_series_for_summary) > 0:
            # 2) All cells/state lag-corrected FR vs real FR
            nrows = len(all_series_for_summary)
            labels = [d["label"] for d in all_series_for_summary]
            fig_all = make_subplots(
                rows=nrows,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=max(0.002, min(0.03, 0.15 / max(1, nrows))),
                subplot_titles=labels,
            )
            for i, d in enumerate(all_series_for_summary, start=1):
                show_leg = True
                corr_txt = f"{d.get('fit_corr', np.nan):.3f}" if np.isfinite(d.get("fit_corr", np.nan)) else "nan"
                fig_all.add_trace(
                    go.Scatter(
                        x=d["t"],
                        y=d["fr_real"],
                        mode="lines",
                        line=dict(color="black", width=1.6),
                        name=f"{d['label']} | real FR",
                        showlegend=show_leg,
                    ),
                    row=i,
                    col=1,
                )
                fig_all.add_trace(
                    go.Scatter(
                        x=d["t"],
                        y=d["fr_pred_lag"],
                        mode="lines",
                        line=dict(color="firebrick", width=1.6),
                        name=f"{d['label']} | lag-corrected fitted FR (r={corr_txt})",
                        showlegend=show_leg,
                    ),
                    row=i,
                    col=1,
                )
                fig_all.update_yaxes(title_text="Hz", row=i, col=1)
            fig_all.update_xaxes(title_text="time (s)", row=nrows, col=1)
            fig_all.update_layout(
                template="simple_white",
                width=1800,
                height=max(550, 220 * nrows),
                title=f"All cells/states | lag-corrected fitted FR from spks vs real FR | z_mode={z_mode_title_tag}",
            )
            all_html = os.path.join(summary_out_dir, f"all_cells_lag_corrected_fr_vs_real_{z_mode_file_tag}.html")
            fig_all.write_html(all_html)
            all_svg = None
            if bool(save_svg):
                all_svg = os.path.join(summary_out_dir, f"all_cells_lag_corrected_fr_vs_real_{z_mode_file_tag}.svg")
                try:
                    fig_all.write_image(all_svg)
                except Exception:
                    all_svg = None
            res_df.attrs["summary_all_fr_html"] = all_html
            res_df.attrs["summary_all_fr_svg"] = all_svg

            # 3) All cells/state errors summary
            fig_err = make_subplots(
                rows=nrows,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=max(0.002, min(0.03, 0.15 / max(1, nrows))),
                subplot_titles=labels,
            )
            for i, d in enumerate(all_series_for_summary, start=1):
                fig_err.add_trace(
                    go.Scatter(
                        x=d["t"],
                        y=d["err_lag"],
                        mode="lines",
                        line=dict(color="royalblue", width=1.5),
                        name="error (real - pred)",
                        showlegend=(i == 1),
                    ),
                    row=i,
                    col=1,
                )
                fig_err.update_yaxes(title_text="Hz", row=i, col=1)
            fig_err.update_xaxes(title_text="time (s)", row=nrows, col=1)
            fig_err.update_layout(
                template="simple_white",
                width=1800,
                height=max(550, 220 * nrows),
                title=f"All cells/states | error (real FR - lag-corrected predicted FR) | z_mode={z_mode_title_tag}",
            )
            err_html = os.path.join(summary_out_dir, f"all_cells_error_real_minus_pred_{z_mode_file_tag}.html")
            fig_err.write_html(err_html)
            err_svg = None
            if bool(save_svg):
                err_svg = os.path.join(summary_out_dir, f"all_cells_error_real_minus_pred_{z_mode_file_tag}.svg")
                try:
                    fig_err.write_image(err_svg)
                except Exception:
                    err_svg = None
            res_df.attrs["summary_all_error_html"] = err_html
            res_df.attrs["summary_all_error_svg"] = err_svg

        # Event-under/over stacked bars summary across all cells/states
        all_evt_df = pd.DataFrame(all_event_rows)
        thr_src = corrlation_thr if (corrlation_thr is not None) else correlation_thr
        try:
            thr_val = float(thr_src) if (thr_src is not None) else np.nan
        except Exception:
            thr_val = np.nan
        thr_tag = ""
        if np.isfinite(thr_val):
            thr_tag = f"_corrthr_{thr_val:.3f}".replace(".", "p").replace("-", "m")

        all_evt_csv = os.path.join(summary_out_dir, f"all_cells_event_under_over_by_type_size{thr_tag}.csv")
        if len(all_evt_df) > 0:
            all_evt_csv = _safe_to_csv(all_evt_df, all_evt_csv, index=False)
        else:
            all_evt_csv = _safe_to_csv(pd.DataFrame(
                columns=["cell_path", "cell_name", "state", "category", "event_class", "n_spikes", "cal_idx", "vol_idx"]
            ), all_evt_csv, index=False)
        all_evt_html = os.path.join(
            summary_out_dir,
            f"all_cells_event_under_over_stacked_bars_{z_mode_file_tag}{thr_tag}.html",
        )
        all_evt_svg = (
            os.path.join(
                summary_out_dir,
                f"all_cells_event_under_over_stacked_bars_{z_mode_file_tag}{thr_tag}.svg",
            )
            if bool(save_svg)
            else None
        )
        all_evt_plot_df = all_evt_df.copy()
        total_counts_for_plot = dict(all_total_counts_by_class)
        if np.isfinite(thr_val) and len(all_evt_plot_df) > 0 and len(valid) > 0:
            valid_fit = valid.loc[:, ["cell_path", "state", "fit_corr_pred_lag_vs_real"]].copy()
            valid_fit["fit_corr_pred_lag_vs_real"] = pd.to_numeric(valid_fit["fit_corr_pred_lag_vs_real"], errors="coerce")
            keep_keys = set(
                zip(
                    valid_fit.loc[valid_fit["fit_corr_pred_lag_vs_real"] >= thr_val, "cell_path"].astype(str),
                    valid_fit.loc[valid_fit["fit_corr_pred_lag_vs_real"] >= thr_val, "state"].astype(str),
                )
            )
            if len(keep_keys) > 0:
                key_series = list(
                    zip(
                        all_evt_plot_df["cell_path"].astype(str),
                        all_evt_plot_df["state"].astype(str),
                    )
                )
                keep_mask_evt = np.array([kk in keep_keys for kk in key_series], dtype=bool)
                all_evt_plot_df = all_evt_plot_df.loc[keep_mask_evt].copy()
            else:
                all_evt_plot_df = all_evt_plot_df.iloc[0:0].copy()

            # Recompute class totals for filtered states so pie percentages remain consistent.
            total_counts_for_plot = {"simple": 0, "complex": 0, "plateau": 0}
            if len(keep_keys) > 0:
                for _cp, _st in keep_keys:
                    st_rows = valid.loc[
                        (valid["cell_path"].astype(str) == str(_cp))
                        & (valid["state"].astype(str) == str(_st))
                    ]
                    if len(st_rows) <= 0:
                        continue
                    simple_n = pd.to_numeric(st_rows.get("n_simple_events_total", np.nan), errors="coerce").to_numpy(dtype=float)
                    complex_n = pd.to_numeric(st_rows.get("n_complex_events_total", np.nan), errors="coerce").to_numpy(dtype=float)
                    plateau_n = pd.to_numeric(st_rows.get("n_plateau_events_total", np.nan), errors="coerce").to_numpy(dtype=float)
                    if simple_n.size > 0 and np.isfinite(simple_n[0]):
                        total_counts_for_plot["simple"] += int(max(0, round(float(simple_n[0]))))
                    if complex_n.size > 0 and np.isfinite(complex_n[0]):
                        total_counts_for_plot["complex"] += int(max(0, round(float(complex_n[0]))))
                    if plateau_n.size > 0 and np.isfinite(plateau_n[0]):
                        total_counts_for_plot["plateau"] += int(max(0, round(float(plateau_n[0]))))
        _plot_under_over_stacked_bars(
            event_df=all_evt_plot_df,
            title=(
                f"All cells/states | events in under/over windows | z_mode={z_mode_title_tag}"
                + (f" | fit_corr_thr>={thr_val:.3f}" if np.isfinite(thr_val) else "")
            ),
            save_html=all_evt_html,
            save_svg=all_evt_svg,
            total_counts_by_class=total_counts_for_plot,
        )
        res_df.attrs["summary_event_under_over_html"] = all_evt_html
        res_df.attrs["summary_event_under_over_svg"] = all_evt_svg
        res_df.attrs["summary_event_under_over_csv"] = all_evt_csv

        # Fallback pooled summary:
        # closest event within 500 ms before empty under/over window onset.
        all_evt_fb_df = pd.DataFrame(all_event_rows_closest_pre500)
        all_evt_fb_csv = os.path.join(summary_out_dir, f"all_cells_event_under_over_closest_pre500ms_by_type_size{thr_tag}.csv")
        if len(all_evt_fb_df) > 0:
            all_evt_fb_csv = _safe_to_csv(all_evt_fb_df, all_evt_fb_csv, index=False)
        else:
            all_evt_fb_csv = _safe_to_csv(pd.DataFrame(
                columns=["cell_path", "cell_name", "state", "category", "event_class", "n_spikes", "cal_idx", "vol_idx", "match_mode", "window_start_idx", "window_end_idx"]
            ), all_evt_fb_csv, index=False)
        all_evt_fb_html = os.path.join(
            summary_out_dir,
            f"all_cells_event_under_over_closest_pre500ms_stacked_bars_{z_mode_file_tag}{thr_tag}.html",
        )
        all_evt_fb_svg = (
            os.path.join(
                summary_out_dir,
                f"all_cells_event_under_over_closest_pre500ms_stacked_bars_{z_mode_file_tag}{thr_tag}.svg",
            )
            if bool(save_svg)
            else None
        )
        all_evt_fb_plot_df = all_evt_fb_df.copy()
        if np.isfinite(thr_val) and len(all_evt_fb_plot_df) > 0 and len(valid) > 0:
            valid_fit = valid.loc[:, ["cell_path", "state", "fit_corr_pred_lag_vs_real"]].copy()
            valid_fit["fit_corr_pred_lag_vs_real"] = pd.to_numeric(valid_fit["fit_corr_pred_lag_vs_real"], errors="coerce")
            keep_keys = set(
                zip(
                    valid_fit.loc[valid_fit["fit_corr_pred_lag_vs_real"] >= thr_val, "cell_path"].astype(str),
                    valid_fit.loc[valid_fit["fit_corr_pred_lag_vs_real"] >= thr_val, "state"].astype(str),
                )
            )
            if len(keep_keys) > 0:
                key_series = list(
                    zip(
                        all_evt_fb_plot_df["cell_path"].astype(str),
                        all_evt_fb_plot_df["state"].astype(str),
                    )
                )
                keep_mask_evt = np.array([kk in keep_keys for kk in key_series], dtype=bool)
                all_evt_fb_plot_df = all_evt_fb_plot_df.loc[keep_mask_evt].copy()
            else:
                all_evt_fb_plot_df = all_evt_fb_plot_df.iloc[0:0].copy()
        _plot_under_over_stacked_bars(
            event_df=all_evt_fb_plot_df,
            title=(
                f"All cells/states | closest event to underestimate/over estimate empty window | z_mode={z_mode_title_tag}"
                + (f" | fit_corr_thr>={thr_val:.3f}" if np.isfinite(thr_val) else "")
            ),
            save_html=all_evt_fb_html,
            save_svg=all_evt_fb_svg,
            total_counts_by_class=total_counts_for_plot,
        )
        res_df.attrs["summary_event_under_over_closest_pre500ms_html"] = all_evt_fb_html
        res_df.attrs["summary_event_under_over_closest_pre500ms_svg"] = all_evt_fb_svg
        res_df.attrs["summary_event_under_over_closest_pre500ms_csv"] = all_evt_fb_csv

    return res_df


def evaluate_cascade_as_continuous_activity(
    loaded_cells,
    out_subdir="cascade_eval_continuous",
    smooth_sigma_s=0.2,
    max_lag_s=0.5,
    save_svg=False,
    z_thresh=2.0,
    error_z_mode="standard",
    shift_null_window_sizes_s=(0.25, 0.50, 1.00),
    shift_null_n_shifts=200,
    shift_null_min_supported_scales=2,
    shift_null_min_valid_frac=0.80,
    shift_null_merge_gap_s=0.0,
    shift_null_seed=0,
    corrlation_thr=None,
    correlation_thr=None,
    use_cal_mask=True,
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\cascade_vs_realfr\countineous_aproch",
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    cascade_model_folder=None,
    cascade_threshold=0,
    cascade_verbosity=0,
    cascade_convert_to_hz=True,
    match_real_fr_smoothing_to_model=True,
    apply_linear_scaling=False,
    spike_idx_offset_frames=0,
):
    """
    Evaluate CASCADE-predicted activity vs real FR using the same analysis pipeline as suite2p.

    CASCADE is run on each state's calcium dF/F trace (`calDff`) at native sampling (typically 30 Hz).

    After CASCADE prediction:
      - predictions are aligned back to the calcium trace length;
      - by default `cascade_convert_to_hz=True` converts predicted spikes/bin to instantaneous Hz
        by multiplying by the calcium sampling rate;
      - if the model name contains a smoothing value and
        `match_real_fr_smoothing_to_model=True`, the real-FR smoothing is matched to the model.

    The resulting CASCADE prediction is passed through
    `evaluate_suite2p_spks_as_continuous_activity`, so `error_z_mode`,
    shift-null parameters, lag correction, plotting, and correlation filtering behave the same
    as in the suite2p workflow.
    """
    model_smooth_s = _parse_cascade_model_smoothing_s(cascade_model_name)
    if bool(match_real_fr_smoothing_to_model) and np.isfinite(model_smooth_s):
        smooth_sigma_s_eff = float(model_smooth_s)
    else:
        smooth_sigma_s_eff = float(smooth_sigma_s)

    cells_cascade = copy.deepcopy(list(loaded_cells))
    cascade_errors = []
    n_pred_ok = 0

    for cell in cells_cascade:
        if not bool(cell.get("loaded", False)):
            continue
        cal_sr_cell = float(cell.get("cal_sr_hz", FS_CA_HZ_FALLBACK))
        for st in cell.get("states", []):
            cal_dff = np.asarray(st.get("calDff", []), dtype=float).ravel()
            if cal_dff.size == 0:
                cal_dff = np.asarray(st.get("calRaw", []), dtype=float).ravel()
            n_cal = int(cal_dff.size)
            if n_cal <= 0:
                st["suite2pSpks"] = np.asarray([], dtype=float)
                st["suite2pSpksMasked"] = np.asarray([], dtype=float)
                continue

            try:
                pred = _predict_cascade_for_single_trace(
                    cal_trace_dff=cal_dff,
                    model_name=str(cascade_model_name),
                    model_folder=cascade_model_folder,
                    threshold=cascade_threshold,
                    verbosity=cascade_verbosity,
                )
                n_pred_ok += 1
            except Exception as exc:
                pred = np.full(n_cal, np.nan, dtype=float)
                cascade_errors.append(
                    {
                        "cell_path": str(cell.get("cell_path", "")),
                        "state": str(st.get("state", "main")),
                        "error": str(exc),
                    }
                )

            pred = _align_1d_len(pred, n_cal, fill_value=np.nan)
            if bool(cascade_convert_to_hz):
                pred = np.asarray(pred, dtype=float) * float(cal_sr_cell)
            st["suite2pSpks"] = np.asarray(pred, dtype=float)
            cal_mask = _normalize_bool_mask(st.get("calMask", np.ones(n_cal, dtype=bool)), n_cal)
            st["suite2pSpksMasked"] = np.where(np.asarray(cal_mask, dtype=bool), np.asarray(pred, dtype=float), np.nan)

    if n_pred_ok <= 0 and len(cascade_errors) > 0:
        first_err = cascade_errors[0]
        raise RuntimeError(
            "CASCADE prediction failed for all states. "
            f"First failure: cell={first_err.get('cell_path','')}, state={first_err.get('state','')}, "
            f"error={first_err.get('error','')}"
        )

    res_df = evaluate_suite2p_spks_as_continuous_activity(
        loaded_cells=cells_cascade,
        out_subdir=out_subdir,
        smooth_sigma_s=smooth_sigma_s_eff,
        max_lag_s=max_lag_s,
        save_svg=save_svg,
        z_thresh=z_thresh,
        error_z_mode=error_z_mode,
        shift_null_window_sizes_s=shift_null_window_sizes_s,
        shift_null_n_shifts=shift_null_n_shifts,
        shift_null_min_supported_scales=shift_null_min_supported_scales,
        shift_null_min_valid_frac=shift_null_min_valid_frac,
        shift_null_merge_gap_s=shift_null_merge_gap_s,
        shift_null_seed=shift_null_seed,
        corrlation_thr=corrlation_thr,
        correlation_thr=correlation_thr,
        apply_linear_scaling=apply_linear_scaling,
        use_cal_mask=use_cal_mask,
        summary_out_dir=summary_out_dir,
        spike_idx_offset_frames=spike_idx_offset_frames,
    )
    res_df.attrs["prediction_method"] = "CASCADE"
    res_df.attrs["cascade_model_name"] = str(cascade_model_name)
    res_df.attrs["cascade_model_folder"] = None if cascade_model_folder is None else str(cascade_model_folder)
    res_df.attrs["cascade_threshold"] = cascade_threshold
    res_df.attrs["cascade_convert_to_hz"] = bool(cascade_convert_to_hz)
    res_df.attrs["real_fr_smoothing_sigma_s_requested"] = float(smooth_sigma_s)
    res_df.attrs["real_fr_smoothing_sigma_s_effective"] = float(smooth_sigma_s_eff)
    res_df.attrs["cascade_model_smoothing_sigma_s_parsed"] = float(model_smooth_s) if np.isfinite(model_smooth_s) else np.nan
    res_df.attrs["cascade_errors"] = cascade_errors
    return res_df


def _lag_correct_prediction_after_true(pred_in, lag_frames):
    x = np.asarray(pred_in, dtype=float).ravel()
    n = int(x.size)
    lag = int(max(0, lag_frames))
    if lag <= 0:
        return x.copy()
    out = np.full(n, np.nan, dtype=float)
    if lag < n:
        out[: n - lag] = x[lag:]
    return out


def _attach_cascade_predictions_to_loaded_cells(
    loaded_cells,
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    cascade_model_folder=None,
    cascade_threshold=0,
    cascade_verbosity=0,
    cascade_convert_to_hz=True,
):
    cells_cascade = copy.deepcopy(list(loaded_cells))
    errors = []
    n_ok = 0
    for cell in cells_cascade:
        if not bool(cell.get("loaded", False)):
            continue
        cal_sr_cell = float(cell.get("cal_sr_hz", FS_CA_HZ_FALLBACK))
        for st in cell.get("states", []):
            cal_dff = np.asarray(st.get("calDff", []), dtype=float).ravel()
            if cal_dff.size == 0:
                cal_dff = np.asarray(st.get("calRaw", []), dtype=float).ravel()
            n_cal = int(cal_dff.size)
            if n_cal <= 0:
                st["suite2pSpks"] = np.asarray([], dtype=float)
                st["suite2pSpksMasked"] = np.asarray([], dtype=float)
                continue
            try:
                pred = _predict_cascade_for_single_trace(
                    cal_trace_dff=cal_dff,
                    model_name=str(cascade_model_name),
                    model_folder=cascade_model_folder,
                    threshold=cascade_threshold,
                    verbosity=cascade_verbosity,
                )
                n_ok += 1
            except Exception as exc:
                pred = np.full(n_cal, np.nan, dtype=float)
                errors.append(
                    {
                        "cell_path": str(cell.get("cell_path", "")),
                        "state": str(st.get("state", "main")),
                        "error": str(exc),
                    }
                )
            pred = _align_1d_len(pred, n_cal, fill_value=np.nan)
            if bool(cascade_convert_to_hz):
                pred = np.asarray(pred, dtype=float) * float(cal_sr_cell)
            st["suite2pSpks"] = np.asarray(pred, dtype=float)
            cal_mask = _normalize_bool_mask(st.get("calMask", np.ones(n_cal, dtype=bool)), n_cal)
            st["suite2pSpksMasked"] = np.where(np.asarray(cal_mask, dtype=bool), np.asarray(pred, dtype=float), np.nan)
    return cells_cascade, errors, int(n_ok)


def _complex_like_events_from_state_pkl(pkl_data, vol_len, vol_sr):
    try:
        ev_list = _events_from_pkl_same_as_pyr_event_cal2(
            pkl_data,
            trace_len_v=int(vol_len),
            fs_v_hz=float(vol_sr),
            isi_ms=EVENT_ISI_MS_DEFAULT,
        )
    except Exception:
        return []
    out = []
    for ev in ev_list:
        et = str(ev.get("event_type", "")).strip().lower()
        if et in ("complex", "plateau") or bool(ev.get("is_complex_event", False)) or bool(ev.get("is_plateau_event", False)):
            out.append(ev)
    return out


def _build_state_true_pred_for_compare(
    cell,
    st,
    smooth_sigma_s=0.05,
    max_lag_s=0.5,
    scale_pred_to_true=True,
    use_lag_correction=True,
    use_cal_mask=True,
):
    cal_sr = float(cell.get("cal_sr_hz", FS_CA_HZ_FALLBACK))
    vol_sr = float(cell.get("vol_sr_hz", FS_VOL_HZ_DEFAULT))
    pkl_data = st.get("spike_pkl_data", None)
    if not isinstance(pkl_data, dict):
        return None, "Missing spike_pkl_data"

    pred_raw = np.asarray(st.get("suite2pSpks", []), dtype=float).ravel()
    cal_trace = np.asarray(st.get("calDff", []), dtype=float).ravel()
    if cal_trace.size == 0:
        cal_trace = np.asarray(st.get("calRaw", []), dtype=float).ravel()
    vol_trace = np.asarray(st.get("volRaw", []), dtype=float).ravel()
    n_cal = int(min(pred_raw.size, cal_trace.size if cal_trace.size else pred_raw.size))
    if n_cal <= 2:
        return None, "Empty prediction/calcium trace"
    pred_raw = _align_1d_len(pred_raw, n_cal, fill_value=np.nan)
    cal_trace = _align_1d_len(cal_trace, n_cal, fill_value=np.nan)

    vol_len = int(vol_trace.size)
    if vol_len <= 0:
        vol_len = int(np.asarray(pkl_data.get("trace_vol", []), dtype=float).size)
        vol_trace = np.asarray(pkl_data.get("trace_vol", []), dtype=float).ravel()
    if vol_len <= 0:
        return None, "Empty voltage trace"

    cal_mask = _normalize_bool_mask(st.get("calMask", np.ones(n_cal, dtype=bool)), n_cal)
    vol_mask = _normalize_bool_mask(st.get("volMask", np.ones(vol_len, dtype=bool)), vol_len)

    spk_idx = _spike_idx_from_state_pkl_aligned(
        pkl_data=pkl_data,
        vol_len=vol_len,
        vol_mask_state=vol_mask,
        vol_raw_state=vol_trace,
        pkl_path=st.get("pkl_path", None),
    )
    if spk_idx.size > 0:
        spk_idx = spk_idx[vol_mask[spk_idx]]
    if spk_idx.size > 0:
        cidx_sp = np.array(
            [_vol_idx_to_cal_idx(vv, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal) for vv in spk_idx],
            dtype=int,
        )
        spk_idx = spk_idx[cal_mask[cidx_sp]]

    true_fr_raw = _spike_idx_to_fr_on_cal(
        spike_idx_vol=spk_idx,
        n_cal=n_cal,
        vol_sr_hz=vol_sr,
        cal_sr_hz=cal_sr,
    )
    sigma_frames = max(0.0, float(smooth_sigma_s) * float(cal_sr))
    if sigma_frames > 0:
        true_fr = gaussian_filter1d(true_fr_raw.astype(float), sigma=sigma_frames, mode="nearest")
        pred_smooth = gaussian_filter1d(pred_raw.astype(float), sigma=sigma_frames, mode="nearest")
        cal_smooth = gaussian_filter1d(cal_trace.astype(float), sigma=sigma_frames, mode="nearest")
    else:
        true_fr = np.asarray(true_fr_raw, dtype=float)
        pred_smooth = np.asarray(pred_raw, dtype=float)
        cal_smooth = np.asarray(cal_trace, dtype=float)

    max_lag_frames = int(max(0, round(float(max_lag_s) * float(cal_sr))))
    corr_best_lag, lag_frames = _best_lag_corr_spks_after_fr(
        fr_smooth=true_fr,
        spks_smooth=pred_smooth,
        max_lag_frames=max_lag_frames,
    )
    if bool(use_lag_correction):
        pred_aligned = _lag_correct_prediction_after_true(pred_smooth, lag_frames)
    else:
        pred_aligned = np.asarray(pred_smooth, dtype=float)
        lag_frames = 0

    if bool(scale_pred_to_true):
        scale, offset, pred_eval = _linear_fit_scale_offset(pred_aligned, true_fr)
    else:
        scale, offset = 1.0, 0.0
        pred_eval = np.asarray(pred_aligned, dtype=float)

    if bool(use_cal_mask):
        invalid = ~cal_mask
        true_fr = np.asarray(true_fr, dtype=float)
        true_fr_raw = np.asarray(true_fr_raw, dtype=float)
        pred_eval = np.asarray(pred_eval, dtype=float)
        pred_raw = np.asarray(pred_raw, dtype=float)
        pred_smooth = np.asarray(pred_smooth, dtype=float)
        cal_trace = np.asarray(cal_trace, dtype=float)
        true_fr[invalid] = np.nan
        true_fr_raw[invalid] = np.nan
        pred_eval[invalid] = np.nan
        pred_raw[invalid] = np.nan
        pred_smooth[invalid] = np.nan
        cal_trace[invalid] = np.nan
        cal_smooth = np.asarray(cal_smooth, dtype=float)
        cal_smooth[invalid] = np.nan

    # Calcium-vs-true-FR correlation: keep both no-lag and lag-corrected
    # using the same lag-search convention as model predictions.
    corr_calcium_vs_true_fr_nolag = _pearson_corr_valid(cal_smooth, true_fr)
    corr_cal_best_lag, cal_lag_frames = _best_lag_corr_spks_after_fr(
        fr_smooth=true_fr,
        spks_smooth=cal_smooth,
        max_lag_frames=max_lag_frames,
    )
    cal_smooth_lag = _lag_correct_prediction_after_true(cal_smooth, cal_lag_frames)
    corr_calcium_vs_true_fr = _pearson_corr_valid(cal_smooth_lag, true_fr)

    return {
        "cal_sr": float(cal_sr),
        "vol_sr": float(vol_sr),
        "n_cal": int(n_cal),
        "vol_len": int(vol_len),
        "cal_trace": np.asarray(cal_trace, dtype=float),
        "vol_trace": np.asarray(vol_trace, dtype=float).ravel(),
        "spk_idx": np.asarray(spk_idx, dtype=int),
        "true_fr_raw": np.asarray(true_fr_raw, dtype=float),
        "true_fr": np.asarray(true_fr, dtype=float),
        "pred_raw": np.asarray(pred_raw, dtype=float),
        "pred_smooth": np.asarray(pred_smooth, dtype=float),
        "pred_eval": np.asarray(pred_eval, dtype=float),
        "cal_smooth": np.asarray(cal_smooth, dtype=float),
        "cal_smooth_lag": np.asarray(cal_smooth_lag, dtype=float),
        "cal_mask": np.asarray(cal_mask, dtype=bool),
        "lag_frames": int(lag_frames),
        "lag_s": float(lag_frames) / float(cal_sr),
        "corr_best_lag": float(corr_best_lag) if np.isfinite(corr_best_lag) else np.nan,
        "corr_calcium_vs_true_fr": float(corr_calcium_vs_true_fr) if np.isfinite(corr_calcium_vs_true_fr) else np.nan,
        "corr_calcium_vs_true_fr_nolag": float(corr_calcium_vs_true_fr_nolag)
        if np.isfinite(corr_calcium_vs_true_fr_nolag)
        else np.nan,
        "cal_best_lag_frames": int(cal_lag_frames),
        "cal_best_lag_s": float(cal_lag_frames) / float(cal_sr),
        "cal_best_lag_corr": float(corr_cal_best_lag) if np.isfinite(corr_cal_best_lag) else np.nan,
        "scale": float(scale) if np.isfinite(scale) else np.nan,
        "offset": float(offset) if np.isfinite(offset) else np.nan,
    }, None


def _relative_error_bias_from_rates(true_fr, pred_fr, true_fr_raw, cal_sr):
    true = np.asarray(true_fr, dtype=float).ravel()
    pred = np.asarray(pred_fr, dtype=float).ravel()
    raw = np.asarray(true_fr_raw, dtype=float).ravel()
    n = min(true.size, pred.size, raw.size)
    if n <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    true = true[:n]
    pred = pred[:n]
    raw = raw[:n]
    ok = np.isfinite(true) & np.isfinite(pred) & np.isfinite(raw)
    if np.sum(ok) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    true_counts = np.clip(true[ok] / float(cal_sr), 0.0, None)
    pred_counts = np.clip(pred[ok] / float(cal_sr), 0.0, None)
    raw_true_counts = np.clip(raw[ok] / float(cal_sr), 0.0, None)
    denom = float(np.nansum(raw_true_counts))
    if (not np.isfinite(denom)) or denom <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    fp = float(np.nansum(np.clip(pred_counts - true_counts, 0.0, None)))
    fn = float(np.nansum(np.clip(true_counts - pred_counts, 0.0, None)))
    rel_error = (fp + fn) / denom
    rel_bias = (fp - fn) / denom
    return float(rel_error), float(rel_bias), float(fp), float(fn), float(denom)


def _extract_by_cal_offsets(x, center_idx, offsets):
    arr = np.asarray(x, dtype=float).ravel()
    out = np.full(int(len(offsets)), np.nan, dtype=float)
    idx = int(center_idx) + np.asarray(offsets, dtype=int)
    ok = (idx >= 0) & (idx < arr.size)
    if np.any(ok):
        out[ok] = arr[idx[ok]]
    return out


def _align_idx_for_complex_event(ev, series, align_to="true_fr_peak"):
    mode = str(align_to).strip().lower()
    cal_sr = float(series["cal_sr"])
    vol_sr = float(series["vol_sr"])
    n_cal = int(series["n_cal"])
    vol_trace = np.asarray(series["vol_trace"], dtype=float).ravel()
    true_fr = np.asarray(series["true_fr"], dtype=float).ravel()

    sp = pe._as_sorted_unique_int(ev.get("spikes", []))
    s_v = int(ev.get("start_frame", int(sp[0]) if sp.size else 0))
    e_v = int(ev.get("end_frame", int(sp[-1]) if sp.size else s_v))
    if e_v < s_v:
        s_v, e_v = e_v, s_v
    s_v = int(max(0, min(max(0, vol_trace.size - 1), s_v)))
    e_v = int(max(s_v, min(max(0, vol_trace.size - 1), e_v)))

    if mode in ("complex_peak", "vm_peak", "voltage_peak", "peak"):
        if vol_trace.size > 0 and e_v >= s_v:
            seg = vol_trace[s_v : e_v + 1]
            if seg.size > 0 and np.any(np.isfinite(seg)):
                v_peak = int(s_v + int(np.nanargmax(seg)))
            elif sp.size > 0:
                v_peak = int(sp[0])
            else:
                v_peak = int(s_v)
        elif sp.size > 0:
            v_peak = int(sp[0])
        else:
            v_peak = int(s_v)
        return _vol_idx_to_cal_idx(v_peak, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)

    c0 = _vol_idx_to_cal_idx(s_v, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
    c1 = _vol_idx_to_cal_idx(e_v, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
    if c1 < c0:
        c0, c1 = c1, c0
    c0 = int(max(0, min(n_cal - 1, c0)))
    c1 = int(max(c0, min(n_cal - 1, c1)))
    seg = true_fr[c0 : c1 + 1]
    if seg.size > 0 and np.any(np.isfinite(seg)):
        return int(c0 + int(np.nanargmax(seg)))
    if sp.size > 0:
        return _vol_idx_to_cal_idx(int(sp[0]), vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
    return int(c0)


def _plot_complex_event_temporal_bias(
    rel_t,
    mean_cal,
    mean_true_fr,
    mean_pred_fr,
    mean_error,
    save_html,
    save_svg=None,
    title="complex-event temporal bias",
    auc_window_s=None,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=("Mean complex/plateau event", "Mean signed error (inferred - true)"),
    )
    rel_t = np.asarray(rel_t, dtype=float)
    pre_edge = float(np.nanmin(rel_t))
    post_edge = float(np.nanmax(rel_t))
    if auc_window_s is None or not np.isfinite(float(auc_window_s)):
        auc_window_s = min(abs(pre_edge), abs(post_edge))
    auc_window_s = float(auc_window_s)
    x_pre0, x_pre1 = -auc_window_s, 0.0
    x_post0, x_post1 = 0.0, auc_window_s

    for row in (1, 2):
        fig.add_vrect(x0=x_pre0, x1=x_pre1, fillcolor="rgba(76,120,168,0.12)", line_width=0, row=row, col=1)
        fig.add_vrect(x0=x_post0, x1=x_post1, fillcolor="rgba(245,133,24,0.12)", line_width=0, row=row, col=1)
        fig.add_vline(x=0.0, line=dict(color="black", width=1, dash="dash"), row=row, col=1)

    fig.add_trace(
        go.Scatter(x=rel_t, y=mean_cal, mode="lines", line=dict(color="mediumseagreen", width=2), name="mean calcium dF/F"),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=rel_t, y=mean_true_fr, mode="lines", line=dict(color="black", width=2), name="mean true FR"),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=rel_t, y=mean_pred_fr, mode="lines", line=dict(color="#d62728", width=2), name="mean inferred FR"),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=rel_t, y=mean_error, mode="lines", line=dict(color="#1f77b4", width=2), name="mean error"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="rgba(76,120,168,0.35)", size=10), name="pre AUC region"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="rgba(245,133,24,0.35)", size=10), name="post AUC region"),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Calcium dF/F", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="FR (Hz)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Error (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time from alignment (s)", row=2, col=1)
    fig.update_layout(template="simple_white", width=1050, height=760, title=title)
    os.makedirs(os.path.dirname(save_html), exist_ok=True)
    fig.write_html(save_html)
    if save_svg is not None:
        try:
            fig.write_image(save_svg)
        except Exception:
            pass
    return fig


def _plot_model_compare_bias_metrics(metrics_df, save_html, save_svg=None, title="Model compare bias metrics"):
    metrics = [
        ("pearson_corr_true_vs_pred", "Correlation with ground truth<br>(Pearson r)"),
        ("relative_error", "Relative error<br>(FP + FN) / true spikes"),
        ("relative_bias", "Relative bias (paper)<br>(FP - FN) / true spikes"),
        ("temporal_bias_index", "Temporal bias index<br>(post - pre) / (|post| + |pre|)"),
    ]
    df = metrics_df.copy()
    has_mode = "calibration_mode" in df.columns
    split_rows = False
    if has_mode:
        tmp = df.copy()
        if "model" in tmp.columns:
            tmp["_base_model"] = tmp["model"].astype(str)
        else:
            tmp["_base_model"] = (
                tmp.get("model_display", pd.Series(index=tmp.index, dtype=object))
                .astype(str)
                .str.replace(r"_(scaled|unscaled)$", "", regex=True)
            )
        tmp["_mode"] = tmp["calibration_mode"].astype(str).str.lower()
        by_model_modes = tmp.groupby("_base_model")["_mode"].apply(lambda s: set(s.dropna().tolist()))
        # Split only for true dual-report mode (at least one model has both scaled+unscaled).
        split_rows = any(("scaled" in ms) and ("unscaled" in ms) for ms in by_model_modes.tolist())

    if split_rows:
        row_modes = ["unscaled", "scaled"]
        fig = make_subplots(
            rows=2,
            cols=len(metrics),
            subplot_titles=[f"Unscaled | {m[1]}" for m in metrics] + [f"Scaled | {m[1]}" for m in metrics],
        )
    else:
        row_modes = [None]
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[m[1] for m in metrics])

    colors = {
        "suite2p": "#4C78A8",
        "suite2p_unscaled": "#4C78A8",
        "suite2p_scaled": "#2F5E91",
        "cascade": "#F58518",
        "cascade_unscaled": "#F58518",
        "cascade_scaled": "#C66A11",
        "calcium": "#F58518",
        "calcium_trace": "#666666",
    }
    fill_colors = {
        "suite2p": "rgba(76,120,168,0.28)",
        "suite2p_unscaled": "rgba(76,120,168,0.28)",
        "suite2p_scaled": "rgba(47,94,145,0.32)",
        "cascade": "rgba(245,133,24,0.28)",
        "cascade_unscaled": "rgba(245,133,24,0.28)",
        "cascade_scaled": "rgba(198,106,17,0.32)",
        "calcium": "rgba(245,133,24,0.28)",
        "calcium_trace": "rgba(120,120,120,0.28)",
    }
    for row_idx, mode_name in enumerate(row_modes, start=1):
        if mode_name is None:
            df_row = df.copy()
        else:
            df_row = df[df["calibration_mode"].astype(str).str.lower() == str(mode_name).lower()].copy()

        if "model" in df_row.columns:
            base_model = df_row["model"].astype(str)
        else:
            base_model = df_row.get("model_display", pd.Series(index=df_row.index, dtype=object)).astype(str)
            base_model = base_model.str.replace(r"_(scaled|unscaled)$", "", regex=True)
        df_row["_base_model"] = base_model

        models = [m for m in ["suite2p", "cascade"] if m in set(df_row["_base_model"].astype(str))]
        extra_models = [m for m in sorted(set(df_row["_base_model"].astype(str))) if m not in models]
        models += extra_models

        for col_idx, (metric, _label) in enumerate(metrics, start=1):
            for model in models:
                vals = pd.to_numeric(
                    df_row.loc[df_row["_base_model"].astype(str) == model, metric],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                fig.add_trace(
                    go.Box(
                        y=vals,
                        x=[model] * int(vals.size),
                        name=model,
                        legendgroup=model,
                        showlegend=(row_idx == 1 and col_idx == 1),
                        boxpoints="all",
                        pointpos=0,
                        jitter=0.48,
                        width=0.52,
                        boxmean=True,
                        marker=dict(
                            size=6,
                            opacity=0.78,
                            color=colors.get(model, "#777777"),
                            line=dict(color="rgba(0,0,0,0.45)", width=0.6),
                        ),
                        line=dict(color=colors.get(model, "#777777")),
                        fillcolor=fill_colors.get(model, "rgba(119,119,119,0.28)"),
                        opacity=1.0,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # Merge calcium-vs-true-FR correlation into first panel of each row.
            if col_idx == 1:
                dd = df_row.copy()
                if ("cell_path" in dd.columns) and ("state" in dd.columns):
                    dd = (
                        dd.sort_values(["cell_path", "state"])
                        .groupby(["cell_path", "state"], as_index=False)
                        .first()
                    )
                vals_cal = pd.to_numeric(dd.get("pearson_corr_calcium_vs_true_fr", np.nan), errors="coerce").dropna().to_numpy(dtype=float)
                if vals_cal.size > 0:
                    fig.add_trace(
                        go.Box(
                            y=vals_cal,
                            x=["calcium_trace"] * int(vals_cal.size),
                            name="calcium_trace",
                            legendgroup="calcium_trace",
                            showlegend=(row_idx == 1),
                            boxpoints="all",
                            pointpos=0,
                            jitter=0.48,
                            width=0.52,
                            boxmean=True,
                            marker=dict(
                                size=6,
                                opacity=0.78,
                                color="#666666",
                                line=dict(color="rgba(0,0,0,0.45)", width=0.6),
                            ),
                            line=dict(color="#444444"),
                            fillcolor="rgba(120,120,120,0.28)",
                            opacity=1.0,
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
            fig.update_xaxes(title_text="model", row=row_idx, col=col_idx)

    fig.update_yaxes(title_text="value", row=1, col=1)
    if split_rows:
        fig.update_yaxes(title_text="value", row=2, col=1)
    fig.update_layout(
        template="simple_white",
        width=2450,
        height=(1300 if split_rows else 760),
        title=title,
        boxmode="group",
        font=dict(size=16),
        margin=dict(l=85, r=45, t=115, b=85),
    )
    os.makedirs(os.path.dirname(save_html), exist_ok=True)
    fig.write_html(save_html)
    if save_svg is not None:
        try:
            fig.write_image(save_svg)
        except Exception:
            pass
    return fig


def _auc_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compare_suite2p_and_cascade_bias_metrics(
    loaded_cells,
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\model_compare",
    per_cell_subdir="model_compare",
    include_suite2p=True,
    include_cascade=True,
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    cascade_model_folder=None,
    cascade_threshold=0,
    cascade_verbosity=0,
    cascade_convert_to_hz=True,
    smooth_sigma_s=0.05,
    max_lag_s=0.5,
    scale_pred_to_true=True,
    scale_pred_to_true_suite2p=None,
    scale_pred_to_true_cascade=None,
    use_lag_correction=True,
    report_scaled_and_unscaled=False,
    use_cal_mask=True,
    peri_window_s=(-1.0, 1.0),
    align_to="true_fr_peak",
    bias_auc_window_s=None,
    min_events_for_plot=1,
    save_svg=True,
):
    """
    Compare suite2p and CASCADE/calcium predictions using CASCADE-paper-style metrics.

    Event analysis:
      - uses voltage-defined complex/plateau events from the loaded spike PKL;
      - extracts peri-event calcium, true voltage-derived FR, and inferred FR;
      - aligns to either `align_to="true_fr_peak"` or `align_to="vm_peak"`;
      - computes signed error as inferred FR - true FR;
      - computes temporal bias from pre/post signed-error AUCs.

    Cell/state metrics:
      - Pearson correlation between inferred spike rate and ground-truth spike rate;
      - relative error = (false-positive mass + false-negative mass) / true spike count;
      - relative bias = (false-positive mass - false-negative mass) / true spike count.

    Scaling control:
      - `scale_pred_to_true` is a global default (backward-compatible).
      - `scale_pred_to_true_suite2p` overrides scaling for suite2p only when not None.
      - `scale_pred_to_true_cascade` overrides scaling for cascade only when not None.
      - `report_scaled_and_unscaled=True` reports both variants for each model/state
        in one run (adds `calibration_mode` and `model_display` columns).

    Returns a DataFrame with one row per model/cell/state and saves:
      - per-cell complex-event temporal-bias HTML/SVG plots;
      - summary CSV;
      - summary violin + beeswarm figure in `summary_out_dir/model_compare`.
    """
    model_sets = []
    if bool(include_suite2p):
        model_sets.append(("suite2p", loaded_cells, []))
    if bool(include_cascade):
        cascade_cells, cascade_errors, n_cascade_ok = _attach_cascade_predictions_to_loaded_cells(
            loaded_cells=loaded_cells,
            cascade_model_name=cascade_model_name,
            cascade_model_folder=cascade_model_folder,
            cascade_threshold=cascade_threshold,
            cascade_verbosity=cascade_verbosity,
            cascade_convert_to_hz=cascade_convert_to_hz,
        )
        if n_cascade_ok <= 0 and len(cascade_errors) > 0:
            first_err = cascade_errors[0]
            if bool(include_suite2p):
                print(
                    "[WARN] CASCADE prediction unavailable for this run; "
                    "continuing with suite2p only. "
                    f"First failure: cell={first_err.get('cell_path','')}, "
                    f"state={first_err.get('state','')}, error={first_err.get('error','')}"
                )
            else:
                raise RuntimeError(
                    "CASCADE prediction failed for all states. "
                    f"First failure: cell={first_err.get('cell_path','')}, state={first_err.get('state','')}, "
                    f"error={first_err.get('error','')}"
                )
        else:
            model_sets.append(("cascade", cascade_cells, cascade_errors))

    summary_dir = os.path.join(str(summary_out_dir), "model_compare")
    os.makedirs(summary_dir, exist_ok=True)
    rows = []
    pre_s, post_s = float(peri_window_s[0]), float(peri_window_s[1])
    if pre_s > 0:
        pre_s = -pre_s
    if post_s <= 0:
        post_s = abs(post_s)

    for model_name, cells_for_model, model_errors in model_sets:
        if str(model_name).lower() == "suite2p":
            model_scale_flag = bool(scale_pred_to_true) if scale_pred_to_true_suite2p is None else bool(scale_pred_to_true_suite2p)
        elif str(model_name).lower() == "cascade":
            model_scale_flag = bool(scale_pred_to_true) if scale_pred_to_true_cascade is None else bool(scale_pred_to_true_cascade)
        else:
            model_scale_flag = bool(scale_pred_to_true)
        for cell in cells_for_model:
            if not bool(cell.get("loaded", False)):
                continue
            cell_path = str(cell.get("cell_path", ""))
            cell_name = os.path.basename(cell_path)
            for st in cell.get("states", []):
                state = str(st.get("state", "main"))
                series, reason = _build_state_true_pred_for_compare(
                    cell=cell,
                    st=st,
                    smooth_sigma_s=smooth_sigma_s,
                    max_lag_s=max_lag_s,
                    scale_pred_to_true=(False if bool(report_scaled_and_unscaled) else model_scale_flag),
                    use_lag_correction=use_lag_correction,
                    use_cal_mask=use_cal_mask,
                )
                base_row = {
                    "model": model_name,
                    "cell_path": cell_path,
                    "cell_name": cell_name,
                    "state": state,
                    "pkl_path": st.get("pkl_path", None),
                    "ok": False,
                    "reason": reason if reason is not None else "",
                }
                if series is None:
                    rows.append(base_row)
                    continue
                true_fr = np.asarray(series["true_fr"], dtype=float)
                pred_unscaled = np.asarray(series["pred_eval"], dtype=float)
                scale_fit, offset_fit, pred_scaled = _linear_fit_scale_offset(pred_unscaled, true_fr)
                if bool(report_scaled_and_unscaled):
                    mode_defs = [
                        ("unscaled", pred_unscaled, 1.0, 0.0),
                        ("scaled", pred_scaled, scale_fit, offset_fit),
                    ]
                else:
                    mode_name = "scaled" if bool(model_scale_flag) else "unscaled"
                    if bool(model_scale_flag):
                        mode_defs = [(mode_name, pred_scaled, scale_fit, offset_fit)]
                    else:
                        mode_defs = [(mode_name, pred_unscaled, 1.0, 0.0)]

                ev_list = _complex_like_events_from_state_pkl(
                    st.get("spike_pkl_data", {}),
                    vol_len=int(series["vol_len"]),
                    vol_sr=float(series["vol_sr"]),
                )
                pre_n = int(round(abs(pre_s) * float(series["cal_sr"])))
                post_n = int(round(abs(post_s) * float(series["cal_sr"])))
                offsets = np.arange(-pre_n, post_n + 1, dtype=int)
                rel_t = offsets.astype(float) / float(series["cal_sr"])
                n_events_total = int(len(ev_list))

                for mode_name, pred_mode, scale_mode, offset_mode in mode_defs:
                    rel_error, rel_bias, fp, fn, true_spikes = _relative_error_bias_from_rates(
                        true_fr=series["true_fr"],
                        pred_fr=pred_mode,
                        true_fr_raw=series["true_fr_raw"],
                        cal_sr=float(series["cal_sr"]),
                    )
                    pearson_corr = _pearson_corr_valid(series["true_fr"], pred_mode)

                    cal_stack = []
                    true_stack = []
                    pred_stack = []
                    err_stack = []
                    for ev in ev_list:
                        c_align = _align_idx_for_complex_event(ev, series=series, align_to=align_to)
                        cal_seg = _extract_by_cal_offsets(series["cal_trace"], c_align, offsets)
                        true_seg = _extract_by_cal_offsets(series["true_fr"], c_align, offsets)
                        pred_seg = _extract_by_cal_offsets(pred_mode, c_align, offsets)
                        err_seg = pred_seg - true_seg
                        if np.sum(np.isfinite(true_seg) & np.isfinite(pred_seg)) < max(3, int(0.25 * offsets.size)):
                            continue
                        cal_stack.append(cal_seg)
                        true_stack.append(true_seg)
                        pred_stack.append(pred_seg)
                        err_stack.append(err_seg)

                    n_events_used = int(len(err_stack))
                    temporal_bias_index = np.nan
                    temporal_bias_index_raw_formula = np.nan
                    auc_pre = np.nan
                    auc_post = np.nan
                    fig_html = None
                    fig_svg = None
                    if n_events_used > 0:
                        cal_arr = np.asarray(cal_stack, dtype=float)
                        true_arr = np.asarray(true_stack, dtype=float)
                        pred_arr = np.asarray(pred_stack, dtype=float)
                        err_arr = np.asarray(err_stack, dtype=float)
                        mean_cal = np.nanmean(cal_arr, axis=0)
                        mean_true = np.nanmean(true_arr, axis=0)
                        mean_pred = np.nanmean(pred_arr, axis=0)
                        mean_err = np.nanmean(err_arr, axis=0)

                        if bias_auc_window_s is None:
                            auc_win = float(min(abs(pre_s), abs(post_s)))
                        else:
                            auc_win = float(bias_auc_window_s)
                        pre_mask = (rel_t >= -auc_win) & (rel_t < 0)
                        post_mask = (rel_t > 0) & (rel_t <= auc_win)
                        if np.sum(pre_mask & np.isfinite(mean_err)) >= 2 and np.sum(post_mask & np.isfinite(mean_err)) >= 2:
                            auc_pre = _auc_trapezoid(mean_err[pre_mask], rel_t[pre_mask])
                            auc_post = _auc_trapezoid(mean_err[post_mask], rel_t[post_mask])
                            denom_abs = float(abs(auc_post) + abs(auc_pre))
                            if denom_abs > 0:
                                temporal_bias_index = float((auc_post - auc_pre) / denom_abs)
                            denom_raw = float(auc_post + auc_pre)
                            if abs(denom_raw) > 1e-12:
                                temporal_bias_index_raw_formula = float((auc_post - auc_pre) / denom_raw)

                        if n_events_used >= int(min_events_for_plot):
                            safe_cell = re.sub(r"[^A-Za-z0-9]+", "_", cell_name).strip("_") or "cell"
                            safe_state = re.sub(r"[^A-Za-z0-9]+", "_", state).strip("_") or "state"
                            safe_align = re.sub(r"[^A-Za-z0-9]+", "_", str(align_to)).strip("_") or "align"
                            out_dir = os.path.join(cell_path, str(per_cell_subdir))
                            fig_html = os.path.join(out_dir, f"complex_event_temporal_bias_{model_name}_{mode_name}_{safe_state}_{safe_align}.html")
                            fig_svg = os.path.join(out_dir, f"complex_event_temporal_bias_{model_name}_{mode_name}_{safe_state}_{safe_align}.svg") if bool(save_svg) else None
                            title = (
                                f"{safe_cell} | {state} | {model_name} | {mode_name} | complex/plateau temporal bias "
                                f"(align={align_to}, n={n_events_used}, TBI={temporal_bias_index:.3g})"
                            )
                            _plot_complex_event_temporal_bias(
                                rel_t=rel_t,
                                mean_cal=mean_cal,
                                mean_true_fr=mean_true,
                                mean_pred_fr=mean_pred,
                                mean_error=mean_err,
                                save_html=fig_html,
                                save_svg=fig_svg,
                                title=title,
                                auc_window_s=auc_win,
                            )

                    row = base_row.copy()
                    row.update(
                        {
                            "ok": True,
                            "reason": "",
                            "model_display": f"{model_name}_{mode_name}",
                            "calibration_mode": str(mode_name),
                            "pearson_corr_true_vs_pred": float(pearson_corr) if np.isfinite(pearson_corr) else np.nan,
                            "pearson_corr_calcium_vs_true_fr": float(series.get("corr_calcium_vs_true_fr", np.nan))
                            if np.isfinite(series.get("corr_calcium_vs_true_fr", np.nan))
                            else np.nan,
                            "pearson_corr_calcium_vs_true_fr_nolag": float(series.get("corr_calcium_vs_true_fr_nolag", np.nan))
                            if np.isfinite(series.get("corr_calcium_vs_true_fr_nolag", np.nan))
                            else np.nan,
                            "calcium_best_lag_s": float(series.get("cal_best_lag_s", np.nan))
                            if np.isfinite(series.get("cal_best_lag_s", np.nan))
                            else np.nan,
                            "calcium_best_lag_corr": float(series.get("cal_best_lag_corr", np.nan))
                            if np.isfinite(series.get("cal_best_lag_corr", np.nan))
                            else np.nan,
                            "relative_error": rel_error,
                            "relative_bias": rel_bias,
                            "bias_mass_spikes": float(fp - fn) if (np.isfinite(fp) and np.isfinite(fn)) else np.nan,
                            "false_positive_mass": fp,
                            "false_negative_mass": fn,
                            "true_spike_count": true_spikes,
                            "temporal_bias_index": temporal_bias_index,
                            "temporal_bias_index_raw_formula": temporal_bias_index_raw_formula,
                            "auc_pre_error": auc_pre,
                            "auc_post_error": auc_post,
                            "n_complex_plateau_events_total": n_events_total,
                            "n_complex_plateau_events_used": n_events_used,
                            "align_to": str(align_to),
                            "smooth_sigma_s": float(smooth_sigma_s),
                            "max_lag_s": float(max_lag_s),
                            "lag_s": float(series["lag_s"]),
                            "best_lag_corr": float(series["corr_best_lag"]) if np.isfinite(series["corr_best_lag"]) else np.nan,
                            "scale_pred_to_true": bool(mode_name == "scaled"),
                            "scale_pred_to_true_suite2p": bool(scale_pred_to_true_suite2p) if scale_pred_to_true_suite2p is not None else np.nan,
                            "scale_pred_to_true_cascade": bool(scale_pred_to_true_cascade) if scale_pred_to_true_cascade is not None else np.nan,
                            "scale": float(scale_mode) if np.isfinite(scale_mode) else np.nan,
                            "offset": float(offset_mode) if np.isfinite(offset_mode) else np.nan,
                            "figure_html": fig_html,
                            "figure_svg": fig_svg,
                        }
                    )
                    rows.append(row)

    metrics_df = pd.DataFrame(rows)
    csv_path = os.path.join(summary_dir, "bias_metrics_model_compare.csv")
    metrics_df.to_csv(csv_path, index=False)
    if "ok" in metrics_df.columns:
        ok_df = metrics_df[metrics_df["ok"] == True].copy()
    else:
        ok_df = metrics_df.copy()
    fig_html = os.path.join(summary_dir, "bias_metrics_model_compare.html")
    fig_svg = os.path.join(summary_dir, "bias_metrics_model_compare.svg") if bool(save_svg) else None
    if len(ok_df) > 0:
        _plot_model_compare_bias_metrics(
            ok_df,
            save_html=fig_html,
            save_svg=fig_svg,
            title=(
                f"Model compare bias metrics | align={align_to} | "
                f"smooth={float(smooth_sigma_s):.3g}s | "
                + ("scaled+unscaled" if bool(report_scaled_and_unscaled) else f"scaled={bool(scale_pred_to_true)}")
            ),
        )
    metrics_df.attrs["summary_csv"] = csv_path
    metrics_df.attrs["summary_html"] = fig_html if len(ok_df) > 0 else None
    metrics_df.attrs["summary_svg"] = fig_svg if len(ok_df) > 0 and bool(save_svg) else None
    return metrics_df


def _spike_count_bin_1_to_6p(n_spikes):
    try:
        n = int(n_spikes)
    except Exception:
        return "unknown"
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    return "6+"


def _hex_to_rgb_triplet(hex_color):
    s = str(hex_color).strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _shade_color_rgba(base_hex, level_idx, n_levels):
    r, g, b = _hex_to_rgb_triplet(base_hex)
    if n_levels <= 1:
        t = 0.0
    else:
        t = float(level_idx) / float(n_levels - 1)
    blend = 0.70 * (1.0 - t)
    rr = int(round(r * (1.0 - blend) + 255.0 * blend))
    gg = int(round(g * (1.0 - blend) + 255.0 * blend))
    bb = int(round(b * (1.0 - blend) + 255.0 * blend))
    return f"rgba({rr},{gg},{bb},0.95)"


def plot_isolated_event_inferred_spike_distributions(
    loaded_cells,
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\model_compare",
    include_suite2p=True,
    include_cascade=True,
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    cascade_model_folder=None,
    cascade_threshold=0,
    cascade_verbosity=0,
    cascade_convert_to_hz=True,
    smooth_sigma_s=0.05,
    max_lag_s=0.5,
    scale_pred_to_true=True,
    scale_pred_to_true_suite2p=None,
    scale_pred_to_true_cascade=None,
    use_lag_correction=True,
    use_cal_mask=False,
    selection_mode="event_distance",
    min_events_per_curve=6,
    kde_points=450,
    save_svg=True,
):
    """
    KDE distributions of inferred spike counts per isolated event.

    Isolation rules are reused from pyr_event_cal3:
      build_events(...) + choose_events(..., selection_mode='event_distance').

    Figure layout:
      rows = models (suite2p/cascade), cols = true event spike-count bins (1,2,3,4,5,6+)
      each curve in a panel = event type (simple/complex/plateau).
    """
    if pe3 is None:
        raise RuntimeError("pyr_event_cal3_pipeline is required for event-distance isolation rules.")
    try:
        from scipy.stats import gaussian_kde
    except Exception as exc:
        raise RuntimeError(f"scipy.stats.gaussian_kde is required: {exc}")

    model_sets = []
    if bool(include_suite2p):
        model_sets.append(("suite2p", loaded_cells, []))
    if bool(include_cascade):
        cascade_cells, cascade_errors, n_cascade_ok = _attach_cascade_predictions_to_loaded_cells(
            loaded_cells=loaded_cells,
            cascade_model_name=cascade_model_name,
            cascade_model_folder=cascade_model_folder,
            cascade_threshold=cascade_threshold,
            cascade_verbosity=cascade_verbosity,
            cascade_convert_to_hz=cascade_convert_to_hz,
        )
        if n_cascade_ok <= 0 and len(cascade_errors) > 0:
            first_err = cascade_errors[0]
            if bool(include_suite2p):
                print(
                    "[WARN] CASCADE prediction unavailable for this run; "
                    "continuing with suite2p only. "
                    f"First failure: cell={first_err.get('cell_path','')}, "
                    f"state={first_err.get('state','')}, error={first_err.get('error','')}"
                )
            else:
                raise RuntimeError(
                    "CASCADE prediction failed for all states. "
                    f"First failure: cell={first_err.get('cell_path','')}, state={first_err.get('state','')}, "
                    f"error={first_err.get('error','')}"
                )
        else:
            model_sets.append(("cascade", cascade_cells, cascade_errors))

    rows = []
    event_types = ["simple", "complex", "plateau"]
    spike_bins = ["1", "2", "3", "4", "5", "6+"]

    for model_name, cells_for_model, _model_errors in model_sets:
        if str(model_name).lower() == "suite2p":
            model_scale_flag = bool(scale_pred_to_true) if scale_pred_to_true_suite2p is None else bool(scale_pred_to_true_suite2p)
        elif str(model_name).lower() == "cascade":
            model_scale_flag = bool(scale_pred_to_true) if scale_pred_to_true_cascade is None else bool(scale_pred_to_true_cascade)
        else:
            model_scale_flag = bool(scale_pred_to_true)

        for cell in cells_for_model:
            if not bool(cell.get("loaded", False)):
                continue
            cell_path = str(cell.get("cell_path", ""))
            cell_name = os.path.basename(cell_path)
            for st in cell.get("states", []):
                state = str(st.get("state", "main"))
                pkl_data = st.get("spike_pkl_data", None)
                if not isinstance(pkl_data, dict):
                    continue

                series, _reason = _build_state_true_pred_for_compare(
                    cell=cell,
                    st=st,
                    smooth_sigma_s=smooth_sigma_s,
                    max_lag_s=max_lag_s,
                    scale_pred_to_true=bool(model_scale_flag),
                    use_lag_correction=use_lag_correction,
                    use_cal_mask=use_cal_mask,
                )
                if series is None:
                    continue

                cal_sr = float(series["cal_sr"])
                vol_sr = float(series["vol_sr"])
                n_cal = int(series["n_cal"])
                pred_fr = np.asarray(series["pred_eval"], dtype=float).ravel()
                pred_fr = _align_1d_len(pred_fr, n_cal, fill_value=np.nan)

                cal_for_select = np.asarray(st.get("calDff", []), dtype=float).ravel()
                if cal_for_select.size == 0:
                    cal_for_select = np.asarray(series["cal_trace"], dtype=float).ravel()
                if cal_for_select.size <= 3:
                    continue
                cal_for_select = _align_1d_len(cal_for_select, n_cal, fill_value=np.nan)

                try:
                    ev_all = pe3.build_events(pkl_data)
                    ev_chosen = pe3.choose_events(
                        ev_all,
                        gdf=cal_for_select,
                        fluor=cal_for_select,
                        cal_sr=float(cal_sr),
                        vol_sr=float(vol_sr),
                        selection_mode=str(selection_mode),
                    )
                    ev_selected = [ev for ev in ev_chosen if bool(getattr(ev, "include", False))]
                except Exception:
                    ev_selected = []

                for ev in ev_selected:
                    et = str(getattr(ev, "event_type", "")).strip().lower()
                    if et not in event_types:
                        continue
                    n_true = int(getattr(ev, "n_spikes", 0))
                    s_v = int(getattr(ev, "start_frame", 0))
                    e_v = int(getattr(ev, "end_frame", s_v))
                    if e_v < s_v:
                        s_v, e_v = e_v, s_v
                    c0 = _vol_idx_to_cal_idx(s_v, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
                    c1 = _vol_idx_to_cal_idx(e_v, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=n_cal)
                    if c1 < c0:
                        c0, c1 = c1, c0
                    seg = pred_fr[int(c0) : int(c1) + 1]
                    if seg.size == 0:
                        continue
                    seg = np.asarray(seg, dtype=float)
                    seg = seg[np.isfinite(seg)]
                    if seg.size == 0:
                        continue
                    inferred_spikes = float(np.nansum(np.clip(seg, 0.0, None)) / float(cal_sr))
                    rows.append(
                        {
                            "model": str(model_name),
                            "cell_path": str(cell_path),
                            "cell_name": str(cell_name),
                            "state": str(state),
                            "event_type": str(et),
                            "n_spikes_true": int(n_true),
                            "n_spike_bin": _spike_count_bin_1_to_6p(int(n_true)),
                            "inferred_spikes_in_event": inferred_spikes,
                            "scale_pred_to_true": bool(model_scale_flag),
                            "selection_mode": str(selection_mode),
                        }
                    )

    out_df = pd.DataFrame(rows)
    out_dir = os.path.join(str(summary_out_dir), "model_compare")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "isolated_event_inferred_spike_distributions.csv")
    _safe_to_csv(out_df, out_csv, index=False)

    if len(out_df) == 0:
        out_df.attrs["summary_csv"] = out_csv
        out_df.attrs["summary_html"] = None
        out_df.attrs["summary_svg"] = None
        return out_df

    model_order = [m for m in ["suite2p", "cascade"] if m in set(out_df["model"].astype(str))]
    model_order += [m for m in sorted(set(out_df["model"].astype(str))) if m not in model_order]
    nrows = int(max(1, len(model_order)))
    ncols = int(len(spike_bins))

    subplot_titles = []
    for m in model_order:
        for sb in spike_bins:
            subplot_titles.append(f"{m} | n={sb}")
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

    base_hex = {"simple": "#000000", "complex": "#d62728", "plateau": "#8000ff"}

    for ri, m in enumerate(model_order, start=1):
        for ci, sb in enumerate(spike_bins, start=1):
            sub = out_df[
                (out_df["model"].astype(str) == str(m))
                & (out_df["n_spike_bin"].astype(str) == str(sb))
            ].copy()
            if len(sub) == 0:
                continue
            all_vals = pd.to_numeric(sub["inferred_spikes_in_event"], errors="coerce").to_numpy(float)
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size == 0:
                continue
            x_max = float(np.nanpercentile(all_vals, 99.5))
            if (not np.isfinite(x_max)) or x_max <= 0:
                x_max = float(max(1.0, np.nanmax(all_vals)))
            x_grid = np.linspace(0.0, x_max * 1.05, int(max(150, kde_points)))

            for et in event_types:
                vals = pd.to_numeric(
                    sub.loc[sub["event_type"].astype(str) == str(et), "inferred_spikes_in_event"],
                    errors="coerce",
                ).dropna().to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size < int(max(2, min_events_per_curve)):
                    continue
                try:
                    kde = gaussian_kde(vals)
                    y = kde(x_grid)
                except Exception:
                    mu = float(np.nanmean(vals))
                    sd = float(np.nanstd(vals))
                    sd = max(sd, 1e-3)
                    y = np.exp(-0.5 * ((x_grid - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))

                shade = _shade_color_rgba(base_hex.get(et, "#000000"), 5, 6)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y,
                        mode="lines",
                        line=dict(color=shade, width=2.0),
                        name=f"{et}",
                        legendgroup=f"{et}",
                        showlegend=(ri == 1 and ci == 1),
                    ),
                    row=ri,
                    col=ci,
                )

            fig.update_xaxes(title_text="# inferred spikes in event", row=ri, col=ci)
            fig.update_yaxes(title_text="kernel density", row=ri, col=ci)

    fig.update_layout(
        template="simple_white",
        width=540 * ncols,
        height=max(420, 360 * nrows),
        title=(
            "Isolated event distributions of inferred spike numbers | "
            f"selection={selection_mode} | panels by true spike count (1,2,3,4,5,6+)"
        ),
        legend=dict(title="event type"),
    )

    out_html = os.path.join(out_dir, "isolated_event_inferred_spike_distributions.html")
    out_svg = os.path.join(out_dir, "isolated_event_inferred_spike_distributions.svg") if bool(save_svg) else None
    fig.write_html(out_html)
    if out_svg is not None:
        try:
            fig.write_image(out_svg)
        except Exception:
            pass

    out_df.attrs["summary_csv"] = out_csv
    out_df.attrs["summary_html"] = out_html
    out_df.attrs["summary_svg"] = out_svg
    return out_df


def diagnose_cascade_input_quality(
    loaded_cells,
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\cascade_vs_realfr\countineous_aproch",
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    pre_window_s=0.25,
    post_window_s=1.0,
    baseline_window_s=(-0.20, -0.05),
    peak_window_s=(0.00, 0.50),
    min_single_spike_events=5,
    save_svg=True,
):
    """
    Diagnostics to explain CASCADE scale mismatch, per cell/state.

    Per row (cell + state) computes:
      1) median dF/F (unit sanity check)
      2) single-spike calcium amplitude
      3) single-spike mean-shape rise and decay times
      4) noise level and coarse model-noise match (high/low)
    """
    rows = []
    pre_window_s = float(abs(pre_window_s))
    post_window_s = float(abs(post_window_s))
    bw0, bw1 = float(baseline_window_s[0]), float(baseline_window_s[1])
    pw0, pw1 = float(peak_window_s[0]), float(peak_window_s[1])

    for cell in loaded_cells:
        if not bool(cell.get("loaded", False)):
            continue
        cell_path = str(cell.get("cell_path", ""))
        cell_name = os.path.basename(cell_path)
        cal_sr = float(cell.get("cal_sr_hz", FS_CA_HZ_FALLBACK))
        vol_sr = float(cell.get("vol_sr_hz", FS_VOL_HZ_DEFAULT))

        for st in cell.get("states", []):
            state = str(st.get("state", "main"))
            pkl_data = st.get("spike_pkl_data", None)
            if not isinstance(pkl_data, dict):
                continue

            cal = np.asarray(st.get("calDff", []), dtype=float).ravel()
            if cal.size == 0:
                cal = np.asarray(st.get("calRaw", []), dtype=float).ravel()
            n_cal = int(cal.size)
            if n_cal <= 3:
                continue

            median_dff = float(np.nanmedian(cal)) if np.any(np.isfinite(cal)) else np.nan

            cfin = cal[np.isfinite(cal)]
            if cfin.size >= 4:
                d1 = np.diff(cfin)
                d1_med = float(np.nanmedian(d1))
                d1_mad = float(np.nanmedian(np.abs(d1 - d1_med)))
                noise_sigma = (1.4826 * d1_mad) / np.sqrt(2.0) if np.isfinite(d1_mad) else np.nan
            else:
                noise_sigma = np.nan

            vol_len = int(np.asarray(st.get("volRaw", []), dtype=float).size)
            if vol_len <= 0:
                vol_len = int(np.asarray(pkl_data.get("trace_vol", []), dtype=float).size)
            if vol_len <= 0:
                continue

            ev = _events_from_pkl_same_as_pyr_event_cal2(
                pkl_data,
                trace_len_v=int(vol_len),
                fs_v_hz=float(vol_sr),
                isi_ms=EVENT_ISI_MS_DEFAULT,
            )

            single_v_idx = []
            for e in ev:
                sp = pe._as_sorted_unique_int(e.get("spikes", []))
                if sp.size != 1:
                    continue
                et = str(e.get("event_type", "")).strip().lower()
                if et != "simple":
                    continue
                single_v_idx.append(int(sp[0]))

            pre_n = int(round(pre_window_s * cal_sr))
            post_n = int(round(post_window_s * cal_sr))
            offsets = np.arange(-pre_n, post_n + 1, dtype=int)
            t_rel = offsets.astype(float) / float(cal_sr)
            base_mask = (t_rel >= bw0) & (t_rel <= bw1)
            peak_mask = (t_rel >= pw0) & (t_rel <= pw1)

            amp_vals = []
            segs = []
            for vv in single_v_idx:
                cc = _vol_idx_to_cal_idx(vv, vol_sr=float(vol_sr), cal_sr=float(cal_sr), cal_len=n_cal)
                seg = _extract_by_cal_offsets(cal, cc, offsets)
                if np.sum(np.isfinite(seg)) < max(5, int(0.40 * seg.size)):
                    continue
                b = float(np.nanmedian(seg[base_mask])) if np.any(base_mask) else np.nan
                p = float(np.nanmax(seg[peak_mask])) if np.any(peak_mask) else np.nan
                if np.isfinite(b) and np.isfinite(p):
                    amp_vals.append(float(p - b))
                segs.append(seg)

            amp_med = float(np.nanmedian(amp_vals)) if len(amp_vals) > 0 else np.nan
            amp_mean = float(np.nanmean(amp_vals)) if len(amp_vals) > 0 else np.nan
            n_single_events_used = int(len(segs))
            rise_t_s = np.nan
            decay_t_s = np.nan

            if n_single_events_used >= int(max(1, min_single_spike_events)):
                mseg = np.nanmean(np.asarray(segs, dtype=float), axis=0)
                if np.any(np.isfinite(mseg)):
                    b0 = float(np.nanmedian(mseg[base_mask])) if np.any(base_mask) else float(np.nanmin(mseg))
                    p0 = float(np.nanmax(mseg[peak_mask])) if np.any(peak_mask) else float(np.nanmax(mseg))
                    a0 = p0 - b0
                    if np.isfinite(a0) and a0 > 0:
                        y10 = b0 + 0.10 * a0
                        y90 = b0 + 0.90 * a0
                        peak_idx = int(np.nanargmax(mseg))

                        pre_part = mseg[: peak_idx + 1]
                        pre_t = t_rel[: peak_idx + 1]
                        i10 = np.where(pre_part >= y10)[0]
                        i90 = np.where(pre_part >= y90)[0]
                        if i10.size > 0 and i90.size > 0:
                            t10 = float(pre_t[int(i10[0])])
                            t90 = float(pre_t[int(i90[0])])
                            if np.isfinite(t10) and np.isfinite(t90) and t90 >= t10:
                                rise_t_s = float(t90 - t10)

                        post_part = mseg[peak_idx:]
                        post_t = t_rel[peak_idx:]
                        target = b0 + (a0 / np.e)
                        idc = np.where(post_part <= target)[0]
                        if idc.size > 0:
                            tdc = float(post_t[int(idc[0])])
                            tpk = float(t_rel[peak_idx])
                            if np.isfinite(tdc) and np.isfinite(tpk) and tdc >= tpk:
                                decay_t_s = float(tdc - tpk)

            rows.append(
                {
                    "cell_path": cell_path,
                    "cell_name": cell_name,
                    "state": state,
                    "cal_sr_hz": float(cal_sr),
                    "vol_sr_hz": float(vol_sr),
                    "median_dff": median_dff,
                    "noise_sigma_dff": float(noise_sigma) if np.isfinite(noise_sigma) else np.nan,
                    "n_simple_single_events_total": int(len(single_v_idx)),
                    "n_simple_single_events_used": int(n_single_events_used),
                    "single_spike_amp_median_dff": amp_med,
                    "single_spike_amp_mean_dff": amp_mean,
                    "single_spike_rise_time_s": float(rise_t_s) if np.isfinite(rise_t_s) else np.nan,
                    "single_spike_decay_time_s": float(decay_t_s) if np.isfinite(decay_t_s) else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    out_dir = os.path.join(str(summary_out_dir), "cascade_input_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    if len(df) == 0:
        out_csv = os.path.join(out_dir, "cascade_input_diagnostics_summary.csv")
        pd.DataFrame(columns=["cell_path", "state"]).to_csv(out_csv, index=False)
        df.attrs["summary_csv"] = out_csv
        df.attrs["summary_html"] = None
        df.attrs["summary_svg"] = None
        return df

    model_noise = "unknown"
    mn = str(cascade_model_name).strip().lower()
    if "high_noise" in mn:
        model_noise = "high"
    elif "low_noise" in mn:
        model_noise = "low"
    df["cascade_model_noise_label"] = model_noise

    nz = pd.to_numeric(df["noise_sigma_dff"], errors="coerce").to_numpy(float)
    noise_med = float(np.nanmedian(nz)) if np.any(np.isfinite(nz)) else np.nan
    noise_cls = np.full(len(df), "unknown", dtype=object)
    if np.isfinite(noise_med):
        noise_cls = np.where(nz >= noise_med, "high", "low").astype(object)
        noise_cls[~np.isfinite(nz)] = "unknown"
    df["data_noise_label"] = noise_cls
    if model_noise in ("high", "low"):
        df["model_noise_match"] = df["data_noise_label"].astype(str) == model_noise
    else:
        df["model_noise_match"] = np.nan

    out_csv = os.path.join(out_dir, "cascade_input_diagnostics_summary.csv")
    _safe_to_csv(df, out_csv, index=False)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "1) median dF/F per cell-state",
            "2) single-spike Ca amplitude (median dF/F)",
            "3) event shape: rise vs decay",
            "4) noise level (robust sigma dF/F)",
            "5) noise vs single-spike amplitude",
            "6) model-noise match",
        ),
    )

    m1 = pd.to_numeric(df["median_dff"], errors="coerce").to_numpy(float)
    m2 = pd.to_numeric(df["single_spike_amp_median_dff"], errors="coerce").to_numpy(float)
    m3x = pd.to_numeric(df["single_spike_rise_time_s"], errors="coerce").to_numpy(float)
    m3y = pd.to_numeric(df["single_spike_decay_time_s"], errors="coerce").to_numpy(float)
    m4 = pd.to_numeric(df["noise_sigma_dff"], errors="coerce").to_numpy(float)

    fig.add_trace(go.Histogram(x=m1[np.isfinite(m1)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=m2[np.isfinite(m2)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=m3x[np.isfinite(m3x) & np.isfinite(m3y)],
            y=m3y[np.isfinite(m3x) & np.isfinite(m3y)],
            mode="markers",
            marker=dict(size=6, color="rgba(75,75,75,0.75)"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(go.Histogram(x=m4[np.isfinite(m4)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=m4[np.isfinite(m4) & np.isfinite(m2)],
            y=m2[np.isfinite(m4) & np.isfinite(m2)],
            mode="markers",
            marker=dict(size=6, color="rgba(75,75,75,0.75)"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    m_ok = df["model_noise_match"].astype(str)
    n_true = int(np.sum(m_ok == "True"))
    n_false = int(np.sum(m_ok == "False"))
    n_unk = int(np.sum(~df["model_noise_match"].isin([True, False])))
    fig.add_trace(
        go.Bar(
            x=["match", "mismatch", "unknown"],
            y=[n_true, n_false, n_unk],
            marker=dict(color=["#2ca02c", "#d62728", "#8a8a8a"]),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    fig.update_xaxes(title_text="median dF/F", row=1, col=1)
    fig.update_yaxes(title_text="# rows", row=1, col=1)
    fig.update_xaxes(title_text="single-spike amplitude (dF/F)", row=1, col=2)
    fig.update_yaxes(title_text="# rows", row=1, col=2)
    fig.update_xaxes(title_text="rise time (s)", row=1, col=3)
    fig.update_yaxes(title_text="decay time (s)", row=1, col=3)
    fig.update_xaxes(title_text="noise sigma (dF/F)", row=2, col=1)
    fig.update_yaxes(title_text="# rows", row=2, col=1)
    fig.update_xaxes(title_text="noise sigma (dF/F)", row=2, col=2)
    fig.update_yaxes(title_text="single-spike amp (dF/F)", row=2, col=2)
    fig.update_xaxes(title_text="model-vs-data noise class", row=2, col=3)
    fig.update_yaxes(title_text="# rows", row=2, col=3)

    fig.update_layout(
        template="simple_white",
        width=1650,
        height=980,
        title=(
            "CASCADE input diagnostics | "
            f"model={cascade_model_name} | n_rows={int(len(df))} | n_cells={int(df['cell_path'].astype(str).nunique())}"
        ),
    )
    out_html = os.path.join(out_dir, "cascade_input_diagnostics_summary.html")
    out_svg = os.path.join(out_dir, "cascade_input_diagnostics_summary.svg") if bool(save_svg) else None
    fig.write_html(out_html)
    if out_svg is not None:
        try:
            fig.write_image(out_svg)
        except Exception:
            pass

    df.attrs["summary_csv"] = out_csv
    df.attrs["summary_html"] = out_html
    df.attrs["summary_svg"] = out_svg
    return df


def diagnose_cascade_training_dataset_input_quality(
    summary_out_dir=r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\cascade_vs_realfr\countineous_aproch",
    cascade_repo_root=None,
    dataset_names=("DS30-GCaMP8f-m-V1", "DS31-GCaMP8m-m-V1", "DS32-GCaMP8s-m-V1"),
    cascade_model_name="GC8_EXC_30Hz_smoothing50ms_high_noise",
    isi_ms=30.0,
    pre_window_s=0.25,
    post_window_s=1.0,
    baseline_window_s=(-0.20, -0.05),
    peak_window_s=(0.00, 0.50),
    min_single_spike_events=5,
    save_svg=True,
):
    """
    Run the same input diagnostics on the original CASCADE training datasets.

    Expected source:
      <cascade_repo_root>/Ground_truth/DS30..., DS31..., DS32...
    """
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError(f"scipy.io.loadmat is required for CASCADE ground-truth .mat files: {exc}")

    if cascade_repo_root is None:
        cascade_repo_root = os.path.join(os.path.dirname(__file__), "_cascade_ref_repo")
    gt_root = os.path.join(str(cascade_repo_root), "Ground_truth")
    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"Missing CASCADE Ground_truth folder: {gt_root}")

    pre_window_s = float(abs(pre_window_s))
    post_window_s = float(abs(post_window_s))
    bw0, bw1 = float(baseline_window_s[0]), float(baseline_window_s[1])
    pw0, pw1 = float(peak_window_s[0]), float(peak_window_s[1])
    isi_s = float(isi_ms) / 1000.0
    rows = []

    def _events_to_time_s(events_ap, tmax_s):
        ev = np.asarray(events_ap, dtype=float).ravel()
        ev = ev[np.isfinite(ev)]
        if ev.size == 0:
            return ev
        cands = [1.0, 1e-3, 1e-4]
        best = ev
        best_score = np.inf
        for sc in cands:
            e = ev * float(sc)
            inside = np.sum((e >= 0) & (e <= float(tmax_s) + 1.0))
            frac_in = float(inside) / float(max(1, e.size))
            score = abs(np.nanmax(e) - float(tmax_s)) - (2.0 * frac_in)
            if score < best_score:
                best_score = score
                best = e
        best = np.asarray(best, dtype=float)
        best = best[(best >= 0) & (best <= float(tmax_s) + 1.0)]
        return np.sort(best)

    for ds_name in dataset_names:
        ds_dir = os.path.join(gt_root, str(ds_name))
        if not os.path.isdir(ds_dir):
            continue
        mat_paths = sorted(glob.glob(os.path.join(ds_dir, "*.mat")))
        for mp in mat_paths:
            try:
                mm = loadmat(mp, squeeze_me=True, struct_as_record=False)
                if "CAttached" not in mm:
                    continue
                attached = np.ravel(mm["CAttached"])
            except Exception:
                continue

            for rec_i, rec in enumerate(attached):
                try:
                    cal = np.asarray(getattr(rec, "fluo_mean"), dtype=float).ravel()
                    t = np.asarray(getattr(rec, "fluo_time"), dtype=float).ravel()
                    ev_ap = np.asarray(getattr(rec, "events_AP"), dtype=float).ravel()
                except Exception:
                    continue

                n = int(min(cal.size, t.size))
                if n <= 3:
                    continue
                cal = cal[:n]
                t = t[:n]
                ok_t = np.isfinite(t)
                if np.sum(ok_t) <= 3:
                    continue
                t = t - float(np.nanmin(t[ok_t]))
                dt = float(np.nanmedian(np.diff(t[np.isfinite(t)])))
                if (not np.isfinite(dt)) or dt <= 0:
                    continue
                cal_sr = 1.0 / dt
                tmax = float(np.nanmax(t))
                sp_t = _events_to_time_s(ev_ap, tmax_s=tmax)

                median_dff = float(np.nanmedian(cal)) if np.any(np.isfinite(cal)) else np.nan
                cfin = cal[np.isfinite(cal)]
                if cfin.size >= 4:
                    d1 = np.diff(cfin)
                    d1_med = float(np.nanmedian(d1))
                    d1_mad = float(np.nanmedian(np.abs(d1 - d1_med)))
                    noise_sigma = (1.4826 * d1_mad) / np.sqrt(2.0) if np.isfinite(d1_mad) else np.nan
                else:
                    noise_sigma = np.nan

                if sp_t.size == 0:
                    single_idx = np.array([], dtype=int)
                else:
                    groups = []
                    cur = [float(sp_t[0])]
                    for s in sp_t[1:]:
                        if float(s - cur[-1]) <= float(isi_s):
                            cur.append(float(s))
                        else:
                            groups.append(cur)
                            cur = [float(s)]
                    groups.append(cur)
                    singles = [g[0] for g in groups if len(g) == 1]
                    single_idx = np.searchsorted(t, np.asarray(singles, dtype=float), side="left")
                    single_idx = np.clip(single_idx, 0, n - 1).astype(int)

                pre_n = int(round(pre_window_s * cal_sr))
                post_n = int(round(post_window_s * cal_sr))
                offsets = np.arange(-pre_n, post_n + 1, dtype=int)
                t_rel = offsets.astype(float) / float(cal_sr)
                base_mask = (t_rel >= bw0) & (t_rel <= bw1)
                peak_mask = (t_rel >= pw0) & (t_rel <= pw1)

                amp_vals = []
                segs = []
                for ci in single_idx:
                    seg = _extract_by_cal_offsets(cal, int(ci), offsets)
                    if np.sum(np.isfinite(seg)) < max(5, int(0.40 * seg.size)):
                        continue
                    b = float(np.nanmedian(seg[base_mask])) if np.any(base_mask) else np.nan
                    p = float(np.nanmax(seg[peak_mask])) if np.any(peak_mask) else np.nan
                    if np.isfinite(b) and np.isfinite(p):
                        amp_vals.append(float(p - b))
                    segs.append(seg)

                amp_med = float(np.nanmedian(amp_vals)) if len(amp_vals) > 0 else np.nan
                amp_mean = float(np.nanmean(amp_vals)) if len(amp_vals) > 0 else np.nan
                n_single_events_used = int(len(segs))
                rise_t_s = np.nan
                decay_t_s = np.nan

                if n_single_events_used >= int(max(1, min_single_spike_events)):
                    mseg = np.nanmean(np.asarray(segs, dtype=float), axis=0)
                    if np.any(np.isfinite(mseg)):
                        b0 = float(np.nanmedian(mseg[base_mask])) if np.any(base_mask) else float(np.nanmin(mseg))
                        p0 = float(np.nanmax(mseg[peak_mask])) if np.any(peak_mask) else float(np.nanmax(mseg))
                        a0 = p0 - b0
                        if np.isfinite(a0) and a0 > 0:
                            y10 = b0 + 0.10 * a0
                            y90 = b0 + 0.90 * a0
                            peak_idx = int(np.nanargmax(mseg))
                            pre_part = mseg[: peak_idx + 1]
                            pre_t = t_rel[: peak_idx + 1]
                            i10 = np.where(pre_part >= y10)[0]
                            i90 = np.where(pre_part >= y90)[0]
                            if i10.size > 0 and i90.size > 0:
                                t10 = float(pre_t[int(i10[0])])
                                t90 = float(pre_t[int(i90[0])])
                                if np.isfinite(t10) and np.isfinite(t90) and t90 >= t10:
                                    rise_t_s = float(t90 - t10)
                            post_part = mseg[peak_idx:]
                            post_t = t_rel[peak_idx:]
                            target = b0 + (a0 / np.e)
                            idc = np.where(post_part <= target)[0]
                            if idc.size > 0:
                                tdc = float(post_t[int(idc[0])])
                                tpk = float(t_rel[peak_idx])
                                if np.isfinite(tdc) and np.isfinite(tpk) and tdc >= tpk:
                                    decay_t_s = float(tdc - tpk)

                rows.append(
                    {
                        "dataset_name": str(ds_name),
                        "source_file": str(mp),
                        "source_entry_idx": int(rec_i),
                        "cell_type": str(getattr(rec, "cell_type", "")),
                        "cal_sr_hz": float(cal_sr),
                        "median_dff": median_dff,
                        "noise_sigma_dff": float(noise_sigma) if np.isfinite(noise_sigma) else np.nan,
                        "n_simple_single_events_total": int(single_idx.size),
                        "n_simple_single_events_used": int(n_single_events_used),
                        "single_spike_amp_median_dff": amp_med,
                        "single_spike_amp_mean_dff": amp_mean,
                        "single_spike_rise_time_s": float(rise_t_s) if np.isfinite(rise_t_s) else np.nan,
                        "single_spike_decay_time_s": float(decay_t_s) if np.isfinite(decay_t_s) else np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = os.path.join(str(summary_out_dir), "cascade_input_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    if len(df) == 0:
        out_csv = os.path.join(out_dir, "cascade_input_diagnostics_summary_casDS.csv")
        pd.DataFrame(columns=["dataset_name", "source_file"]).to_csv(out_csv, index=False)
        df.attrs["summary_csv"] = out_csv
        df.attrs["summary_html"] = None
        df.attrs["summary_svg"] = None
        return df

    model_noise = "unknown"
    mn = str(cascade_model_name).strip().lower()
    if "high_noise" in mn:
        model_noise = "high"
    elif "low_noise" in mn:
        model_noise = "low"
    df["cascade_model_noise_label"] = model_noise
    nz = pd.to_numeric(df["noise_sigma_dff"], errors="coerce").to_numpy(float)
    noise_med = float(np.nanmedian(nz)) if np.any(np.isfinite(nz)) else np.nan
    noise_cls = np.full(len(df), "unknown", dtype=object)
    if np.isfinite(noise_med):
        noise_cls = np.where(nz >= noise_med, "high", "low").astype(object)
        noise_cls[~np.isfinite(nz)] = "unknown"
    df["data_noise_label"] = noise_cls
    if model_noise in ("high", "low"):
        df["model_noise_match"] = df["data_noise_label"].astype(str) == model_noise
    else:
        df["model_noise_match"] = np.nan

    out_csv = os.path.join(out_dir, "cascade_input_diagnostics_summary_casDS.csv")
    _safe_to_csv(df, out_csv, index=False)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "1) median dF/F (CASCADE DS)",
            "2) single-spike Ca amplitude (median dF/F)",
            "3) event shape: rise vs decay",
            "4) noise level (robust sigma dF/F)",
            "5) noise vs single-spike amplitude",
            "6) model-noise match",
        ),
    )
    m1 = pd.to_numeric(df["median_dff"], errors="coerce").to_numpy(float)
    m2 = pd.to_numeric(df["single_spike_amp_median_dff"], errors="coerce").to_numpy(float)
    m3x = pd.to_numeric(df["single_spike_rise_time_s"], errors="coerce").to_numpy(float)
    m3y = pd.to_numeric(df["single_spike_decay_time_s"], errors="coerce").to_numpy(float)
    m4 = pd.to_numeric(df["noise_sigma_dff"], errors="coerce").to_numpy(float)

    fig.add_trace(go.Histogram(x=m1[np.isfinite(m1)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=m2[np.isfinite(m2)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=m3x[np.isfinite(m3x) & np.isfinite(m3y)],
            y=m3y[np.isfinite(m3x) & np.isfinite(m3y)],
            mode="markers",
            marker=dict(size=5, color="rgba(75,75,75,0.75)"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(go.Histogram(x=m4[np.isfinite(m4)], marker=dict(color="rgba(0,0,0,0.55)"), nbinsx=24, showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=m4[np.isfinite(m4) & np.isfinite(m2)],
            y=m2[np.isfinite(m4) & np.isfinite(m2)],
            mode="markers",
            marker=dict(size=5, color="rgba(75,75,75,0.75)"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    m_ok = df["model_noise_match"].astype(str)
    n_true = int(np.sum(m_ok == "True"))
    n_false = int(np.sum(m_ok == "False"))
    n_unk = int(np.sum(~df["model_noise_match"].isin([True, False])))
    fig.add_trace(
        go.Bar(
            x=["match", "mismatch", "unknown"],
            y=[n_true, n_false, n_unk],
            marker=dict(color=["#2ca02c", "#d62728", "#8a8a8a"]),
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.update_xaxes(title_text="median dF/F", row=1, col=1)
    fig.update_yaxes(title_text="# rows", row=1, col=1)
    fig.update_xaxes(title_text="single-spike amplitude (dF/F)", row=1, col=2)
    fig.update_yaxes(title_text="# rows", row=1, col=2)
    fig.update_xaxes(title_text="rise time (s)", row=1, col=3)
    fig.update_yaxes(title_text="decay time (s)", row=1, col=3)
    fig.update_xaxes(title_text="noise sigma (dF/F)", row=2, col=1)
    fig.update_yaxes(title_text="# rows", row=2, col=1)
    fig.update_xaxes(title_text="noise sigma (dF/F)", row=2, col=2)
    fig.update_yaxes(title_text="single-spike amp (dF/F)", row=2, col=2)
    fig.update_xaxes(title_text="model-vs-data noise class", row=2, col=3)
    fig.update_yaxes(title_text="# rows", row=2, col=3)
    fig.update_layout(
        template="simple_white",
        width=1650,
        height=980,
        title=(
            "CASCADE training-dataset diagnostics | "
            f"model={cascade_model_name} | datasets={','.join([str(x) for x in dataset_names])} | "
            f"n_rows={int(len(df))}"
        ),
    )

    out_html = os.path.join(out_dir, "cascade_input_diagnostics_summary_casDS.html")
    out_svg = os.path.join(out_dir, "cascade_input_diagnostics_summary_casDS.svg") if bool(save_svg) else None
    fig.write_html(out_html)
    if out_svg is not None:
        try:
            fig.write_image(out_svg)
        except Exception:
            pass

    df.attrs["summary_csv"] = out_csv
    df.attrs["summary_html"] = out_html
    df.attrs["summary_svg"] = out_svg
    return df


def _load_suite2p_deconv_fr(cell_folder, fs_ca_hz, roi_idx=None):
    suite2p_dir = os.path.join(os.path.dirname(cell_folder), "Sync", "cal", "suite2p", "plane0")
    spks_path = os.path.join(suite2p_dir, "spks.npy")
    if not os.path.isfile(spks_path):
        raise FileNotFoundError(f"Missing suite2p deconvolution file: {spks_path}")

    raw_path = os.path.join(cell_folder, "calTrace.csv")
    cal_raw = None
    if os.path.isfile(raw_path):
        try:
            cal_raw = pe._read_trace_csv_1d(raw_path)
        except Exception:
            cal_raw = None

    selected_idx = None
    match_corr = np.nan
    if roi_idx is not None:
        selected_idx = int(roi_idx)
    else:
        if cal_raw is not None and np.asarray(cal_raw).size > 0:
            selected_idx, match_corr = pe._resolve_suite2p_row_idx(cell_folder, np.asarray(cal_raw, float).ravel(), suite2p_dir)
        else:
            selected_idx = pe._cell_idx_from_folder(cell_folder)
            match_corr = np.nan

    spks = np.asarray(np.load(spks_path, mmap_mode="r"), dtype=float)
    if spks.ndim == 1:
        spks = spks.reshape(1, -1)
    if spks.ndim < 2 or spks.shape[0] == 0:
        raise RuntimeError(f"Invalid spks.npy shape: {spks.shape}")

    if selected_idx is None or (selected_idx < 0) or (selected_idx >= int(spks.shape[0])):
        selected_idx = 0
        match_corr = np.nan

    deconv = np.asarray(spks[int(selected_idx)], dtype=float).ravel()
    pred_fr = deconv * float(fs_ca_hz)
    return pred_fr, {
        "suite2p_dir": suite2p_dir,
        "spks_path": spks_path,
        "roi_idx": int(selected_idx),
        "match_corr_to_calTrace": float(match_corr) if np.isfinite(match_corr) else np.nan,
    }


def _real_fr_on_ca_from_pkl(d, n_ca, fs_v_hz, fs_ca_hz, smooth_sigma_s=REAL_FR_SMOOTH_SIGMA_S_DEFAULT):
    sp = pe._as_sorted_unique_int(d.get("vm_all_spikes", []))
    if sp.size == 0 or int(n_ca) <= 0:
        fr = np.zeros(int(max(0, n_ca)), dtype=float)
        return fr, fr.copy()

    spike_t_s = np.asarray(sp, float) / float(fs_v_hz)
    edges_s = np.arange(int(n_ca) + 1, dtype=float) / float(fs_ca_hz)
    counts, _ = np.histogram(spike_t_s, bins=edges_s)
    fr = counts.astype(float) * float(fs_ca_hz)

    if smooth_sigma_s is None or float(smooth_sigma_s) <= 0:
        return fr, fr.copy()
    sigma_samples = float(smooth_sigma_s) * float(fs_ca_hz)
    if sigma_samples <= 0:
        return fr, fr.copy()
    fr_sm = gaussian_filter1d(fr.astype(float), sigma=sigma_samples, mode="nearest")
    return fr, fr_sm


def _events_from_pkl_same_as_pyr_event_cal2(d, trace_len_v, fs_v_hz=FS_VOL_HZ_DEFAULT, isi_ms=EVENT_ISI_MS_DEFAULT):
    trace_vol = np.asarray(d.get("trace_vol", []), dtype=float).ravel()
    ev, _, _ = pe._events_from_saved_labels(
        d=d,
        trace_len=int(trace_len_v),
        trace_vol=trace_vol if trace_vol.size == int(trace_len_v) else None,
        vol_sr=float(fs_v_hz),
        cs_z_start_thr=getattr(pe, "CS_Z_START_THR", 1.5),
        cs_z_end_thr=getattr(pe, "CS_Z_END_THR", 0.5),
        non_cs_pad_frames=5,
        simple_isi_ms=float(isi_ms),
    )
    return ev


def analyze_suite2p_vs_real_fr(
    cell_folder,
    pkl_path=None,
    cal_sr_hz=None,
    fs_v_hz=FS_VOL_HZ_DEFAULT,
    real_fr_smooth_sigma_s=REAL_FR_SMOOTH_SIGMA_S_DEFAULT,
    pred_fr_smooth_sigma_s=PRED_FR_SMOOTH_SIGMA_S_DEFAULT,
    resid_z_k=RESID_Z_K_DEFAULT,
    min_seg_dur_s=MIN_SEG_DUR_S_DEFAULT,
    calibrate_pred_to_real=True,
    calibration_clip_q=99.5,
    lag_search_max_s=2.0,
    db_path=None,
    out_dir=None,
    save_outputs=True,
    file_tag="suite2p_vs_realfr",
):
    cell_folder = str(cell_folder)
    if not os.path.isdir(cell_folder):
        raise FileNotFoundError(f"Missing cell folder: {cell_folder}")

    pkl_path = _pick_pkl(cell_folder, pkl_path=pkl_path)
    suffix = _suffix_from_pkl_name(pkl_path)
    with open(pkl_path, "rb") as f:
        d = _safe_pickle_load(f)

    if cal_sr_hz is None:
        cal_sr_hz = _safe_cal_sr_from_db(cell_folder, db_path=db_path, fallback=FS_CA_HZ_FALLBACK)
    cal_sr_hz = float(cal_sr_hz)

    pred_fr_raw, suite2p_info = _load_suite2p_deconv_fr(cell_folder, fs_ca_hz=cal_sr_hz, roi_idx=None)
    n_ca = int(pred_fr_raw.size)
    if n_ca <= 0:
        raise RuntimeError("Empty predicted FR from suite2p deconvolution.")

    # If pkl has trace_cal, trim to common length
    trace_cal = np.asarray(d.get("trace_cal", []), dtype=float).ravel()
    if trace_cal.size > 0:
        n_ca = int(min(n_ca, trace_cal.size))
    pred_fr_raw = np.asarray(pred_fr_raw[:n_ca], dtype=float)

    real_fr_raw, real_fr_smooth = _real_fr_on_ca_from_pkl(
        d=d,
        n_ca=n_ca,
        fs_v_hz=float(fs_v_hz),
        fs_ca_hz=float(cal_sr_hz),
        smooth_sigma_s=float(real_fr_smooth_sigma_s),
    )

    pred_raw = np.asarray(pred_fr_raw, dtype=float)
    real = np.asarray(real_fr_smooth, dtype=float)
    n = int(min(pred_raw.size, real.size))
    pred_raw = pred_raw[:n]
    real = real[:n]
    real_fr_raw = real_fr_raw[:n]
    t_s = np.arange(n, dtype=float) / float(cal_sr_hz)

    # Suite2p spks amplitude is often in arbitrary units; calibrate to real-FR scale.
    pred_scale_alpha = 1.0
    if bool(calibrate_pred_to_real):
        pred_scale_alpha = _fit_nonneg_scale(pred_raw, real, clip_q=float(calibration_clip_q))
    pred_cal = pred_raw * float(pred_scale_alpha)
    if pred_fr_smooth_sigma_s is None:
        pred_fr_smooth_sigma_s = real_fr_smooth_sigma_s
    pred_sigma_s = float(pred_fr_smooth_sigma_s)
    if pred_sigma_s > 0:
        pred_sigma_samples = pred_sigma_s * float(cal_sr_hz)
        pred = gaussian_filter1d(pred_cal.astype(float), sigma=pred_sigma_samples, mode="nearest")
    else:
        pred = pred_cal.copy()
    trace_len_v = int(np.asarray(d.get("trace_vol", []), dtype=float).size)
    if trace_len_v <= 0:
        sp = pe._as_sorted_unique_int(d.get("vm_all_spikes", []))
        trace_len_v = int(sp.max() + 1) if sp.size > 0 else 0
    events = _events_from_pkl_same_as_pyr_event_cal2(
        d=d,
        trace_len_v=trace_len_v,
        fs_v_hz=float(fs_v_hz),
        isi_ms=float(EVENT_ISI_MS_DEFAULT),
    )
    lag_frames, lag_scan_r = _best_lag_by_corr(real, pred, fs_hz=float(cal_sr_hz), max_lag_s=float(lag_search_max_s))
    lag_seconds = float(lag_frames / float(cal_sr_hz)) if float(cal_sr_hz) > 0 else np.nan
    pred_lag = _shift_with_nan(pred, int(lag_frames))

    def _compute_mode(pred_mode, mode_key, mode_title):
        valid = np.isfinite(pred_mode) & np.isfinite(real)
        pearson_r = np.nan
        if np.sum(valid) >= 3:
            sx = np.nanstd(real[valid])
            sy = np.nanstd(pred_mode[valid])
            if sx > 0 and sy > 0:
                pearson_r = float(np.corrcoef(real[valid], pred_mode[valid])[0, 1])

        resid = real - pred_mode
        resid_z, resid_med, resid_scale = _robust_z(resid)
        k = float(resid_z_k)
        missed_mask = resid_z >= k
        over_mask = resid_z <= -k
        min_len = int(max(1, np.ceil(float(min_seg_dur_s) * float(cal_sr_hz))))
        missed_segs = _mask_to_segments(missed_mask, min_len=min_len)
        over_segs = _mask_to_segments(over_mask, min_len=min_len)

        seg_rows = []
        for s, e in missed_segs:
            seg_rows.append(
                {
                    "kind": "missed_underestimated",
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "start_s": float(s / cal_sr_hz),
                    "end_s": float(e / cal_sr_hz),
                    "duration_s": float((e - s + 1) / cal_sr_hz),
                    "mode": mode_key,
                }
            )
        for s, e in over_segs:
            seg_rows.append(
                {
                    "kind": "overestimated",
                    "start_idx": int(s),
                    "end_idx": int(e),
                    "start_s": float(s / cal_sr_hz),
                    "end_s": float(e / cal_sr_hz),
                    "duration_s": float((e - s + 1) / cal_sr_hz),
                    "mode": mode_key,
                }
            )
        seg_df = pd.DataFrame(seg_rows)

        evt_rows = []
        for i, ev in enumerate(events):
            sp = pe._as_sorted_unique_int(ev.get("spikes", []))
            if sp.size == 0:
                continue
            first_v = int(sp[0])
            first_ca = int(np.clip(np.round((float(first_v) / float(fs_v_hz)) * float(cal_sr_hz)), 0, n - 1))
            in_missed = _in_any_segment(first_ca, missed_segs)
            in_over = _in_any_segment(first_ca, over_segs)
            real_at_first = float(real[first_ca]) if 0 <= first_ca < n else np.nan
            pred_at_first = float(pred_mode[first_ca]) if 0 <= first_ca < n else np.nan
            resid_hz_at_first = float(resid[first_ca]) if 0 <= first_ca < n else np.nan
            resid_z_at_first = float(resid_z[first_ca]) if 0 <= first_ca < n else np.nan
            evt_rows.append(
                {
                    "event_idx": int(i),
                    "event_type": str(ev.get("event_type", "simple")),
                    "event_kind": str(ev.get("event_kind", "single")),
                    "n_spikes_in_event": int(sp.size),
                    "first_spike_v_idx": int(first_v),
                    "first_spike_time_s": float(first_v / float(fs_v_hz)),
                    "first_spike_ca_idx": int(first_ca),
                    "in_missed_underestimated_area": bool(in_missed),
                    "in_overestimated_area": bool(in_over),
                    "real_fr_at_first_spike_hz": real_at_first,
                    "pred_fr_at_first_spike_hz": pred_at_first,
                    "residual_hz_at_first_spike": resid_hz_at_first,
                    "residual_z_at_first_spike": resid_z_at_first,
                    "mode": mode_key,
                }
            )
        evt_df = pd.DataFrame(evt_rows)
        if len(evt_df) > 0:
            evt_df["area_kind"] = "none"
            in_pos = evt_df["in_missed_underestimated_area"].astype(bool)
            real0 = pd.to_numeric(evt_df["real_fr_at_first_spike_hz"], errors="coerce")
            pred0 = pd.to_numeric(evt_df["pred_fr_at_first_spike_hz"], errors="coerce")
            missed_like = in_pos & np.isfinite(real0) & np.isfinite(pred0) & (pred0 <= np.maximum(1.0, 0.10 * real0))
            under_like = in_pos & (~missed_like)
            evt_df.loc[missed_like, "area_kind"] = "missed"
            evt_df.loc[under_like, "area_kind"] = "underestimated"
            evt_df.loc[evt_df["in_overestimated_area"], "area_kind"] = "overestimated"
            both = evt_df["in_missed_underestimated_area"] & evt_df["in_overestimated_area"]
            evt_df.loc[both, "area_kind"] = "both"
        else:
            evt_df["area_kind"] = []

        compare_df = pd.DataFrame(
            {
                "time_s": t_s,
                "predicted_fr_hz_raw": pred_raw,
                "predicted_fr_hz_calibrated": pred_cal,
                "predicted_fr_hz": pred_mode,
                "real_fr_hz_raw": real_fr_raw[:n],
                "real_fr_hz_smooth": real,
                "residual_real_minus_pred_hz": resid[:n],
                "residual_robust_z": resid_z[:n],
                "missed_mask": missed_mask[:n],
                "overestimated_mask": over_mask[:n],
                "mode": mode_key,
            }
        )

        fig_fr = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.72, 0.28],
            subplot_titles=("Real vs Suite2p-predicted FR", "Difference over time (Real - Predicted)"),
        )
        fig_fr.add_trace(
            go.Scatter(
                x=t_s,
                y=real,
                mode="lines",
                name=f"Real FR (smooth, sigma={real_fr_smooth_sigma_s:.3g}s)",
                line=dict(color="#4C78A8", width=2.0),
            ),
            row=1,
            col=1,
        )
        fig_fr.add_trace(
            go.Scatter(
                x=t_s,
                y=pred_raw,
                mode="lines",
                name="Suite2p predicted FR raw (spks * fs_ca)",
                line=dict(color="#F58518", width=1.2),
                opacity=0.45,
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
        fig_fr.add_trace(
            go.Scatter(
                x=t_s,
                y=pred_cal,
                mode="lines",
                name="Suite2p predicted FR calibrated (unsmoothed)",
                line=dict(color="#F58518", width=1.2, dash="dot"),
                opacity=0.55,
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
        fig_fr.add_trace(
            go.Scatter(
                x=t_s,
                y=pred_mode,
                mode="lines",
                name=f"Suite2p predicted FR [{mode_key}]",
                line=dict(color="#F58518", width=2.0),
            ),
            row=1,
            col=1,
        )
        fig_fr.add_trace(
            go.Scatter(
                x=t_s,
                y=resid,
                mode="lines",
                name="Difference (real - predicted)",
                line=dict(color="#6A3D9A", width=1.8),
            ),
            row=2,
            col=1,
        )
        for s, e in missed_segs:
            fig_fr.add_vrect(
                x0=float(s / cal_sr_hz),
                x1=float(e / cal_sr_hz),
                fillcolor="rgba(220,20,60,0.18)",
                line_width=0,
                annotation_text="missed",
                annotation_position="top left",
                row="all",
                col=1,
            )
        for s, e in over_segs:
            fig_fr.add_vrect(
                x0=float(s / cal_sr_hz),
                x1=float(e / cal_sr_hz),
                fillcolor="rgba(54,162,235,0.18)",
                line_width=0,
                annotation_text="over",
                annotation_position="bottom left",
                row="all",
                col=1,
            )
        thr_pos = float(resid_med + k * resid_scale) if np.isfinite(resid_med) and np.isfinite(resid_scale) else np.nan
        thr_neg = float(resid_med - k * resid_scale) if np.isfinite(resid_med) and np.isfinite(resid_scale) else np.nan
        fig_fr.add_hline(y=0.0, line_width=1.2, line_dash="solid", line_color="#555555", row=2, col=1)
        if np.isfinite(thr_pos):
            fig_fr.add_hline(y=thr_pos, line_width=1.2, line_dash="dash", line_color="crimson", row=2, col=1)
        if np.isfinite(thr_neg):
            fig_fr.add_hline(y=thr_neg, line_width=1.2, line_dash="dash", line_color="#1F77B4", row=2, col=1)
        fig_fr.update_layout(
            template="simple_white",
            width=1450,
            height=860,
            title=(
                f"{os.path.basename(cell_folder)} | {os.path.basename(pkl_path)} | {mode_title}"
                f"<br><sup>Pearson r={pearson_r:.4f} | lag={lag_seconds:.4f}s ({int(lag_frames)} frames) | "
                f"fs_ca={cal_sr_hz:.4f} Hz | ROI idx={suite2p_info.get('roi_idx')} | "
                f"pred alpha={pred_scale_alpha:.4g} | pred smooth sigma={pred_sigma_s:.3g}s | resid robust-z k={k:.2f}</sup>"
            ),
            legend=dict(orientation="h"),
        )
        fig_fr.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_fr.update_yaxes(title_text="Firing rate (Hz)", row=1, col=1)
        fig_fr.update_yaxes(title_text="Real - Pred (Hz)", row=2, col=1)

        fig_evt = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Real spike events in missed areas",
                "Real spike events in underestimated areas",
                "Real spike events in overestimated areas",
            ),
            horizontal_spacing=0.08,
        )
        color_map = {"simple": "#2ca02c", "complex": "#e377c2"}
        for col, area_kind in ((1, "missed"), (2, "underestimated"), (3, "overestimated")):
            sub = evt_df[evt_df["area_kind"] == area_kind].copy() if len(evt_df) else evt_df.copy()
            if len(sub) == 0:
                continue
            for et in ("simple", "complex"):
                ss = sub[sub["event_type"] == et]
                if len(ss) == 0:
                    continue
                cnt = ss.groupby("n_spikes_in_event").size().reset_index(name="count")
                fig_evt.add_trace(
                    go.Bar(
                        x=cnt["n_spikes_in_event"],
                        y=cnt["count"],
                        name=f"{et} ({area_kind})",
                        marker_color=color_map.get(et, "#666666"),
                        showlegend=True,
                    ),
                    row=1,
                    col=col,
                )
            fig_evt.update_xaxes(title_text="# spikes in event", row=1, col=col)
            fig_evt.update_yaxes(title_text="# events", row=1, col=col)
        fig_evt.update_layout(
            template="simple_white",
            width=1850,
            height=650,
            barmode="group",
            title=(
                f"{os.path.basename(cell_folder)} | {mode_title} | event-size distribution in significant-difference areas"
                "<br><sup>event clustering: ISI<=30ms from vm_all_spikes; complex if any spike is vm_complex_spike</sup>"
            ),
            legend=dict(orientation="h"),
        )
        return {
            "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
            "resid_robust_median": float(resid_med) if np.isfinite(resid_med) else np.nan,
            "resid_robust_scale": float(resid_scale) if np.isfinite(resid_scale) else np.nan,
            "n_missed_segments": int(len(missed_segs)),
            "n_over_segments": int(len(over_segs)),
            "compare_df": compare_df,
            "segments_df": seg_df,
            "events_df": evt_df,
            "figure_fr": fig_fr,
            "figure_events": fig_evt,
        }

    raw_out = _compute_mode(pred_mode=pred, mode_key="nolag", mode_title="No lag correction")
    lag_out = _compute_mode(pred_mode=pred_lag, mode_key="lagcorr", mode_title="Lag corrected")

    outputs = {
        "cell_folder": cell_folder,
        "pkl_path": pkl_path,
        "pkl_suffix": suffix,
        "cal_sr_hz": float(cal_sr_hz),
        "fs_v_hz": float(fs_v_hz),
        "lag_frames": int(lag_frames),
        "lag_seconds": float(lag_seconds) if np.isfinite(lag_seconds) else np.nan,
        "lag_search_max_s": float(lag_search_max_s),
        "lag_search_peak_r": float(lag_scan_r) if np.isfinite(lag_scan_r) else np.nan,
        "pearson_r": raw_out["pearson_r"],
        "pearson_r_lagcorr": lag_out["pearson_r"],
        "resid_robust_median": raw_out["resid_robust_median"],
        "resid_robust_scale": raw_out["resid_robust_scale"],
        "resid_robust_median_lagcorr": lag_out["resid_robust_median"],
        "resid_robust_scale_lagcorr": lag_out["resid_robust_scale"],
        "n_missed_segments": raw_out["n_missed_segments"],
        "n_over_segments": raw_out["n_over_segments"],
        "n_missed_segments_lagcorr": lag_out["n_missed_segments"],
        "n_over_segments_lagcorr": lag_out["n_over_segments"],
        "suite2p_roi_idx": int(suite2p_info.get("roi_idx", -1)),
        "suite2p_roi_match_corr_to_calTrace": suite2p_info.get("match_corr_to_calTrace", np.nan),
        "suite2p_spks_path": suite2p_info.get("spks_path", None),
        "pred_scale_alpha": float(pred_scale_alpha),
        "pred_smooth_sigma_s": float(pred_sigma_s),
        "calibrate_pred_to_real": bool(calibrate_pred_to_real),
        # Legacy keys (nolag)
        "figure_fr": raw_out["figure_fr"],
        "figure_events": raw_out["figure_events"],
        "compare_df": raw_out["compare_df"],
        "segments_df": raw_out["segments_df"],
        "events_df": raw_out["events_df"],
        # Lag corrected
        "figure_fr_lagcorr": lag_out["figure_fr"],
        "figure_events_lagcorr": lag_out["figure_events"],
        "compare_df_lagcorr": lag_out["compare_df"],
        "segments_df_lagcorr": lag_out["segments_df"],
        "events_df_lagcorr": lag_out["events_df"],
    }

    if save_outputs:
        if out_dir is None:
            out_dir = cell_folder
        os.makedirs(str(out_dir), exist_ok=True)
        stem = f"{file_tag}_{suffix}"

        def _save_mode(mode_out, mode_key, keep_legacy=False):
            compare_csv_m = os.path.join(out_dir, f"{stem}_{mode_key}_framewise.csv")
            seg_csv_m = os.path.join(out_dir, f"{stem}_{mode_key}_segments.csv")
            evt_csv_m = os.path.join(out_dir, f"{stem}_{mode_key}_events_in_diff_areas.csv")
            fig1_html_m = os.path.join(out_dir, f"{stem}_{mode_key}_fr_compare.html")
            fig1_svg_m = os.path.join(out_dir, f"{stem}_{mode_key}_fr_compare.svg")
            fig2_html_m = os.path.join(out_dir, f"{stem}_{mode_key}_event_size_diff_areas.html")
            fig2_svg_m = os.path.join(out_dir, f"{stem}_{mode_key}_event_size_diff_areas.svg")
            mode_out["compare_df"].to_csv(compare_csv_m, index=False)
            mode_out["segments_df"].to_csv(seg_csv_m, index=False)
            mode_out["events_df"].to_csv(evt_csv_m, index=False)
            mode_out["figure_fr"].write_html(fig1_html_m)
            mode_out["figure_events"].write_html(fig2_html_m)
            try:
                mode_out["figure_fr"].write_image(fig1_svg_m)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(fig1_svg_m)}): {e}")
            try:
                mode_out["figure_events"].write_image(fig2_svg_m)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(fig2_svg_m)}): {e}")
            outputs.update(
                {
                    f"compare_csv_{mode_key}": compare_csv_m,
                    f"segments_csv_{mode_key}": seg_csv_m,
                    f"events_csv_{mode_key}": evt_csv_m,
                    f"figure_fr_html_{mode_key}": fig1_html_m,
                    f"figure_fr_svg_{mode_key}": fig1_svg_m,
                    f"figure_events_html_{mode_key}": fig2_html_m,
                    f"figure_events_svg_{mode_key}": fig2_svg_m,
                }
            )
            if keep_legacy:
                compare_csv = os.path.join(out_dir, f"{stem}_framewise.csv")
                seg_csv = os.path.join(out_dir, f"{stem}_segments.csv")
                evt_csv = os.path.join(out_dir, f"{stem}_events_in_diff_areas.csv")
                fig1_html = os.path.join(out_dir, f"{stem}_fr_compare.html")
                fig1_svg = os.path.join(out_dir, f"{stem}_fr_compare.svg")
                fig2_html = os.path.join(out_dir, f"{stem}_event_size_diff_areas.html")
                fig2_svg = os.path.join(out_dir, f"{stem}_event_size_diff_areas.svg")
                mode_out["compare_df"].to_csv(compare_csv, index=False)
                mode_out["segments_df"].to_csv(seg_csv, index=False)
                mode_out["events_df"].to_csv(evt_csv, index=False)
                mode_out["figure_fr"].write_html(fig1_html)
                mode_out["figure_events"].write_html(fig2_html)
                try:
                    mode_out["figure_fr"].write_image(fig1_svg)
                except Exception as e:
                    print(f"[WARN] SVG export failed ({os.path.basename(fig1_svg)}): {e}")
                try:
                    mode_out["figure_events"].write_image(fig2_svg)
                except Exception as e:
                    print(f"[WARN] SVG export failed ({os.path.basename(fig2_svg)}): {e}")
                outputs.update(
                    {
                        "compare_csv": compare_csv,
                        "segments_csv": seg_csv,
                        "events_csv": evt_csv,
                        "figure_fr_html": fig1_html,
                        "figure_fr_svg": fig1_svg,
                        "figure_events_html": fig2_html,
                        "figure_events_svg": fig2_svg,
                    }
                )

        _save_mode(raw_out, "nolag", keep_legacy=True)
        _save_mode(lag_out, "lagcorr", keep_legacy=False)
        print(
            f"[OK] Saved suite2p-vs-realFR outputs for {os.path.basename(cell_folder)} to: {out_dir} | "
            f"lag={lag_seconds:.4f}s ({int(lag_frames)} frames)"
        )

    return outputs


def _area_group_from_kind(area_kind):
    ak = str(area_kind).strip().lower()
    if ak in ("missed", "underestimated", "overestimated", "both"):
        return ak
    return "correct"


def _compute_binned_means(x, y, nbins=12):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.array([]), np.array([]), np.array([])
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_max <= x_min):
        return np.array([]), np.array([]), np.array([])
    nb = int(max(3, nbins))
    edges = np.linspace(x_min, x_max, nb + 1)
    bx = []
    by = []
    bn = []
    for i in range(nb):
        lo = edges[i]
        hi = edges[i + 1]
        if i < (nb - 1):
            mm = (x >= lo) & (x < hi)
        else:
            mm = (x >= lo) & (x <= hi)
        n = int(np.sum(mm))
        if n <= 0:
            continue
        bx.append(float(np.nanmean(x[mm])))
        by.append(float(np.nanmean(y[mm])))
        bn.append(n)
    return np.asarray(bx, float), np.asarray(by, float), np.asarray(bn, int)


def _shift_with_nan(x, lag_frames):
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    y = np.full(n, np.nan, dtype=float)
    lag = int(lag_frames)
    if n <= 0:
        return y
    if lag == 0:
        y[:] = x
        return y
    if lag > 0:
        # positive lag: shift prediction later in time
        if lag < n:
            y[lag:] = x[: n - lag]
        return y
    k = -lag
    # negative lag: shift prediction earlier in time
    if k < n:
        y[: n - k] = x[k:]
    return y


def _best_lag_by_corr(real, pred, fs_hz, max_lag_s=2.0):
    real = np.asarray(real, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    n = int(min(real.size, pred.size))
    if n < 5 or (not np.isfinite(fs_hz)) or fs_hz <= 0:
        return 0, np.nan
    real = real[:n]
    pred = pred[:n]
    max_lag = int(max(0, round(float(max_lag_s) * float(fs_hz))))
    if max_lag <= 0:
        m = np.isfinite(real) & np.isfinite(pred)
        if np.sum(m) < 3:
            return 0, np.nan
        sx = float(np.nanstd(real[m]))
        sy = float(np.nanstd(pred[m]))
        if sx <= 0 or sy <= 0:
            return 0, np.nan
        return 0, float(np.corrcoef(real[m], pred[m])[0, 1])
    best_lag = 0
    best_r = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        p = _shift_with_nan(pred, lag)
        m = np.isfinite(real) & np.isfinite(p)
        if np.sum(m) < 3:
            continue
        sx = float(np.nanstd(real[m]))
        sy = float(np.nanstd(p[m]))
        if sx <= 0 or sy <= 0:
            continue
        rr = float(np.corrcoef(real[m], p[m])[0, 1])
        if np.isfinite(rr) and rr > best_r:
            best_r = rr
            best_lag = int(lag)
    if not np.isfinite(best_r):
        return 0, np.nan
    return int(best_lag), float(best_r)


def run_suite2p_vs_realfr_on_db(
    db_path=None,
    max_cells=None,
    out_dir=None,
    save_cell_outputs_in_place=True,
    all_cells_ncols=3,
    **kwargs,
):
    if db_path is None:
        db_path = DB_PATH
    if (db_path is None) or (not os.path.isfile(str(db_path))):
        raise FileNotFoundError(f"Missing DB CSV: {db_path}")
    db = pd.read_csv(db_path)
    if max_cells is not None:
        db = db.iloc[: int(max_cells)].copy()
    if "Link" not in db.columns:
        raise KeyError("DB must contain 'Link' column")

    rows = []
    panel_rows_by_mode = {"nolag": [], "lagcorr": []}
    all_event_rows_by_mode = {"nolag": [], "lagcorr": []}
    diag_rows = []
    for i, r in db.iterrows():
        cell_folder = str(r["Link"])
        if not os.path.isdir(cell_folder):
            print(f"[SKIP] missing folder: {cell_folder}")
            continue
        try:
            res = analyze_suite2p_vs_real_fr(
                cell_folder=cell_folder,
                cal_sr_hz=_safe_float(r.get("CALsr", FS_CA_HZ_FALLBACK), default=FS_CA_HZ_FALLBACK),
                out_dir=cell_folder if bool(save_cell_outputs_in_place) else (out_dir if out_dir is not None else cell_folder),
                save_outputs=True,
                **kwargs,
            )

            for mode_key in ("nolag", "lagcorr"):
                cmp_key = "compare_df" if mode_key == "nolag" else "compare_df_lagcorr"
                ev_key = "events_df" if mode_key == "nolag" else "events_df_lagcorr"
                r_key = "pearson_r" if mode_key == "nolag" else "pearson_r_lagcorr"
                cmp = res.get(cmp_key, pd.DataFrame())
                if len(cmp) > 0:
                    panel_rows_by_mode[mode_key].append(
                        {
                            "cell_folder": cell_folder,
                            "cell_name": os.path.basename(cell_folder),
                            "pkl_suffix": str(res.get("pkl_suffix", "main")),
                            "pearson_r": _safe_float(res.get(r_key, np.nan), default=np.nan),
                            "time_s": np.asarray(cmp["time_s"], dtype=float).ravel(),
                            "real_fr_hz_smooth": np.asarray(cmp["real_fr_hz_smooth"], dtype=float).ravel(),
                            "predicted_fr_hz": np.asarray(cmp["predicted_fr_hz"], dtype=float).ravel(),
                        }
                    )
                evdf = res.get(ev_key, pd.DataFrame())
                if len(evdf) > 0:
                    evx = evdf.copy()
                    evx["cell_folder"] = str(cell_folder)
                    evx["cell_name"] = os.path.basename(str(cell_folder))
                    evx["pkl_suffix"] = str(res.get("pkl_suffix", "main"))
                    evx["analysis_mode"] = mode_key
                    all_event_rows_by_mode[mode_key].append(evx)

            cmp_nolag = res.get("compare_df", pd.DataFrame())
            if len(cmp_nolag) > 0:
                real_raw_arr = pd.to_numeric(cmp_nolag.get("real_fr_hz_raw", np.nan), errors="coerce").to_numpy(dtype=float)
                real_sm_arr = pd.to_numeric(cmp_nolag.get("real_fr_hz_smooth", np.nan), errors="coerce").to_numpy(dtype=float)
                pred_raw_arr = pd.to_numeric(cmp_nolag.get("predicted_fr_hz_raw", np.nan), errors="coerce").to_numpy(dtype=float)
                pred_cal_arr = pd.to_numeric(cmp_nolag.get("predicted_fr_hz_calibrated", np.nan), errors="coerce").to_numpy(dtype=float)
                cal_sr_hz_val = _safe_float(res.get("cal_sr_hz", np.nan), default=np.nan)
                alpha_global = _safe_float(res.get("pred_scale_alpha", np.nan), default=np.nan)
                alpha_active = _fit_nonneg_scale_active(pred_raw_arr, real_sm_arr, active_q=80.0, clip_q=99.5)
                pred_cal_active_arr = pred_raw_arr * float(alpha_active)
                sigma_s = _safe_float(res.get("pred_smooth_sigma_s", np.nan), default=np.nan)
                if np.isfinite(sigma_s) and np.isfinite(cal_sr_hz_val) and sigma_s > 0 and cal_sr_hz_val > 0:
                    sigma_samples = float(sigma_s) * float(cal_sr_hz_val)
                    pred_active_sm_arr = gaussian_filter1d(pred_cal_active_arr.astype(float), sigma=sigma_samples, mode="nearest")
                else:
                    pred_active_sm_arr = pred_cal_active_arr.copy()
                valid_a = np.isfinite(real_sm_arr) & np.isfinite(pred_active_sm_arr)
                if np.sum(valid_a) >= 3 and np.nanstd(real_sm_arr[valid_a]) > 0 and np.nanstd(pred_active_sm_arr[valid_a]) > 0:
                    r_active = float(np.corrcoef(real_sm_arr[valid_a], pred_active_sm_arr[valid_a])[0, 1])
                else:
                    r_active = np.nan
                if np.isfinite(cal_sr_hz_val) and cal_sr_hz_val > 0:
                    total_real = float(np.nansum(real_raw_arr) / cal_sr_hz_val)
                    total_pred_raw = float(np.nansum(pred_raw_arr) / cal_sr_hz_val)
                    total_pred_cal = float(np.nansum(pred_cal_arr) / cal_sr_hz_val)
                    total_pred_active = float(np.nansum(pred_cal_active_arr) / cal_sr_hz_val)
                else:
                    total_real = np.nan
                    total_pred_raw = np.nan
                    total_pred_cal = np.nan
                    total_pred_active = np.nan
                diag_rows.append(
                    {
                        "cell_folder": cell_folder,
                        "cell_name": os.path.basename(cell_folder),
                        "pkl_suffix": str(res.get("pkl_suffix", "main")),
                        "cal_sr_hz": cal_sr_hz_val,
                        "roi_match_corr": _safe_float(res.get("suite2p_roi_match_corr_to_calTrace", np.nan), default=np.nan),
                        "alpha_global": alpha_global,
                        "alpha_active": _safe_float(alpha_active, default=np.nan),
                        "pearson_r_nolag": _safe_float(res.get("pearson_r", np.nan), default=np.nan),
                        "pearson_r_lagcorr": _safe_float(res.get("pearson_r_lagcorr", np.nan), default=np.nan),
                        "pearson_r_active_alpha": _safe_float(r_active, default=np.nan),
                        "total_real_spikes_equiv": total_real,
                        "total_pred_raw_spikes_equiv": total_pred_raw,
                        "total_pred_cal_spikes_equiv": total_pred_cal,
                        "total_pred_active_spikes_equiv": total_pred_active,
                        "real_sm_arr": real_sm_arr,
                        "pred_raw_arr": pred_raw_arr,
                        "pred_cal_arr": pred_cal_arr,
                    }
                )

            rows.append(
                {
                    "cell_folder": cell_folder,
                    "pkl_path": res["pkl_path"],
                    "pkl_suffix": res["pkl_suffix"],
                    "cal_sr_hz": res["cal_sr_hz"],
                    "pearson_r": res["pearson_r"],
                    "pearson_r_lagcorr": res.get("pearson_r_lagcorr", np.nan),
                    "lag_frames": res.get("lag_frames", np.nan),
                    "lag_seconds": res.get("lag_seconds", np.nan),
                    "n_missed_segments": res["n_missed_segments"],
                    "n_over_segments": res["n_over_segments"],
                    "n_missed_segments_lagcorr": res.get("n_missed_segments_lagcorr", np.nan),
                    "n_over_segments_lagcorr": res.get("n_over_segments_lagcorr", np.nan),
                    "suite2p_roi_idx": res["suite2p_roi_idx"],
                    "suite2p_roi_match_corr_to_calTrace": res["suite2p_roi_match_corr_to_calTrace"],
                }
            )
        except Exception as e:
            print(f"[ERROR] {cell_folder}: {e}")

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        roi_corr_vals = pd.to_numeric(out_df.get("suite2p_roi_match_corr_to_calTrace", np.nan), errors="coerce")
        r_nolag_vals = pd.to_numeric(out_df.get("pearson_r", np.nan), errors="coerce")
        r_lag_vals = pd.to_numeric(out_df.get("pearson_r_lagcorr", np.nan), errors="coerce")
        low_roi_mask = np.isfinite(roi_corr_vals) & (roi_corr_vals < 0.4)
        low_predreal_nolag_mask = np.isfinite(r_nolag_vals) & (r_nolag_vals < 0.4)
        low_predreal_lag_mask = np.isfinite(r_lag_vals) & (r_lag_vals < 0.4)
        low_any_mask = low_roi_mask | low_predreal_nolag_mask | low_predreal_lag_mask

        low_corr_df = out_df.loc[low_roi_mask].copy()
        if len(low_corr_df) > 0:
            low_corr_df = low_corr_df.sort_values(
                by=["suite2p_roi_match_corr_to_calTrace", "cell_folder", "pkl_suffix"],
                ascending=[True, True, True],
            )
            print("\n[INFO] Traces with suite2p ROI match correlation < 0.4:")
            for _, rr in low_corr_df.iterrows():
                print(
                    "  - "
                    f"{rr.get('cell_folder', '')} | suffix={rr.get('pkl_suffix', 'main')} | "
                    f"roi_idx={rr.get('suite2p_roi_idx', np.nan)} | "
                    f"roi_match_corr={_safe_float(rr.get('suite2p_roi_match_corr_to_calTrace', np.nan), default=np.nan):.4f} | "
                    f"r_nolag={_safe_float(rr.get('pearson_r', np.nan), default=np.nan):.4f} | "
                    f"r_lagcorr={_safe_float(rr.get('pearson_r_lagcorr', np.nan), default=np.nan):.4f}"
                )
        else:
            print("\n[INFO] No traces with suite2p ROI match correlation < 0.4.")

        low_pred_df = out_df.loc[low_predreal_nolag_mask | low_predreal_lag_mask].copy()
        if len(low_pred_df) > 0:
            low_pred_df = low_pred_df.sort_values(
                by=["pearson_r_lagcorr", "pearson_r", "cell_folder", "pkl_suffix"],
                ascending=[True, True, True, True],
            )
            print("\n[INFO] Traces with predicted-vs-real FR correlation < 0.4:")
            for _, rr in low_pred_df.iterrows():
                print(
                    "  - "
                    f"{rr.get('cell_folder', '')} | suffix={rr.get('pkl_suffix', 'main')} | "
                    f"r_nolag={_safe_float(rr.get('pearson_r', np.nan), default=np.nan):.4f} | "
                    f"r_lagcorr={_safe_float(rr.get('pearson_r_lagcorr', np.nan), default=np.nan):.4f} | "
                    f"roi_match_corr={_safe_float(rr.get('suite2p_roi_match_corr_to_calTrace', np.nan), default=np.nan):.4f}"
                )
        else:
            print("\n[INFO] No traces with predicted-vs-real FR correlation < 0.4.")

    if out_dir is not None and len(out_df) > 0:
        os.makedirs(str(out_dir), exist_ok=True)
        out_csv = os.path.join(str(out_dir), "suite2p_vs_realfr_summary.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"[OK] Saved summary: {out_csv}")
        roi_corr_vals = pd.to_numeric(out_df.get("suite2p_roi_match_corr_to_calTrace", np.nan), errors="coerce")
        r_nolag_vals = pd.to_numeric(out_df.get("pearson_r", np.nan), errors="coerce")
        r_lag_vals = pd.to_numeric(out_df.get("pearson_r_lagcorr", np.nan), errors="coerce")
        low_roi_mask = np.isfinite(roi_corr_vals) & (roi_corr_vals < 0.4)
        low_predreal_nolag_mask = np.isfinite(r_nolag_vals) & (r_nolag_vals < 0.4)
        low_predreal_lag_mask = np.isfinite(r_lag_vals) & (r_lag_vals < 0.4)
        low_any_mask = low_roi_mask | low_predreal_nolag_mask | low_predreal_lag_mask

        low_corr_df = out_df.loc[low_roi_mask].copy()
        low_corr_csv = os.path.join(str(out_dir), "suite2p_vs_realfr_low_roi_match_corr_lt_0p4.csv")
        low_corr_df.to_csv(low_corr_csv, index=False)
        print(f"[OK] Saved low ROI-match correlation list: {low_corr_csv}")

        low_pred_df = out_df.loc[low_predreal_nolag_mask | low_predreal_lag_mask].copy()
        low_pred_csv = os.path.join(str(out_dir), "suite2p_vs_realfr_low_pred_real_corr_lt_0p4.csv")
        low_pred_df.to_csv(low_pred_csv, index=False)
        print(f"[OK] Saved low predicted-vs-real correlation list: {low_pred_csv}")

        attention_df = out_df.loc[low_any_mask].copy()
        if len(attention_df) > 0:
            attention_df["low_roi_match_corr_lt_0p4"] = low_roi_mask[low_any_mask].astype(bool)
            attention_df["low_pred_real_corr_nolag_lt_0p4"] = low_predreal_nolag_mask[low_any_mask].astype(bool)
            attention_df["low_pred_real_corr_lagcorr_lt_0p4"] = low_predreal_lag_mask[low_any_mask].astype(bool)
        else:
            attention_df["low_roi_match_corr_lt_0p4"] = []
            attention_df["low_pred_real_corr_nolag_lt_0p4"] = []
            attention_df["low_pred_real_corr_lagcorr_lt_0p4"] = []
        attention_csv = os.path.join(str(out_dir), "suite2p_vs_realfr_attention_list_lt_0p4.csv")
        attention_df.to_csv(attention_csv, index=False)
        print(f"[OK] Saved combined attention list (<0.4): {attention_csv}")

        if len(diag_rows) > 0:
            diag_df = pd.DataFrame(diag_rows)
            diag_csv = os.path.join(str(out_dir), "suite2p_vs_realfr_scaling_diagnostics_per_trace.csv")
            diag_df.drop(columns=["real_sm_arr", "pred_raw_arr"], errors="ignore").to_csv(diag_csv, index=False)
            print(f"[OK] Saved scaling diagnostics table: {diag_csv}")

            n_panels = len(diag_rows)
            ncols = int(max(1, all_cells_ncols))
            nrows = int(np.ceil(n_panels / ncols))
            subplot_titles = [f"{d['cell_name']} ({d['pkl_suffix']})" for d in diag_rows]
            subplot_titles += [""] * int(max(0, nrows * ncols - len(subplot_titles)))

            # 1) Totals comparison per trace
            fig_tot = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=min(0.08, max(0.02, 0.28 / max(1, nrows))),
            )
            for i, d in enumerate(diag_rows):
                rr = i // ncols + 1
                cc = i % ncols + 1
                show_legend = i == 0
                x = ["real_total", "pred_raw_total", "pred_alpha_total", "pred_active_total"]
                y = [
                    _safe_float(d.get("total_real_spikes_equiv", np.nan), default=np.nan),
                    _safe_float(d.get("total_pred_raw_spikes_equiv", np.nan), default=np.nan),
                    _safe_float(d.get("total_pred_cal_spikes_equiv", np.nan), default=np.nan),
                    _safe_float(d.get("total_pred_active_spikes_equiv", np.nan), default=np.nan),
                ]
                fig_tot.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        marker_color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"],
                        showlegend=show_legend,
                        name="totals",
                    ),
                    row=rr,
                    col=cc,
                )
                fig_tot.update_yaxes(title_text="spike-equivalent count", row=rr, col=cc)
            fig_tot.update_layout(
                template="simple_white",
                width=max(1300, 510 * ncols),
                height=max(700, 320 * nrows),
                title="Diagnostics #1: total spike-equivalent counts per trace",
                barmode="group",
            )
            tot_html = os.path.join(str(out_dir), "suite2p_vs_realfr_diag1_totals_per_trace.html")
            tot_svg = os.path.join(str(out_dir), "suite2p_vs_realfr_diag1_totals_per_trace.svg")
            fig_tot.write_html(tot_html)
            try:
                fig_tot.write_image(tot_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(tot_svg)}): {e}")
            print(f"[OK] Saved diagnostics figure #1: {tot_html}")

            # 2) ROI match correlation per trace
            fig_roi = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=min(0.08, max(0.02, 0.28 / max(1, nrows))),
            )
            for i, d in enumerate(diag_rows):
                rr = i // ncols + 1
                cc = i % ncols + 1
                show_legend = i == 0
                roi = _safe_float(d.get("roi_match_corr", np.nan), default=np.nan)
                fig_roi.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[roi],
                        mode="markers",
                        marker=dict(size=11, color="#1f77b4"),
                        name="ROI match corr",
                        showlegend=show_legend,
                    ),
                    row=rr,
                    col=cc,
                )
                fig_roi.add_trace(
                    go.Scatter(
                        x=[-0.5, 0.5],
                        y=[0.4, 0.4],
                        mode="lines",
                        line=dict(color="crimson", dash="dash", width=1.4),
                        name="threshold 0.4",
                        showlegend=show_legend,
                    ),
                    row=rr,
                    col=cc,
                )
                fig_roi.update_xaxes(showticklabels=False, range=[-0.6, 0.6], row=rr, col=cc)
                fig_roi.update_yaxes(title_text="corr", row=rr, col=cc)
            fig_roi.update_layout(
                template="simple_white",
                width=max(1300, 510 * ncols),
                height=max(700, 320 * nrows),
                title="Diagnostics #2: ROI match correlation to calTrace",
            )
            roi_html = os.path.join(str(out_dir), "suite2p_vs_realfr_diag2_roi_match_corr_per_trace.html")
            roi_svg = os.path.join(str(out_dir), "suite2p_vs_realfr_diag2_roi_match_corr_per_trace.svg")
            fig_roi.write_html(roi_html)
            try:
                fig_roi.write_image(roi_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(roi_svg)}): {e}")
            print(f"[OK] Saved diagnostics figure #2: {roi_html}")

            # 3) Active-frame alpha diagnostic
            fig_alpha = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=min(0.08, max(0.02, 0.28 / max(1, nrows))),
            )
            for i, d in enumerate(diag_rows):
                rr = i // ncols + 1
                cc = i % ncols + 1
                show_legend = i == 0
                x = [0, 1, 2, 3]
                y = [
                    _safe_float(d.get("alpha_global", np.nan), default=np.nan),
                    _safe_float(d.get("alpha_active", np.nan), default=np.nan),
                    _safe_float(d.get("pearson_r_nolag", np.nan), default=np.nan),
                    _safe_float(d.get("pearson_r_active_alpha", np.nan), default=np.nan),
                ]
                fig_alpha.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers+lines",
                        marker=dict(size=8, color="#9467bd"),
                        line=dict(width=2.0, color="#9467bd"),
                        name="alpha/r diagnostics",
                        showlegend=show_legend,
                    ),
                    row=rr,
                    col=cc,
                )
                fig_alpha.update_xaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["alpha_global", "alpha_active", "r_global", "r_active"],
                    row=rr,
                    col=cc,
                )
                fig_alpha.update_yaxes(title_text="value", row=rr, col=cc)
            fig_alpha.update_layout(
                template="simple_white",
                width=max(1300, 510 * ncols),
                height=max(700, 320 * nrows),
                title="Diagnostics #3: active-frame alpha and resulting correlation",
            )
            alpha_html = os.path.join(str(out_dir), "suite2p_vs_realfr_diag3_active_alpha_per_trace.html")
            alpha_svg = os.path.join(str(out_dir), "suite2p_vs_realfr_diag3_active_alpha_per_trace.svg")
            fig_alpha.write_html(alpha_html)
            try:
                fig_alpha.write_image(alpha_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(alpha_svg)}): {e}")
            print(f"[OK] Saved diagnostics figure #3: {alpha_html}")

            # 4) pred_calibrated vs real_smooth scatter + reference/fitted lines
            fig_sc = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=min(0.08, max(0.02, 0.28 / max(1, nrows))),
            )
            for i, d in enumerate(diag_rows):
                rr = i // ncols + 1
                cc = i % ncols + 1
                show_legend = i == 0
                x = np.asarray(d.get("pred_cal_arr", []), dtype=float).ravel()
                y = np.asarray(d.get("real_sm_arr", []), dtype=float).ravel()
                m = np.isfinite(x) & np.isfinite(y)
                x = x[m]
                y = y[m]
                if x.size > 0:
                    fig_sc.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            marker=dict(size=3, color="rgba(31,119,180,0.35)"),
                            name="points",
                            showlegend=show_legend,
                        ),
                        row=rr,
                        col=cc,
                    )
                    xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 120)
                    a_g = _safe_float(d.get("alpha_global", np.nan), default=np.nan)
                    a_a = _safe_float(d.get("alpha_active", np.nan), default=np.nan)
                    # global-alpha corrected prediction should align to identity line
                    fig_sc.add_trace(
                        go.Scatter(x=xx, y=xx, mode="lines", line=dict(color="#2ca02c", width=2.0), name="y=x (global corrected)", showlegend=show_legend),
                        row=rr,
                        col=cc,
                    )
                    if np.isfinite(a_a):
                        slope_active_on_cal = float(a_a / a_g) if (np.isfinite(a_g) and abs(a_g) > 1e-12) else np.nan
                        if not np.isfinite(slope_active_on_cal):
                            slope_active_on_cal = 1.0
                        fig_sc.add_trace(
                            go.Scatter(x=xx, y=slope_active_on_cal * xx, mode="lines", line=dict(color="#ff7f0e", width=2.0, dash="dash"), name="active-corrected line", showlegend=show_legend),
                            row=rr,
                            col=cc,
                        )
                fig_sc.update_xaxes(title_text="pred_calibrated", row=rr, col=cc)
                fig_sc.update_yaxes(title_text="real_fr_smooth", row=rr, col=cc)
            fig_sc.update_layout(
                template="simple_white",
                width=max(1300, 510 * ncols),
                height=max(700, 320 * nrows),
                title="Diagnostics #4: pred_calibrated vs real_fr_smooth per trace (with reference/fitted lines)",
            )
            sc_html = os.path.join(str(out_dir), "suite2p_vs_realfr_diag4_predraw_vs_real_scatter_per_trace.html")
            sc_svg = os.path.join(str(out_dir), "suite2p_vs_realfr_diag4_predraw_vs_real_scatter_per_trace.svg")
            fig_sc.write_html(sc_html)
            try:
                fig_sc.write_image(sc_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(sc_svg)}): {e}")
            print(f"[OK] Saved diagnostics figure #4: {sc_html}")

        def _save_corr_scatter(mode_key, col_name):
            x_idx = np.arange(len(out_df), dtype=int)
            y_r = pd.to_numeric(out_df[col_name], errors="coerce").to_numpy(dtype=float)
            text_lbl = []
            for _, rr in out_df.iterrows():
                lag_txt = rr.get("lag_seconds", np.nan)
                if np.isfinite(lag_txt):
                    lag_txt = f"{float(lag_txt):.4f}s"
                else:
                    lag_txt = "nan"
                text_lbl.append(
                    f"{os.path.basename(str(rr.get('cell_folder', '')))} | {rr.get('pkl_suffix', 'main')} | lag={lag_txt}"
                )
            fig_corr = go.Figure()
            fig_corr.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=y_r,
                    mode="markers",
                    marker=dict(size=8, color="#4C78A8", line=dict(width=0.5, color="white")),
                    text=text_lbl,
                    hovertemplate="%{text}<br>Pearson r=%{y:.4f}<extra></extra>",
                    name=f"Per-trace correlation ({mode_key})",
                )
            )
            fig_corr.add_hline(y=0.0, line_width=1.0, line_dash="dash", line_color="#777777")
            fig_corr.update_layout(
                template="simple_white",
                width=1150,
                height=580,
                title=f"Suite2p vs real FR correlations (all traces) | {mode_key}",
                xaxis_title="Trace index",
                yaxis_title="Pearson r",
            )
            corr_html = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_correlations_scatter_{mode_key}.html")
            corr_svg = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_correlations_scatter_{mode_key}.svg")
            fig_corr.write_html(corr_html)
            try:
                fig_corr.write_image(corr_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(corr_svg)}): {e}")
            print(f"[OK] Saved correlation scatter ({mode_key}): {corr_html}")

        def _save_all_cells_panel(mode_key):
            panel_rows = panel_rows_by_mode[mode_key]
            if len(panel_rows) == 0:
                return
            n_panels = len(panel_rows)
            ncols = int(max(1, all_cells_ncols))
            nrows = int(np.ceil(n_panels / ncols))
            subplot_titles = [
                f"{p['cell_name']} ({p['pkl_suffix']}) | r={p['pearson_r']:.3f}" if np.isfinite(p["pearson_r"]) else f"{p['cell_name']} ({p['pkl_suffix']})"
                for p in panel_rows
            ]
            subplot_titles += [""] * int(max(0, nrows * ncols - len(subplot_titles)))
            fig_all = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.06,
                vertical_spacing=0.08,
            )
            for i, p in enumerate(panel_rows):
                rr = i // ncols + 1
                cc = i % ncols + 1
                show_legend = i == 0
                fig_all.add_trace(
                    go.Scatter(
                        x=p["time_s"],
                        y=p["real_fr_hz_smooth"],
                        mode="lines",
                        line=dict(color="#4C78A8", width=1.8),
                        name="Real FR (smooth)",
                        showlegend=show_legend,
                    ),
                    row=rr,
                    col=cc,
                )
                fig_all.add_trace(
                    go.Scatter(
                        x=p["time_s"],
                        y=p["predicted_fr_hz"],
                        mode="lines",
                        line=dict(color="#F58518", width=1.6),
                        name=f"Suite2p predicted FR ({mode_key})",
                        showlegend=show_legend,
                    ),
                    row=rr,
                    col=cc,
                )
                fig_all.update_xaxes(title_text="Time (s)", row=rr, col=cc)
                fig_all.update_yaxes(title_text="FR (Hz)", row=rr, col=cc)
            fig_all.update_layout(
                template="simple_white",
                width=max(1200, 520 * ncols),
                height=max(650, 280 * nrows),
                title=f"Suite2p predicted FR vs real FR (all cells) | {mode_key}",
                legend=dict(orientation="h"),
            )
            all_html = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_cells_fr_compare_{mode_key}.html")
            all_svg = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_cells_fr_compare_{mode_key}.svg")
            all_pdf = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_cells_fr_compare_{mode_key}.pdf")
            fig_all.write_html(all_html)
            try:
                fig_all.write_image(all_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(all_svg)}): {e}")
            try:
                fig_all.write_image(all_pdf)
            except Exception as e:
                print(f"[WARN] PDF export failed ({os.path.basename(all_pdf)}): {e}")
            print(f"[OK] Saved all-cells FR comparison figure ({mode_key}): {all_html}")

        def _save_event_population(mode_key):
            all_event_rows = all_event_rows_by_mode[mode_key]
            if len(all_event_rows) == 0:
                return
            evt_all = pd.concat(all_event_rows, ignore_index=True)
            if "area_kind" not in evt_all.columns:
                evt_all["area_kind"] = "none"
            evt_all["area_group"] = evt_all["area_kind"].apply(_area_group_from_kind)
            evt_all_csv = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_events_{mode_key}.csv")
            evt_all.to_csv(evt_all_csv, index=False)
            print(f"[OK] Saved all-events table ({mode_key}): {evt_all_csv}")

            fig_sum = make_subplots(
                rows=2,
                cols=4,
                specs=[
                    [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                    [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
                ],
                subplot_titles=(
                    "Missed events: size distribution",
                    "Underestimated events: size distribution",
                    "Overestimated events: size distribution",
                    "Correctly estimated events: size distribution",
                    "Missed: simple vs complex",
                    "Underestimated: simple vs complex",
                    "Overestimated: simple vs complex",
                    "All events: missed vs under vs over vs correct",
                ),
                horizontal_spacing=0.06,
                vertical_spacing=0.14,
            )
            clr = {"simple": "#2ca02c", "complex": "#e377c2"}
            area_cols = [("missed", 1), ("underestimated", 2), ("overestimated", 3), ("correct", 4)]
            for area_name, col in area_cols:
                ss = evt_all[evt_all["area_group"] == area_name].copy()
                if len(ss) == 0:
                    continue
                for et in ("simple", "complex"):
                    s2 = ss[ss["event_type"].astype(str).str.lower() == et]
                    if len(s2) == 0:
                        continue
                    cnt = s2.groupby("n_spikes_in_event").size().reset_index(name="count")
                    fig_sum.add_trace(
                        go.Bar(
                            x=cnt["n_spikes_in_event"],
                            y=cnt["count"],
                            marker_color=clr.get(et, "#666666"),
                            name=f"{et}",
                            legendgroup=f"type_{et}",
                            showlegend=(col == 1),
                        ),
                        row=1,
                        col=col,
                    )
                fig_sum.update_xaxes(title_text="# spikes in event", row=1, col=col)
                fig_sum.update_yaxes(title_text="# events", row=1, col=col)

            miss_like = evt_all[evt_all["area_group"] == "missed"]
            under_like = evt_all[evt_all["area_group"] == "underestimated"]
            over_like = evt_all[evt_all["area_group"] == "overestimated"]
            corr_like = evt_all[evt_all["area_group"] == "correct"]
            miss_simple = int(np.sum(miss_like["event_type"].astype(str).str.lower() == "simple"))
            miss_complex = int(np.sum(miss_like["event_type"].astype(str).str.lower() == "complex"))
            under_simple = int(np.sum(under_like["event_type"].astype(str).str.lower() == "simple"))
            under_complex = int(np.sum(under_like["event_type"].astype(str).str.lower() == "complex"))
            over_simple = int(np.sum(over_like["event_type"].astype(str).str.lower() == "simple"))
            over_complex = int(np.sum(over_like["event_type"].astype(str).str.lower() == "complex"))
            fig_sum.add_trace(
                go.Pie(labels=["simple", "complex"], values=[miss_simple, miss_complex], marker=dict(colors=["#2ca02c", "#e377c2"]), textinfo="label+percent+value", sort=False, showlegend=False),
                row=2,
                col=1,
            )
            fig_sum.add_trace(
                go.Pie(labels=["simple", "complex"], values=[under_simple, under_complex], marker=dict(colors=["#2ca02c", "#e377c2"]), textinfo="label+percent+value", sort=False, showlegend=False),
                row=2,
                col=2,
            )
            fig_sum.add_trace(
                go.Pie(labels=["simple", "complex"], values=[over_simple, over_complex], marker=dict(colors=["#2ca02c", "#e377c2"]), textinfo="label+percent+value", sort=False, showlegend=False),
                row=2,
                col=3,
            )
            fig_sum.add_trace(
                go.Pie(
                    labels=["missed", "underestimated", "overestimated", "correct"],
                    values=[len(miss_like), len(under_like), len(over_like), len(corr_like)],
                    marker=dict(colors=["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]),
                    textinfo="label+percent+value",
                    sort=False,
                    showlegend=False,
                ),
                row=2,
                col=4,
            )
            fig_sum.update_layout(
                template="simple_white",
                width=2400,
                height=1050,
                barmode="group",
                title=f"Suite2p vs real FR: event-mismatch summary (all traces) | {mode_key}",
                legend=dict(orientation="h"),
            )
            sum_html = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_events_summary_with_pies_{mode_key}.html")
            sum_svg = os.path.join(str(out_dir), f"suite2p_vs_realfr_all_events_summary_with_pies_{mode_key}.svg")
            fig_sum.write_html(sum_html)
            try:
                fig_sum.write_image(sum_svg)
            except Exception as e:
                print(f"[WARN] SVG export failed ({os.path.basename(sum_svg)}): {e}")
            print(f"[OK] Saved event summary + pies ({mode_key}): {sum_html}")

            evt_bad = evt_all[evt_all["area_group"].isin(["missed", "underestimated", "overestimated"])].copy()
            evt_bad["residual_z_at_first_spike"] = pd.to_numeric(evt_bad["residual_z_at_first_spike"], errors="coerce")
            evt_bad = evt_bad[np.isfinite(evt_bad["residual_z_at_first_spike"])].reset_index(drop=True)
            if len(evt_bad) > 0:
                fig_z = go.Figure()
                color_map = {"missed": "#d62728", "underestimated": "#ff7f0e", "overestimated": "#1f77b4"}
                for g in ("missed", "underestimated", "overestimated"):
                    ss = evt_bad[evt_bad["area_group"] == g].copy()
                    if len(ss) == 0:
                        continue
                    fig_z.add_trace(
                        go.Scatter(
                            x=np.arange(len(ss), dtype=int),
                            y=ss["residual_z_at_first_spike"],
                            mode="markers",
                            marker=dict(size=7, color=color_map[g], line=dict(width=0.5, color="white")),
                            name=g,
                            text=ss["cell_name"].astype(str) + " | " + ss["event_type"].astype(str),
                            hovertemplate="%{text}<br>z diff=%{y:.3f}<extra></extra>",
                        )
                    )
                fig_z.add_hline(y=0.0, line_width=1.0, line_dash="dash", line_color="#777777")
                fig_z.update_layout(
                    template="simple_white",
                    width=1250,
                    height=620,
                    title=f"Event-level difference (z-score): missed vs underestimated vs overestimated | {mode_key}",
                    xaxis_title="Event index (within class)",
                    yaxis_title="Residual z at first spike (real - predicted)",
                    legend=dict(orientation="h"),
                )
                z_html = os.path.join(str(out_dir), f"suite2p_vs_realfr_missed_over_residual_z_scatter_{mode_key}.html")
                z_svg = os.path.join(str(out_dir), f"suite2p_vs_realfr_missed_over_residual_z_scatter_{mode_key}.svg")
                fig_z.write_html(z_html)
                try:
                    fig_z.write_image(z_svg)
                except Exception as e:
                    print(f"[WARN] SVG export failed ({os.path.basename(z_svg)}): {e}")
                print(f"[OK] Saved residual-z scatter ({mode_key}): {z_html}")

            evt_bp = evt_bad.copy()
            evt_bp["real_fr_at_first_spike_hz"] = pd.to_numeric(evt_bp["real_fr_at_first_spike_hz"], errors="coerce")
            evt_bp["pred_fr_at_first_spike_hz"] = pd.to_numeric(evt_bp["pred_fr_at_first_spike_hz"], errors="coerce")
            evt_bp = evt_bp[np.isfinite(evt_bp["real_fr_at_first_spike_hz"]) & np.isfinite(evt_bp["pred_fr_at_first_spike_hz"])].copy()
            if len(evt_bp) > 0:
                fig_bin = go.Figure()
                color_map = {"missed": "#d62728", "underestimated": "#ff7f0e", "overestimated": "#1f77b4"}
                for g in ("missed", "underestimated", "overestimated"):
                    ss = evt_bp[evt_bp["area_group"] == g].copy()
                    if len(ss) == 0:
                        continue
                    bx, by, bn = _compute_binned_means(
                        ss["real_fr_at_first_spike_hz"].to_numpy(dtype=float),
                        ss["pred_fr_at_first_spike_hz"].to_numpy(dtype=float),
                        nbins=12,
                    )
                    if bx.size == 0:
                        continue
                    fig_bin.add_trace(
                        go.Scatter(
                            x=bx,
                            y=by,
                            mode="markers+lines",
                            marker=dict(size=np.clip(5 + 2 * np.sqrt(np.maximum(bn, 1)), 6, 18), color=color_map[g], line=dict(width=0.8, color="white")),
                            line=dict(color=color_map[g], width=2.2),
                            name=f"{g} (binned mean)",
                            hovertemplate="real=%{x:.3f} Hz<br>pred=%{y:.3f} Hz<br>n=%{customdata}<extra></extra>",
                            customdata=bn,
                        )
                    )
                xy = np.concatenate(
                    [
                        evt_bp["real_fr_at_first_spike_hz"].to_numpy(dtype=float),
                        evt_bp["pred_fr_at_first_spike_hz"].to_numpy(dtype=float),
                    ]
                )
                if xy.size > 0 and np.any(np.isfinite(xy)):
                    lo = float(np.nanmin(xy))
                    hi = float(np.nanmax(xy))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        fig_bin.add_trace(
                            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#666666", width=1.2, dash="dash"), name="y=x")
                        )
                fig_bin.update_layout(
                    template="simple_white",
                    width=1050,
                    height=700,
                    title=f"Real FR vs predicted FR (binned means): missed vs underestimated vs overestimated | {mode_key}",
                    xaxis_title="Real FR at first spike (Hz)",
                    yaxis_title="Predicted FR at first spike (Hz)",
                    legend=dict(orientation="h"),
                )
                bin_html = os.path.join(str(out_dir), f"suite2p_vs_realfr_missed_over_real_vs_pred_binned_{mode_key}.html")
                bin_svg = os.path.join(str(out_dir), f"suite2p_vs_realfr_missed_over_real_vs_pred_binned_{mode_key}.svg")
                fig_bin.write_html(bin_html)
                try:
                    fig_bin.write_image(bin_svg)
                except Exception as e:
                    print(f"[WARN] SVG export failed ({os.path.basename(bin_svg)}): {e}")
                print(f"[OK] Saved binned real-vs-pred scatter ({mode_key}): {bin_html}")

        _save_corr_scatter("nolag", "pearson_r")
        _save_corr_scatter("lagcorr", "pearson_r_lagcorr")
        _save_all_cells_panel("nolag")
        _save_all_cells_panel("lagcorr")
        _save_event_population("nolag")
        _save_event_population("lagcorr")

        # legacy names keep nolag versions for compatibility
        try:
            legacy_map = {
                "suite2p_vs_realfr_all_correlations_scatter.html": "suite2p_vs_realfr_all_correlations_scatter_nolag.html",
                "suite2p_vs_realfr_all_correlations_scatter.svg": "suite2p_vs_realfr_all_correlations_scatter_nolag.svg",
                "suite2p_vs_realfr_all_cells_fr_compare.html": "suite2p_vs_realfr_all_cells_fr_compare_nolag.html",
                "suite2p_vs_realfr_all_cells_fr_compare.svg": "suite2p_vs_realfr_all_cells_fr_compare_nolag.svg",
                "suite2p_vs_realfr_all_cells_fr_compare.pdf": "suite2p_vs_realfr_all_cells_fr_compare_nolag.pdf",
                "suite2p_vs_realfr_all_events.csv": "suite2p_vs_realfr_all_events_nolag.csv",
                "suite2p_vs_realfr_all_events_summary_with_pies.html": "suite2p_vs_realfr_all_events_summary_with_pies_nolag.html",
                "suite2p_vs_realfr_all_events_summary_with_pies.svg": "suite2p_vs_realfr_all_events_summary_with_pies_nolag.svg",
                "suite2p_vs_realfr_missed_over_residual_z_scatter.html": "suite2p_vs_realfr_missed_over_residual_z_scatter_nolag.html",
                "suite2p_vs_realfr_missed_over_residual_z_scatter.svg": "suite2p_vs_realfr_missed_over_residual_z_scatter_nolag.svg",
                "suite2p_vs_realfr_missed_over_real_vs_pred_binned.html": "suite2p_vs_realfr_missed_over_real_vs_pred_binned_nolag.html",
                "suite2p_vs_realfr_missed_over_real_vs_pred_binned.svg": "suite2p_vs_realfr_missed_over_real_vs_pred_binned_nolag.svg",
            }
            for legacy_name, nolag_name in legacy_map.items():
                src = os.path.join(str(out_dir), nolag_name)
                dst = os.path.join(str(out_dir), legacy_name)
                if os.path.isfile(src):
                    with open(src, "rb") as fs:
                        data = fs.read()
                    with open(dst, "wb") as fd:
                        fd.write(data)
        except Exception as e:
            print(f"[WARN] Legacy copy failed: {e}")
    return out_df
