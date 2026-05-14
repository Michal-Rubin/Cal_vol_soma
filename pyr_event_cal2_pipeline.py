import os
import re
import glob
import pickle
import shutil
import threading
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:
    _scipy_find_peaks = None
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

# Use existing code-folder function when available
try:
    from AnalasysFunction import BurstC as _BurstC
except Exception:
    _BurstC = None

VOL_SR = 500.0
SIMPLE_ISI_MS = 30.0
TAIL_RATIO_THR = 0.30   # exclude later event when (abs(prev_tail_min) / current_event_max) >= this threshold
GAP_THR_PREV_SIMPLE_S = 0.250
GAP_THR_PREV_COMPLEX_S = 0.500
HARD_MIN_GAP_S = 0.100  # hard reject when gap(prev last spike -> curr first spike) < 100 ms
COMPLEX_FOLLOWUP_EXCLUDE_S = 0.150  # exclude non-complex event if a complex/plateau starts within 150 ms after it
MAX_PEAK_SEARCH_S = 0.150  # peak search within 150 ms from event start
SIMPLE_PEAK_SEARCH_S = 0.150
COMPLEX_PEAK_SEARCH_S = 0.300
CS_Z_START_THR = 1.5
CS_Z_END_THR = 0.5
SUBTHR_BOUND_FRAC = 0.20
SUBTHR_BOUND_PRE_MS = 150.0
SUBTHR_BOUND_POST_MS = 300.0
EVENT_BOUND_SPIKE_PAD_FRAMES = 5
EVENT_END_FRAC = 0.20
PRE_BASELINE_S = 0.5
ROBUST_CAL_F0_PERCENTILE = 15.0
PEAK_DUP_TOL_FRAMES = 0
PIPELINE_VERSION = "2026-04-27-a"
SPIKE_COUNT_CAP = 6
COMPLEX4_REF_N_SPIKES = 4
CAL_NEUROPIL_R = 0.7
CAL_F0_PERCENTILE = 8.0
CAL_Z_NON_ROBUST_LOW_PERCENTILE = 15.0
AUC_BIN_COUNT = 8
STATIC_IMAGE_TIMEOUT_S = 120.0
EXPORT_PDF = False
SAVE_EVENT_SELECTION_FIGURES = False
PERM_N = 1000
PERM_MAX_PER_GROUP = 300


EVENT_TYPES = ("simple", "complex", "plateau")
EVENT_COLOR_MAP = {"simple": "red", "complex": "black", "plateau": "purple"}
EVENT_LABEL_MAP = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}
EVENT_WINNER_PRIORITY = {"simple": 0, "complex": 1, "plateau": 2}
# Plotly uses CSS/hex colors; matplotlib tab:red equivalent is #d62728.
VOLTAGE_TRACE_COLOR = "#d62728"   # tab:red
CALCIUM_TRACE_COLOR = "mediumseagreen"


def _norm_event_type(x):
    t = str(x).strip().lower()
    if t in ("simple", "complex", "plateau"):
        return t
    return "simple"


def _is_complex_like_event_type(x):
    return _norm_event_type(x) in ("complex", "plateau")


def _is_dual_plateau_complex(ev):
    return bool(ev.get("is_plateau_event", False)) and bool(ev.get("is_complex_event", False))


def _as_sorted_unique_int(x):
    if x is None:
        return np.array([], dtype=int)
    a = np.asarray(x, dtype=int).ravel()
    if a.size == 0:
        return np.array([], dtype=int)
    return np.unique(a)
def _safe_cal_sr(row_val, default=30.0):
    try:
        v = float(row_val)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        pass
    return float(default)
def _zscore_low8(x, low_percentile=8.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    try:
        p = float(low_percentile)
    except Exception:
        p = 8.0
    p = min(100.0, max(0.0, p))
    n = max(1, int(round((p / 100.0) * x.size)))
    lo = np.sort(x)[:n]
    return float(np.mean(lo)), float(np.std(lo))
def _robust_center_scale(x, center=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    c = float(np.nanmedian(x) if center is None else center)
    mad = float(np.nanmedian(np.abs(x - c)))
    s = 1.4826 * mad
    if (not np.isfinite(s)) or (s <= 0):
        s = float(np.nanstd(x))
    if (not np.isfinite(s)) or (s <= 0):
        s = np.nan
    return float(c), float(s)
def _calcium_p8_center_scale(x, p=8.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    c = float(np.nanpercentile(x, float(p)))
    _, s = _robust_center_scale(x, center=c)
    return float(c), float(s)
def _build_quiet_masks(trace_vol_len, trace_cal_len, class_events, vol_sr, cal_sr):
    qv = np.ones(int(trace_vol_len), dtype=bool)
    qc = np.ones(int(trace_cal_len), dtype=bool)
    if trace_vol_len <= 0 or trace_cal_len <= 0:
        return qv, qc
    for ev in class_events:
        vs = int(max(0, min(int(trace_vol_len) - 1, int(ev.get("start_frame", 0)))))
        ve = int(max(vs, min(int(trace_vol_len) - 1, int(ev.get("end_frame", vs)))))
        qv[vs:ve + 1] = False
        cs = _vol_idx_to_cal_idx(vs, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal_len)
        ce = _vol_idx_to_cal_idx(ve, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal_len)
        qc[cs:ce + 1] = False
    return qv, qc
def _suffix_from_pkl_name(path):
    name = os.path.basename(path)
    for pat in (r"final_correct_spike_detection(.*?)\.pkl$", r"spike_detection_refined_new(.*?)\.pkl$"):
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            s = m.group(1)
            return "main" if s == "" else s
    return os.path.splitext(name)[0]

def _normalize_suffix_tag(suffix):
    s = str(suffix).strip()
    if s == "":
        return "main"
    s = re.sub(r"^[\s_]+", "", s)
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = s.strip("_")
    return s if s else "main"

def _motor_state_from_suffix(suffix):
    s = _normalize_suffix_tag(suffix).lower()
    m = re.fullmatch(r"([mr])(\d+)", s)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"

def _is_motor_row(row):
    vals = [
        str(row.get("motor", "")).strip().lower(),
        str(row.get("brainState", "")).strip().lower(),
    ]
    for v in vals:
        if "motor" in v:
            return True
    return False

def _natural_pkl_key(path):
    name = os.path.basename(path)
    m = re.search(r"(final_correct_spike_detection|spike_detection_refined_new)(.*?)\.pkl$", name, flags=re.IGNORECASE)
    if not m:
        return (2, 1, name.lower())

    prefix, suffix = m.group(1).lower(), m.group(2)
    pref_rank = 0 if prefix.startswith("final_correct") else 1

    if suffix == "":
        return (pref_rank, 0, -1)
    try:
        return (pref_rank, 0, int(suffix))
    except Exception:
        return (pref_rank, 1, suffix.lower())

def _read_trace_csv_1d(csv_path):
    arr = pd.read_csv(csv_path).to_numpy(dtype=float).ravel()
    return np.asarray(arr, dtype=float)

def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 3:
        return -np.inf
    x = x[:n]
    y = y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 3:
        return -np.inf
    x = x[ok]
    y = y[ok]
    sx = float(np.nanstd(x))
    sy = float(np.nanstd(y))
    if sx <= 0 or sy <= 0:
        return -np.inf
    return float(np.corrcoef(x, y)[0, 1])

def _cell_idx_from_folder(cell_folder):
    base = os.path.basename(os.path.normpath(cell_folder))
    m = re.match(r"cell(\d+)$", str(base), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _resolve_suite2p_row_idx(cell_folder, raw_trace, suite2p_dir):
    f_path = os.path.join(suite2p_dir, "F.npy")
    if not os.path.isfile(f_path):
        cell_idx = _cell_idx_from_folder(cell_folder)
        return (cell_idx, np.nan)
    try:
        F = np.asarray(np.load(f_path, mmap_mode="r"))
    except Exception:
        cell_idx = _cell_idx_from_folder(cell_folder)
        return (cell_idx, np.nan)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if F.ndim < 2 or F.shape[0] == 0:
        cell_idx = _cell_idx_from_folder(cell_folder)
        return (cell_idx, np.nan)

    best_idx = None
    best_corr = -np.inf
    for ridx in range(int(F.shape[0])):
        corr = _safe_corrcoef(raw_trace, np.asarray(F[int(ridx)], dtype=float).ravel())
        if corr > best_corr:
            best_corr = corr
            best_idx = int(ridx)

    if best_idx is not None and np.isfinite(best_corr):
        return (int(best_idx), float(best_corr))

    cell_idx = _cell_idx_from_folder(cell_folder)
    if cell_idx is not None and 0 <= int(cell_idx) < int(F.shape[0]):
        return (int(cell_idx), np.nan)
    return (None, np.nan)

def _compute_calcium_nb_and_df(cell_folder, neuropil_r=CAL_NEUROPIL_R, f0_percentile=CAL_F0_PERCENTILE, save_outputs=True):
    raw_path = os.path.join(cell_folder, "calTrace.csv")
    if not os.path.isfile(raw_path):
        return None, None

    cal_raw = _read_trace_csv_1d(raw_path)
    if cal_raw.size == 0:
        return None, None

    suite2p_dir = os.path.join(os.path.dirname(cell_folder), "Sync", "cal", "suite2p", "plane0")
    fneu_path = os.path.join(suite2p_dir, "Fneu.npy")
    if os.path.isfile(fneu_path):
        row_idx, row_corr = _resolve_suite2p_row_idx(cell_folder, cal_raw, suite2p_dir)
    else:
        row_idx, row_corr = (None, np.nan)

    if row_idx is not None:
        Fneu = np.asarray(np.load(fneu_path, mmap_mode="r"))
        if Fneu.ndim == 1:
            neu = Fneu.ravel().astype(float)
        elif 0 <= int(row_idx) < int(Fneu.shape[0]):
            neu = np.asarray(Fneu[int(row_idx)], dtype=float).ravel()
        else:
            neu = np.zeros_like(cal_raw)
        n = min(cal_raw.size, neu.size)
        cal_raw = cal_raw[:n]
        neu = neu[:n]
        cal_nb = cal_raw - float(neuropil_r) * neu
    else:
        cal_nb = cal_raw.copy()

    finite = np.isfinite(cal_nb)
    if np.any(finite):
        baseline = float(np.nanpercentile(cal_nb[finite], float(f0_percentile)))
    else:
        baseline = 0.0
    denom = baseline
    if (not np.isfinite(denom)) or (abs(denom) < 1e-9):
        denom = 1e-9 if (not np.isfinite(denom) or denom >= 0) else -1e-9
    cal_df = (cal_nb - baseline) / denom

    if save_outputs:
        pd.Series(cal_nb).to_csv(os.path.join(cell_folder, "calTraceNB.csv"), index=False)
        pd.Series(cal_df).to_csv(os.path.join(cell_folder, "calTraceDF.csv"), index=False)
        idx_txt = os.path.join(cell_folder, "calcium_neuropil_selected_roi_idx.txt")
        with open(idx_txt, "w", encoding="utf-8") as f:
            f.write(f"selected_suite2p_roi_idx: {row_idx}\n")
            f.write(f"match_corr_to_calTrace: {row_corr}\n")
            f.write(f"neuropil_r: {float(neuropil_r)}\n")
            f.write(f"f0_percentile: {float(f0_percentile)}\n")
            f.write(f"suite2p_dir: {suite2p_dir}\n")
            f.write(f"fneu_exists: {os.path.isfile(fneu_path)}\n")

    return cal_nb, cal_df
def _find_spike_pkls(cell_folder):
    final_pkls = glob.glob(os.path.join(cell_folder, "final_correct_spike_detection*.pkl"))
    if final_pkls:
        return sorted(list(dict.fromkeys(final_pkls)), key=_natural_pkl_key)
    refined_pkls = glob.glob(os.path.join(cell_folder, "spike_detection_refined_new*.pkl"))
    return sorted(list(dict.fromkeys(refined_pkls)), key=_natural_pkl_key)

def _segment_cal_trace_candidates_from_suffix(suffix):
    s = _normalize_suffix_tag(suffix).lower()
    m = re.fullmatch(r"([mr])(\d+)", s)
    if not m:
        return []
    kind = m.group(1)
    idx = m.group(2)
    if kind == "m":
        return [
            f"calTraceMot{idx}.csv",
            f"calTraceM{idx}.csv",
            f"calTrace{s}.csv",
            f"caTrace{idx}.csv",
            f"calTrace{idx}.csv",
        ]
    return [
        f"calTraceRest{idx}.csv",
        f"calTraceRes{idx}.csv",
        f"calTraceResr{idx}.csv",
        f"calTraceR{idx}.csv",
        f"calTrace{s}.csv",
        f"caTrace{idx}.csv",
        f"calTrace{idx}.csv",
    ]

def _load_segment_cal_trace_by_suffix(cell_folder, suffix):
    for fn in _segment_cal_trace_candidates_from_suffix(suffix):
        p = os.path.join(cell_folder, fn)
        if os.path.isfile(p):
            try:
                arr = _read_trace_csv_1d(p)
                if arr is not None and np.asarray(arr).size > 0:
                    return np.asarray(arr, dtype=float).ravel(), p
            except Exception:
                pass
    return None, None

def _interp_nan_1d(x):
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return x
    idx = np.arange(n)
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if np.sum(good) == 1:
        y = np.full_like(x, x[good][0])
        return y
    y = x.copy()
    y[~good] = np.interp(idx[~good], idx[good], x[good])
    return y

def _moving_average_1d(x, w):
    x = np.asarray(x, float)
    w = max(1, int(w))
    if x.size == 0 or w <= 1:
        return x.copy()
    ker = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, ker, mode="same")

def _build_subthreshold_z(trace_vol, spike_idx, vol_sr=500.0, remove_radius=3, smooth_ms=20.0):
    v = np.asarray(trace_vol, float).copy()
    n = v.size
    if n == 0:
        return v

    sidx = _as_sorted_unique_int(spike_idx)
    sidx = sidx[(sidx >= 0) & (sidx < n)]
    for s in sidx:
        a = max(0, int(s) - int(remove_radius))
        b = min(n, int(s) + int(remove_radius) + 1)
        v[a:b] = np.nan

    v = _interp_nan_1d(v)
    w = max(1, int(round((float(smooth_ms) / 1000.0) * float(vol_sr))))
    v_lp = _moving_average_1d(v, w)

    mu, sd = _zscore_low8(v_lp)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(v_lp)
    return (v_lp - float(mu)) / float(sd)

def _build_subthreshold_signal(trace_vol, spike_idx, vol_sr=500.0, remove_radius=3, smooth_ms=20.0):
    v = np.asarray(trace_vol, float).copy()
    n = v.size
    if n == 0:
        return v, np.nan

    sidx = _as_sorted_unique_int(spike_idx)
    sidx = sidx[(sidx >= 0) & (sidx < n)]
    for s in sidx:
        a = max(0, int(s) - int(remove_radius))
        b = min(n, int(s) + int(remove_radius) + 1)
        v[a:b] = np.nan

    v = _interp_nan_1d(v)
    w = max(1, int(round((float(smooth_ms) / 1000.0) * float(vol_sr))))
    v_lp = _moving_average_1d(v, w)
    mu, _ = _zscore_low8(v_lp)
    return v_lp, float(mu) if np.isfinite(mu) else np.nan

def _cs_bounds_from_subz(subz, first_sp, last_sp, trace_len, z_start_thr=1.5, z_end_thr=0.5, vol_sr=500.0,
                         pre_ms=150.0, post_ms=200.0):
    n = int(trace_len)
    if n <= 0:
        return 0, 0

    fs = int(max(0, min(n - 1, int(first_sp))))
    ls = int(max(fs, min(n - 1, int(last_sp))))

    pad_pre = int(round((float(pre_ms) / 1000.0) * float(vol_sr)))
    pad_post = int(round((float(post_ms) / 1000.0) * float(vol_sr)))

    a = max(0, fs - pad_pre)
    b = min(n - 1, ls + pad_post)

    z = np.asarray(subz, float)
    if z.size != n or b < a:
        return max(0, fs - 5), min(n - 1, ls + 5)

    seg = z[a:b + 1]
    if seg.size == 0 or not np.any(np.isfinite(seg)):
        return max(0, fs - 5), min(n - 1, ls + 5)

    i_peak = int(a + np.nanargmax(seg))
    if not np.isfinite(z[i_peak]) or z[i_peak] < float(z_start_thr):
        return max(0, fs - 5), min(n - 1, ls + 5)

    i = i_peak
    while i > a and np.isfinite(z[i]) and z[i] >= float(z_start_thr):
        i -= 1
    s = int(i + 1 if (np.isfinite(z[i]) and z[i] < float(z_start_thr)) else a)

    j = i_peak
    while j < b and np.isfinite(z[j]) and z[j] >= float(z_end_thr):
        j += 1
    e = int(j - 1 if (np.isfinite(z[j]) and z[j] < float(z_end_thr)) else b)

    # Keep bounds consistent with spike span
    s = min(s, fs)
    e = max(e, ls)

    s = int(max(0, min(n - 1, s)))
    e = int(max(s, min(n - 1, e)))
    return s, e

def _bounds_from_subthreshold_frac(
    sub_sig,
    first_sp,
    last_sp,
    trace_len,
    baseline=None,
    frac=0.20,
    vol_sr=500.0,
    pre_ms=150.0,
    post_ms=300.0,
    spike_pad_frames=5,
):
    n = int(trace_len)
    if n <= 0:
        return 0, 0, np.nan, -1

    fs = int(max(0, min(n - 1, int(first_sp))))
    ls = int(max(fs, min(n - 1, int(last_sp))))

    pad_pre = int(round((float(pre_ms) / 1000.0) * float(vol_sr)))
    pad_post = int(round((float(post_ms) / 1000.0) * float(vol_sr)))
    a = max(0, fs - pad_pre)
    b = min(n - 1, ls + pad_post)

    x = np.asarray(sub_sig, float).ravel()
    if x.size != n or b < a:
        s = max(0, fs - int(spike_pad_frames))
        e = min(n - 1, ls + int(spike_pad_frames))
        return int(s), int(max(s, e)), np.nan, -1

    seg = x[a:b + 1]
    if seg.size == 0 or (not np.any(np.isfinite(seg))):
        s = max(0, fs - int(spike_pad_frames))
        e = min(n - 1, ls + int(spike_pad_frames))
        return int(s), int(max(s, e)), np.nan, -1

    # Stable peak search in finite signal
    if np.any(~np.isfinite(seg)):
        idx = np.arange(seg.size)
        good = np.isfinite(seg)
        seg_work = seg.copy()
        seg_work[~good] = np.interp(idx[~good], idx[good], seg[good]) if np.any(good) else 0.0
    else:
        seg_work = seg

    i_peak = int(a + int(np.nanargmax(seg_work)))
    peak_val = float(x[i_peak]) if np.isfinite(x[i_peak]) else float(np.nanmax(seg_work))

    if not np.isfinite(baseline):
        if np.any(np.isfinite(x)):
            baseline = float(np.nanmedian(x))
        else:
            baseline = 0.0
    else:
        baseline = float(baseline)

    frac = float(frac)
    frac = min(1.0, max(0.0, frac))
    thr = float(baseline + frac * (peak_val - baseline))
    if not np.isfinite(thr):
        thr = baseline

    s = int(a)
    i = int(i_peak)
    while i > a:
        xv = x[i]
        if np.isfinite(xv) and (xv < thr):
            s = int(i + 1)
            break
        i -= 1

    e = int(b)
    j = int(i_peak)
    while j < b:
        xv = x[j]
        if np.isfinite(xv) and (xv <= thr):
            e = int(j)
            break
        j += 1

    # Guard against bounds contradicting spike span.
    if s > fs:
        s = max(0, fs - int(spike_pad_frames))
    if e < ls:
        e = min(n - 1, ls + int(spike_pad_frames))

    s = int(max(0, min(n - 1, s)))
    e = int(max(s, min(n - 1, e)))
    return s, e, thr, i_peak

def _vol_idx_to_cal_idx(v_idx, vol_sr, cal_sr, cal_len):
    if cal_len <= 0:
        return 0
    t = float(v_idx) / float(vol_sr)
    c = int(round(t * float(cal_sr)))
    return int(max(0, min(int(cal_len) - 1, c)))

def _ensure_main_overlay_for_cell(cell_folder, out_infos):
    target_html = os.path.join(cell_folder, "event_spike_overlay_main.html")
    target_svg = os.path.join(cell_folder, "event_spike_overlay_main.svg")

    if len(out_infos) == 0:
        return target_html, target_svg, None

    best = None
    for info in out_infos:
        if _suffix_from_pkl_name(str(info.get("pkl_path", ""))) == "main":
            best = info
            break
    if best is None:
        best = sorted(out_infos, key=lambda x: _natural_pkl_key(str(x.get("pkl_path", ""))))[0]

    src_html = str(best.get("html_path", ""))
    src_svg = str(best.get("svg_path", ""))

    if src_html and os.path.exists(src_html) and os.path.abspath(src_html) != os.path.abspath(target_html):
        shutil.copy2(src_html, target_html)
    if src_svg and os.path.exists(src_svg) and os.path.abspath(src_svg) != os.path.abspath(target_svg):
        shutil.copy2(src_svg, target_svg)

    return target_html, target_svg, src_html

def _ensure_main_voltage_overlay_for_cell(cell_folder, out_infos):
    target_html = os.path.join(cell_folder, "event_voltage_overlay_main.html")
    target_svg = os.path.join(cell_folder, "event_voltage_overlay_main.svg")

    if len(out_infos) == 0:
        return target_html, target_svg, None

    best = None
    for info in out_infos:
        if _suffix_from_pkl_name(str(info.get("pkl_path", ""))) == "main":
            best = info
            break
    if best is None:
        best = sorted(out_infos, key=lambda x: _natural_pkl_key(str(x.get("pkl_path", ""))))[0]

    src_html = str(best.get("voltage_html_path", ""))
    src_svg = str(best.get("voltage_svg_path", ""))

    if src_html and os.path.exists(src_html) and os.path.abspath(src_html) != os.path.abspath(target_html):
        shutil.copy2(src_html, target_html)
    if src_svg and os.path.exists(src_svg) and os.path.abspath(src_svg) != os.path.abspath(target_svg):
        shutil.copy2(src_svg, target_svg)

    return target_html, target_svg, src_html

def _save_placeholder_main_overlay(cell_folder, note="No valid events/spikes for overlay"):
    fig = go.Figure()
    fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(template="simple_white", width=1200, height=500,
                      title=f"{os.path.basename(cell_folder)} | event_spike_overlay_main")

    target_html = os.path.join(cell_folder, "event_spike_overlay_main.html")
    target_svg = os.path.join(cell_folder, "event_spike_overlay_main.svg")
    fig.write_html(target_html)
    _safe_write_image(fig, target_svg, warn_prefix="Placeholder SVG")

    return target_html, target_svg

def _group_events_by_isi(spikes, isi_ms=30.0, fs=500.0):
    spikes = _as_sorted_unique_int(spikes)
    if spikes.size == 0:
        return []
    threshold_frames = max(1, int(round((isi_ms / 1000.0) * fs)))

    if _BurstC is not None:
        grouped, _ = _BurstC(spikes.tolist(), threshold_frames)
        return [np.asarray(g, dtype=int) for g in grouped if len(g) > 0]

    events = [[int(spikes[0])]]
    for s in spikes[1:]:
        if (s - events[-1][-1]) <= threshold_frames:
            events[-1].append(int(s))
        else:
            events.append([int(s)])
    return [np.asarray(g, dtype=int) for g in events]

def _events_from_saved_labels(d, trace_len, trace_vol=None, vol_sr=500.0, cs_z_start_thr=1.5, cs_z_end_thr=0.5, non_cs_pad_frames=5, simple_isi_ms=30.0):
    """
    Build events from saved labels with this priority:
    1) Plateau events from vm_plateaus_dict windows (treated as complex-like for choosing rules).
       If a plateau overlaps complex spikes, event_type is plateau and flags keep dual classification.
    2) Complex events from vm_burst_dict windows (excluding spikes already consumed by plateaus).
    3) Simple events grouped from vm_simple_spikes using ISI<=simple_isi_ms.
    4) Any remaining unlabeled spikes grouped by the same ISI rule as a fallback.
    """
    trace_len = int(trace_len)
    if trace_len <= 0:
        return [], np.array([], dtype=int), np.array([], dtype=int)

    simple_spikes = _as_sorted_unique_int(d.get("vm_simple_spikes", []))
    complex_spikes = _as_sorted_unique_int(d.get("vm_complex_spikes", []))
    all_spikes = _as_sorted_unique_int(d.get("vm_all_spikes", []))

    simple_spikes = simple_spikes[(simple_spikes >= 0) & (simple_spikes < trace_len)]
    complex_spikes = complex_spikes[(complex_spikes >= 0) & (complex_spikes < trace_len)]
    all_spikes = all_spikes[(all_spikes >= 0) & (all_spikes < trace_len)]
    if all_spikes.size == 0:
        all_spikes = _as_sorted_unique_int(np.r_[simple_spikes, complex_spikes])

    events = []
    used_spikes = set()

    # 0) Plateau events: use saved plateau windows directly when available.
    vm_plateaus_dict = d.get("vm_plateaus_dict", None)
    plateau_spikes_all = np.array([], dtype=int)
    if isinstance(vm_plateaus_dict, dict):
        # Gather all known plateau-associated spikes for fallback or overlap checks.
        locs = _as_sorted_unique_int(vm_plateaus_dict.get("locs", []))
        spikes_flat = []
        raw_si = vm_plateaus_dict.get("spike_indices", [])
        if isinstance(raw_si, (list, tuple)):
            for si in raw_si:
                si_arr = _as_sorted_unique_int(si)
                if si_arr.size:
                    spikes_flat.extend(si_arr.tolist())
        plateau_spikes_all = _as_sorted_unique_int(np.r_[locs, np.asarray(spikes_flat, dtype=int) if len(spikes_flat) else np.array([], dtype=int)])
        plateau_spikes_all = plateau_spikes_all[(plateau_spikes_all >= 0) & (plateau_spikes_all < trace_len)]

        starts = _as_sorted_unique_int(vm_plateaus_dict.get("starts", []))
        ends = _as_sorted_unique_int(vm_plateaus_dict.get("ends", []))
        per_event_sp = raw_si if isinstance(raw_si, (list, tuple)) else []

        if starts.size > 0 and ends.size > 0 and starts.size == ends.size:
            for i_evt, (s0, e0) in enumerate(zip(starts.tolist(), ends.tolist())):
                s = int(max(0, min(trace_len - 1, s0)))
                e = int(max(s, min(trace_len - 1, e0)))

                # Prefer explicit spike_indices for each plateau, fallback to locs/window overlap.
                sp = np.array([], dtype=int)
                if i_evt < len(per_event_sp):
                    sp = _as_sorted_unique_int(per_event_sp[i_evt])
                    sp = sp[(sp >= 0) & (sp < trace_len)]
                if sp.size == 0:
                    sp = plateau_spikes_all[(plateau_spikes_all >= s) & (plateau_spikes_all <= e)]
                if sp.size == 0:
                    # keep plateau event even when no explicit spike list exists: use nearest loc in window
                    loc_in = locs[(locs >= s) & (locs <= e)]
                    if loc_in.size > 0:
                        sp = np.array([int(loc_in[0])], dtype=int)

                sp = _as_sorted_unique_int(sp)
                if sp.size == 0:
                    continue

                has_complex_overlap = bool(np.intersect1d(sp, complex_spikes).size > 0)
                for x in sp.tolist():
                    used_spikes.add(int(x))
                events.append({
                    "spikes": sp,
                    "n_spikes": int(sp.size),
                    "event_type": "plateau",
                    "event_kind": "plateau",
                    "start_frame": s,
                    "end_frame": e,
                    "source": "plateau_windows",
                    "is_plateau_event": True,
                    "is_complex_event": bool(has_complex_overlap),
                })

        # Fallback: plateau spikes grouped by ISI if no explicit windows produced events
        if len([ev for ev in events if str(ev.get("event_type", "")) == "plateau"]) == 0 and plateau_spikes_all.size > 0:
            grouped_plateau = _group_events_by_isi(plateau_spikes_all, isi_ms=float(simple_isi_ms), fs=vol_sr)
            for sp in grouped_plateau:
                sp = _as_sorted_unique_int(sp)
                if sp.size == 0:
                    continue
                has_complex_overlap = bool(np.intersect1d(sp, complex_spikes).size > 0)
                for x in sp.tolist():
                    used_spikes.add(int(x))
                events.append({
                    "spikes": sp,
                    "n_spikes": int(sp.size),
                    "event_type": "plateau",
                    "event_kind": "plateau",
                    "start_frame": int(sp[0]),
                    "end_frame": int(sp[-1]),
                    "source": "plateau_spikes_isi_grouped_fallback",
                    "is_plateau_event": True,
                    "is_complex_event": bool(has_complex_overlap),
                })

    plateau_windows = []
    for evp in events:
        if str(evp.get("event_type", "")).lower() != "plateau":
            continue
        ps = int(evp.get("start_frame", -1))
        pe = int(evp.get("end_frame", -1))
        if ps < 0 or pe < 0:
            continue
        if pe < ps:
            ps, pe = pe, ps
        plateau_windows.append((ps, pe))

    # 1) Complex events: use saved complex burst windows directly.
    vm_burst_dict = d.get("vm_burst_dict", None)
    if isinstance(vm_burst_dict, dict):
        starts = _as_sorted_unique_int(vm_burst_dict.get("starts", []))
        ends = _as_sorted_unique_int(vm_burst_dict.get("ends", []))
        if starts.size > 0 and ends.size > 0 and starts.size == ends.size:
            for s, e in zip(starts.tolist(), ends.tolist()):
                s = int(max(0, min(trace_len - 1, s)))
                e = int(max(s, min(trace_len - 1, e)))
                sp = complex_spikes[(complex_spikes >= s) & (complex_spikes <= e)]
                sp = _as_sorted_unique_int(sp)
                if sp.size == 0:
                    continue
                # Keep complex event spikes exactly from PKL complex windows (no overlap trimming).
                overlap_by_spikes = bool(np.intersect1d(sp, plateau_spikes_all).size > 0)
                overlap_by_window = any((not (e < ps or s > pe)) for (ps, pe) in plateau_windows)
                classify_as_plateau = bool(overlap_by_spikes or overlap_by_window)

                for x in sp.tolist():
                    used_spikes.add(int(x))
                events.append({
                    "spikes": sp,
                    "n_spikes": int(sp.size),
                    "event_type": "plateau" if classify_as_plateau else "complex",
                    "event_kind": "plateau" if classify_as_plateau else "complex",
                    "start_frame": s,
                    "end_frame": e,
                    "source": "complex_burst_windows_reclassified_plateau" if classify_as_plateau else "complex_burst_windows",
                    "is_plateau_event": bool(classify_as_plateau),
                    "is_complex_event": True,
                })

    # Fallback for datasets without vm_burst_dict: keep old behavior for complex spikes.
    if len([ev for ev in events if str(ev.get("event_type", "")) == "complex"]) == 0 and complex_spikes.size > 0:
        grouped_complex = _group_events_by_isi(complex_spikes, isi_ms=float(simple_isi_ms), fs=vol_sr)
        for sp in grouped_complex:
            sp = _as_sorted_unique_int(sp)
            if sp.size == 0:
                continue
            # Keep fallback complex grouping untouched by overlap; only class label may change.
            s0 = int(sp[0])
            e0 = int(sp[-1])
            overlap_by_spikes = bool(np.intersect1d(sp, plateau_spikes_all).size > 0)
            overlap_by_window = any((not (e0 < ps or s0 > pe)) for (ps, pe) in plateau_windows)
            classify_as_plateau = bool(overlap_by_spikes or overlap_by_window)

            for x in sp.tolist():
                used_spikes.add(int(x))
            events.append({
                "spikes": sp,
                "n_spikes": int(sp.size),
                "event_type": "plateau" if classify_as_plateau else "complex",
                "event_kind": "plateau" if classify_as_plateau else "complex",
                "start_frame": int(sp[0]),
                "end_frame": int(sp[-1]),
                "source": "complex_spikes_isi_grouped_fallback_reclassified_plateau" if classify_as_plateau else "complex_spikes_isi_grouped_fallback",
                "is_plateau_event": bool(classify_as_plateau),
                "is_complex_event": True,
            })

    # 2) Simple events: group only simple spikes by ISI rule.
    if simple_spikes.size > 0:
        grouped_simple = _group_events_by_isi(simple_spikes, isi_ms=float(simple_isi_ms), fs=vol_sr)
        for sp in grouped_simple:
            sp = _as_sorted_unique_int(sp)
            if sp.size == 0:
                continue
            # avoid duplicating spikes already consumed by plateau/complex events
            sp = np.asarray([int(x) for x in sp.tolist() if int(x) not in used_spikes], dtype=int)
            sp = _as_sorted_unique_int(sp)
            if sp.size == 0:
                continue
            for x in sp.tolist():
                used_spikes.add(int(x))
            events.append({
                "spikes": sp,
                "n_spikes": int(sp.size),
                "event_type": "simple",
                "event_kind": "single" if int(sp.size) == 1 else "simple_burst",
                "start_frame": int(sp[0]),
                "end_frame": int(sp[-1]),
                "source": "simple_spikes_isi_grouped",
                "is_plateau_event": False,
                "is_complex_event": False,
            })

    # 3) Fallback: include any remaining unlabeled spikes so nothing is dropped.
    if all_spikes.size > 0:
        leftover = np.asarray([int(x) for x in all_spikes.tolist() if int(x) not in used_spikes], dtype=int)
        leftover = _as_sorted_unique_int(leftover)
        if leftover.size > 0:
            grouped_leftover = _group_events_by_isi(leftover, isi_ms=float(simple_isi_ms), fs=vol_sr)
            for sp in grouped_leftover:
                sp = _as_sorted_unique_int(sp)
                if sp.size == 0:
                    continue
                for x in sp.tolist():
                    used_spikes.add(int(x))
                events.append({
                    "spikes": sp,
                    "n_spikes": int(sp.size),
                    "event_type": "simple",
                    "event_kind": "single" if int(sp.size) == 1 else "simple_burst",
                    "start_frame": int(sp[0]),
                    "end_frame": int(sp[-1]),
                    "source": "leftover_all_spikes_isi_grouped",
                    "is_plateau_event": False,
                    "is_complex_event": False,
                })

    events = sorted(events, key=lambda x: (int(x.get("start_frame", 0)), int(x.get("end_frame", 0))))

    if trace_vol is not None and len(events) > 0:
        sub_sig, sub_baseline = _build_subthreshold_signal(
            trace_vol,
            all_spikes,
            vol_sr=vol_sr,
            remove_radius=3,
            smooth_ms=20.0,
        )
        pad = int(max(1, non_cs_pad_frames))

        for ev in events:
            sp = _as_sorted_unique_int(ev.get("spikes", []))
            if sp.size == 0:
                continue
            fs = int(sp[0])
            ls = int(sp[-1])

            s0, e0, thr, i_peak_sub = _bounds_from_subthreshold_frac(
                sub_sig,
                fs,
                ls,
                trace_len,
                baseline=sub_baseline,
                frac=SUBTHR_BOUND_FRAC,
                vol_sr=vol_sr,
                pre_ms=SUBTHR_BOUND_PRE_MS,
                post_ms=SUBTHR_BOUND_POST_MS,
                spike_pad_frames=pad,
            )
            ev["bound_method"] = "subthreshold_baseline_frac20_crossing"
            ev["bound_thr_sub"] = float(thr) if np.isfinite(thr) else np.nan
            ev["bound_sub_peak_idx"] = int(i_peak_sub) if int(i_peak_sub) >= 0 else -1
            ev["start_frame"] = int(s0)
            ev["end_frame"] = int(e0)

    return events, simple_spikes, complex_spikes


def _nearest_idx(t_axis, t):
    if t_axis.size == 0:
        return 0
    return int(np.argmin(np.abs(t_axis - t)))

def _fit_decay_tau_loglin(t, y):
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(t) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    t = t[m]
    y = y[m]
    if t.size < 3:
        return np.nan
    y_peak = float(np.nanmax(y))
    if (not np.isfinite(y_peak)) or (y_peak <= 0):
        return np.nan
    keep = y > max(1e-12, 0.10 * y_peak)
    if np.sum(keep) < 3:
        return np.nan
    t_fit = t[keep]
    y_fit = y[keep]
    if np.unique(t_fit).size < 2:
        return np.nan
    try:
        p = np.polyfit(t_fit, np.log(y_fit), 1)
    except Exception:
        return np.nan
    slope = float(p[0])
    if (not np.isfinite(slope)) or slope >= 0:
        return np.nan
    tau = -1.0 / slope
    return float(tau) if np.isfinite(tau) and tau > 0 else np.nan

def _calc_event_features(cal_s, cal_t, start_idx, end_idx, next_start_idx=None,
                         mu_low8=np.nan, sd_low8=np.nan,
                         max_peak_search_s=0.150, pre_baseline_s=0.5, end_frac=0.20,
                         use_baseline_frac_end=False,
                         peak_search_start_idx=None, peak_search_end_idx=None,
                         first_spike_idx=None):
    cal_s = np.asarray(cal_s, float)
    cal_t = np.asarray(cal_t, float)
    n = cal_s.size
    if n == 0:
        return None

    dt = float(np.nanmedian(np.diff(cal_t))) if cal_t.size >= 2 else (1.0 / 30.0)
    cal_sr = (1.0 / dt) if dt > 0 else 30.0

    i_start = int(max(0, min(n - 1, int(start_idx))))
    i_end = int(max(i_start, min(n - 1, int(end_idx))))
    i_first_sp = int(max(0, min(n - 1, int(first_spike_idx)))) if first_spike_idx is not None else i_start

    if next_start_idx is None:
        i_next_start = n - 1
        i_next_cap = n - 1
    else:
        i_next_start = int(max(i_start, min(n - 1, int(next_start_idx))))
        i_next_cap = int(max(i_start, min(n - 1, i_next_start - 1)))
    i_end = int(max(i_start, min(i_end, i_next_cap)))
    # Do not search for a calcium peak before the first spike of the event.
    if peak_search_start_idx is not None:
        i_peak_start = int(max(i_start, min(i_end, int(peak_search_start_idx))))
    else:
        i_peak_start = i_start
    i_peak_start = int(max(i_peak_start, i_first_sp))

    pre_n = max(1, int(round(pre_baseline_s * cal_sr)))
    b0 = max(0, i_start - pre_n)
    b1 = i_start
    if b1 > b0:
        baseline = float(np.nanmedian(cal_s[b0:b1]))
    else:
        baseline = float(mu_low8) if np.isfinite(mu_low8) else float(np.nanmedian(cal_s))

    if peak_search_end_idx is not None:
        i_peak_user_end = int(max(i_start, min(n - 1, int(peak_search_end_idx))))
        peak_lim = int(min(i_end, i_peak_user_end, max(i_start, i_next_start - 1)))
    elif (max_peak_search_s is None) or (not np.isfinite(max_peak_search_s)) or (float(max_peak_search_s) <= 0):
        peak_lim = int(min(i_end, max(i_start, i_next_start - 1)))
    else:
        peak_n = max(1, int(round(float(max_peak_search_s) * cal_sr)))
        peak_lim = int(min(i_end, i_peak_start + peak_n, max(i_start, i_next_start - 1)))
    if peak_lim < i_peak_start:
        peak_lim = i_peak_start

    seg = np.asarray(cal_s[i_peak_start:peak_lim + 1], dtype=float)
    clear_peak = False
    peak_before_first_spike = False
    if seg.size > 0 and np.any(np.isfinite(seg)):
        # Prefer a true local maximum inside the search window.
        if np.any(~np.isfinite(seg)):
            idx = np.arange(seg.size)
            good = np.isfinite(seg)
            if np.any(good):
                seg_work = seg.copy()
                seg_work[~good] = np.interp(idx[~good], idx[good], seg[good])
            else:
                seg_work = np.zeros_like(seg)
        else:
            seg_work = seg

        # Peak ID rule: use absolute maximum inside the search window
        # (includes edge samples; avoids missing end-of-window peaks).
        rel_peak = int(np.nanargmax(seg_work))

        i_peak = i_peak_start + rel_peak
        peak_raw = float(cal_s[i_peak])
        peak_df_f = float(peak_raw - baseline)
        peak_before_first_spike = bool(i_peak < i_first_sp)
        clear_peak = np.isfinite(peak_df_f) and (peak_df_f > 0) and (not peak_before_first_spike)
    else:
        i_peak = i_peak_start
        peak_raw = baseline
        peak_df_f = 0.0

    if not clear_peak:
        peak_raw = float(cal_s[i_peak]) if (0 <= int(i_peak) < n and np.isfinite(cal_s[i_peak])) else baseline
        peak_df_f = 0.0 if (peak_before_first_spike or not np.isfinite(peak_df_f) or peak_df_f <= 0) else float(peak_df_f)
        peak_z = 0.0
        hwhm_s = np.nan
        auc = 0.0
        return {
            "start_idx": int(i_start),
            "peak_idx": int(i_peak),
            "end_idx": int(i_end),
            "first_spike_idx": int(i_first_sp),
            "baseline": float(baseline),
            "peak_raw": float(peak_raw),
            "peak_df_f": float(peak_df_f),
            "peak_z": float(peak_z),
            "hwhm_s": float(hwhm_s) if np.isfinite(hwhm_s) else np.nan,
            "auc": float(auc),
            "rise_time_s": np.nan,
            "decay_time_s": np.nan,
            "tau_decay_s": np.nan,
            "clear_peak": False,
            "peak_before_first_spike": bool(peak_before_first_spike),
        }

    peak_z = float((peak_raw - mu_low8) / sd_low8) if np.isfinite(sd_low8) and sd_low8 > 0 else np.nan

    # Burst calcium end rule:
    # end at first return to baseline + end_frac*(peak-baseline),
    # or at next event start - 1 (whichever comes first).
    if use_baseline_frac_end:
        end_level = float(baseline + float(end_frac) * peak_df_f)
        i_end_upper = int(max(i_peak, i_next_cap))
        right_decay = cal_s[i_peak:i_end_upper + 1]
        if right_decay.size > 0 and np.any(np.isfinite(right_decay)):
            decay_cross = np.where(right_decay <= end_level)[0]
            if decay_cross.size > 0:
                i_end = int(i_peak + int(decay_cross[0]))
            else:
                i_end = int(i_end_upper)
        else:
            i_end = int(i_end_upper)

    hwhm_s = np.nan
    half_level = baseline + 0.5 * peak_df_f
    left_seg = cal_s[i_start:i_peak + 1]
    right_seg = cal_s[i_peak:i_end + 1]
    left_cross = np.where(left_seg <= half_level)[0]
    right_cross = np.where(right_seg <= half_level)[0]
    if left_cross.size > 0 and right_cross.size > 0:
        i_left = i_start + int(left_cross[-1])
        i_right = i_peak + int(right_cross[0])
        if i_right > i_left:
            hwhm_s = float(cal_t[i_right] - cal_t[i_left])

    if i_end > i_start:
        y = np.clip(cal_s[i_start:i_end + 1] - baseline, 0, None)
        auc = float(np.trapezoid(y, x=cal_t[i_start:i_end + 1]))
    else:
        auc = 0.0

    # Kinetics metrics
    rise_time_s = np.nan
    if i_peak > i_first_sp:
        rise_time_s = float(cal_t[i_peak] - cal_t[i_first_sp])

    decay_time_s = np.nan
    if i_end > i_peak:
        decay_time_s = float(cal_t[i_end] - cal_t[i_peak])

    tau_decay_s = np.nan
    if i_end > i_peak:
        t_decay = np.asarray(cal_t[i_peak:i_end + 1], float) - float(cal_t[i_peak])
        y_decay = np.asarray(cal_s[i_peak:i_end + 1], float) - float(baseline)
        y_decay = np.clip(y_decay, 0, None)
        tau_decay_s = _fit_decay_tau_loglin(t_decay, y_decay)

    return {
        "start_idx": int(i_start),
        "peak_idx": int(i_peak),
        "end_idx": int(i_end),
        "first_spike_idx": int(i_first_sp),
        "baseline": float(baseline),
        "peak_raw": float(peak_raw),
        "peak_df_f": float(peak_df_f),
        "peak_z": float(peak_z),
        "hwhm_s": float(hwhm_s) if np.isfinite(hwhm_s) else np.nan,
        "auc": float(auc),
        "rise_time_s": float(rise_time_s) if np.isfinite(rise_time_s) else np.nan,
        "decay_time_s": float(decay_time_s) if np.isfinite(decay_time_s) else np.nan,
        "tau_decay_s": float(tau_decay_s) if np.isfinite(tau_decay_s) else np.nan,
        "clear_peak": True,
        "peak_before_first_spike": False,
    }

def _merge_events_same_start(events):
    """
    Merge events that share the same calcium start index into a single event.
    Keep one representative feature profile (best-scored event) and merge spike lists.
    Plateau has highest class priority, then complex, then simple.
    IMPORTANT: simple events are NOT merged here; simple burst grouping must remain
    voltage-spike ISI based only (from _group_events_by_isi).
    """
    if events is None or len(events) <= 1:
        return events, 0

    def _k_start(ev):
        return int(ev.get("start_idx", ev.get("start_frame", 0)))

    def _type_priority(ev):
        t = _norm_event_type(ev.get("event_type", "simple"))
        if bool(ev.get("is_plateau_event", False)):
            t = "plateau"
        return int(EVENT_WINNER_PRIORITY.get(t, 0))

    def _k_score(ev):
        return (
            1 if bool(ev.get("clear_peak", False)) else 0,
            float(ev.get("peak_df_f", -np.inf)) if np.isfinite(ev.get("peak_df_f", np.nan)) else -np.inf,
            int(ev.get("n_spikes", 0)),
            _type_priority(ev),
        )

    def _k_type(ev):
        t = _norm_event_type(ev.get("event_type", "simple"))
        if bool(ev.get("is_plateau_event", False)):
            t = "plateau"
        return t

    ev_sorted = sorted(events, key=lambda ev: (_k_start(ev), _k_type(ev), int(ev.get("event_idx", 0))))
    merged = []
    n_merged_out = 0
    i = 0
    while i < len(ev_sorted):
        s0 = _k_start(ev_sorted[i])
        grp = [ev_sorted[i]]
        i += 1
        t0 = _k_type(ev_sorted[i-1])
        while i < len(ev_sorted) and _k_start(ev_sorted[i]) == s0 and _k_type(ev_sorted[i]) == t0:
            grp.append(ev_sorted[i])
            i += 1

        # Keep simple events split; their burst/single identity must come only
        # from voltage ISI grouping, not calcium-boundary alignment artifacts.
        if t0 == "simple":
            merged.extend(grp)
            continue

        if len(grp) == 1:
            merged.append(grp[0])
            continue

        n_merged_out += (len(grp) - 1)
        winner = max(grp, key=_k_score).copy()

        all_spikes = []
        for ev in grp:
            all_spikes.extend(_as_sorted_unique_int(ev.get("spikes", [])).tolist())
        all_spikes = _as_sorted_unique_int(all_spikes)

        has_plateau = any((_norm_event_type(ev.get("event_type", "simple")) == "plateau") or bool(ev.get("is_plateau_event", False)) for ev in grp)
        has_complex = any((_norm_event_type(ev.get("event_type", "simple")) == "complex") or bool(ev.get("is_complex_event", False)) for ev in grp)

        if has_plateau:
            out_type = "plateau"
            out_kind = "plateau"
        elif has_complex:
            out_type = "complex"
            out_kind = "complex"
        else:
            out_type = "simple"
            out_kind = "single" if all_spikes.size <= 1 else "simple_burst"

        winner["spikes"] = all_spikes
        winner["n_spikes"] = int(all_spikes.size)
        winner["event_type"] = out_type
        winner["event_kind"] = out_kind
        winner["start_idx"] = int(s0)
        winner["end_idx"] = int(max(int(ev.get("end_idx", s0)) for ev in grp))
        winner["start_frame_v"] = int(min(int(ev.get("start_frame_v", 0)) for ev in grp))
        winner["end_frame_v"] = int(max(int(ev.get("end_frame_v", winner["start_frame_v"])) for ev in grp))
        winner["is_plateau_event"] = bool(has_plateau)
        winner["is_complex_event"] = bool(has_complex or has_plateau)
        winner["is_burst_event"] = bool((has_complex or has_plateau) or all_spikes.size > 1)
        winner["is_dual_class_event"] = bool(has_plateau and has_complex)
        winner["merged_same_start_count"] = int(len(grp))
        winner["merged_from_event_idx"] = ",".join(str(int(ev.get("event_idx", -1))) for ev in grp)
        merged.append(winner)

    merged = sorted(merged, key=lambda ev: (int(ev.get("start_idx", 0)), int(ev.get("peak_idx", 0))))
    for new_idx, ev in enumerate(merged):
        ev["event_idx"] = int(new_idx)

    return merged, int(n_merged_out)


def _evaluate_include_rule(events, trace_cal=None, ratio_thr=0.30, cal_sr=30.0,
                           ratio_max_gap_simple_s=0.300, ratio_max_gap_complex_s=0.500,
                           vol_sr=500.0, min_inter_event_gap_s=HARD_MIN_GAP_S,
                           complex_followup_exclude_s=COMPLEX_FOLLOWUP_EXCLUDE_S):
    """
    Include rule (OR):
    include when EITHER condition passes:
    1) gap(prev_last_spike -> curr_first_spike) > type-specific threshold
    2) tail ratio = abs(min(prev tail)) / max(curr event) < threshold
    """
    cal = np.asarray(trace_cal, dtype=float).ravel() if trace_cal is not None else np.array([], dtype=float)

    def _event_order_key(ev):
        sp = _as_sorted_unique_int(ev.get("spikes", []))
        if sp.size > 0:
            return int(sp[0])
        return int(ev.get("start_idx", ev.get("start_frame", 0)))

    # Order by first spike timing (more reliable than calcium start frame for overlap handling).
    events = sorted(events, key=_event_order_key)
    if len(events) == 0:
        return events

    for i, ev in enumerate(events):
        ev_sp = _as_sorted_unique_int(ev.get("spikes", []))
        ev["tail_ratio"] = np.nan
        ev["tail_min_prev"] = np.nan
        ev["curr_max"] = np.nan
        ev["tail_ratio_mode"] = "abs_prev_tail_min_local3mean_to_curr_max"

        if ev_sp.size == 0:
            ev["include"] = False
            ev["include_reason"] = "empty_event"
            continue

        if i == 0:
            ev["include"] = True
            ev["include_reason"] = "first_event"
            continue

        # Hard gate: if this event is too close to the previous event (by spikes), reject.
        prev = events[i - 1]
        prev_sp_hard = _as_sorted_unique_int(prev.get("spikes", []))
        curr_sp_hard = _as_sorted_unique_int(ev.get("spikes", []))
        gap_hard_s = np.nan
        if prev_sp_hard.size > 0 and curr_sp_hard.size > 0 and float(vol_sr) > 0:
            gap_hard_s = float(int(curr_sp_hard[0]) - int(prev_sp_hard[-1])) / float(vol_sr)
            ev["gap_mode"] = "prev_last_spike_to_curr_first_spike"
        else:
            ev["gap_mode"] = "prev_last_spike_to_curr_first_spike_unavailable"
        ev["gap_s"] = float(gap_hard_s) if np.isfinite(gap_hard_s) else np.nan
        ev["gap_hard_min_s"] = float(min_inter_event_gap_s)
        if np.isfinite(gap_hard_s) and (gap_hard_s < float(min_inter_event_gap_s)):
            ev["include"] = False
            ev["gap_pass"] = False
            ev["ratio_pass"] = np.nan
            ev["tail_ratio"] = np.nan
            ev["tail_min_prev"] = np.nan
            ev["curr_max"] = np.nan
            ev["include_reason"] = f"gap_lt_hard_min_{float(min_inter_event_gap_s):.3f}s"
            continue

        if cal.size == 0:
            ev["include"] = True
            ev["include_reason"] = "no_cal_trace_gap_ge_hard_min"
            continue

        prev = events[i - 1]
        s_cur = int(ev.get("start_idx", ev.get("start_frame", int(ev_sp[0]))))
        e_cur = int(ev.get("end_idx", s_cur))

        p_prev = int(prev.get("peak_idx", prev.get("start_idx", 0)))
        e_prev = int(prev.get("end_idx", p_prev))

        p_prev = max(0, min(cal.size - 1, p_prev))
        e_prev = max(p_prev, min(cal.size - 1, e_prev))
        s_cur = max(0, min(cal.size - 1, s_cur))
        e_cur = max(s_cur, min(cal.size - 1, e_cur))

        prev_type = _norm_event_type(prev.get("event_type", "simple"))
        gap_thr_s = float(ratio_max_gap_complex_s) if _is_complex_like_event_type(prev_type) else float(ratio_max_gap_simple_s)
        ev["gap_thr_s"] = gap_thr_s

        prev_sp = _as_sorted_unique_int(prev.get("spikes", []))
        curr_sp = _as_sorted_unique_int(ev.get("spikes", []))
        if prev_sp.size > 0 and curr_sp.size > 0 and float(vol_sr) > 0:
            gap_s = float(int(curr_sp[0]) - int(prev_sp[-1])) / float(vol_sr)
            ev["gap_mode"] = "prev_last_spike_to_curr_first_spike"
        else:
            gap_s = float(s_cur - p_prev) / float(cal_sr) if float(cal_sr) > 0 else np.inf
            ev["gap_mode"] = "fallback_prev_peak_to_curr_start_cal"
        gap_ok = bool(np.isfinite(gap_s) and (gap_s > gap_thr_s))
        ev["gap_s"] = float(gap_s) if np.isfinite(gap_s) else np.nan
        ev["gap_pass"] = bool(gap_ok)

        prev_tail_raw = np.asarray(cal[p_prev:e_prev + 1], dtype=float)
        prev_tail_good = np.isfinite(prev_tail_raw)
        if np.any(prev_tail_good):
            rel_candidates = np.where(prev_tail_good)[0]
            rel_min = int(rel_candidates[np.argmin(prev_tail_raw[prev_tail_good])])
            min_idx = int(p_prev + rel_min)
            w0 = int(max(0, min_idx - 1))
            w1 = int(min(cal.size - 1, min_idx + 1))
            local3 = np.asarray(cal[w0:w1 + 1], dtype=float)
            local3 = local3[np.isfinite(local3)]
            if local3.size > 0:
                prev_tail_min = float(np.nanmean(local3))
            else:
                prev_tail_min = float(prev_tail_raw[rel_min])
            ev["tail_min_idx"] = int(min_idx)
            ev["tail_min_local_window"] = f"{w0}:{w1}"
        else:
            prev_tail_min = np.nan

        curr_seg = np.asarray(cal[s_cur:e_cur + 1], dtype=float)
        curr_seg = curr_seg[np.isfinite(curr_seg)]
        curr_max = float(np.nanmax(curr_seg)) if curr_seg.size > 0 else np.nan

        ev["tail_min_prev"] = prev_tail_min
        ev["curr_max"] = curr_max

        if (not np.isfinite(prev_tail_min)) or (not np.isfinite(curr_max)) or (curr_max <= 0):
            ev["include"] = bool(gap_ok)
            ev["ratio_pass"] = np.nan
            ev["include_reason"] = "gap_gt_type_thr_ratio_input_invalid" if gap_ok else "ratio_input_invalid"
            ev["tail_ratio"] = np.nan
            continue

        ratio = float(abs(prev_tail_min) / curr_max)
        ev["tail_ratio"] = ratio
        ratio_ok = bool(ratio < float(ratio_thr))
        ev["ratio_pass"] = bool(ratio_ok)

        if gap_ok and ratio_ok:
            ev["include"] = True
            ev["include_reason"] = "gap_gt_type_thr_and_tail_ratio_lt_thr"
        elif gap_ok:
            ev["include"] = True
            ev["include_reason"] = "gap_gt_type_thr"
        elif ratio_ok:
            ev["include"] = True
            ev["include_reason"] = "tail_ratio_lt_thr"
        elif not gap_ok:
            ev["include"] = False
            ev["include_reason"] = "gap_le_type_thr"
        else:
            ev["include"] = False
            ev["include_reason"] = "tail_ratio_ge_thr"

    # Extra rejection rule:
    # Exclude a non-complex event when a later complex/plateau starts very soon after it.
    # Gap is measured from current last spike -> next complex first spike.
    try:
        followup_thr_s = float(complex_followup_exclude_s)
    except Exception:
        followup_thr_s = float(COMPLEX_FOLLOWUP_EXCLUDE_S)
    if np.isfinite(followup_thr_s) and followup_thr_s > 0 and float(vol_sr) > 0:
        n = len(events)
        first_sp = [None] * n
        last_sp = [None] * n
        is_complex_like = [False] * n
        for k, evk in enumerate(events):
            spk = _as_sorted_unique_int(evk.get("spikes", []))
            if spk.size > 0:
                first_sp[k] = int(spk[0])
                last_sp[k] = int(spk[-1])
            t_k = _norm_event_type(evk.get("event_type", "simple"))
            is_complex_like[k] = bool(evk.get("is_complex_event", False) or _is_complex_like_event_type(t_k))

        for i, ev in enumerate(events):
            if not bool(ev.get("include", False)):
                continue
            if is_complex_like[i]:
                # Keep complex/plateau candidates; this rule is for earlier non-complex events.
                continue
            if last_sp[i] is None:
                continue

            min_gap_s = np.inf
            for j in range(i + 1, n):
                if not is_complex_like[j]:
                    continue
                if first_sp[j] is None:
                    continue
                gap_s = float(int(first_sp[j]) - int(last_sp[i])) / float(vol_sr)
                if np.isfinite(gap_s):
                    if gap_s < min_gap_s:
                        min_gap_s = gap_s
                    if gap_s < followup_thr_s:
                        ev["include"] = False
                        ev["include_reason"] = f"complex_followup_lt_{followup_thr_s:.3f}s"
                        ev["complex_followup_gap_s"] = float(gap_s)
                        break
            if np.isfinite(min_gap_s):
                ev["complex_followup_gap_s"] = float(min_gap_s)

    return events

def _resolve_duplicate_event_peaks(events, tol_frames=0):
    """
    If multiple included events point to the same (or near-same) calcium peak index,
    keep the strongest one and exclude the rest.
    """
    if events is None or len(events) <= 1:
        return events, 0

    tol = int(max(0, tol_frames))
    chosen = []
    for i, ev in enumerate(events):
        if not bool(ev.get("include", False)):
            continue
        p = ev.get("peak_idx", np.nan)
        if not np.isfinite(p):
            continue
        chosen.append((i, int(p)))

    if len(chosen) <= 1:
        return events, 0

    chosen = sorted(chosen, key=lambda t: t[1])
    groups = []
    cur = [chosen[0]]
    for it in chosen[1:]:
        if abs(int(it[1]) - int(cur[-1][1])) <= tol:
            cur.append(it)
        else:
            groups.append(cur)
            cur = [it]
    groups.append(cur)

    n_excluded = 0
    for g in groups:
        if len(g) <= 1:
            continue

        def _score(ev):
            return (
                1 if bool(ev.get("clear_peak", False)) else 0,
                float(ev.get("peak_df_f", -np.inf)) if np.isfinite(ev.get("peak_df_f", np.nan)) else -np.inf,
                int(ev.get("n_spikes", 0)),
                int(EVENT_WINNER_PRIORITY.get(_norm_event_type(ev.get("event_type", "simple")), 0)),
            )

        winner_i, _ = max(g, key=lambda t: _score(events[t[0]]))
        winner_peak = int(events[winner_i].get("peak_idx", -1))
        for loser_i, _ in g:
            if loser_i == winner_i:
                continue
            events[loser_i]["include"] = False
            events[loser_i]["include_reason"] = f"duplicate_peak_conflict_w_event_{winner_i}"
            events[loser_i]["duplicate_peak_with"] = int(winner_peak)
            n_excluded += 1

    return events, int(n_excluded)

def _safe_linear_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan
    if np.nanstd(x) <= 1e-12:
        return np.nan
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan

def _safe_linear_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan, np.nan
    if np.nanstd(x) <= 1e-12:
        return np.nan, np.nan
    try:
        p = np.polyfit(x, y, 1)
        return float(p[0]), float(p[1])
    except Exception:
        return np.nan, np.nan

def _sanitize_scale(scale, default=1.0, lo=0.05, hi=20.0):
    try:
        s = float(scale)
    except Exception:
        return float(default)
    if (not np.isfinite(s)) or (s <= 0):
        return float(default)
    return float(min(float(hi), max(float(lo), s)))

def _append_suffix_before_ext(path, suffix):
    if path is None:
        return None
    p = str(path)
    root, ext = os.path.splitext(p)
    return root + str(suffix) + ext

def _has_plateau_events(df):
    if df is None or len(df) == 0:
        return False
    try:
        if "event_type" in df.columns:
            et = df["event_type"].astype(str).str.lower().map(_norm_event_type)
            if bool((et == "plateau").any()):
                return True
        if "is_plateau_event" in df.columns:
            if bool(df["is_plateau_event"].astype(bool).any()):
                return True
    except Exception:
        return False
    return False

def _tag_figure_paths(html_path=None, svg_path=None, pdf_path=None, include_plateau=False, uses_robust_calcium=False):
    plateau_tag = "plateau_yes" if bool(include_plateau) else "plateau_no"
    robust_tag = "robcal_yes" if bool(uses_robust_calcium) else "robcal_no"
    suffix = f"_{plateau_tag}_{robust_tag}"
    return (
        _append_suffix_before_ext(html_path, suffix) if html_path else None,
        _append_suffix_before_ext(svg_path, suffix) if svg_path else None,
        _append_suffix_before_ext(pdf_path, suffix) if pdf_path else None,
    )

def _copy_if_exists(src, dst):
    try:
        if src and dst and os.path.isfile(str(src)):
            os.makedirs(os.path.dirname(str(dst)), exist_ok=True)
            shutil.copy2(str(src), str(dst))
            return True
    except Exception:
        return False
    return False

def _copy_glob_to_dir(pattern, dst_dir):
    try:
        if (pattern is None) or (dst_dir is None):
            return 0
        os.makedirs(str(dst_dir), exist_ok=True)
        n = 0
        for src in glob.glob(str(pattern)):
            if not os.path.isfile(src):
                continue
            dst = os.path.join(str(dst_dir), os.path.basename(str(src)))
            try:
                shutil.copy2(src, dst)
                n += 1
            except Exception:
                pass
        return int(n)
    except Exception:
        return 0

def _safe_write_image(fig, out_path, warn_prefix="Figure", timeout_s=STATIC_IMAGE_TIMEOUT_S):
    if not out_path:
        return False
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass

    err = {}
    tmp_file = None
    _, ext = os.path.splitext(out_path)
    ext = ext if ext else ".svg"
    try:
        fd, tmp_file = tempfile.mkstemp(prefix="plotly_export_", suffix=ext)
        os.close(fd)
    except Exception:
        tmp_file = out_path

    base_name = os.path.basename(out_path)
    print(f"[SAVE] {warn_prefix} start: {base_name}")

    def _worker():
        try:
            fig.write_image(tmp_file)
        except Exception as e:
            err["e"] = e

    try:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        if timeout_s is None:
            t.join()
        else:
            total_wait = max(0.0, float(timeout_s))
            step = 10.0
            waited = 0.0
            while t.is_alive() and waited < total_wait:
                wait_now = min(step, total_wait - waited)
                t.join(wait_now)
                waited += wait_now
                if t.is_alive():
                    print(f"[SAVE] {warn_prefix} waiting: {base_name} ({int(round(waited))}s)")
            if t.is_alive():
                print(
                    f"[WARN] {warn_prefix} static export timed out after "
                    f"{float(timeout_s):.0f}s ({base_name}); skipping."
                )
                return False
    except Exception as e:
        print(f"[WARN] {warn_prefix} static export launch failed ({os.path.basename(out_path)}): {e}")
        return False

    if "e" in err:
        print(f"[WARN] {warn_prefix} static export failed ({base_name}): {err['e']}")
        try:
            if tmp_file and (tmp_file != out_path) and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        return False

    try:
        if tmp_file and (tmp_file != out_path):
            shutil.copy2(tmp_file, out_path)
            try:
                os.remove(tmp_file)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] {warn_prefix} copy-to-destination failed ({base_name}): {e}")
        return False

    print(f"[SAVE] {warn_prefix} done: {base_name}")
    return True

def _save_fig_pair(fig, html_path=None, svg_path=None, warn_prefix="Figure"):
    if html_path:
        fig.write_html(html_path)
    if svg_path:
        _safe_write_image(fig, svg_path, warn_prefix=f"{warn_prefix} SVG")

def _save_fig_triplet(fig, html_path=None, svg_path=None, pdf_path=None, warn_prefix="Figure"):
    if html_path:
        fig.write_html(html_path)
    if svg_path:
        _safe_write_image(fig, svg_path, warn_prefix=f"{warn_prefix} SVG")
    if EXPORT_PDF and pdf_path:
        _safe_write_image(fig, pdf_path, warn_prefix=f"{warn_prefix} PDF")

def _grid_shape(n_items, n_cols=5):
    n_items = int(max(0, n_items))
    n_cols = int(max(1, n_cols))
    n_rows = int(np.ceil(n_items / float(n_cols))) if n_items > 0 else 1
    return n_rows, n_cols

def _short_cell_label(cell_folder):
    try:
        cell = os.path.basename(os.path.normpath(str(cell_folder)))
        fov = os.path.basename(os.path.dirname(os.path.normpath(str(cell_folder))))
        if fov and cell:
            return f"{fov}/{cell}"
        return cell if cell else str(cell_folder)
    except Exception:
        return str(cell_folder)

def _cap_n_spikes(x, cap=SPIKE_COUNT_CAP):
    a = np.asarray(x, float).ravel()
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a)
    if np.any(m):
        out[m] = np.minimum(a[m], float(cap))
    return out

def _spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP):
    vals = list(range(1, int(cap) + 1))
    txt = [str(v) for v in vals[:-1]] + [f"{int(cap)}+"]
    return dict(title_text="# spikes in event", tickmode="array", tickvals=vals, ticktext=txt)

def _beeswarm_x(n_spikes, ev_type, seed=0, jitter=0.12, type_offset=0.08):
    x = np.asarray(n_spikes, float).ravel()
    if x.size == 0:
        return x
    rng = np.random.default_rng(int(seed))
    noise = rng.uniform(-float(jitter), float(jitter), size=x.size)
    et = _norm_event_type(ev_type)
    if et == "simple":
        offset = -float(type_offset)
    elif et == "plateau":
        offset = float(type_offset)
    else:
        offset = 0.0
    return x + noise + offset

def _fit_quality_metrics(y_true, y_pred):
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if np.sum(m) < 2:
        return np.nan, np.nan
    yt = yt[m]
    yp = yp[m]
    rmse = float(np.sqrt(np.nanmean((yt - yp) ** 2)))
    var = float(np.nanvar(yt))
    if (not np.isfinite(var)) or var <= 0:
        return np.nan, rmse
    r2 = float(1.0 - (np.nansum((yt - yp) ** 2) / np.nansum((yt - np.nanmean(yt)) ** 2)))
    return r2, rmse

def _fit_linear_model(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    out = {
        "xx": None,
        "yy": None,
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
    }
    if x.size < 2 or np.unique(x).size < 2:
        return out
    try:
        p = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
        yy = p[0] * xx + p[1]
        y_hat = p[0] * x + p[1]
        r2, rmse = _fit_quality_metrics(y, y_hat)
        out.update(
            {
                "xx": xx,
                "yy": yy,
                "slope": float(p[0]),
                "intercept": float(p[1]),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
            }
        )
    except Exception:
        pass
    return out

def _sat_model(x, y0, a, k):
    x = np.asarray(x, float)
    x = np.maximum(x, 0.0)
    return y0 + (a * x) / (k + x + 1e-12)

def _fit_saturating_model(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    out = {
        "xx": None,
        "yy": None,
        "y0": np.nan,
        "a": np.nan,
        "k": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
    }
    if x.size < 3 or np.unique(x).size < 2 or (curve_fit is None):
        return out
    try:
        x0 = np.maximum(x, 0.0)
        y0_init = float(np.nanpercentile(y, 10))
        a_init = float(np.nanmax(y) - y0_init)
        if not np.isfinite(a_init):
            a_init = 1.0
        if abs(a_init) < 1e-6:
            a_init = 1.0
        k_init = float(np.nanmedian(np.unique(x0)))
        if (not np.isfinite(k_init)) or k_init <= 0:
            k_init = 1.0
        p0 = [y0_init, a_init, k_init]
        bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e3])
        popt, _ = curve_fit(_sat_model, x0, y, p0=p0, bounds=bounds, maxfev=5000)
        xx = np.linspace(float(np.nanmin(x0)), float(np.nanmax(x0)), 100)
        yy = _sat_model(xx, *popt)
        y_hat = _sat_model(x0, *popt)
        r2, rmse = _fit_quality_metrics(y, y_hat)
        out.update(
            {
                "xx": xx,
                "yy": yy,
                "y0": float(popt[0]),
                "a": float(popt[1]),
                "k": float(popt[2]),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
            }
        )
    except Exception:
        pass
    return out

def _fit_line_xy(x, y):
    lin = _fit_linear_model(x, y)
    if lin["xx"] is None:
        return None, None, np.nan
    return lin["xx"], lin["yy"], float(lin["slope"])

def _add_fit_traces(fig, x, y, color, label, legendgroup=None, showlegend=False, row=None, col=None):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return {"linear": None, "nonlinear": None}
    x = x[m]
    y = y[m]
    lin = _fit_linear_model(x, y)
    nonlin = _fit_saturating_model(x, y)

    if lin["xx"] is not None:
        name_lin = f"{label} linear (slope={lin['slope']:.3g}, R2={lin['r2']:.3g}, RMSE={lin['rmse']:.3g})"
        tr_lin = go.Scatter(
            x=lin["xx"],
            y=lin["yy"],
            mode="lines",
            name=name_lin,
            legendgroup=str(legendgroup) if legendgroup is not None else None,
            line=dict(color=color, width=2.0, dash="dash"),
            showlegend=bool(showlegend),
        )
        if row is None or col is None:
            fig.add_trace(tr_lin)
        else:
            fig.add_trace(tr_lin, row=row, col=col)

    if nonlin["xx"] is not None:
        name_nl = (
            f"{label} sat (K={nonlin['k']:.3g}, A={nonlin['a']:.3g}, "
            f"R2={nonlin['r2']:.3g}, RMSE={nonlin['rmse']:.3g})"
        )
        tr_nl = go.Scatter(
            x=nonlin["xx"],
            y=nonlin["yy"],
            mode="lines",
            name=name_nl,
            legendgroup=str(legendgroup) if legendgroup is not None else None,
            line=dict(color=color, width=2.0, dash="dot"),
            showlegend=bool(showlegend),
        )
        if row is None or col is None:
            fig.add_trace(tr_nl)
        else:
            fig.add_trace(tr_nl, row=row, col=col)

    return {"linear": lin, "nonlinear": nonlin}

def _fit_models_text(prefix, fit_out):
    if fit_out is None:
        return ""
    parts = []
    lin = fit_out.get("linear", None)
    if isinstance(lin, dict) and lin.get("xx", None) is not None:
        parts.append(f"{prefix} lin m={lin['slope']:.3g},R2={lin['r2']:.3g}")
    nl = fit_out.get("nonlinear", None)
    if isinstance(nl, dict) and nl.get("xx", None) is not None:
        parts.append(f"{prefix} sat K={nl['k']:.3g},A={nl['a']:.3g},R2={nl['r2']:.3g}")
    return " | ".join(parts)

def _add_beeswarm_points_and_fit(fig, sub_df, ycol, ev_type, color, label, showlegend=False, row=None, col=None, seed=0):
    x_raw = np.asarray(sub_df.get("n_spikes", np.array([], dtype=float)), dtype=float).ravel()
    x_raw = _cap_n_spikes(x_raw, cap=SPIKE_COUNT_CAP)
    y = np.asarray(sub_df.get(ycol, np.array([], dtype=float)), dtype=float).ravel()
    m = np.isfinite(x_raw) & np.isfinite(y)
    if not np.any(m):
        return
    x_raw = x_raw[m]
    y = y[m]
    x_plot = _beeswarm_x(x_raw, ev_type=ev_type, seed=seed, jitter=0.12, type_offset=0.08)

    # Violin behind beeswarm points (narrow; no box), plus horizontal mean line.
    if str(color).lower() == "red":
        vfill = "rgba(220,20,60,0.20)"
    elif str(color).lower() == "black":
        vfill = "rgba(0,0,0,0.16)"
    else:
        vfill = "rgba(100,100,100,0.18)"

    tr_v = go.Violin(
        x=x_raw,
        y=y,
        name=f"{label} dist",
        legendgroup=str(ev_type),
        line_color=color,
        fillcolor=vfill,
        points=False,
        box_visible=False,
        meanline_visible=False,
        width=0.50,
        showlegend=False,
    )
    if row is None or col is None:
        fig.add_trace(tr_v)
    else:
        fig.add_trace(tr_v, row=row, col=col)

    # Mean per beeswarm group (per spike-count bin), not one global type-level mean.
    x_groups = np.unique(np.asarray(x_raw, int))
    for xg in x_groups:
        mg = (np.asarray(x_raw, int) == int(xg))
        if not np.any(mg):
            continue
        yg = y[mg]
        if yg.size == 0 or not np.any(np.isfinite(yg)):
            continue
        y_mean_g = float(np.nanmean(yg))
        x_center = float(_beeswarm_x(np.array([xg], dtype=float), ev_type=ev_type, seed=seed, jitter=0.0, type_offset=0.08)[0])
        half_w = 0.11
        tr_mean_g = go.Scatter(
            x=[x_center - half_w, x_center + half_w],
            y=[y_mean_g, y_mean_g],
            mode="lines",
            name=f"{label} mean (per n_spikes)",
            legendgroup=str(ev_type),
            line=dict(color=color, width=1.4, dash="dot"),
            showlegend=False,
        )
        if row is None or col is None:
            fig.add_trace(tr_mean_g)
        else:
            fig.add_trace(tr_mean_g, row=row, col=col)

    tr_pts = go.Scatter(
        x=x_plot,
        y=y,
        mode="markers",
        name=label,
        legendgroup=str(ev_type),
        marker=dict(symbol="circle-open", size=8, color=color, line=dict(color=color, width=1.4)),
        showlegend=bool(showlegend),
    )
    if row is None or col is None:
        fig.add_trace(tr_pts)
    else:
        fig.add_trace(tr_pts, row=row, col=col)

    _add_fit_traces(
        fig=fig,
        x=x_raw,
        y=y,
        color=color,
        label=label,
        legendgroup=str(ev_type),
        showlegend=bool(showlegend),
        row=row,
        col=col,
    )

def _save_summary_subplots(metrics_df, save_html=None, save_svg=None, title_prefix="Chosen calcium events vs spike count"):
    if save_html is None and save_svg is None:
        return
    if metrics_df is None or len(metrics_df) == 0:
        return

    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}

    panel_specs = [
        ("peak_df_f", "Peak (dF/F)", "peak_df_f"),
        ("peak_df_f_global", "Peak (global-baseline dF/F)", "peak_df_f_global"),
        (
            "peak_norm_complex4_mean",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "peak_norm_complex4_mean",
        ),
        ("peak_z", "Peak (z-score)", "peak_z"),
        ("hwhm_s", "HWHM (s)", "hwhm_s"),
        ("auc", "AUC", "auc"),
        ("decay_time_s", "Decay time (s)", "decay_time_s"),
        ("tau_decay_s", "Decay tau (s)", "tau_decay_s"),
    ]
    for col, ytitle, tag in panel_specs:
        fig = go.Figure()
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
        for ev_type in EVENT_TYPES:
            sub = metrics_df[metrics_df["event_type"] == ev_type] if "event_type" in metrics_df.columns else metrics_df.iloc[0:0]
            _add_beeswarm_points_and_fit(
                fig=fig,
                sub_df=sub,
                ycol=col,
                ev_type=ev_type,
                color=color_map[ev_type],
                label=label_map[ev_type],
                showlegend=True,
                row=None,
                col=None,
                seed=1000 + (17 if ev_type == "simple" else 31) + len(tag),
            )
        fig.update_layout(
            template="simple_white",
            width=850,
            height=650,
            title=f"{title_prefix} | {ytitle}",
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        fig.update_yaxes(title_text=ytitle)
        html_out = _append_suffix_before_ext(save_html, f"_subplot_{tag}") if save_html else None
        svg_out = _append_suffix_before_ext(save_svg, f"_subplot_{tag}") if save_svg else None
        _save_fig_pair(fig, html_out, svg_out, warn_prefix="Summary subplot")

    # Violin subplot (by spike bins)
    fig_v = go.Figure()
    violin_categories = []
    vcol = "peak_z"
    vdf = metrics_df.copy()
    vdf = vdf[np.isfinite(vdf.get(vcol, np.nan)) & np.isfinite(vdf.get("n_spikes", np.nan))]
    if len(vdf) > 0:
        vdf["n_spikes"] = vdf["n_spikes"].astype(int)
        vdf["n_spikes_cap"] = _cap_n_spikes(vdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        simple = vdf[vdf["event_type"] == "simple"].copy()
        complex_df = vdf[vdf["event_type"] == "complex"].copy()
        plateau_df = vdf[vdf["event_type"] == "plateau"].copy()

        simple_counts = sorted(simple["n_spikes_cap"].unique().tolist()) if len(simple) else []
        complex_counts = sorted(complex_df["n_spikes_cap"].unique().tolist()) if len(complex_df) else []
        plateau_counts = sorted(plateau_df["n_spikes_cap"].unique().tolist()) if len(plateau_df) else []
        shared_counts = sorted(set(simple_counts).union(set(complex_counts)).union(set(plateau_counts)))
        violin_categories = list(shared_counts)
        simple_means = simple.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(simple) else {}
        complex_means = complex_df.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(complex_df) else {}
        plateau_means = plateau_df.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(plateau_df) else {}

        if len(simple):
            fig_v.add_trace(
                go.Violin(
                    x=simple["n_spikes_cap"],
                    y=simple[vcol],
                    name="Simple/Burst",
                    line_color="red",
                    fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(complex_df):
            fig_v.add_trace(
                go.Violin(
                    x=complex_df["n_spikes_cap"],
                    y=complex_df[vcol],
                    name="Complex",
                    line_color="black",
                    fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(plateau_df):
            fig_v.add_trace(
                go.Violin(
                    x=plateau_df["n_spikes_cap"],
                    y=plateau_df[vcol],
                    name="Plateau",
                    line_color="purple",
                    fillcolor="rgba(128,0,128,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        fig_v.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        y_max = float(np.nanmax(vdf[vcol])) if np.any(np.isfinite(vdf[vcol])) else 1.0
        y_min = float(np.nanmin(vdf[vcol])) if np.any(np.isfinite(vdf[vcol])) else 0.0
        y_span = max(1e-6, y_max - y_min)
        y_text = y_max + 0.08 * y_span
        fig_v.update_yaxes(range=[y_min - 0.05 * y_span, y_max + 0.22 * y_span])

        fit_s = _add_fit_traces(
            fig=fig_v,
            x=simple["n_spikes_cap"].values if len(simple) else [],
            y=simple[vcol].values if len(simple) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=True,
        )
        fit_c = _add_fit_traces(
            fig=fig_v,
            x=complex_df["n_spikes_cap"].values if len(complex_df) else [],
            y=complex_df[vcol].values if len(complex_df) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=True,
        )

        fit_txt = []
        s_txt = _fit_models_text("simple", fit_s)
        c_txt = _fit_models_text("complex", fit_c)
        if s_txt:
            fit_txt.append(s_txt)
        if c_txt:
            fit_txt.append(c_txt)
        if len(fit_txt) > 0:
            fig_v.add_annotation(
                x=float(SPIKE_COUNT_CAP),
                y=y_max + 0.19 * y_span,
                text=" | ".join(fit_txt),
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10, color="black"),
            )

        for n in simple_counts:
            if n in simple_means:
                fig_v.add_annotation(
                    x=float(n) - 0.08,
                    y=y_text,
                    text=f"mean={float(simple_means[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="red"),
                )
        for n in complex_counts:
            if n in complex_means:
                fig_v.add_annotation(
                    x=float(n) + 0.08,
                    y=y_text,
                    text=f"mean={float(complex_means[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="black"),
                )
    fig_v.update_layout(
        template="simple_white",
        width=max(1200, 120 * max(1, len(violin_categories)) + 450),
        height=650,
        title=f"{title_prefix} | Peak (z-score) violin by spike count",
        legend=dict(orientation="h"),
        violinmode="overlay",
        violingap=0.02,
    )
    fig_v.update_yaxes(title_text="Peak (z-score)")
    html_out = _append_suffix_before_ext(save_html, "_subplot_peak_z_violin") if save_html else None
    svg_out = _append_suffix_before_ext(save_svg, "_subplot_peak_z_violin") if save_svg else None
    _save_fig_pair(fig_v, html_out, svg_out, warn_prefix="Summary violin subplot")

    # Additional violin subplot: peak amplitude from global baseline
    fig_vg = go.Figure()
    vgcol = "peak_df_f_global"
    vgdf = metrics_df.copy()
    if vgcol not in vgdf.columns:
        vgdf[vgcol] = np.nan
    vgdf = vgdf[np.isfinite(vgdf.get(vgcol, np.nan)) & np.isfinite(vgdf.get("n_spikes", np.nan))]
    if len(vgdf) > 0:
        vgdf["n_spikes"] = vgdf["n_spikes"].astype(int)
        vgdf["n_spikes_cap"] = _cap_n_spikes(vgdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        s2 = vgdf[vgdf["event_type"] == "simple"].copy()
        c2 = vgdf[vgdf["event_type"] == "complex"].copy()
        p2 = vgdf[vgdf["event_type"] == "plateau"].copy()

        if len(s2):
            fig_vg.add_trace(
                go.Violin(
                    x=s2["n_spikes_cap"], y=s2[vgcol], name="Simple/Burst",
                    line_color="red", fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True, meanline_visible=True, points=False, width=0.95,
                )
            )
        if len(c2):
            fig_vg.add_trace(
                go.Violin(
                    x=c2["n_spikes_cap"], y=c2[vgcol], name="Complex",
                    line_color="black", fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True, meanline_visible=True, points=False, width=0.95,
                )
            )
        if len(p2):
            fig_vg.add_trace(
                go.Violin(
                    x=p2["n_spikes_cap"], y=p2[vgcol], name="Plateau",
                    line_color="purple", fillcolor="rgba(128,0,128,0.25)",
                    box_visible=True, meanline_visible=True, points=False, width=0.95,
                )
            )
        _add_fit_traces(
            fig=fig_vg,
            x=s2["n_spikes_cap"].values if len(s2) else [],
            y=s2[vgcol].values if len(s2) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_vg,
            x=c2["n_spikes_cap"].values if len(c2) else [],
            y=c2[vgcol].values if len(c2) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_vg,
            x=p2["n_spikes_cap"].values if len(p2) else [],
            y=p2[vgcol].values if len(p2) else [],
            color="purple",
            label="Plateau",
            legendgroup="plateau",
            showlegend=True,
        )
    fig_vg.update_layout(
        template="simple_white",
        width=1200,
        height=650,
        title=f"{title_prefix} | Peak (global-baseline dF/F) violin by spike count",
        legend=dict(orientation="h"),
        violinmode="overlay",
        violingap=0.02,
    )
    fig_vg.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
    fig_vg.update_yaxes(title_text="Peak (global-baseline dF/F)")
    html_out = _append_suffix_before_ext(save_html, "_subplot_peak_df_f_global_violin") if save_html else None
    svg_out = _append_suffix_before_ext(save_svg, "_subplot_peak_df_f_global_violin") if save_svg else None
    _save_fig_pair(fig_vg, html_out, svg_out, warn_prefix="Summary global-baseline violin subplot")
    if save_html:
        print(f"[SUMMARY-SUBPLOTS] saved HTML subplots with stem: {os.path.splitext(str(save_html))[0]}_subplot_*")
    if save_svg:
        print(f"[SUMMARY-SUBPLOTS] saved SVG subplots with stem: {os.path.splitext(str(save_svg))[0]}_subplot_*")

def _save_decay_time_individual_summary(metrics_df, save_html=None, save_svg=None, save_pdf=None,
                                        title_prefix="Chosen calcium events vs spike count"):
    if save_html is None and save_svg is None and save_pdf is None:
        return
    if metrics_df is None or len(metrics_df) == 0:
        return

    if "decay_time_s" not in metrics_df.columns:
        return

    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}
    fig = go.Figure()
    for ev_type in EVENT_TYPES:
        sub = metrics_df[metrics_df["event_type"] == ev_type] if "event_type" in metrics_df.columns else metrics_df.iloc[0:0]
        _add_beeswarm_points_and_fit(
            fig=fig,
            sub_df=sub,
            ycol="decay_time_s",
            ev_type=ev_type,
            color=color_map[ev_type],
            label=label_map[ev_type],
            showlegend=True,
            row=None,
            col=None,
            seed=2048 + (11 if ev_type == "simple" else 29),
        )

    fig.update_layout(
        template="simple_white",
        width=900,
        height=680,
        title=f"{title_prefix} | Decay time (s) individual summary",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
    fig.update_yaxes(title_text="Decay time (s)")
    _save_fig_triplet(
        fig,
        html_path=save_html,
        svg_path=save_svg,
        pdf_path=save_pdf,
        warn_prefix="Decay-time individual summary",
    )

def _plot_voltage_auc_vs_calcium_peak(metrics_df, save_html=None, save_svg=None, save_pdf=None,
                                      title_prefix="Voltage AUC vs calcium peak"):
    if metrics_df is None or len(metrics_df) == 0:
        return None
    if ("event_type" not in metrics_df.columns) or ("peak_df_f" not in metrics_df.columns):
        return None

    xcol = "vol_auc"
    ycol = "peak_df_f"
    if xcol not in metrics_df.columns:
        return None

    df = metrics_df.copy()
    for c in (xcol, ycol):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])].copy()
    if len(df) == 0:
        return None

    fig = go.Figure()
    type_specs = [
        ("simple", "black", "Simple/Burst"),
        ("complex", "red", "Complex"),
        ("plateau", "purple", "Plateau"),
    ]
    for ev_type, color, label in type_specs:
        sub = df[df["event_type"] == ev_type].copy()
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[xcol],
                y=sub[ycol],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.80),
            )
        )

    fig.update_layout(
        template="simple_white",
        width=1050,
        height=700,
        title=f"{title_prefix}",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="Voltage AUC (baseline-subtracted, positive area, a.u.*s)")
    fig.update_yaxes(title_text="Calcium peak (dF/F)")

    tagged_html, tagged_svg, tagged_pdf = _tag_figure_paths(
        html_path=save_html,
        svg_path=save_svg,
        pdf_path=save_pdf,
        include_plateau=_has_plateau_events(df),
        uses_robust_calcium=False,
    )
    _save_fig_triplet(
        fig,
        html_path=tagged_html,
        svg_path=tagged_svg,
        pdf_path=tagged_pdf,
        warn_prefix="Voltage-AUC-vs-calcium-peak",
    )
    return fig

def _save_summary_subplots_normalized(metrics_df, save_html=None, save_svg=None, title_prefix="Chosen calcium events vs spike count"):
    if save_html is None and save_svg is None:
        return
    if metrics_df is None or len(metrics_df) == 0:
        return

    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}

    panel_specs = [
        ("peak_norm_p95", "Peak normalized by p95", "peak_norm_p95"),
        (
            "peak_norm_complex4_mean",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "peak_norm_complex4_mean",
        ),
        ("peak_z_cal_p8_robust", "Calcium robust z (8th percentile center)", "peak_z_cal_p8_robust"),
        ("vol_peak_z_quiet_robust", "Voltage robust z (quiet windows)", "vol_peak_z_quiet_robust"),
    ]
    for col, ytitle, tag in panel_specs:
        fig = go.Figure()
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
        for ev_type in EVENT_TYPES:
            sub = metrics_df[metrics_df["event_type"] == ev_type] if "event_type" in metrics_df.columns else metrics_df.iloc[0:0]
            _add_beeswarm_points_and_fit(
                fig=fig,
                sub_df=sub,
                ycol=col,
                ev_type=ev_type,
                color=color_map[ev_type],
                label=label_map[ev_type],
                showlegend=True,
                row=None,
                col=None,
                seed=2000 + (17 if ev_type == "simple" else 31) + len(tag),
            )
        fig.update_layout(
            template="simple_white",
            width=850,
            height=650,
            title=f"{title_prefix}_normalized_data | {ytitle}",
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        fig.update_yaxes(title_text=col)
        html_out = _append_suffix_before_ext(save_html, f"_subplot_{tag}") if save_html else None
        svg_out = _append_suffix_before_ext(save_svg, f"_subplot_{tag}") if save_svg else None
        _save_fig_pair(fig, html_out, svg_out, warn_prefix="Normalized summary subplot")

    # Normalized violin by spike bins
    fig_v = go.Figure()
    vcol = "peak_norm_p95"
    violin_categories = []
    vdf = metrics_df.copy()
    vdf = vdf[np.isfinite(vdf.get(vcol, np.nan)) & np.isfinite(vdf.get("n_spikes", np.nan))]
    if len(vdf) > 0:
        vdf["n_spikes"] = vdf["n_spikes"].astype(int)
        vdf["n_spikes_cap"] = _cap_n_spikes(vdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        simple = vdf[vdf["event_type"] == "simple"].copy()
        complex_df = vdf[vdf["event_type"] == "complex"].copy()
        plateau_df = vdf[vdf["event_type"] == "plateau"].copy()

        simple_counts = sorted(simple["n_spikes_cap"].unique().tolist()) if len(simple) else []
        complex_counts = sorted(complex_df["n_spikes_cap"].unique().tolist()) if len(complex_df) else []
        plateau_counts = sorted(plateau_df["n_spikes_cap"].unique().tolist()) if len(plateau_df) else []
        shared_counts = sorted(set(simple_counts).union(set(complex_counts)).union(set(plateau_counts)))
        violin_categories = list(shared_counts)
        simple_means = simple.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(simple) else {}
        complex_means = complex_df.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(complex_df) else {}
        plateau_means = plateau_df.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(plateau_df) else {}

        if len(simple):
            fig_v.add_trace(
                go.Violin(
                    x=simple["n_spikes_cap"],
                    y=simple[vcol],
                    name="Simple/Burst",
                    line_color="red",
                    fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(complex_df):
            fig_v.add_trace(
                go.Violin(
                    x=complex_df["n_spikes_cap"],
                    y=complex_df[vcol],
                    name="Complex",
                    line_color="black",
                    fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(plateau_df):
            fig_v.add_trace(
                go.Violin(
                    x=plateau_df["n_spikes_cap"],
                    y=plateau_df[vcol],
                    name="Plateau",
                    line_color="purple",
                    fillcolor="rgba(128,0,128,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        fig_v.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        y_max = float(np.nanmax(vdf[vcol])) if np.any(np.isfinite(vdf[vcol])) else 1.0
        y_min = float(np.nanmin(vdf[vcol])) if np.any(np.isfinite(vdf[vcol])) else 0.0
        y_span = max(1e-6, y_max - y_min)
        y_text = y_max + 0.08 * y_span
        fig_v.update_yaxes(range=[y_min - 0.05 * y_span, y_max + 0.22 * y_span])

        _add_fit_traces(
            fig=fig_v,
            x=simple["n_spikes_cap"].values if len(simple) else [],
            y=simple[vcol].values if len(simple) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_v,
            x=complex_df["n_spikes_cap"].values if len(complex_df) else [],
            y=complex_df[vcol].values if len(complex_df) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_v,
            x=plateau_df["n_spikes_cap"].values if len(plateau_df) else [],
            y=plateau_df[vcol].values if len(plateau_df) else [],
            color="purple",
            label="Plateau",
            legendgroup="plateau",
            showlegend=True,
        )

        for n in simple_counts:
            if n in simple_means:
                fig_v.add_annotation(
                    x=float(n) - 0.08,
                    y=y_text,
                    text=f"mean={float(simple_means[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="red"),
                )
        for n in complex_counts:
            if n in complex_means:
                fig_v.add_annotation(
                    x=float(n) + 0.08,
                    y=y_text,
                    text=f"mean={float(complex_means[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="black"),
                )
    fig_v.update_layout(
        template="simple_white",
        width=max(1200, 120 * max(1, len(violin_categories)) + 450),
        height=650,
        title=f"{title_prefix}_normalized_data | Peak normalized distribution by spike count",
        legend=dict(orientation="h"),
        violinmode="overlay",
        violingap=0.02,
    )
    fig_v.update_yaxes(title_text=vcol)
    html_out = _append_suffix_before_ext(save_html, "_subplot_peak_norm_p95_violin") if save_html else None
    svg_out = _append_suffix_before_ext(save_svg, "_subplot_peak_norm_p95_violin") if save_svg else None
    _save_fig_pair(fig_v, html_out, svg_out, warn_prefix="Normalized violin subplot")

    # Separate violin + fit-line plot for complex-4 normalized response
    fig_v_c4 = go.Figure()
    vcol_c4 = "peak_norm_complex4_mean"
    violin_categories_c4 = []
    vdf_c4 = metrics_df.copy()
    vdf_c4 = vdf_c4[np.isfinite(vdf_c4.get(vcol_c4, np.nan)) & np.isfinite(vdf_c4.get("n_spikes", np.nan))]
    if len(vdf_c4) > 0:
        vdf_c4["n_spikes"] = vdf_c4["n_spikes"].astype(int)
        vdf_c4["n_spikes_cap"] = _cap_n_spikes(vdf_c4["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        simple_c4 = vdf_c4[vdf_c4["event_type"] == "simple"].copy()
        complex_c4 = vdf_c4[vdf_c4["event_type"] == "complex"].copy()
        plateau_c4 = vdf_c4[vdf_c4["event_type"] == "plateau"].copy()
        simple_counts_c4 = sorted(simple_c4["n_spikes_cap"].unique().tolist()) if len(simple_c4) else []
        complex_counts_c4 = sorted(complex_c4["n_spikes_cap"].unique().tolist()) if len(complex_c4) else []
        plateau_counts_c4 = sorted(plateau_c4["n_spikes_cap"].unique().tolist()) if len(plateau_c4) else []
        violin_categories_c4 = sorted(set(simple_counts_c4).union(set(complex_counts_c4)).union(set(plateau_counts_c4)))
        simple_means_c4 = simple_c4.groupby("n_spikes_cap")[vcol_c4].mean().to_dict() if len(simple_c4) else {}
        complex_means_c4 = complex_c4.groupby("n_spikes_cap")[vcol_c4].mean().to_dict() if len(complex_c4) else {}
        plateau_means_c4 = plateau_c4.groupby("n_spikes_cap")[vcol_c4].mean().to_dict() if len(plateau_c4) else {}

        if len(simple_c4):
            fig_v_c4.add_trace(
                go.Violin(
                    x=simple_c4["n_spikes_cap"],
                    y=simple_c4[vcol_c4],
                    name="Simple/Burst",
                    line_color="red",
                    fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(complex_c4):
            fig_v_c4.add_trace(
                go.Violin(
                    x=complex_c4["n_spikes_cap"],
                    y=complex_c4[vcol_c4],
                    name="Complex",
                    line_color="black",
                    fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        if len(plateau_c4):
            fig_v_c4.add_trace(
                go.Violin(
                    x=plateau_c4["n_spikes_cap"],
                    y=plateau_c4[vcol_c4],
                    name="Plateau",
                    line_color="purple",
                    fillcolor="rgba(128,0,128,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                )
            )
        fig_v_c4.update_xaxes(**_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))

        _add_fit_traces(
            fig=fig_v_c4,
            x=simple_c4["n_spikes_cap"].values if len(simple_c4) else [],
            y=simple_c4[vcol_c4].values if len(simple_c4) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_v_c4,
            x=complex_c4["n_spikes_cap"].values if len(complex_c4) else [],
            y=complex_c4[vcol_c4].values if len(complex_c4) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=True,
        )
        _add_fit_traces(
            fig=fig_v_c4,
            x=plateau_c4["n_spikes_cap"].values if len(plateau_c4) else [],
            y=plateau_c4[vcol_c4].values if len(plateau_c4) else [],
            color="purple",
            label="Plateau",
            legendgroup="plateau",
            showlegend=True,
        )

        y_max_c4 = float(np.nanmax(vdf_c4[vcol_c4])) if np.any(np.isfinite(vdf_c4[vcol_c4])) else 1.0
        y_min_c4 = float(np.nanmin(vdf_c4[vcol_c4])) if np.any(np.isfinite(vdf_c4[vcol_c4])) else 0.0
        y_span_c4 = max(1e-6, y_max_c4 - y_min_c4)
        y_text_c4 = y_max_c4 + 0.08 * y_span_c4
        fig_v_c4.update_yaxes(range=[y_min_c4 - 0.05 * y_span_c4, y_max_c4 + 0.22 * y_span_c4])
        for n in simple_counts_c4:
            if n in simple_means_c4:
                fig_v_c4.add_annotation(
                    x=float(n) - 0.08,
                    y=y_text_c4,
                    text=f"mean={float(simple_means_c4[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="red"),
                )
        for n in complex_counts_c4:
            if n in complex_means_c4:
                fig_v_c4.add_annotation(
                    x=float(n) + 0.08,
                    y=y_text_c4,
                    text=f"mean={float(complex_means_c4[n]):.3f}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="black"),
                )

    fig_v_c4.update_layout(
        template="simple_white",
        width=max(1200, 120 * max(1, len(violin_categories_c4)) + 450),
        height=650,
        title=f"{title_prefix}_normalized_data | Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)} distribution by spike count",
        legend=dict(orientation="h"),
        violinmode="overlay",
        violingap=0.02,
    )
    fig_v_c4.update_yaxes(title_text=vcol_c4)
    html_out_c4 = _append_suffix_before_ext(save_html, "_subplot_peak_norm_complex4_mean_violin") if save_html else None
    svg_out_c4 = _append_suffix_before_ext(save_svg, "_subplot_peak_norm_complex4_mean_violin") if save_svg else None
    _save_fig_pair(fig_v_c4, html_out_c4, svg_out_c4, warn_prefix="Normalized violin subplot")
    if save_html:
        print(f"[SUMMARY-SUBPLOTS-NORM] saved HTML subplots with stem: {os.path.splitext(str(save_html))[0]}_subplot_*")
    if save_svg:
        print(f"[SUMMARY-SUBPLOTS-NORM] saved SVG subplots with stem: {os.path.splitext(str(save_svg))[0]}_subplot_*")

def _plot_summary_normalized(metrics_df, save_html=None, save_svg=None, show_plot=False,
                             title_prefix="Chosen calcium events vs spike count",
                             save_subplots=False):
    if metrics_df is None or len(metrics_df) == 0:
        return None

    n_cells = int(metrics_df["cell_folder"].nunique()) if "cell_folder" in metrics_df.columns else np.nan
    n_complex = int((metrics_df["event_type"] == "complex").sum()) if "event_type" in metrics_df.columns else 0
    n_simple = int((metrics_df["event_type"] == "simple").sum()) if "event_type" in metrics_df.columns else 0
    n_plateau = int((metrics_df["event_type"] == "plateau").sum()) if "event_type" in metrics_df.columns else 0

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Peak normalized by p95 (peak_norm_p95)",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "Calcium robust z (8th percentile center)",
            "Voltage robust z (quiet windows)",
            "Peak normalized distribution by event type (p95)",
            "",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}

    panels = [
        (1, 1, "peak_norm_p95", "peak_norm_p95"),
        (1, 2, "peak_norm_complex4_mean", "peak_norm_complex4_mean"),
        (1, 3, "peak_z_cal_p8_robust", "peak_z_cal_p8_robust"),
        (2, 1, "vol_peak_z_quiet_robust", "vol_peak_z_quiet_robust"),
    ]
    for r, c, ycol, ytitle in panels:
        if ycol not in metrics_df.columns:
            metrics_df[ycol] = np.nan
        for ev_type in EVENT_TYPES:
            sub = metrics_df[metrics_df["event_type"] == ev_type] if "event_type" in metrics_df.columns else metrics_df.iloc[0:0]
            _add_beeswarm_points_and_fit(
                fig=fig,
                sub_df=sub,
                ycol=ycol,
                ev_type=ev_type,
                color=color_map[ev_type],
                label=label_map[ev_type],
                showlegend=(r == 1 and c == 1),
                row=r, col=c,
                seed=3000 + (17 if ev_type == "simple" else 31) + (10 * r + c),
            )
        fig.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        fig.update_yaxes(title_text=ytitle, row=r, col=c)

    vcol = "peak_norm_p95"
    violin_categories = []
    if vcol in metrics_df.columns and "event_type" in metrics_df.columns:
        vdf = metrics_df[np.isfinite(metrics_df[vcol])].copy()
        if len(vdf) > 0:
            vdf = vdf[np.isfinite(vdf["n_spikes"])].copy()
            if len(vdf) > 0:
                vdf["n_spikes"] = vdf["n_spikes"].astype(int)
                vdf["n_spikes_cap"] = _cap_n_spikes(vdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
                simple = vdf[vdf["event_type"] == "simple"].copy()
                complex_df = vdf[vdf["event_type"] == "complex"].copy()
                plateau_df = vdf[vdf["event_type"] == "plateau"].copy()

                simple_counts = sorted(simple["n_spikes_cap"].unique().tolist()) if len(simple) else []
                complex_counts = sorted(complex_df["n_spikes_cap"].unique().tolist()) if len(complex_df) else []
                plateau_counts = sorted(plateau_df["n_spikes_cap"].unique().tolist()) if len(plateau_df) else []
                shared_counts = sorted(set(simple_counts).union(set(complex_counts)).union(set(plateau_counts)))
                violin_categories = list(shared_counts)

                if len(simple) > 0:
                    fig.add_trace(
                        go.Violin(
                            x=simple["n_spikes_cap"],
                            y=simple[vcol],
                            name="Simple/Burst",
                            line_color="red",
                            fillcolor="rgba(220,20,60,0.35)",
                            box_visible=True,
                            meanline_visible=True,
                            points=False,
                            width=0.95,
                            showlegend=False,
                        ),
                        row=2, col=2,
                    )
                if len(complex_df) > 0:
                    fig.add_trace(
                        go.Violin(
                            x=complex_df["n_spikes_cap"],
                            y=complex_df[vcol],
                            name="Complex",
                            line_color="black",
                            fillcolor="rgba(0,0,0,0.25)",
                            box_visible=True,
                            meanline_visible=True,
                            points=False,
                            width=0.95,
                            showlegend=False,
                        ),
                        row=2, col=2,
                    )
                if len(plateau_df) > 0:
                    fig.add_trace(
                        go.Violin(
                            x=plateau_df["n_spikes_cap"],
                            y=plateau_df[vcol],
                            name="Plateau",
                            line_color="purple",
                            fillcolor="rgba(128,0,128,0.25)",
                            box_visible=True,
                            meanline_visible=True,
                            points=False,
                            width=0.95,
                            showlegend=False,
                        ),
                        row=2, col=2,
                    )

                _add_fit_traces(
                    fig=fig,
                    x=simple["n_spikes_cap"].values if len(simple) else [],
                    y=simple[vcol].values if len(simple) else [],
                    color="red",
                    label="Simple/Burst",
                    legendgroup="simple",
                    showlegend=False,
                    row=2,
                    col=2,
                )
                _add_fit_traces(
                    fig=fig,
                    x=complex_df["n_spikes_cap"].values if len(complex_df) else [],
                    y=complex_df[vcol].values if len(complex_df) else [],
                    color="black",
                    label="Complex",
                    legendgroup="complex",
                    showlegend=False,
                    row=2,
                    col=2,
                )
                _add_fit_traces(
                    fig=fig,
                    x=plateau_df["n_spikes_cap"].values if len(plateau_df) else [],
                    y=plateau_df[vcol].values if len(plateau_df) else [],
                    color="purple",
                    label="Plateau",
                    legendgroup="plateau",
                    showlegend=False,
                    row=2,
                    col=2,
                )

    fig.update_xaxes(row=2, col=2, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
    fig.update_yaxes(title_text=vcol, row=2, col=2)

    fig.update_layout(
        template="simple_white",
        width=max(1300, 115 * max(1, len(violin_categories)) + 900),
        height=900,
        title=(
            f"{title_prefix}_normalized_data"
            f"<br><sup>n_cells={n_cells} | n_complex_events={n_complex} | n_simple_events={n_simple} | n_plateau_events={n_plateau}</sup>"
        ),
        violinmode="overlay",
        violingap=0.02,
    )

    if save_html:
        fig.write_html(save_html)
    if save_svg:
        _safe_write_image(fig, save_svg, warn_prefix="Normalized summary SVG")

    if show_plot:
        fig.show()
    if save_subplots:
        _save_summary_subplots_normalized(
            metrics_df=metrics_df,
            save_html=save_html,
            save_svg=save_svg,
            title_prefix=title_prefix,
        )
    return fig

def _compute_robust_scaling_from_cs(trace_vol, trace_cal, class_events, event_rows,
                                    vol_sr=500.0, cal_sr=30.0, pre_baseline_s=0.5):
    """
    Trace-level robust dF/F approximation using CS events.
    We use per-CS-event scatter of (F0, dF) and slope(dF vs F0),
    then scale = slope / mean_raw_dF_CS.
    """
    v = np.asarray(trace_vol, float).ravel()
    c = np.asarray(trace_cal, float).ravel()
    nv = v.size
    nc = c.size

    out = {
        "scale_vol": 1.0,
        "scale_cal": 1.0,
        "n_cs_used": 0,
        "slope_vol": np.nan,
        "intercept_vol": np.nan,
        "slope_cal": np.nan,
        "intercept_cal": np.nan,
        "raw_cs_amp_vol": np.nan,
        "raw_cs_amp_cal": np.nan,
        "f0_vol": np.array([], dtype=float),
        "df_vol": np.array([], dtype=float),
        "f0_cal": np.array([], dtype=float),
        "df_cal": np.array([], dtype=float),
    }

    if nv == 0 or nc == 0 or len(event_rows) == 0:
        return out

    # Quiet periods are defined only from voltage-labeled activity windows,
    # then projected to calcium indices by time mapping.
    silent_v, silent_c = _build_quiet_masks(
        trace_vol_len=nv,
        trace_cal_len=nc,
        class_events=class_events,
        vol_sr=vol_sr,
        cal_sr=cal_sr,
    )

    g_f0_v = float(np.nanmedian(v[silent_v])) if np.any(silent_v) else float(np.nanmedian(v))
    c_fin = c[np.isfinite(c)]
    g_f0_c = float(np.nanpercentile(c_fin, float(ROBUST_CAL_F0_PERCENTILE))) if c_fin.size > 0 else float(np.nanmedian(c))

    pre_nv = max(1, int(round(float(pre_baseline_s) * float(vol_sr))))
    pre_nc = max(1, int(round(float(pre_baseline_s) * float(cal_sr))))

    cs_events = [ev for ev in event_rows if str(ev.get("event_type", "")) == "complex"]

    f0_v, df_v = [], []
    f0_c, df_c = [], []

    for ev in cs_events:
        sp = _as_sorted_unique_int(ev.get("spikes", []))
        if sp.size == 0:
            continue

        first_sp = int(max(0, min(nv - 1, int(sp[0]))))
        b0v = max(0, first_sp - pre_nv)
        b1v = first_sp
        if b1v > b0v and np.any(np.isfinite(v[b0v:b1v])):
            base_v = float(np.nanmedian(v[b0v:b1v]))
        else:
            base_v = g_f0_v

        w0v = max(0, first_sp - 1)
        w1v = min(nv, first_sp + 2)
        if w1v <= w0v or not np.any(np.isfinite(v[w0v:w1v])):
            continue
        amp_v = float(np.nanmean(v[w0v:w1v]) - base_v)

        p_idx = int(max(0, min(nc - 1, int(ev.get("peak_idx", ev.get("start_idx", 0))))))
        s_idx = int(max(0, min(nc - 1, int(ev.get("start_idx", p_idx)))))
        # Calcium F0 for robust scaling: global percentile baseline (not quiet-mask based).
        base_c = g_f0_c

        w0c = max(0, p_idx - 1)
        w1c = min(nc, p_idx + 2)
        if w1c <= w0c or not np.any(np.isfinite(c[w0c:w1c])):
            continue
        amp_c = float(np.nanmean(c[w0c:w1c]) - base_c)

        f0_v.append(base_v)
        df_v.append(amp_v)
        f0_c.append(base_c)
        df_c.append(amp_c)

    f0_v = np.asarray(f0_v, float)
    df_v = np.asarray(df_v, float)
    f0_c = np.asarray(f0_c, float)
    df_c = np.asarray(df_c, float)

    n_use = int(min(f0_v.size, f0_c.size))
    out["n_cs_used"] = n_use

    slope_v, intercept_v = _safe_linear_fit(f0_v, df_v)
    slope_c, intercept_c = _safe_linear_fit(f0_c, df_c)
    raw_v = float(np.nanmean(df_v)) if df_v.size else np.nan
    raw_c = float(np.nanmean(df_c)) if df_c.size else np.nan

    out["slope_vol"] = slope_v
    out["intercept_vol"] = intercept_v
    out["slope_cal"] = slope_c
    out["intercept_cal"] = intercept_c
    out["raw_cs_amp_vol"] = raw_v
    out["raw_cs_amp_cal"] = raw_c
    out["f0_vol"] = f0_v
    out["df_vol"] = df_v
    out["f0_cal"] = f0_c
    out["df_cal"] = df_c

    sc_v = (slope_v / raw_v) if (np.isfinite(slope_v) and np.isfinite(raw_v) and abs(raw_v) > 1e-12) else np.nan
    sc_c = (slope_c / raw_c) if (np.isfinite(slope_c) and np.isfinite(raw_c) and abs(raw_c) > 1e-12) else np.nan

    out["scale_vol"] = _sanitize_scale(sc_v, default=1.0)
    out["scale_cal"] = _sanitize_scale(sc_c, default=1.0)
    return out

def _save_robust_scaling_diagnostic(cell_folder, pkl_path, robust_info, suffix_tag):
    if robust_info is None:
        return None, None

    f0_v = np.asarray(robust_info.get("f0_vol", []), float)
    df_v = np.asarray(robust_info.get("df_vol", []), float)
    f0_c = np.asarray(robust_info.get("f0_cal", []), float)
    df_c = np.asarray(robust_info.get("df_cal", []), float)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calcium robust dF/F scaling (CS events)", "Voltage robust dF/F scaling (CS events)"),
        horizontal_spacing=0.10,
    )

    def _add_panel(col, x, y, slope, intercept, raw_amp, scale_val, color_pts):
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=color_pts, size=6, opacity=0.8),
                    name="CS events",
                    showlegend=False,
                    hovertemplate="F0=%{x:.4f}<br>dF=%{y:.4f}<extra></extra>",
                ),
                row=1,
                col=col,
            )

        if x.size >= 2 and np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 120)
            yy = slope * xx + intercept
            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    mode="lines",
                    line=dict(color="#d62728", width=2),
                    name="linear fit",
                    showlegend=False,
                    hovertemplate="fit<extra></extra>",
                ),
                row=1,
                col=col,
            )

        txt = (
            f"n={int(x.size)}<br>"
            f"Raw mean dF={raw_amp:.4g}<br>"
            f"Slope={slope:.4g}<br>"
            f"Scale={scale_val:.4g}"
        )
        xref = "x domain" if int(col) == 1 else f"x{int(col)} domain"
        yref = "y domain" if int(col) == 1 else f"y{int(col)} domain"
        fig.add_annotation(
            x=0.98,
            y=0.03,
            xref=xref,
            yref=yref,
            text=txt,
            showarrow=False,
            align="right",
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11),
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
        )
        fig.update_xaxes(title_text="F0", row=1, col=col)
        fig.update_yaxes(title_text="dF", row=1, col=col)

    _add_panel(
        col=1,
        x=f0_c,
        y=df_c,
        slope=float(robust_info.get("slope_cal", np.nan)),
        intercept=float(robust_info.get("intercept_cal", np.nan)),
        raw_amp=float(robust_info.get("raw_cs_amp_cal", np.nan)),
        scale_val=float(robust_info.get("scale_cal", np.nan)),
        color_pts="black",
    )
    _add_panel(
        col=2,
        x=f0_v,
        y=df_v,
        slope=float(robust_info.get("slope_vol", np.nan)),
        intercept=float(robust_info.get("intercept_vol", np.nan)),
        raw_amp=float(robust_info.get("raw_cs_amp_vol", np.nan)),
        scale_val=float(robust_info.get("scale_vol", np.nan)),
        color_pts="#1f77b4",
    )

    fig.update_layout(
        template="simple_white",
        width=1250,
        height=520,
        title=(
            f"{os.path.basename(cell_folder)} | {os.path.basename(pkl_path)}"
            f"<br><sup>Robust dF/F scaling from CS events (voltage quiet windows for voltage; calcium F0 = global p{ROBUST_CAL_F0_PERCENTILE:g})</sup>"
        ),
    )

    html_path = os.path.join(cell_folder, f"robust_dff_scaling_by_{suffix_tag}.html")
    svg_path = os.path.join(cell_folder, f"robust_dff_scaling_by_{suffix_tag}.svg")
    fig.write_html(html_path)
    _safe_write_image(fig, svg_path, warn_prefix="Robust dF/F scaling SVG")
    return html_path, svg_path

def _analyze_single_pkl(cell_folder, pkl_path, cal_sr, vol_sr=500.0, ratio_thr=0.30, trace_cal_override=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    trace_vol = np.asarray(d.get("trace_vol", []), dtype=float).ravel()
    if trace_cal_override is not None:
        trace_cal = np.asarray(trace_cal_override, dtype=float).ravel()
    else:
        trace_cal = np.asarray(d.get("trace_cal", []), dtype=float).ravel()
    if trace_vol.size == 0 or trace_cal.size == 0:
        return pd.DataFrame(), None, pd.DataFrame()

    class_events, _simple_spikes, _complex_spikes = _events_from_saved_labels(
        d,
        trace_vol.size,
        trace_vol=trace_vol,
        vol_sr=vol_sr,
        cs_z_start_thr=CS_Z_START_THR,
        cs_z_end_thr=CS_Z_END_THR,
        non_cs_pad_frames=5,
        simple_isi_ms=SIMPLE_ISI_MS,
    )
    if len(class_events) == 0:
        return pd.DataFrame(), None, pd.DataFrame()

    # Enforce chronological order so "next event" capping is always correct.
    class_events = sorted(
        class_events,
        key=lambda ev: int(ev.get("start_frame", 0)),
    )

    vol_t = np.arange(trace_vol.size, dtype=float) / float(vol_sr)
    cal_t = np.arange(trace_cal.size, dtype=float) / float(cal_sr)
    cal_finite = trace_cal[np.isfinite(trace_cal)]
    cal_global_baseline = float(np.nanmedian(cal_finite)) if cal_finite.size > 0 else np.nan
    mu_low8, sd_low8 = _zscore_low8(trace_cal, low_percentile=CAL_Z_NON_ROBUST_LOW_PERCENTILE)
    quiet_v_mask, _quiet_c_mask = _build_quiet_masks(
        trace_vol_len=trace_vol.size,
        trace_cal_len=trace_cal.size,
        class_events=class_events,
        vol_sr=vol_sr,
        cal_sr=cal_sr,
    )

    # Added normalization diagnostics (without replacing existing metrics):
    # 1) voltage robust z: center/scale from quiet voltage windows only
    # 2) calcium robust z: center at 8th percentile of calcium trace
    if np.any(quiet_v_mask):
        v_quiet_center, v_quiet_scale = _robust_center_scale(trace_vol[quiet_v_mask], center=None)
    else:
        v_quiet_center, v_quiet_scale = _robust_center_scale(trace_vol, center=None)
    cal_p8_center, cal_p8_scale = _calcium_p8_center_scale(trace_cal, p=8.0)

    # Build spike-valid events and use this list for "next event" capping.
    # This guarantees capping is based on the next real spike event, regardless of later inclusion/exclusion.
    usable_events = []
    for ev in class_events:
        ev_spk = _as_sorted_unique_int(ev.get("spikes", []))
        ev_spk = ev_spk[(ev_spk >= 0) & (ev_spk < trace_vol.size)]
        if ev_spk.size == 0:
            continue
        ev_start = int(max(0, min(trace_vol.size - 1, int(ev.get("start_frame", int(ev_spk[0]))))))
        usable_events.append((ev_start, int(ev_spk[0]), ev, ev_spk))

    # Order by first spike timing for robust next-event capping.
    usable_events = sorted(usable_events, key=lambda x: int(x[1]))

    event_rows = []
    for i, (_ev_start, _first_sp, ev, ev_spk) in enumerate(usable_events):

        v_start = int(max(0, min(trace_vol.size - 1, int(ev.get("start_frame", int(ev_spk[0]))))))
        v_end = int(max(v_start, min(trace_vol.size - 1, int(ev.get("end_frame", int(ev_spk[-1]))))))

        c_start = _vol_idx_to_cal_idx(v_start, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal.size)
        c_first_sp = _vol_idx_to_cal_idx(int(ev_spk[0]), vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal.size)
        c_last_sp = _vol_idx_to_cal_idx(int(ev_spk[-1]), vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal.size)

        c_next_start = None
        c_next_start_for_feat = None
        if i < len(usable_events) - 1:
            _next_start, _next_first_sp, next_ev, _next_spk = usable_events[i + 1]
            # Cap by the next event's first spike (not start_frame), so boundaries
            # are spike-consistent even when start_frame extends earlier.
            if _next_spk is not None and np.asarray(_next_spk).size > 0:
                next_v_start = int(max(0, min(trace_vol.size - 1, int(_next_spk[0]))))
            else:
                next_v_start = int(max(0, min(trace_vol.size - 1, int(next_ev.get("start_frame", _next_start)))))

            curr_last_sp = int(ev_spk[-1])
            if next_v_start > curr_last_sp:
                if v_end >= next_v_start:
                    v_end = int(max(v_start, next_v_start - 1))
                c_next_start = _vol_idx_to_cal_idx(next_v_start, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal.size)
                c_next_start_for_feat = c_next_start

        c_end = _vol_idx_to_cal_idx(v_end, vol_sr=vol_sr, cal_sr=cal_sr, cal_len=trace_cal.size)
        # Never end earlier than the event's own last spike in calcium time.
        c_end = int(max(c_end, c_last_sp))
        if c_next_start is not None:
            c_end_cap = int(max(c_start, min(trace_cal.size - 1, c_next_start - 1)))
            if c_end > c_end_cap and c_end_cap > c_start:
                c_end = c_end_cap
            elif c_end_cap <= c_start:
                # Keep capping active even when neighbor maps to the same calcium bin.
                # Use virtual next-start at +1 frame so peak/end remain capped to current frame.
                c_end = int(c_start)
                c_next_start_for_feat = int(min(trace_cal.size - 1, c_start + 1))

        ev_type_norm_now = _norm_event_type(ev.get("event_type", "simple"))
        is_plateau_event = (ev_type_norm_now == "plateau") or (str(ev.get("event_kind", "")).lower() == "plateau") or bool(ev.get("is_plateau_event", False))
        is_complex_event = _is_complex_like_event_type(ev_type_norm_now) or (str(ev.get("event_kind", "")).lower() in ("complex", "plateau")) or bool(ev.get("is_complex_event", False))
        is_burst_event = (str(ev.get("event_kind", "")) == "simple_burst") or is_complex_event

        # Peak search rule:
        # simple  -> after first spike, capped by next event start or +150 ms
        # complex -> after first spike, capped by next event start or +300 ms
        peak_window_s = float(COMPLEX_PEAK_SEARCH_S if is_complex_event else SIMPLE_PEAK_SEARCH_S)
        peak_search_start_idx = int(c_first_sp)
        peak_search_end_idx = None
        c_end_for_feat = c_end
        peak_n = max(1, int(round(float(peak_window_s) * float(cal_sr))))
        peak_search_end_idx = int(min(trace_cal.size - 1, peak_search_start_idx + peak_n))
        if c_next_start_for_feat is not None:
            peak_search_end_idx = int(min(peak_search_end_idx, max(peak_search_start_idx, c_next_start_for_feat - 1)))
        c_end_for_feat = int(max(c_end_for_feat, peak_search_end_idx))
        if c_next_start_for_feat is not None:
            c_end_for_feat = int(min(c_end_for_feat, max(c_start, c_next_start_for_feat - 1)))

        feat = _calc_event_features(
            trace_cal,
            cal_t,
            start_idx=c_start,
            end_idx=c_end_for_feat,
            next_start_idx=c_next_start_for_feat,
            mu_low8=mu_low8,
            sd_low8=sd_low8,
            max_peak_search_s=peak_window_s,
            pre_baseline_s=PRE_BASELINE_S,
            end_frac=EVENT_END_FRAC,
            use_baseline_frac_end=is_burst_event,
            peak_search_start_idx=peak_search_start_idx,
            peak_search_end_idx=peak_search_end_idx,
            first_spike_idx=c_first_sp,
        )
        if feat is None:
            continue

        event_rows.append({
            "event_idx": len(event_rows),
            "spikes": ev_spk,
            "n_spikes": int(ev_spk.size),
            "event_type": _norm_event_type(ev.get("event_type", "simple")),
            "event_kind": str(ev.get("event_kind", "single")),
            "bound_method": str(ev.get("bound_method", "")),
            "start_frame_v": int(v_start),
            "end_frame_v": int(v_end),
            "is_burst_event": bool(is_burst_event),
            "is_complex_event": bool(is_complex_event),
            "is_plateau_event": bool(is_plateau_event),
            "is_dual_class_event": bool(is_plateau_event and is_complex_event),
            **feat,
        })

    if len(event_rows) == 0:
        return pd.DataFrame(), None, pd.DataFrame()

    event_rows, n_same_start_merged = _merge_events_same_start(event_rows)
    if n_same_start_merged > 0:
        print(f"[INFO] {os.path.basename(pkl_path)} merged same-start events: merged={n_same_start_merged}")

    event_rows = _evaluate_include_rule(
        event_rows,
        trace_cal=trace_cal,
        ratio_thr=ratio_thr,
        cal_sr=cal_sr,
        ratio_max_gap_simple_s=GAP_THR_PREV_SIMPLE_S,
        ratio_max_gap_complex_s=GAP_THR_PREV_COMPLEX_S,
        vol_sr=vol_sr,
    )

    event_rows, n_dup_excluded = _resolve_duplicate_event_peaks(
        event_rows,
        tol_frames=PEAK_DUP_TOL_FRAMES,
    )
    if n_dup_excluded > 0:
        print(f"[INFO] {os.path.basename(pkl_path)} duplicate-peak conflicts resolved: excluded={n_dup_excluded}")

    # 95th percentile reference for peak normalization (per recording / per pkl)
    incl_peak_vals = []
    for ev in event_rows:
        if not bool(ev.get("include", False)):
            continue
        pv = float(ev.get("peak_df_f", np.nan))
        if np.isfinite(pv):
            incl_peak_vals.append(pv)
    incl_peak_vals = np.asarray(incl_peak_vals, dtype=float)
    incl_peak_vals = incl_peak_vals[np.isfinite(incl_peak_vals)]
    if incl_peak_vals.size > 0:
        peak_ref_p95 = float(np.nanpercentile(incl_peak_vals, 95))
    else:
        peak_ref_p95 = np.nan

    # Mean complex-event calcium response for events with exactly 4 spikes.
    # This serves as an additional normalization reference.
    incl_complex4_vals = []
    for ev in event_rows:
        if not bool(ev.get("include", False)):
            continue
        if not _is_complex_like_event_type(ev.get("event_type", "simple")):
            continue
        if int(ev.get("n_spikes", 0)) != int(COMPLEX4_REF_N_SPIKES):
            continue
        if bool(ev.get("peak_before_first_spike", False)):
            continue
        pv = float(ev.get("peak_df_f", np.nan))
        if np.isfinite(pv):
            incl_complex4_vals.append(pv)
    incl_complex4_vals = np.asarray(incl_complex4_vals, dtype=float)
    incl_complex4_vals = incl_complex4_vals[np.isfinite(incl_complex4_vals)]
    if incl_complex4_vals.size > 0:
        peak_ref_complex4_mean = float(np.nanmean(incl_complex4_vals))
    else:
        peak_ref_complex4_mean = np.nan

    include_single_simple = []
    include_simple_burst = []
    include_complex = []
    include_plateau = []
    excluded = []
    metric_rows = []
    all_event_rows = []
    suffix = _suffix_from_pkl_name(pkl_path)
    pkl_name = os.path.basename(pkl_path)

    for ev in event_rows:
        sp = _as_sorted_unique_int(ev["spikes"])
        ev_type_norm = _norm_event_type(ev.get("event_type", "simple"))
        if bool(ev.get("is_plateau_event", False)):
            ev_type_norm = "plateau"
        n_sp = int(ev.get("n_spikes", sp.size))
        n_sp_cap = int(min(max(1, n_sp), int(SPIKE_COUNT_CAP)))
        all_event_rows.append({
            "cell_folder": cell_folder,
            "pkl_name": pkl_name,
            "suffix": suffix,
            "event_idx": int(ev.get("event_idx", -1)),
            "event_type": ev_type_norm,
            "n_spikes": int(n_sp),
            "n_spikes_cap": int(n_sp_cap),
            "include": bool(ev.get("include", False)),
            "include_reason": str(ev.get("include_reason", "")),
            "start_idx": int(ev.get("start_idx", -1)),
            "peak_idx": int(ev.get("peak_idx", -1)),
            "end_idx": int(ev.get("end_idx", -1)),
            "gap_s": float(ev.get("gap_s", np.nan)) if np.isfinite(ev.get("gap_s", np.nan)) else np.nan,
            "gap_thr_s": float(ev.get("gap_thr_s", np.nan)) if np.isfinite(ev.get("gap_thr_s", np.nan)) else np.nan,
            "tail_ratio": float(ev.get("tail_ratio", np.nan)) if np.isfinite(ev.get("tail_ratio", np.nan)) else np.nan,
            "complex_followup_gap_s": float(ev.get("complex_followup_gap_s", np.nan)) if np.isfinite(ev.get("complex_followup_gap_s", np.nan)) else np.nan,
            "is_complex_event": bool(ev.get("is_complex_event", False) or _is_complex_like_event_type(ev_type_norm)),
            "is_plateau_event": bool(ev.get("is_plateau_event", False) or (ev_type_norm == "plateau")),
            "is_dual_class_event": bool((ev.get("is_plateau_event", False) or (ev_type_norm == "plateau")) and (ev.get("is_complex_event", False) or _is_complex_like_event_type(ev_type_norm))),
        })
        if bool(ev.get("include", False)):
            ev_type_this = _norm_event_type(ev.get("event_type", "simple"))
            is_plateau_this = bool(ev.get("is_plateau_event", False) or (ev_type_this == "plateau"))
            is_complex_this = bool(ev.get("is_complex_event", False) or _is_complex_like_event_type(ev_type_this))
            # Plateau has priority for choosing spikes.
            if is_plateau_this:
                include_plateau.extend(sp.tolist())
            elif is_complex_this:
                include_complex.extend(sp.tolist())
            else:
                if str(ev.get("event_kind", "single")) == "single":
                    include_single_simple.extend(sp.tolist())
                else:
                    include_simple_burst.extend(sp.tolist())

            peak_before_first = bool(ev.get("peak_before_first_spike", False))
            peak_df_f_out = 0.0 if peak_before_first else float(ev.get("peak_df_f", np.nan))
            peak_z_out = 0.0 if peak_before_first else float(ev.get("peak_z", np.nan))
            peak_raw_out = float(ev.get("peak_raw", np.nan))

            # Voltage event peak (chosen among event spikes)
            vol_peak_idx = int(sp[0]) if sp.size > 0 else -1
            if sp.size > 0:
                try:
                    vol_vals = trace_vol[sp]
                    if np.any(np.isfinite(vol_vals)):
                        rel = int(np.nanargmax(vol_vals))
                        vol_peak_idx = int(sp[rel])
                except Exception:
                    vol_peak_idx = int(sp[0])
            vol_peak_raw = float(trace_vol[vol_peak_idx]) if (0 <= vol_peak_idx < trace_vol.size and np.isfinite(trace_vol[vol_peak_idx])) else np.nan
            if np.isfinite(v_quiet_center) and np.isfinite(v_quiet_scale) and v_quiet_scale > 0 and np.isfinite(vol_peak_raw):
                vol_peak_z_quiet = float((vol_peak_raw - v_quiet_center) / v_quiet_scale)
            else:
                vol_peak_z_quiet = np.nan

            # Voltage AUC over event window (baseline-subtracted, positive area)
            vol_auc = np.nan
            vol_start = int(ev.get("start_frame_v", sp[0] if sp.size > 0 else -1))
            vol_end = int(ev.get("end_frame_v", vol_start))
            if trace_vol.size > 0 and vol_start >= 0:
                vol_start = int(max(0, min(trace_vol.size - 1, vol_start)))
                vol_end = int(max(vol_start, min(trace_vol.size - 1, vol_end)))
                pre_n_v = max(1, int(round(PRE_BASELINE_S * float(vol_sr))))
                vb0 = max(0, vol_start - pre_n_v)
                vb1 = vol_start
                if vb1 > vb0 and np.any(np.isfinite(trace_vol[vb0:vb1])):
                    vol_baseline = float(np.nanmedian(trace_vol[vb0:vb1]))
                elif np.isfinite(v_quiet_center):
                    vol_baseline = float(v_quiet_center)
                elif np.any(np.isfinite(trace_vol)):
                    vol_baseline = float(np.nanmedian(trace_vol))
                else:
                    vol_baseline = 0.0
                if vol_end > vol_start:
                    yv = np.asarray(trace_vol[vol_start:vol_end + 1], float) - float(vol_baseline)
                    yv = np.clip(yv, 0, None)
                    xv = np.arange(vol_start, vol_end + 1, dtype=float) / float(vol_sr)
                    vol_auc = float(np.trapezoid(yv, x=xv))
                else:
                    vol_auc = 0.0

            # Calcium robust z using 8th-percentile center
            if np.isfinite(cal_p8_center) and np.isfinite(cal_p8_scale) and cal_p8_scale > 0 and np.isfinite(peak_raw_out):
                peak_z_cal_p8 = float((peak_raw_out - cal_p8_center) / cal_p8_scale)
            else:
                peak_z_cal_p8 = np.nan

            # Peak normalization using recording-level p95
            if np.isfinite(peak_ref_p95) and peak_ref_p95 > 0 and np.isfinite(peak_df_f_out):
                peak_norm_p95 = float(peak_df_f_out / peak_ref_p95)
            else:
                peak_norm_p95 = np.nan

            # Peak normalization using mean response of complex events with n_spikes==4
            if (
                np.isfinite(peak_ref_complex4_mean)
                and (peak_ref_complex4_mean > 0)
                and np.isfinite(peak_df_f_out)
            ):
                peak_norm_complex4_mean = float(peak_df_f_out / peak_ref_complex4_mean)
            else:
                peak_norm_complex4_mean = np.nan

            # Global-baseline amplitude: peak_raw - median(trace_cal over full recording)
            if np.isfinite(peak_raw_out) and np.isfinite(cal_global_baseline):
                peak_df_f_global = float(peak_raw_out - cal_global_baseline)
            else:
                peak_df_f_global = np.nan

            metric_rows.append({
                "cell_folder": cell_folder,
                "pkl_name": os.path.basename(pkl_path),
                "suffix": _suffix_from_pkl_name(pkl_path),
                "event_idx": int(ev["event_idx"]),
                "event_type": ev_type_norm,
                "n_spikes": int(ev.get("n_spikes", 0)),
                "peak_df_f": peak_df_f_out,
                "peak_df_f_global": peak_df_f_global,
                "peak_z": peak_z_out,
                "peak_z_cal_p8_robust": peak_z_cal_p8,
                "peak_norm_p95": peak_norm_p95,
                "peak_norm_complex4_mean": peak_norm_complex4_mean,
                "hwhm_s": float(ev["hwhm_s"]) if np.isfinite(ev.get("hwhm_s", np.nan)) else np.nan,
                "auc": float(ev.get("auc", np.nan)),
                "rise_time_s": float(ev.get("rise_time_s", np.nan)) if np.isfinite(ev.get("rise_time_s", np.nan)) else np.nan,
                "decay_time_s": float(ev.get("decay_time_s", np.nan)) if np.isfinite(ev.get("decay_time_s", np.nan)) else np.nan,
                "tau_decay_s": float(ev.get("tau_decay_s", np.nan)) if np.isfinite(ev.get("tau_decay_s", np.nan)) else np.nan,
                "tail_ratio": float(ev["tail_ratio"]) if np.isfinite(ev.get("tail_ratio", np.nan)) else np.nan,
                "peak_ref_p95": float(peak_ref_p95) if np.isfinite(peak_ref_p95) else np.nan,
                "peak_ref_complex4_mean": float(peak_ref_complex4_mean) if np.isfinite(peak_ref_complex4_mean) else np.nan,
                "cal_p8_center": float(cal_p8_center) if np.isfinite(cal_p8_center) else np.nan,
                "cal_p8_scale": float(cal_p8_scale) if np.isfinite(cal_p8_scale) else np.nan,
                "cal_global_baseline": float(cal_global_baseline) if np.isfinite(cal_global_baseline) else np.nan,
                "vol_quiet_center": float(v_quiet_center) if np.isfinite(v_quiet_center) else np.nan,
                "vol_quiet_scale": float(v_quiet_scale) if np.isfinite(v_quiet_scale) else np.nan,
                "vol_peak_idx": int(vol_peak_idx),
                "vol_peak_raw": float(vol_peak_raw) if np.isfinite(vol_peak_raw) else np.nan,
                "vol_peak_z_quiet_robust": float(vol_peak_z_quiet) if np.isfinite(vol_peak_z_quiet) else np.nan,
                "vol_auc": float(vol_auc) if np.isfinite(vol_auc) else np.nan,
                "bound_method": str(ev.get("bound_method", "")),
                "include_reason": str(ev.get("include_reason", "")),
                "start_idx": int(ev.get("start_idx", -1)),
                "peak_idx": int(ev.get("peak_idx", -1)),
                "end_idx": int(ev.get("end_idx", -1)),
            })
        else:
            excluded.extend(sp.tolist())

    include_single_simple = _as_sorted_unique_int(include_single_simple)
    include_simple_burst = _as_sorted_unique_int(include_simple_burst)
    include_complex = _as_sorted_unique_int(include_complex)
    include_plateau = _as_sorted_unique_int(include_plateau)
    excluded = _as_sorted_unique_int(excluded)

    include_single_simple = include_single_simple[(include_single_simple >= 0) & (include_single_simple < trace_vol.size)]
    include_simple_burst = include_simple_burst[(include_simple_burst >= 0) & (include_simple_burst < trace_vol.size)]
    include_complex = include_complex[(include_complex >= 0) & (include_complex < trace_vol.size)]
    include_plateau = include_plateau[(include_plateau >= 0) & (include_plateau < trace_vol.size)]
    excluded = excluded[(excluded >= 0) & (excluded < trace_vol.size)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=vol_t,
            y=trace_vol,
            mode="lines",
            name="Voltage trace",
            line=dict(color=VOLTAGE_TRACE_COLOR, width=1.0),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cal_t,
            y=trace_cal,
            mode="lines",
            name="Calcium trace",
            line=dict(color=CALCIUM_TRACE_COLOR, width=1.0),
        ),
        secondary_y=True,
    )

    # Event-window rectangles on voltage axis (all clustered events after boundary detection).
    v_finite = trace_vol[np.isfinite(trace_vol)]
    if v_finite.size > 0:
        v_lo = float(np.nanmin(v_finite))
        v_hi = float(np.nanmax(v_finite))
    else:
        v_lo, v_hi = 0.0, 1.0
    v_span = max(1e-9, (v_hi - v_lo))
    v_y0 = v_lo - 0.02 * v_span
    v_y1 = v_hi + 0.02 * v_span
    for ev in event_rows:
        sv = int(ev.get("start_frame_v", -1))
        evv = int(ev.get("end_frame_v", -1))
        if not (0 <= sv < trace_vol.size):
            continue
        if not (0 <= evv < trace_vol.size):
            continue
        if evv < sv:
            sv, evv = evv, sv
        x0 = float(vol_t[sv])
        x1 = float(vol_t[evv])
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=v_y0,
            y1=v_y1,
            fillcolor="rgba(255, 235, 59, 0.16)",
            line=dict(color="rgba(204, 170, 0, 0.95)", width=1.0),
            layer="below",
        )
    # Add calcium z-score trace (based on low-8% center/scale) to each overlay figure.
    if np.isfinite(sd_low8) and sd_low8 > 0 and np.isfinite(mu_low8):
        cal_z = (np.asarray(trace_cal, float) - float(mu_low8)) / float(sd_low8)
        fig.add_trace(
            go.Scatter(
                x=cal_t,
                y=cal_z,
                mode="lines",
                name=f"Calcium z-score (low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g})",
                line=dict(color="#6f42c1", width=1.2, dash="dot"),
                opacity=0.9,
                hovertemplate="t=%{x:.3f}s<br>Cal z=%{y:.3f}<extra></extra>",
            ),
            secondary_y=True,
        )
    # Keep current z-score and add robust calcium z-score for direct comparison.
    if np.isfinite(cal_p8_scale) and cal_p8_scale > 0 and np.isfinite(cal_p8_center):
        cal_z_robust = (np.asarray(trace_cal, float) - float(cal_p8_center)) / float(cal_p8_scale)
        fig.add_trace(
            go.Scatter(
                x=cal_t,
                y=cal_z_robust,
                mode="lines",
                name="Calcium robust z (p8+MAD)",
                line=dict(color="#8a2be2", width=1.2, dash="dash"),
                opacity=0.9,
                hovertemplate="t=%{x:.3f}s<br>Cal robust z=%{y:.3f}<extra></extra>",
            ),
            secondary_y=True,
        )

    def _add_spike_idx_trace(spike_idx_arr, name, color, size):
        if spike_idx_arr.size == 0:
            return
        txt = [str(int(i)) for i in spike_idx_arr.tolist()]
        fig.add_trace(
            go.Scatter(
                x=vol_t[spike_idx_arr],
                y=trace_vol[spike_idx_arr],
                mode="markers",
                text=txt,
                name=name,
                marker=dict(color=color, size=size, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<br>spike_idx=%{text}<extra></extra>",
            ),
            secondary_y=False,
        )

    if excluded.size:
        _add_spike_idx_trace(excluded, name="Non-chosen spikes", color="gray", size=6)
    if include_single_simple.size:
        _add_spike_idx_trace(include_single_simple, name="Chosen simple single", color="#1f77b4", size=7)
    if include_simple_burst.size:
        _add_spike_idx_trace(include_simple_burst, name="Chosen simple burst", color="green", size=7)
    if include_complex.size:
        _add_spike_idx_trace(include_complex, name="Chosen complex", color="#ff69b4", size=7)
    if include_plateau.size:
        _add_spike_idx_trace(include_plateau, name="Chosen plateau", color="purple", size=8)

    chosen_events = [ev for ev in event_rows if bool(ev.get("include", False))]
    if len(chosen_events) > 0:
        start_events = [ev for ev in chosen_events if 0 <= int(ev.get("start_idx", -1)) < trace_cal.size]
        end_events = [ev for ev in chosen_events if 0 <= int(ev.get("end_idx", -1)) < trace_cal.size]
        peak_events = [ev for ev in chosen_events if 0 <= int(ev.get("peak_idx", -1)) < trace_cal.size]

        s_idx = np.array([int(ev["start_idx"]) for ev in start_events], dtype=int)
        e_idx = np.array([int(ev["end_idx"]) for ev in end_events], dtype=int)
        p_idx = np.array([int(ev["peak_idx"]) for ev in peak_events], dtype=int)

        s_txt = [
            f"event_idx={int(ev.get('event_idx', -1))}<br>start_idx={int(ev.get('start_idx', -1))}"
            for ev in start_events
        ]
        e_txt = [
            f"event_idx={int(ev.get('event_idx', -1))}<br>end_idx={int(ev.get('end_idx', -1))}"
            for ev in end_events
        ]
        p_txt = [
            f"event_idx={int(ev.get('event_idx', -1))}<br>peak_idx={int(ev.get('peak_idx', -1))}"
            for ev in peak_events
        ]

        if s_idx.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=cal_t[s_idx],
                    y=trace_cal[s_idx],
                    mode="markers",
                    text=s_txt,
                    name="Chosen event start",
                    marker=dict(color="black", size=10, symbol="triangle-up"),
                    hovertemplate="t=%{x:.3f}s<br>cal=%{y:.3f}<br>%{text}<extra></extra>",
                ),
                secondary_y=True,
            )
        if e_idx.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=cal_t[e_idx],
                    y=trace_cal[e_idx],
                    mode="markers",
                    text=e_txt,
                    name="Chosen event end",
                    marker=dict(color="#444444", size=10, symbol="triangle-down"),
                    hovertemplate="t=%{x:.3f}s<br>cal=%{y:.3f}<br>%{text}<extra></extra>",
                ),
                secondary_y=True,
            )
        if p_idx.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=cal_t[p_idx],
                    y=trace_cal[p_idx],
                    mode="markers",
                    text=p_txt,
                    name="Chosen event peak",
                    marker=dict(color="purple", size=11, symbol="x"),
                    hovertemplate="t=%{x:.3f}s<br>cal=%{y:.3f}<br>%{text}<extra></extra>",
                ),
                secondary_y=True,
            )

    out_base = os.path.join(cell_folder, f"event_spike_overlay_{suffix}")

    # Voltage-only diagnostic: spikes + event rectangles.
    vfig = go.Figure()
    vfig.add_trace(
        go.Scatter(
            x=vol_t,
            y=trace_vol,
            mode="lines",
            name="Voltage trace",
            line=dict(color=VOLTAGE_TRACE_COLOR, width=1.1),
        )
    )
    for ev in event_rows:
        sv = int(ev.get("start_frame_v", -1))
        evv = int(ev.get("end_frame_v", -1))
        if not (0 <= sv < trace_vol.size):
            continue
        if not (0 <= evv < trace_vol.size):
            continue
        if evv < sv:
            sv, evv = evv, sv
        x0 = float(vol_t[sv])
        x1 = float(vol_t[evv])
        vfig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=v_y0,
            y1=v_y1,
            fillcolor="rgba(255, 235, 59, 0.18)",
            line=dict(color="rgba(204, 170, 0, 0.95)", width=1.1),
            layer="below",
        )
    if excluded.size:
        vfig.add_trace(
            go.Scatter(
                x=vol_t[excluded],
                y=trace_vol[excluded],
                mode="markers",
                name="Non-chosen spikes",
                marker=dict(color="gray", size=6, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<extra></extra>",
            )
        )
    if include_single_simple.size:
        vfig.add_trace(
            go.Scatter(
                x=vol_t[include_single_simple],
                y=trace_vol[include_single_simple],
                mode="markers",
                name="Chosen simple single",
                marker=dict(color="#1f77b4", size=7, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<extra></extra>",
            )
        )
    if include_simple_burst.size:
        vfig.add_trace(
            go.Scatter(
                x=vol_t[include_simple_burst],
                y=trace_vol[include_simple_burst],
                mode="markers",
                name="Chosen simple burst",
                marker=dict(color="green", size=7, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<extra></extra>",
            )
        )
    if include_complex.size:
        vfig.add_trace(
            go.Scatter(
                x=vol_t[include_complex],
                y=trace_vol[include_complex],
                mode="markers",
                name="Chosen complex",
                marker=dict(color="#ff69b4", size=7, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<extra></extra>",
            )
        )
    if include_plateau.size:
        vfig.add_trace(
            go.Scatter(
                x=vol_t[include_plateau],
                y=trace_vol[include_plateau],
                mode="markers",
                name="Chosen plateau",
                marker=dict(color="purple", size=8, opacity=0.95),
                hovertemplate="t=%{x:.3f}s<br>V=%{y:.3f}<extra></extra>",
            )
        )
    vfig.update_layout(
        template="simple_white",
        title=(
            f"{os.path.basename(cell_folder)} | {os.path.basename(pkl_path)}"
            f"<br><sup>Voltage with detected spikes and event windows (light yellow)</sup>"
        ),
        width=1400,
        height=520,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    vfig.update_xaxes(title_text="Time (s)")
    vfig.update_yaxes(title_text="Voltage", range=[v_y0, v_y1])

    v_html_path = out_base + "_voltage_event_windows.html"
    v_svg_path = out_base + "_voltage_event_windows.svg"
    vfig.write_html(v_html_path)
    v_svg_saved = bool(_safe_write_image(vfig, v_svg_path, warn_prefix="Voltage windows SVG"))

    fig.update_layout(
        template="simple_white",
        title=(
            f"{os.path.basename(cell_folder)} | {os.path.basename(pkl_path)}"
            f"<br><sup>v={PIPELINE_VERSION} | events grouped by ISI<={SIMPLE_ISI_MS:.0f}ms from vm_all_spikes; event is complex if any spike is vm_complex_spike; "
            f"bounds from subthreshold baseline-fraction crossing ({SUBTHR_BOUND_FRAC*100:.0f}% of peak from baseline), with spike guard ֲ±{int(EVENT_BOUND_SPIKE_PAD_FRAMES)} frames; "
            f"same-start merge only for complex/plateau (simple stays ISI-only); hard reject if gap(prev last spike -> curr first spike) < {HARD_MIN_GAP_S:.3f}s; include if EITHER: tail ratio abs(mean(cal[min_tail_idx-1:min_tail_idx+2]))/max(curr) < {TAIL_RATIO_THR:.2f} OR gap(prev last spike -> curr first spike)>{GAP_THR_PREV_SIMPLE_S:.3f}s after simple prev or >{GAP_THR_PREV_COMPLEX_S:.3f}s after complex prev; additionally exclude non-complex when a complex/plateau starts within {COMPLEX_FOLLOWUP_EXCLUDE_S:.3f}s after it</sup>"
        ),
        width=1400,
        height=650,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Voltage", secondary_y=False)
    fig.update_yaxes(title_text="Calcium (dF/F) / z-score", secondary_y=True)

    html_path = out_base + ".html"
    svg_path = out_base + ".svg"
    fig.write_html(html_path)
    svg_saved = bool(_safe_write_image(fig, svg_path, warn_prefix="Event overlay SVG"))

    robust_info = _compute_robust_scaling_from_cs(
        trace_vol=trace_vol,
        trace_cal=trace_cal,
        class_events=class_events,
        event_rows=event_rows,
        vol_sr=vol_sr,
        cal_sr=cal_sr,
        pre_baseline_s=PRE_BASELINE_S,
    )
    robust_html_path, robust_svg_path = _save_robust_scaling_diagnostic(
        cell_folder=cell_folder,
        pkl_path=pkl_path,
        robust_info=robust_info,
        suffix_tag=suffix,
    )

    n_chosen = int(sum(bool(ev.get("include", False)) for ev in event_rows))
    print(
        f"[OK] {os.path.basename(pkl_path)} | events={len(event_rows)} | chosen={n_chosen} | "
        f"single_simple_pts={len(include_single_simple)} | simple_burst_pts={len(include_simple_burst)} | complex_pts={len(include_complex)} | plateau_pts={len(include_plateau)} | "
        f"saved_html={os.path.basename(html_path)} | saved_svg={svg_saved}"
    )

    return pd.DataFrame(metric_rows), {
        "figure": fig,
        "html_path": html_path,
        "svg_path": svg_path,
        "voltage_html_path": v_html_path,
        "voltage_svg_path": v_svg_path,
        "cell_folder": cell_folder,
        "pkl_path": pkl_path,
        "suffix": suffix,
        "robust_scaling": robust_info,
        "robust_scaling_html_path": robust_html_path,
        "robust_scaling_svg_path": robust_svg_path,
    }, pd.DataFrame(all_event_rows)

def _plot_summary(metrics_df, save_html=None, save_svg=None, show_plot=False, title_prefix="Chosen calcium events vs spike count",
                  save_subplots=False):
    if metrics_df is None or len(metrics_df) == 0:
        raise ValueError("No chosen events for summary plot.")

    n_cells = int(metrics_df["cell_folder"].nunique()) if "cell_folder" in metrics_df.columns else np.nan
    n_complex = int((metrics_df["event_type"] == "complex").sum())
    n_simple = int((metrics_df["event_type"] == "simple").sum())
    n_plateau = int((metrics_df["event_type"] == "plateau").sum())

    fig = make_subplots(
        rows=6,
        cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ],
        subplot_titles=(
            "Peak (dF/F)",
            "Peak (z-score)",
            "HWHM (s)",
            "AUC",
            "Peak (z-score) violin by spike count",
            "Peak (global-baseline dF/F)",
            "Decay time (s)",
            "Peak (global-baseline dF/F) violin by spike count",
            "Decay tau (s)",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.09,
    )

    panels = [
        (1, 1, "peak_df_f", "Peak (dF/F)"),
        (1, 2, "peak_z", "Peak (z-score)"),
        (2, 1, "hwhm_s", "HWHM (s)"),
        (2, 2, "auc", "AUC"),
        (4, 1, "peak_df_f_global", "Peak (global-baseline dF/F)"),
        (4, 2, "decay_time_s", "Decay time (s)"),
        (5, 2, "tau_decay_s", "Decay tau (s)"),
        (
            6,
            1,
            "peak_norm_complex4_mean",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
        ),
    ]

    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}

    for r, c, ycol, ytitle in panels:
        if ycol not in metrics_df.columns:
            metrics_df[ycol] = np.nan

        for ev_type in EVENT_TYPES:
            sub = metrics_df[metrics_df["event_type"] == ev_type]
            _add_beeswarm_points_and_fit(
                fig=fig,
                sub_df=sub,
                ycol=ycol,
                ev_type=ev_type,
                color=color_map[ev_type],
                label=label_map[ev_type],
                showlegend=(r == 1 and c == 1),
                row=r, col=c,
                seed=4000 + (17 if ev_type == "simple" else 31) + (10 * r + c),
            )
        fig.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        fig.update_yaxes(title_text=ytitle, row=r, col=c)

    def _p_to_stars(p):
        if not np.isfinite(p):
            return "n/a"
        if p < 1e-4:
            return "****"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "ns"

    # Violin subplot: simple and complex share the same x position (same spike count)
    violin_categories = []
    vcol = "peak_z"
    vdf = metrics_df.copy()
    vdf = vdf[np.isfinite(vdf[vcol]) & np.isfinite(vdf["n_spikes"])]
    if len(vdf) > 0:
        vdf["n_spikes"] = vdf["n_spikes"].astype(int)
        vdf["n_spikes_cap"] = _cap_n_spikes(vdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        simple = vdf[vdf["event_type"] == "simple"].copy()
        complex_df = vdf[vdf["event_type"] == "complex"].copy()
        plateau_df = vdf[vdf["event_type"] == "plateau"].copy()

        simple_counts = sorted(simple["n_spikes_cap"].unique().tolist()) if len(simple) else []
        complex_counts = sorted(complex_df["n_spikes_cap"].unique().tolist()) if len(complex_df) else []
        plateau_counts = sorted(plateau_df["n_spikes_cap"].unique().tolist()) if len(plateau_df) else []
        violin_categories = sorted(set(simple_counts).union(set(complex_counts)).union(set(plateau_counts)))

        simple_means = simple.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(simple) else {}
        complex_means = complex_df.groupby("n_spikes_cap")[vcol].mean().to_dict() if len(complex_df) else {}

        if len(simple):
            fig.add_trace(
                go.Violin(
                    x=simple["n_spikes_cap"],
                    y=simple[vcol],
                    name="Simple/Burst",
                    legendgroup="simple",
                    line_color="red",
                    fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                    showlegend=False,
                ),
                row=3, col=1,
            )

        if len(complex_df):
            fig.add_trace(
                go.Violin(
                    x=complex_df["n_spikes_cap"],
                    y=complex_df[vcol],
                    name="Complex",
                    legendgroup="complex",
                    line_color="black",
                    fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                    showlegend=False,
                ),
                row=3, col=1,
            )
        if len(plateau_df):
            fig.add_trace(
                go.Violin(
                    x=plateau_df["n_spikes_cap"],
                    y=plateau_df[vcol],
                    name="Plateau",
                    legendgroup="plateau",
                    line_color="purple",
                    fillcolor="rgba(128,0,128,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                    showlegend=False,
                ),
                row=3, col=1,
            )

        fig.update_xaxes(row=3, col=1, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        fig.update_yaxes(title_text="Peak (z-score)", row=3, col=1)

        y_max = float(np.nanmax(vdf[vcol])) if len(vdf) else 1.0
        y_min = float(np.nanmin(vdf[vcol])) if len(vdf) else 0.0
        y_span = max(1e-6, y_max - y_min)
        y_low = y_min - 0.05 * y_span
        y_high = y_max + 0.25 * y_span

        fit_s = _add_fit_traces(
            fig=fig,
            x=simple["n_spikes_cap"].values if len(simple) else [],
            y=simple[vcol].values if len(simple) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=False,
            row=3,
            col=1,
        )
        fit_c = _add_fit_traces(
            fig=fig,
            x=complex_df["n_spikes_cap"].values if len(complex_df) else [],
            y=complex_df[vcol].values if len(complex_df) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=False,
            row=3,
            col=1,
        )
        fit_p = _add_fit_traces(
            fig=fig,
            x=plateau_df["n_spikes_cap"].values if len(plateau_df) else [],
            y=plateau_df[vcol].values if len(plateau_df) else [],
            color="purple",
            label="Plateau",
            legendgroup="plateau",
            showlegend=False,
            row=3,
            col=1,
        )

        fit_txt = []
        s_txt = _fit_models_text("simple", fit_s)
        c_txt = _fit_models_text("complex", fit_c)
        if s_txt:
            fit_txt.append(s_txt)
        if c_txt:
            fit_txt.append(c_txt)
        if len(fit_txt) > 0:
            fig.add_annotation(
                x=float(SPIKE_COUNT_CAP),
                y=y_high - 0.02 * max(1e-6, (y_high - y_low)),
                text=" | ".join(fit_txt),
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=10, color="black"),
                row=3,
                col=1,
            )

        # Significance between simple and complex for matched spike-count bins
        common_counts = [n for n in sorted(set(simple_counts).intersection(set(complex_counts)))]
        if len(common_counts) > 0:
            max_for_sig = float(np.nanmax(vdf[vcol])) if len(vdf) else 1.0
            sig_base = max_for_sig + 0.10 * y_span
            sig_step = 0.08 * y_span

            for i_sig, n in enumerate(common_counts):
                s_vals = simple.loc[simple["n_spikes_cap"] == n, vcol].values
                c_vals = complex_df.loc[complex_df["n_spikes_cap"] == n, vcol].values
                s_vals = s_vals[np.isfinite(s_vals)]
                c_vals = c_vals[np.isfinite(c_vals)]

                p_val = np.nan
                if mannwhitneyu is not None and len(s_vals) >= 2 and len(c_vals) >= 2:
                    try:
                        _, p_val = mannwhitneyu(s_vals, c_vals, alternative="two-sided")
                    except Exception:
                        p_val = np.nan

                stars = _p_to_stars(float(p_val) if np.isfinite(p_val) else np.nan)
                y_sig = sig_base + (i_sig * sig_step)
                x0 = float(n) - 0.14
                x1 = float(n) + 0.14

                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=y_sig,
                    y1=y_sig,
                    line=dict(color="black", width=1.2),
                    row=3,
                    col=1,
                )
                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x0,
                    y0=y_sig - 0.015 * y_span,
                    y1=y_sig,
                    line=dict(color="black", width=1.2),
                    row=3,
                    col=1,
                )
                fig.add_shape(
                    type="line",
                    x0=x1,
                    x1=x1,
                    y0=y_sig - 0.015 * y_span,
                    y1=y_sig,
                    line=dict(color="black", width=1.2),
                    row=3,
                    col=1,
                )
                fig.add_annotation(
                    x=float(n),
                    y=y_sig + 0.01 * y_span,
                    text=f"{stars} (n={int(n)})",
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=10, color="black"),
                    row=3,
                    col=1,
                )

            y_high = max(y_high, sig_base + (len(common_counts) + 1) * sig_step)

        fig.update_yaxes(range=[y_low, y_high], row=3, col=1)

        # Mean labels near top, slightly shifted by event type at same spike count
        panel_span = max(1e-6, y_high - y_low)
        mean_text_y = y_high - 0.04 * panel_span
        for n in simple_counts:
            if n in simple_means:
                fig.add_annotation(
                    x=float(n) - 0.08,
                    y=mean_text_y,
                    text=f"mean={float(simple_means[n]):.3f}",
                    showarrow=False,
                    yanchor="top",
                    font=dict(size=10, color="red"),
                    row=3,
                    col=1,
                )
        for n in complex_counts:
            if n in complex_means:
                fig.add_annotation(
                    x=float(n) + 0.08,
                    y=mean_text_y,
                    text=f"mean={float(complex_means[n]):.3f}",
                    showarrow=False,
                    yanchor="top",
                    font=dict(size=10, color="black"),
                    row=3,
                    col=1,
                )

    # Additional subplot: global-baseline amplitude violin by spike count
    vgcol = "peak_df_f_global"
    vgdf = metrics_df.copy()
    if vgcol not in vgdf.columns:
        vgdf[vgcol] = np.nan
    vgdf = vgdf[np.isfinite(vgdf[vgcol]) & np.isfinite(vgdf["n_spikes"])]
    if len(vgdf) > 0:
        vgdf["n_spikes"] = vgdf["n_spikes"].astype(int)
        vgdf["n_spikes_cap"] = _cap_n_spikes(vgdf["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
        s2 = vgdf[vgdf["event_type"] == "simple"].copy()
        c2 = vgdf[vgdf["event_type"] == "complex"].copy()
        p2 = vgdf[vgdf["event_type"] == "plateau"].copy()

        if len(s2):
            fig.add_trace(
                go.Violin(
                    x=s2["n_spikes_cap"],
                    y=s2[vgcol],
                    name="Simple/Burst",
                    legendgroup="simple",
                    line_color="red",
                    fillcolor="rgba(220,20,60,0.35)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                    showlegend=False,
                ),
                row=5, col=1,
            )
        if len(c2):
            fig.add_trace(
                go.Violin(
                    x=c2["n_spikes_cap"],
                    y=c2[vgcol],
                    name="Complex",
                    legendgroup="complex",
                    line_color="black",
                    fillcolor="rgba(0,0,0,0.25)",
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.95,
                    showlegend=False,
                ),
                row=5, col=1,
            )

        _add_fit_traces(
            fig=fig,
            x=s2["n_spikes_cap"].values if len(s2) else [],
            y=s2[vgcol].values if len(s2) else [],
            color="red",
            label="Simple/Burst",
            legendgroup="simple",
            showlegend=False,
            row=5,
            col=1,
        )
        _add_fit_traces(
            fig=fig,
            x=c2["n_spikes_cap"].values if len(c2) else [],
            y=c2[vgcol].values if len(c2) else [],
            color="black",
            label="Complex",
            legendgroup="complex",
            showlegend=False,
            row=5,
            col=1,
        )

    fig.update_xaxes(row=5, col=1, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
    fig.update_yaxes(title_text="Peak (global-baseline dF/F)", row=5, col=1)

    fig.update_layout(
        template="simple_white",
        width=max(1400, 120 * max(1, len(violin_categories)) + 1000),
        height=2200,
        title=(
            f"{title_prefix}"
            f"<br><sup>n_cells={n_cells} | n_complex_events={n_complex} | n_simple_events={n_simple} | n_plateau_events={n_plateau}</sup>"
        ),
        violinmode="overlay",
        violingap=0.02,
    )

    if save_html:
        fig.write_html(save_html)
    if save_svg:
        _safe_write_image(fig, save_svg, warn_prefix="Summary SVG")

    if show_plot:
        fig.show()

    if save_subplots:
        _save_summary_subplots(
            metrics_df=metrics_df,
            save_html=save_html,
            save_svg=save_svg,
            title_prefix=title_prefix,
        )

    # Also save normalized-data plots (additional, does not replace existing outputs)
    save_html_norm = _append_suffix_before_ext(save_html, "_normalized_data") if save_html else None
    save_svg_norm = _append_suffix_before_ext(save_svg, "_normalized_data") if save_svg else None
    _plot_summary_normalized(
        metrics_df=metrics_df,
        save_html=save_html_norm,
        save_svg=save_svg_norm,
        show_plot=False,
        title_prefix=title_prefix,
        save_subplots=save_subplots,
    )

    return fig

def _build_cell_average_metrics(metrics_df):
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    if ("cell_folder" not in metrics_df.columns) or ("event_type" not in metrics_df.columns):
        return pd.DataFrame()

    df = metrics_df.copy()
    if "n_spikes" not in df.columns:
        return pd.DataFrame()
    df = df[np.isfinite(df["n_spikes"])].copy()
    if len(df) == 0:
        return pd.DataFrame()
    df["n_spikes"] = df["n_spikes"].astype(int)
    df["n_spikes_cap"] = _cap_n_spikes(df["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)

    metric_cols = [
        "peak_df_f",
        "peak_df_f_global",
        "peak_z",
        "hwhm_s",
        "auc",
        "rise_time_s",
        "decay_time_s",
        "tau_decay_s",
        "peak_norm_p95",
        "peak_norm_complex4_mean",
        "peak_z_cal_p8_robust",
        "vol_peak_z_quiet_robust",
        "vol_auc",
    ]
    agg = {}
    for c in metric_cols:
        if c in df.columns:
            agg[c] = "mean"
    if len(agg) == 0:
        return pd.DataFrame()

    g = (
        df.groupby(["cell_folder", "event_type", "n_spikes_cap"], as_index=False)
          .agg(agg)
          .rename(columns={"n_spikes_cap": "n_spikes"})
    )

    # Optional metadata columns
    for c in ("bound_method", "suffix", "pkl_name"):
        if c in df.columns and c not in g.columns:
            first_vals = (
                df.groupby(["cell_folder", "event_type", "n_spikes_cap"], as_index=False)[c]
                  .first()
                  .rename(columns={"n_spikes_cap": "n_spikes"})
            )
            g = g.merge(first_vals, on=["cell_folder", "event_type", "n_spikes"], how="left")

    g["is_cell_average_point"] = True
    return g

def _build_cell_average_metrics_by_vol_auc_bins(metrics_df, n_bins=AUC_BIN_COUNT):
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    need = {"cell_folder", "event_type", "vol_auc"}
    if not need.issubset(set(metrics_df.columns)):
        return pd.DataFrame()

    df = metrics_df.copy()
    df["cell_folder"] = df["cell_folder"].astype(str)
    df["event_type"] = df["event_type"].astype(str)
    df["vol_auc"] = pd.to_numeric(df["vol_auc"], errors="coerce")
    df = df[np.isfinite(df["vol_auc"])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    metric_cols = [
        "peak_df_f",
        "peak_df_f_global",
        "peak_z",
        "hwhm_s",
        "auc",
        "rise_time_s",
        "decay_time_s",
        "tau_decay_s",
        "peak_norm_p95",
        "peak_norm_complex4_mean",
        "peak_z_cal_p8_robust",
        "vol_peak_z_quiet_robust",
        "vol_auc",
    ]
    agg = {}
    for c in metric_cols:
        if c in df.columns:
            agg[c] = "mean"
    if len(agg) == 0:
        return pd.DataFrame()

    n_bins = int(max(2, n_bins))
    v = np.asarray(df["vol_auc"], float)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return pd.DataFrame()

    if vmax <= vmin:
        df["vol_auc_bin"] = 0
    else:
        edges = np.linspace(vmin, vmax, n_bins + 1)
        if np.unique(edges).size < 2:
            df["vol_auc_bin"] = 0
        else:
            bins = pd.cut(
                df["vol_auc"],
                bins=edges,
                include_lowest=True,
                labels=False,
                right=True,
            )
            df["vol_auc_bin"] = pd.to_numeric(bins, errors="coerce")
            df = df[np.isfinite(df["vol_auc_bin"])].copy()
            df["vol_auc_bin"] = df["vol_auc_bin"].astype(int)

    g = (
        df.groupby(["cell_folder", "event_type", "vol_auc_bin"], as_index=False)
        .agg(agg)
    )
    g["is_cell_average_point"] = True
    g["is_auc_binned_cell_average_point"] = True
    return g

def _plot_summary_of_summaries(metrics_df, cell_avg_df=None, save_html=None, save_svg=None, save_pdf=None,
                               title_prefix="Summary of summaries"):
    if metrics_df is None or len(metrics_df) == 0:
        return None
    if cell_avg_df is None or len(cell_avg_df) == 0:
        cell_avg_df = _build_cell_average_metrics(metrics_df)

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "All events: Peak (dF/F) vs # spikes",
            f"All events: Peak z (non-robust, low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g})",
            "All events: Peak z (robust, p8+MAD)",
            f"All events: Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "Cell averages: Peak (dF/F) vs # spikes",
            f"Cell averages: Peak z (non-robust, low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g})",
            "Cell averages: Peak z (robust, p8+MAD)",
            f"Cell averages: Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    # As requested: complex=red, simple=black for points/violin.
    color_map = {"complex": "red", "simple": "black", "plateau": "purple"}
    label_map = {"complex": "Complex", "simple": "Simple/Burst", "plateau": "Plateau"}
    # Mean-line emphasis colors.
    mean_line_color_map = {"complex": "#b30000", "simple": "#555555", "plateau": "#7b1fa2"}

    panel_specs = [
        (1, 1, metrics_df, "peak_df_f", "Peak (dF/F)"),
        (1, 2, metrics_df, "peak_z", "Peak z-score"),
        (1, 3, metrics_df, "peak_z_cal_p8_robust", "Peak robust z-score"),
        (1, 4, metrics_df, "peak_norm_complex4_mean", f"Norm by complex n={int(COMPLEX4_REF_N_SPIKES)}"),
        (2, 1, cell_avg_df, "peak_df_f", "Peak (dF/F)"),
        (2, 2, cell_avg_df, "peak_z", "Peak z-score"),
        (2, 3, cell_avg_df, "peak_z_cal_p8_robust", "Peak robust z-score"),
        (2, 4, cell_avg_df, "peak_norm_complex4_mean", f"Norm by complex n={int(COMPLEX4_REF_N_SPIKES)}"),
    ]

    for r, c, dfi, ycol, ytitle in panel_specs:
        if dfi is None or len(dfi) == 0:
            continue
        if ycol not in dfi.columns or "n_spikes" not in dfi.columns or "event_type" not in dfi.columns:
            continue

        y_panel = np.asarray(dfi[ycol], float)
        y_panel = y_panel[np.isfinite(y_panel)]
        if y_panel.size > 0:
            y_min_panel = float(np.nanmin(y_panel))
            y_max_panel = float(np.nanmax(y_panel))
            y_span_panel = y_max_panel - y_min_panel
            if (not np.isfinite(y_span_panel)) or (y_span_panel <= 0):
                y_span_panel = 1.0
        else:
            y_min_panel, y_max_panel, y_span_panel = 0.0, 1.0, 1.0
        # Place mean tags outside the violin cloud.
        y_tag_top = float(y_max_panel + 0.12 * y_span_panel)
        y_tag_mid = float(y_max_panel + 0.04 * y_span_panel)
        y_tag_bottom = float(y_min_panel - 0.12 * y_span_panel)

        panel_y_all = []
        panel_type_vals = {"complex": [], "simple": [], "plateau": []}
        for ev_type in ("complex", "simple", "plateau"):
            sub = dfi[dfi["event_type"] == ev_type].copy()
            if len(sub) == 0:
                continue
            x_raw = _cap_n_spikes(np.asarray(sub["n_spikes"], float), cap=SPIKE_COUNT_CAP)
            y = np.asarray(sub[ycol], float)
            m = np.isfinite(x_raw) & np.isfinite(y)
            if not np.any(m):
                continue
            x_raw = x_raw[m]
            y = y[m]
            panel_y_all.append(y)
            panel_type_vals[str(ev_type)].append(y)

            # Violin behind beeswarm points.
            if ev_type == "complex":
                vfill = "rgba(220,20,60,0.20)"
            elif ev_type == "plateau":
                vfill = "rgba(128,0,128,0.18)"
            else:
                vfill = "rgba(0,0,0,0.16)"
            fig.add_trace(
                go.Violin(
                    x=x_raw,
                    y=y,
                    name=f"{label_map[ev_type]} dist",
                    legendgroup=f"sos_{ev_type}",
                    line_color=color_map[ev_type],
                    fillcolor=vfill,
                    points=False,
                    box_visible=False,
                    meanline_visible=False,
                    width=0.50,
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            seed_offset = {"complex": 1, "simple": 2, "plateau": 3}.get(ev_type, 0)
            x_plot = _beeswarm_x(x_raw, ev_type=ev_type, seed=7000 + 100 * r + 10 * c + seed_offset, jitter=0.12, type_offset=0.08)
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y,
                    mode="markers",
                    name=label_map[ev_type],
                    legendgroup=f"sos_{ev_type}",
                    marker=dict(symbol="circle-open", size=8, color=color_map[ev_type], line=dict(color=color_map[ev_type], width=1.3)),
                    showlegend=(r == 1 and c == 1),
                ),
                row=r,
                col=c,
            )

            _add_fit_traces(
                fig=fig,
                x=x_raw,
                y=y,
                color=color_map[ev_type],
                label=label_map[ev_type],
                legendgroup=f"sos_{ev_type}",
                showlegend=(r == 1 and c == 1),
                row=r,
                col=c,
            )

            # Bold mean line per spike-count group + mean labels above.
            x_groups = np.unique(np.asarray(x_raw, int))
            y_span = float(np.nanmax(y) - np.nanmin(y)) if y.size > 0 else 1.0
            if (not np.isfinite(y_span)) or (y_span <= 0):
                y_span = 1.0
            for xg in x_groups:
                mg = (np.asarray(x_raw, int) == int(xg))
                if not np.any(mg):
                    continue
                yg = y[mg]
                if yg.size == 0 or not np.any(np.isfinite(yg)):
                    continue
                y_mean_g = float(np.nanmean(yg))
                x_center = float(_beeswarm_x(np.array([xg], dtype=float), ev_type=ev_type, seed=0, jitter=0.0, type_offset=0.08)[0])
                half_w = 0.11
                fig.add_trace(
                    go.Scatter(
                        x=[x_center - half_w, x_center + half_w],
                        y=[y_mean_g, y_mean_g],
                        mode="lines",
                        name=f"{label_map[ev_type]} mean",
                        legendgroup=f"sos_{ev_type}",
                        line=dict(color=mean_line_color_map[ev_type], width=4.2),
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
                fig.add_annotation(
                    x=x_center,
                    y=(y_tag_top if ev_type == "complex" else (y_tag_bottom if ev_type == "simple" else y_tag_mid)),
                    text=f"{y_mean_g:.2f}",
                    showarrow=False,
                    xanchor="center",
                    yanchor=("top" if ev_type in ("complex", "plateau") else "bottom"),
                    font=dict(size=11, color=mean_line_color_map[ev_type]),
                    bgcolor="rgba(255,255,255,0.65)",
                    row=r,
                    col=c,
                )

        fig.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
        # Expand y-range so top/bottom mean tags are outside traces and non-overlapping.
        y_lo = float(y_min_panel - 0.18 * y_span_panel)
        y_hi = float(y_max_panel + 0.22 * y_span_panel)
        fig.update_yaxes(row=r, col=c, title_text=ytitle, range=[y_lo, y_hi])

    fig.update_layout(
        template="simple_white",
        width=1850,
        height=980,
        title=title_prefix,
        legend=dict(orientation="h"),
    )

    tagged_html, tagged_svg, tagged_pdf = _tag_figure_paths(
        html_path=save_html,
        svg_path=save_svg,
        pdf_path=save_pdf,
        include_plateau=_has_plateau_events(metrics_df) or _has_plateau_events(cell_avg_df),
        uses_robust_calcium=True,
    )
    _save_fig_triplet(fig, html_path=tagged_html, svg_path=tagged_svg, pdf_path=tagged_pdf, warn_prefix="Summary-of-summaries")
    return fig


def _critical_level_from_index(idx):
    if not np.isfinite(idx):
        return "n/a"
    if idx >= 45:
        return "high"
    if idx >= 25:
        return "moderate"
    return "low"


def _sep_z_y(y_complex, y_simple):
    y_complex = np.asarray(y_complex, float).ravel()
    y_simple = np.asarray(y_simple, float).ravel()
    y_complex = y_complex[np.isfinite(y_complex)]
    y_simple = y_simple[np.isfinite(y_simple)]
    nc, ns = int(y_complex.size), int(y_simple.size)
    if nc < 2 or ns < 2:
        return np.nan
    mu_c, mu_s = float(np.nanmean(y_complex)), float(np.nanmean(y_simple))
    sd_c, sd_s = float(np.nanstd(y_complex, ddof=1)), float(np.nanstd(y_simple, ddof=1))
    pooled = float(np.sqrt(max(1e-12, (((nc - 1) * sd_c * sd_c) + ((ns - 1) * sd_s * sd_s)) / max(1, nc + ns - 2))))
    return float(abs(mu_c - mu_s) / pooled) if pooled > 0 else np.nan


def _sep_z_2d(x_complex, y_complex, x_simple, y_simple):
    xc = np.asarray(x_complex, float).ravel()
    yc = np.asarray(y_complex, float).ravel()
    xs = np.asarray(x_simple, float).ravel()
    ys = np.asarray(y_simple, float).ravel()
    mc = np.isfinite(xc) & np.isfinite(yc)
    ms = np.isfinite(xs) & np.isfinite(ys)
    xc, yc = xc[mc], yc[mc]
    xs, ys = xs[ms], ys[ms]
    nc, ns = int(xc.size), int(xs.size)
    if nc < 3 or ns < 3:
        return np.nan
    c = np.column_stack([xc, yc])
    s = np.column_stack([xs, ys])
    mu_c = np.nanmean(c, axis=0)
    mu_s = np.nanmean(s, axis=0)
    sc = np.cov(c.T, ddof=1)
    ss = np.cov(s.T, ddof=1)
    sp = (((nc - 1) * sc) + ((ns - 1) * ss)) / max(1, nc + ns - 2)
    try:
        inv_sp = np.linalg.pinv(sp)
        d = mu_c - mu_s
        val = float(np.sqrt(max(0.0, float(d.T @ inv_sp @ d))))
    except Exception:
        val = np.nan
    return val


def _perm_pvalue_for_sep_metrics(
    x_complex, y_complex, x_simple, y_simple,
    n_perm=PERM_N, max_per_group=PERM_MAX_PER_GROUP, return_distributions=False
):
    xc = np.asarray(x_complex, float).ravel()
    yc = np.asarray(y_complex, float).ravel()
    xs = np.asarray(x_simple, float).ravel()
    ys = np.asarray(y_simple, float).ravel()
    mc = np.isfinite(xc) & np.isfinite(yc)
    ms = np.isfinite(xs) & np.isfinite(ys)
    xc, yc = xc[mc], yc[mc]
    xs, ys = xs[ms], ys[ms]
    nc, ns = int(xc.size), int(xs.size)
    if nc < 3 or ns < 3:
        if return_distributions:
            return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=float), np.array([], dtype=float)
        return np.nan, np.nan

    rng = np.random.default_rng(12345 + 97 * nc + 193 * ns)
    if nc > max_per_group:
        idx = rng.choice(nc, size=max_per_group, replace=False)
        xc, yc = xc[idx], yc[idx]
        nc = int(xc.size)
    if ns > max_per_group:
        idx = rng.choice(ns, size=max_per_group, replace=False)
        xs, ys = xs[idx], ys[idx]
        ns = int(xs.size)

    obs_y = _sep_z_y(yc, ys)
    obs_2d = _sep_z_2d(xc, yc, xs, ys)
    if (not np.isfinite(obs_y)) and (not np.isfinite(obs_2d)):
        if return_distributions:
            return np.nan, np.nan, obs_y, obs_2d, np.array([], dtype=float), np.array([], dtype=float)
        return np.nan, np.nan

    all_xy = np.column_stack([np.r_[xc, xs], np.r_[yc, ys]])
    n = int(all_xy.shape[0])
    if n < 6:
        if return_distributions:
            return np.nan, np.nan, obs_y, obs_2d, np.array([], dtype=float), np.array([], dtype=float)
        return np.nan, np.nan

    ge_y = 0
    ge_2d = 0
    done = 0
    perm_y_vals = []
    perm_2d_vals = []
    for _ in range(int(max(10, n_perm))):
        perm = rng.permutation(n)
        ia = perm[:nc]
        ib = perm[nc:]
        a = all_xy[ia]
        b = all_xy[ib]
        sy = _sep_z_y(a[:, 1], b[:, 1])
        s2 = _sep_z_2d(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        if np.isfinite(sy):
            perm_y_vals.append(float(sy))
        if np.isfinite(s2):
            perm_2d_vals.append(float(s2))
        if np.isfinite(obs_y) and np.isfinite(sy) and sy >= obs_y:
            ge_y += 1
        if np.isfinite(obs_2d) and np.isfinite(s2) and s2 >= obs_2d:
            ge_2d += 1
        done += 1
    p_y = float((ge_y + 1) / (done + 1)) if np.isfinite(obs_y) else np.nan
    p_2d = float((ge_2d + 1) / (done + 1)) if np.isfinite(obs_2d) else np.nan
    if return_distributions:
        return p_y, p_2d, obs_y, obs_2d, np.asarray(perm_y_vals, dtype=float), np.asarray(perm_2d_vals, dtype=float)
    return p_y, p_2d


def _auc_overlap_stats_from_xy(x_complex, y_complex, x_simple, y_simple, include_permutation_distributions=False):
    x_complex = np.asarray(x_complex, float).ravel()
    y_complex = np.asarray(y_complex, float).ravel()
    x_simple = np.asarray(x_simple, float).ravel()
    y_simple = np.asarray(y_simple, float).ravel()
    mc = np.isfinite(x_complex) & np.isfinite(y_complex)
    ms = np.isfinite(x_simple) & np.isfinite(y_simple)
    x_complex, y_complex = x_complex[mc], y_complex[mc]
    x_simple, y_simple = x_simple[ms], y_simple[ms]
    nc, ns = int(x_complex.size), int(x_simple.size)
    out = {
        "n_complex": nc,
        "n_simple": ns,
        "point_overlap_est": np.nan,
        "point_overlap_total_pct": np.nan,
        "point_overlap_complex_pct": np.nan,
        "point_overlap_simple_pct": np.nan,
        "bbox_iou_pct": np.nan,
        "y_sep_z": np.nan,
        "sep_2d_z": np.nan,
        "p_sep_y_perm": np.nan,
        "p_sep_2d_perm": np.nan,
        "critical_y_index": np.nan,
        "critical_y_level": "n/a",
        "critical_2d_index": np.nan,
        "critical_2d_level": "n/a",
        "p_critical_y_perm": np.nan,
        "p_critical_2d_perm": np.nan,
        "perm_sep_y": np.array([], dtype=float),
        "perm_sep_2d": np.array([], dtype=float),
        "perm_critical_y": np.array([], dtype=float),
        "perm_critical_2d": np.array([], dtype=float),
    }
    if nc < 2 or ns < 2:
        return out

    x_all = np.r_[x_complex, x_simple]
    y_all = np.r_[y_complex, y_simple]
    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (not np.isfinite(y_min)) or (not np.isfinite(y_max)):
        return out
    if x_max <= x_min or y_max <= y_min:
        return out

    n_bins = int(np.clip(np.sqrt(nc + ns), 10, 30))
    xc_edges = np.linspace(x_min, x_max, n_bins + 1)
    yc_edges = np.linspace(y_min, y_max, n_bins + 1)
    h_c, _, _ = np.histogram2d(x_complex, y_complex, bins=[xc_edges, yc_edges])
    h_s, _, _ = np.histogram2d(x_simple, y_simple, bins=[xc_edges, yc_edges])
    overlap_points_est = float(np.minimum(h_c, h_s).sum())
    out["point_overlap_est"] = overlap_points_est
    out["point_overlap_total_pct"] = float(100.0 * (2.0 * overlap_points_est) / max(1.0, float(nc + ns)))
    out["point_overlap_complex_pct"] = float(100.0 * overlap_points_est / max(1.0, float(nc)))
    out["point_overlap_simple_pct"] = float(100.0 * overlap_points_est / max(1.0, float(ns)))

    x0c, x1c = float(np.nanmin(x_complex)), float(np.nanmax(x_complex))
    y0c, y1c = float(np.nanmin(y_complex)), float(np.nanmax(y_complex))
    x0s, x1s = float(np.nanmin(x_simple)), float(np.nanmax(x_simple))
    y0s, y1s = float(np.nanmin(y_simple)), float(np.nanmax(y_simple))
    ix = max(0.0, min(x1c, x1s) - max(x0c, x0s))
    iy = max(0.0, min(y1c, y1s) - max(y0c, y0s))
    inter = ix * iy
    ac = max(0.0, (x1c - x0c) * (y1c - y0c))
    a_s = max(0.0, (x1s - x0s) * (y1s - y0s))
    union = ac + a_s - inter
    if union > 0:
        out["bbox_iou_pct"] = float(100.0 * inter / union)

    p_y, p_2d, obs_y, obs_2d, perm_y, perm_2d = _perm_pvalue_for_sep_metrics(
        x_complex, y_complex, x_simple, y_simple,
        n_perm=PERM_N, max_per_group=PERM_MAX_PER_GROUP, return_distributions=True,
    )
    out["y_sep_z"] = float(obs_y) if np.isfinite(obs_y) else _sep_z_y(y_complex, y_simple)
    out["sep_2d_z"] = float(obs_2d) if np.isfinite(obs_2d) else _sep_z_2d(x_complex, y_complex, x_simple, y_simple)
    out["p_sep_y_perm"] = p_y
    out["p_sep_2d_perm"] = p_2d

    overlap_avg = np.nanmean([out["point_overlap_complex_pct"], out["point_overlap_simple_pct"]])
    sep_y = out["y_sep_z"] if np.isfinite(out["y_sep_z"]) else 0.0
    sep_2d = out["sep_2d_z"] if np.isfinite(out["sep_2d_z"]) else 0.0
    crit_y = float(overlap_avg / (1.0 + sep_y))
    crit_2d = float(overlap_avg / (1.0 + sep_2d))
    out["critical_y_index"] = crit_y
    out["critical_y_level"] = _critical_level_from_index(crit_y)
    out["critical_2d_index"] = crit_2d
    out["critical_2d_level"] = _critical_level_from_index(crit_2d)
    if perm_y.size > 0:
        crit_y_perm = overlap_avg / (1.0 + perm_y)
        out["p_critical_y_perm"] = float((np.sum(crit_y_perm <= crit_y) + 1) / (crit_y_perm.size + 1))
        if include_permutation_distributions:
            out["perm_critical_y"] = np.asarray(crit_y_perm, dtype=float)
    if perm_2d.size > 0:
        crit_2d_perm = overlap_avg / (1.0 + perm_2d)
        out["p_critical_2d_perm"] = float((np.sum(crit_2d_perm <= crit_2d) + 1) / (crit_2d_perm.size + 1))
        if include_permutation_distributions:
            out["perm_critical_2d"] = np.asarray(crit_2d_perm, dtype=float)
    if include_permutation_distributions:
        out["perm_sep_y"] = np.asarray(perm_y, dtype=float)
        out["perm_sep_2d"] = np.asarray(perm_2d, dtype=float)
    return out


def _fmt_p(v):
    if not np.isfinite(v):
        return "n/a"
    if v < 1e-4:
        return "<1e-4"
    return f"{v:.3f}"


def _save_auc_permutation_tests_figure(panel_perm_records, save_html=None, save_svg=None, title_prefix="AUC permutation diagnostics"):
    if (save_html is None) and (save_svg is None):
        return None
    if panel_perm_records is None or len(panel_perm_records) == 0:
        return None

    test_specs = [
        ("perm_sep_y", "y_sep_z", "p_sep_y_perm", "sepZy permutation", "upper"),
        ("perm_sep_2d", "sep_2d_z", "p_sep_2d_perm", "sepZ2D permutation", "upper"),
        ("perm_critical_y", "critical_y_index", "p_critical_y_perm", "criticalY permutation", "lower"),
        ("perm_critical_2d", "critical_2d_index", "p_critical_2d_perm", "critical2D permutation", "lower"),
    ]
    saved = []
    for rec in panel_perm_records:
        panel_row = int(rec.get("row", -1))
        panel_col = int(rec.get("col", -1))
        ycol = str(rec.get("ycol", "metric"))
        panel_name = str(rec.get("ytitle", ycol))
        ov = rec.get("ov", {})
        panel_tag = f"r{panel_row}c{panel_col}_{_normalize_suffix_tag(ycol)}"

        out_html = _append_suffix_before_ext(save_html, f"_{panel_tag}") if save_html else None
        out_svg = _append_suffix_before_ext(save_svg, f"_{panel_tag}") if save_svg else None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=tuple([t[3] for t in test_specs]),
            horizontal_spacing=0.10,
            vertical_spacing=0.18,
        )
        for i, (perm_key, obs_key, p_key, ytitle, tail) in enumerate(test_specs):
            r = (i // 2) + 1
            c = (i % 2) + 1
            perm_vals = np.asarray(ov.get(perm_key, np.array([], dtype=float)), dtype=float)
            perm_vals = perm_vals[np.isfinite(perm_vals)]
            if perm_vals.size == 0:
                continue

            nbins = int(np.clip(np.sqrt(max(20, perm_vals.size)), 15, 50))
            counts, bin_edges = np.histogram(perm_vals, bins=nbins)
            y_top = float(np.nanmax(counts)) if counts.size else 1.0
            fig.add_trace(
                go.Histogram(
                    x=perm_vals,
                    nbinsx=nbins,
                    marker=dict(color="rgba(31,119,180,0.85)", line=dict(color="rgba(30,30,30,0.6)", width=0.5)),
                    opacity=0.95,
                    name="Permutation scores",
                    showlegend=False,
                ),
                row=r, col=c,
            )

            obs_ref = float(ov.get(obs_key, np.nan)) if np.isfinite(ov.get(obs_key, np.nan)) else np.nan
            sig_ref = float(np.nanpercentile(perm_vals, 95 if tail == "upper" else 5))
            if np.isfinite(sig_ref):
                fig.add_vline(x=sig_ref, line_dash="dash", line_color="red", line_width=2, row=r, col=c)
            if np.isfinite(obs_ref):
                fig.add_vline(x=obs_ref, line_dash="solid", line_color="black", line_width=4, row=r, col=c)
                fig.add_trace(
                    go.Scatter(
                        x=[obs_ref],
                        y=[max(1.0, y_top * 0.96)],
                        mode="markers+text",
                        text=["Observed"],
                        textposition="top center",
                        marker=dict(color="black", size=9, symbol="triangle-up"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=r, col=c,
                )
            x_min_local = float(np.nanmin(np.r_[perm_vals, [sig_ref] if np.isfinite(sig_ref) else [], [obs_ref] if np.isfinite(obs_ref) else []]))
            x_max_local = float(np.nanmax(np.r_[perm_vals, [sig_ref] if np.isfinite(sig_ref) else [], [obs_ref] if np.isfinite(obs_ref) else []]))
            x_pad = 0.04 * max(1e-9, x_max_local - x_min_local)
            fig.update_xaxes(range=[x_min_local - x_pad, x_max_local + x_pad], title_text="Score", row=r, col=c)

            p_val = float(ov.get(p_key, np.nan)) if np.isfinite(ov.get(p_key, np.nan)) else np.nan
            fig.add_annotation(
                x=0.01, y=0.99, xref="x domain", yref="y domain",
                text=f"obs={obs_ref:.3f}<br>sig={sig_ref:.3f}<br>p={_fmt_p(p_val)}",
                showarrow=False, xanchor="left", yanchor="top", align="left",
                bordercolor="rgba(120,120,120,0.5)", borderwidth=1,
                bgcolor="rgba(255,255,255,0.85)", font=dict(size=10),
                row=r, col=c,
            )
        fig.update_yaxes(title_text="Count", row=r, col=c)

        fig.update_layout(
            template="simple_white",
            width=1500,
            height=950,
            title=f"{title_prefix} | panel r{panel_row}c{panel_col} ({panel_name})",
            bargap=0.05,
        )
        _save_fig_pair(fig, html_path=out_html, svg_path=out_svg, warn_prefix="AUC permutation diagnostics")
        saved.append((out_html, out_svg))
    return saved


def _plot_auc_vs_calcium_response_summaries(metrics_df, cell_avg_df=None, save_html=None, save_svg=None, save_pdf=None,
                                            title_prefix="Calcium response vs voltage AUC (summary of summaries)"):
    if metrics_df is None or len(metrics_df) == 0:
        return None
    if (cell_avg_df is None) or (len(cell_avg_df) == 0) or ("vol_auc_bin" not in cell_avg_df.columns):
        cell_avg_df = _build_cell_average_metrics_by_vol_auc_bins(metrics_df, n_bins=AUC_BIN_COUNT)

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "All events: Peak (dF/F) vs voltage AUC",
            f"All events: Peak z (non-robust, low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g}) vs voltage AUC",
            "All events: Peak z (robust, p8+MAD) vs voltage AUC",
            f"All events: Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)} vs voltage AUC",
            f"Cell averages (AUC-binned, n_bins={int(AUC_BIN_COUNT)}): Peak (dF/F) vs voltage AUC",
            f"Cell averages (AUC-binned, n_bins={int(AUC_BIN_COUNT)}): Peak z (non-robust, low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g}) vs voltage AUC",
            f"Cell averages (AUC-binned, n_bins={int(AUC_BIN_COUNT)}): Peak z (robust, p8+MAD) vs voltage AUC",
            f"Cell averages (AUC-binned, n_bins={int(AUC_BIN_COUNT)}): Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)} vs voltage AUC",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    color_map = {"complex": "red", "simple": "black", "plateau": "purple"}
    label_map = {"complex": "Complex", "simple": "Simple/Burst", "plateau": "Plateau"}
    panel_specs = [
        (1, 1, metrics_df, "peak_df_f", "Peak (dF/F)"),
        (1, 2, metrics_df, "peak_z", "Peak z-score"),
        (1, 3, metrics_df, "peak_z_cal_p8_robust", "Peak robust z-score"),
        (1, 4, metrics_df, "peak_norm_complex4_mean", f"Norm by complex n={int(COMPLEX4_REF_N_SPIKES)}"),
        (2, 1, cell_avg_df, "peak_df_f", "Peak (dF/F)"),
        (2, 2, cell_avg_df, "peak_z", "Peak z-score"),
        (2, 3, cell_avg_df, "peak_z_cal_p8_robust", "Peak robust z-score"),
        (2, 4, cell_avg_df, "peak_norm_complex4_mean", f"Norm by complex n={int(COMPLEX4_REF_N_SPIKES)}"),
    ]

    xcol = "vol_auc"
    panel_perm_records = []
    for r, c, dfi, ycol, ytitle in panel_specs:
        if dfi is None or len(dfi) == 0:
            continue
        if (xcol not in dfi.columns) or (ycol not in dfi.columns) or ("event_type" not in dfi.columns):
            continue

        x_by_type = {"complex": np.array([], dtype=float), "simple": np.array([], dtype=float), "plateau": np.array([], dtype=float)}
        y_by_type = {"complex": np.array([], dtype=float), "simple": np.array([], dtype=float), "plateau": np.array([], dtype=float)}
        for ev_type in ("complex", "simple", "plateau"):
            sub = dfi[dfi["event_type"] == ev_type].copy()
            if len(sub) == 0:
                continue
            x = pd.to_numeric(sub[xcol], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            x = x[m]
            y = y[m]
            x_by_type[ev_type] = x
            y_by_type[ev_type] = y

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=label_map[ev_type],
                    legendgroup=f"aucsos_{ev_type}",
                    marker=dict(symbol="circle-open", size=8, color=color_map[ev_type], line=dict(color=color_map[ev_type], width=1.3)),
                    showlegend=(r == 1 and c == 1),
                ),
                row=r,
                col=c,
            )

            _add_fit_traces(
                fig=fig,
                x=x,
                y=y,
                color=color_map[ev_type],
                label=label_map[ev_type],
                legendgroup=f"aucsos_{ev_type}",
                showlegend=(r == 1 and c == 1),
                row=r,
                col=c,
            )

        ov = _auc_overlap_stats_from_xy(
            x_complex=x_by_type.get("complex", np.array([], dtype=float)),
            y_complex=y_by_type.get("complex", np.array([], dtype=float)),
            x_simple=x_by_type.get("simple", np.array([], dtype=float)),
            y_simple=y_by_type.get("simple", np.array([], dtype=float)),
            include_permutation_distributions=True,
        )
        pair_specs = [
            ("C-S", "complex", "simple"),
            ("C-P", "complex", "plateau"),
            ("S-P", "simple", "plateau"),
        ]
        pair_lines = []
        for pair_label, type_a, type_b in pair_specs:
            st = _auc_overlap_stats_from_xy(
                x_complex=x_by_type.get(type_a, np.array([], dtype=float)),
                y_complex=y_by_type.get(type_a, np.array([], dtype=float)),
                x_simple=x_by_type.get(type_b, np.array([], dtype=float)),
                y_simple=y_by_type.get(type_b, np.array([], dtype=float)),
                include_permutation_distributions=False,
            )
            sep2d_txt = f"{float(st['sep_2d_z']):.2f}" if np.isfinite(st.get("sep_2d_z", np.nan)) else "n/a"
            iou_txt = f"{float(st['bbox_iou_pct']):.1f}%" if np.isfinite(st.get("bbox_iou_pct", np.nan)) else "n/a"
            pair_lines.append(
                f"{pair_label}: n={int(st.get('n_complex', 0))}/{int(st.get('n_simple', 0))} "
                f"sep2D={sep2d_txt} p2D={_fmt_p(st.get('p_sep_2d_perm', np.nan))} IoU={iou_txt}"
            )
        panel_perm_records.append({"row": r, "col": c, "ycol": ycol, "ytitle": ytitle, "ov": ov})
        n_plateau_panel = int(x_by_type.get("plateau", np.array([], dtype=float)).size)
        ov_txt = (
            f"nC={ov['n_complex']} nS={ov['n_simple']} nP={n_plateau_panel}<br>"
            f"ptOv~{ov['point_overlap_est']:.0f} ({ov['point_overlap_total_pct']:.1f}%)<br>"
            f"C:{ov['point_overlap_complex_pct']:.1f}% S:{ov['point_overlap_simple_pct']:.1f}%<br>"
            f"boxIoU={ov['bbox_iou_pct']:.1f}%<br>"
            f"sepZy={ov['y_sep_z']:.2f} pY~{_fmt_p(ov['p_sep_y_perm'])}<br>"
            f"sepZ2D={ov['sep_2d_z']:.2f} p2D~{_fmt_p(ov['p_sep_2d_perm'])}<br>"
            f"critY={ov['critical_y_level']} ({ov['critical_y_index']:.1f}) | "
            f"crit2D={ov['critical_2d_level']} ({ov['critical_2d_index']:.1f})<br>"
            f"pCritY~{_fmt_p(ov['p_critical_y_perm'])} pCrit2D~{_fmt_p(ov['p_critical_2d_perm'])}<br>"
            f"{pair_lines[0]}<br>{pair_lines[1]}<br>{pair_lines[2]}"
        )
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="x domain",
            yref="y domain",
            text=ov_txt,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            bordercolor="rgba(120,120,120,0.5)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=10),
            row=r,
            col=c,
        )

        fig.update_xaxes(
            row=r,
            col=c,
            title_text="Voltage AUC (a.u.*s)",
            zeroline=True,
            zerolinewidth=1.0,
            zerolinecolor="rgba(0,0,0,0.25)",
        )
        fig.update_yaxes(
            row=r,
            col=c,
            title_text=ytitle,
            zeroline=True,
            zerolinewidth=1.0,
            zerolinecolor="rgba(0,0,0,0.25)",
        )

    fig.update_layout(
        template="simple_white",
        width=1850,
        height=980,
        title=title_prefix,
        legend=dict(orientation="h"),
    )

    include_plateau = _has_plateau_events(metrics_df) or _has_plateau_events(cell_avg_df)
    tagged_html, tagged_svg, tagged_pdf = _tag_figure_paths(
        html_path=save_html,
        svg_path=save_svg,
        pdf_path=save_pdf,
        include_plateau=include_plateau,
        uses_robust_calcium=True,
    )
    _save_fig_triplet(
        fig,
        html_path=tagged_html,
        svg_path=tagged_svg,
        pdf_path=tagged_pdf,
        warn_prefix="AUC-vs-calcium-response-summary-of-summaries",
    )
    perm_html = _append_suffix_before_ext(tagged_html, "_permutation_tests") if tagged_html else None
    perm_svg = _append_suffix_before_ext(tagged_svg, "_permutation_tests") if tagged_svg else None
    _save_auc_permutation_tests_figure(
        panel_perm_records=panel_perm_records,
        save_html=perm_html,
        save_svg=perm_svg,
        title_prefix=f"{title_prefix} | permutation diagnostics",
    )
    return fig


def save_cell_by_cell_figures(metrics_df, out_dir=None, n_cols=5):
    if metrics_df is None or len(metrics_df) == 0:
        print("[CELL-BY-CELL] skipped (empty metrics_df)")
        return []
    if "cell_folder" not in metrics_df.columns:
        print("[CELL-BY-CELL] skipped (missing 'cell_folder' column)")
        return []

    if out_dir is None:
        out_dir = CELL_BY_CELL_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    df = metrics_df.copy()
    df["cell_folder"] = df["cell_folder"].astype(str)
    cell_list = sorted(df["cell_folder"].dropna().unique().tolist())
    n_cells = len(cell_list)
    if n_cells == 0:
        print("[CELL-BY-CELL] skipped (no cells)")
        return []

    n_rows, n_cols = _grid_shape(n_cells, n_cols=n_cols)
    subplot_titles = [_short_cell_label(c) for c in cell_list]
    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}
    saved_paths = []

    beeswarm_specs = [
        ("peak_df_f", "Peak (dF/F)", "cell_by_cell_peak_dff_beeswarm"),
        (
            "peak_norm_complex4_mean",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "cell_by_cell_peak_norm_complex4_mean_beeswarm",
        ),
        ("peak_z", "Peak (z-score)", "cell_by_cell_peak_z_beeswarm"),
        ("peak_z_cal_p8_robust", "Peak robust z-score (calcium p8+MAD)", "cell_by_cell_peak_z_robust_beeswarm"),
        ("auc", "AUC", "cell_by_cell_auc_beeswarm"),
        ("hwhm_s", "HWHM (s)", "cell_by_cell_hwhm_beeswarm"),
        ("decay_time_s", "Decay time (s)", "cell_by_cell_decay_time_beeswarm"),
        ("tau_decay_s", "Decay tau (s)", "cell_by_cell_tau_decay_beeswarm"),
    ]

    for ycol, ytitle, stem in beeswarm_specs:
        if ycol not in df.columns:
            continue
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=max(0.03, 0.18 / max(1, n_rows)),
        )
        for i, cell_folder in enumerate(cell_list):
            r = (i // n_cols) + 1
            c = (i % n_cols) + 1
            sub_cell = df[df["cell_folder"] == cell_folder]
            for ev_type in EVENT_TYPES:
                if "event_type" not in sub_cell.columns:
                    continue
                sub = sub_cell[sub_cell["event_type"] == ev_type]
                _add_beeswarm_points_and_fit(
                    fig=fig,
                    sub_df=sub,
                    ycol=ycol,
                    ev_type=ev_type,
                    color=color_map[ev_type],
                    label=label_map[ev_type],
                    showlegend=(i == 0),
                    row=r,
                    col=c,
                    seed=7000 + i * 101 + (11 if ev_type == "simple" else 19) + len(stem),
                )
            fig.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
            if c == 1:
                fig.update_yaxes(title_text=ytitle, row=r, col=c)

        fig.update_layout(
            template="simple_white",
            width=max(1500, 360 * n_cols),
            height=max(550, 300 * n_rows + 120),
            title=f"Cell-by-cell | {ytitle}",
            legend=dict(orientation="h"),
        )
        html_path = os.path.join(out_dir, f"{stem}.html")
        svg_path = os.path.join(out_dir, f"{stem}.svg")
        _save_fig_pair(fig, html_path=html_path, svg_path=svg_path, warn_prefix="Cell-by-cell")
        saved_paths.extend([html_path, svg_path])

    if "peak_z" in df.columns and "event_type" in df.columns and "n_spikes" in df.columns:
        fig_v = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=max(0.03, 0.18 / max(1, n_rows)),
        )
        for i, cell_folder in enumerate(cell_list):
            r = (i // n_cols) + 1
            c = (i % n_cols) + 1
            sub_cell = df[df["cell_folder"] == cell_folder].copy()
            sub_cell = sub_cell[np.isfinite(sub_cell["peak_z"]) & np.isfinite(sub_cell["n_spikes"])]
            if len(sub_cell) == 0:
                fig_v.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
                if c == 1:
                    fig_v.update_yaxes(title_text="Peak (z-score)", row=r, col=c)
                continue

            sub_cell["n_spikes_cap"] = _cap_n_spikes(sub_cell["n_spikes"].values, cap=SPIKE_COUNT_CAP).astype(int)
            simple = sub_cell[sub_cell["event_type"] == "simple"].copy()
            complex_df = sub_cell[sub_cell["event_type"] == "complex"].copy()
            plateau_df = sub_cell[sub_cell["event_type"] == "plateau"].copy()

            if len(simple):
                fig_v.add_trace(
                    go.Violin(
                        x=simple["n_spikes_cap"],
                        y=simple["peak_z"],
                        name="Simple/Burst",
                        legendgroup="simple",
                        line_color="red",
                        fillcolor="rgba(220,20,60,0.35)",
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        width=0.95,
                        showlegend=(i == 0),
                    ),
                    row=r, col=c,
                )
            if len(complex_df):
                fig_v.add_trace(
                    go.Violin(
                        x=complex_df["n_spikes_cap"],
                        y=complex_df["peak_z"],
                        name="Complex",
                        legendgroup="complex",
                        line_color="black",
                        fillcolor="rgba(0,0,0,0.25)",
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        width=0.95,
                        showlegend=(i == 0),
                    ),
                    row=r, col=c,
                )
            if len(plateau_df):
                fig_v.add_trace(
                    go.Violin(
                        x=plateau_df["n_spikes_cap"],
                        y=plateau_df["peak_z"],
                        name="Plateau",
                        legendgroup="plateau",
                        line_color="purple",
                        fillcolor="rgba(128,0,128,0.25)",
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        width=0.95,
                        showlegend=(i == 0),
                    ),
                    row=r, col=c,
                )

            _add_fit_traces(
                fig=fig_v,
                x=simple["n_spikes_cap"].values if len(simple) else [],
                y=simple["peak_z"].values if len(simple) else [],
                color="red",
                label="Simple/Burst",
                legendgroup="simple",
                showlegend=(i == 0),
                row=r,
                col=c,
            )
            _add_fit_traces(
                fig=fig_v,
                x=complex_df["n_spikes_cap"].values if len(complex_df) else [],
                y=complex_df["peak_z"].values if len(complex_df) else [],
                color="black",
                label="Complex",
                legendgroup="complex",
                showlegend=(i == 0),
                row=r,
                col=c,
            )
            _add_fit_traces(
                fig=fig_v,
                x=plateau_df["n_spikes_cap"].values if len(plateau_df) else [],
                y=plateau_df["peak_z"].values if len(plateau_df) else [],
                color="purple",
                label="Plateau",
                legendgroup="plateau",
                showlegend=(i == 0),
                row=r,
                col=c,
            )

            fig_v.update_xaxes(row=r, col=c, **_spike_count_axis_kwargs(cap=SPIKE_COUNT_CAP))
            if c == 1:
                fig_v.update_yaxes(title_text="Peak (z-score)", row=r, col=c)

        fig_v.update_layout(
            template="simple_white",
            width=max(1500, 360 * n_cols),
            height=max(550, 300 * n_rows + 120),
            title="Cell-by-cell | Peak (z-score) violin by spike count",
            legend=dict(orientation="h"),
            violinmode="overlay",
            violingap=0.02,
        )
        stem = "cell_by_cell_peak_z_violin"
        html_path = os.path.join(out_dir, f"{stem}.html")
        svg_path = os.path.join(out_dir, f"{stem}.svg")
        _save_fig_pair(fig_v, html_path=html_path, svg_path=svg_path, warn_prefix="Cell-by-cell")
        saved_paths.extend([html_path, svg_path])

    print(f"[CELL-BY-CELL] saved figures to: {out_dir}")
    return saved_paths


def save_cell_by_cell_auc_vs_calcium_response_figures(metrics_df, out_dir=None, n_cols=5):
    """
    Cell-by-cell summary: each subplot is one cell, x=calcium AUC, y=calcium response metric.
    Saves both event-based and cell-average based versions.
    """
    if metrics_df is None or len(metrics_df) == 0:
        print("[CELL-BY-CELL-AUC] skipped (empty metrics_df)")
        return []
    if "cell_folder" not in metrics_df.columns:
        print("[CELL-BY-CELL-AUC] skipped (missing 'cell_folder' column)")
        return []

    if out_dir is None:
        out_dir = CELL_BY_CELL_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    df_events = metrics_df.copy()
    df_events["cell_folder"] = df_events["cell_folder"].astype(str)
    cell_list = sorted(df_events["cell_folder"].dropna().unique().tolist())
    n_cells = len(cell_list)
    if n_cells == 0:
        print("[CELL-BY-CELL-AUC] skipped (no cells)")
        return []

    df_cellavg = _build_cell_average_metrics_by_vol_auc_bins(df_events, n_bins=AUC_BIN_COUNT)
    datasets = [
        (df_events, "events"),
        (df_cellavg, "cellavg"),
    ]

    y_specs = [
        ("peak_df_f", "Peak (dF/F)", "peak_dff"),
        (
            "peak_norm_complex4_mean",
            f"Peak normalized by mean complex n={int(COMPLEX4_REF_N_SPIKES)}",
            "peak_norm_complex4_mean",
        ),
        ("peak_z", f"Peak z-score (non-robust, low{CAL_Z_NON_ROBUST_LOW_PERCENTILE:g})", "peak_z"),
        ("peak_z_cal_p8_robust", "Peak robust z-score (p8+MAD)", "peak_z_robust"),
    ]
    color_map = {"simple": "red", "complex": "black", "plateau": "purple"}
    label_map = {"simple": "Simple/Burst", "complex": "Complex", "plateau": "Plateau"}
    saved_paths = []
    n_rows, n_cols = _grid_shape(n_cells, n_cols=n_cols)
    subplot_titles = [_short_cell_label(c) for c in cell_list]

    for dfi, dtag in datasets:
        if dfi is None or len(dfi) == 0:
            continue
        for ycol, ytitle, stem_tag in y_specs:
            if (ycol not in dfi.columns) or ("vol_auc" not in dfi.columns) or ("event_type" not in dfi.columns):
                continue

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.05,
                vertical_spacing=max(0.03, 0.18 / max(1, n_rows)),
            )

            for i, cell_folder in enumerate(cell_list):
                r = (i // n_cols) + 1
                c = (i % n_cols) + 1
                sub_cell = dfi[dfi["cell_folder"].astype(str) == str(cell_folder)].copy()
                for ev_type in EVENT_TYPES:
                    sub = sub_cell[sub_cell["event_type"] == ev_type].copy()
                    if len(sub) == 0:
                        continue
                    x = pd.to_numeric(sub["vol_auc"], errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if not np.any(m):
                        continue
                    x = x[m]
                    y = y[m]

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            name=label_map[ev_type],
                            legendgroup=f"cb_auc_{ev_type}_{dtag}_{stem_tag}",
                            marker=dict(color=color_map[ev_type], size=6, opacity=0.80),
                            showlegend=(i == 0),
                        ),
                        row=r,
                        col=c,
                    )
                    _add_fit_traces(
                        fig=fig,
                        x=x,
                        y=y,
                        color=color_map[ev_type],
                        label=label_map[ev_type],
                        legendgroup=f"cb_auc_{ev_type}_{dtag}_{stem_tag}",
                        showlegend=(i == 0),
                        row=r,
                        col=c,
                    )

                if c == 1:
                    fig.update_yaxes(title_text=ytitle, row=r, col=c)
                fig.update_xaxes(title_text="Voltage AUC (a.u.*s)", row=r, col=c)

            fig.update_layout(
                template="simple_white",
                width=max(1500, 360 * n_cols),
                height=max(550, 300 * n_rows + 120),
                title=f"Cell-by-cell | {ytitle} vs voltage AUC ({dtag})",
                legend=dict(orientation="h"),
            )
            stem = f"cell_by_cell_vol_auc_vs_{stem_tag}_{dtag}"
            html_base = os.path.join(out_dir, f"{stem}.html")
            svg_base = os.path.join(out_dir, f"{stem}.svg")
            uses_robust = (str(ycol).lower() == "peak_z_cal_p8_robust")
            html_path, svg_path, _ = _tag_figure_paths(
                html_path=html_base,
                svg_path=svg_base,
                pdf_path=None,
                include_plateau=_has_plateau_events(dfi),
                uses_robust_calcium=uses_robust,
            )
            _save_fig_pair(fig, html_path=html_path, svg_path=svg_path, warn_prefix="Cell-by-cell-AUC")
            saved_paths.extend([html_path, svg_path])

    if len(saved_paths) > 0:
        print(f"[CELL-BY-CELL-AUC] saved figures to: {out_dir}")
    return saved_paths


def _prepare_event_selection_df(events_df):
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame()
    df = events_df.copy()
    need_cols = {"cell_folder", "event_type", "n_spikes", "include"}
    if not need_cols.issubset(set(df.columns)):
        return pd.DataFrame()
    df["cell_folder"] = df["cell_folder"].astype(str)
    df["event_type"] = df["event_type"].astype(str).str.lower().map(_norm_event_type)

    # Plateau priority canonicalization:
    # if an event is dual-class (complex+plateau), count it as plateau only.
    raw_plateau = (df["event_type"] == "plateau")
    if "is_plateau_event" in df.columns:
        raw_plateau = raw_plateau | df["is_plateau_event"].astype(bool)
    raw_complex = (df["event_type"] == "complex")
    if "is_complex_event" in df.columns:
        raw_complex = raw_complex | df["is_complex_event"].astype(bool)

    df["is_plateau_event"] = raw_plateau.astype(bool)
    df["is_complex_event"] = (raw_complex & (~df["is_plateau_event"])).astype(bool)
    df["is_dual_class_event"] = (raw_plateau & raw_complex).astype(bool)
    df.loc[df["is_plateau_event"], "event_type"] = "plateau"
    df.loc[(~df["is_plateau_event"]) & df["is_complex_event"], "event_type"] = "complex"
    df.loc[(~df["is_plateau_event"]) & (~df["is_complex_event"]), "event_type"] = "simple"
    df["include"] = df["include"].astype(bool)
    df["n_spikes"] = pd.to_numeric(df["n_spikes"], errors="coerce")
    df = df[np.isfinite(df["n_spikes"])].copy()
    if len(df) == 0:
        return pd.DataFrame()
    df["n_spikes"] = df["n_spikes"].astype(int)
    if "n_spikes_cap" not in df.columns:
        df["n_spikes_cap"] = np.minimum(df["n_spikes"].values, int(SPIKE_COUNT_CAP)).astype(int)
    else:
        df["n_spikes_cap"] = pd.to_numeric(df["n_spikes_cap"], errors="coerce")
        bad = ~np.isfinite(df["n_spikes_cap"])
        if np.any(bad):
            df.loc[bad, "n_spikes_cap"] = df.loc[bad, "n_spikes"]
        df["n_spikes_cap"] = np.minimum(df["n_spikes_cap"].astype(int), int(SPIKE_COUNT_CAP))
    df["n_spikes_cap"] = np.maximum(df["n_spikes_cap"].astype(int), 1)
    return df


def _event_selection_x_labels(cap=SPIKE_COUNT_CAP):
    labels = []
    keys = []
    for n in range(1, int(cap) + 1):
        nlab = f"{int(cap)}+" if n == int(cap) else str(n)
        labels.append(f"S-{nlab}")
        keys.append(("simple", int(n)))
        labels.append(f"C-{nlab}")
        keys.append(("complex", int(n)))
        labels.append(f"P-{nlab}")
        keys.append(("plateau", int(n)))
    return labels, keys


def _event_selection_counts_for_subset(sub_df, cap=SPIKE_COUNT_CAP):
    labels, keys = _event_selection_x_labels(cap=cap)
    sel_vals = []
    nosel_vals = []

    for ev_type, ncap in keys:
        m_type = (sub_df["event_type"] == ev_type)

        m = m_type & (sub_df["n_spikes_cap"] == int(ncap))
        sel_vals.append(int((m & sub_df["include"]).sum()))
        nosel_vals.append(int((m & (~sub_df["include"])).sum()))
    return labels, sel_vals, nosel_vals


def _filter_rows_to_overlay_source_per_cell(df):
    """
    Keep only rows from the same per-cell source used by event_spike_overlay_main:
    prefer suffix 'main'; if unavailable, fallback to the first pkl by natural key.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if isinstance(df, pd.DataFrame) else df
    if "cell_folder" not in df.columns:
        return df.copy()

    w = df.copy()
    w["cell_folder"] = w["cell_folder"].astype(str)
    if "state_suffix" in w.columns:
        w["__suffix__"] = w["state_suffix"].astype(str).str.lower()
    elif "suffix" in w.columns:
        w["__suffix__"] = w["suffix"].astype(str).str.lower()
    elif "pkl_name" in w.columns:
        w["__suffix__"] = w["pkl_name"].astype(str).map(_suffix_from_pkl_name).astype(str).str.lower()
    else:
        w["__suffix__"] = "main"

    if "pkl_name" in w.columns:
        w["__pkl_sort_key__"] = w["pkl_name"].astype(str).map(_natural_pkl_key)
    else:
        w["__pkl_sort_key__"] = ""

    keep_mask = np.zeros(len(w), dtype=bool)
    for _, idx in w.groupby("cell_folder").groups.items():
        sub = w.loc[idx]
        if np.any(sub["__suffix__"] == "main"):
            keep_idx = sub.index[sub["__suffix__"] == "main"].to_numpy()
        else:
            first_pkl = sub.sort_values("__pkl_sort_key__", kind="mergesort").iloc[0]["__pkl_sort_key__"]
            keep_idx = sub.index[sub["__pkl_sort_key__"] == first_pkl].to_numpy()
        keep_mask[w.index.get_indexer(keep_idx)] = True

    out = w.loc[keep_mask].copy()
    out.drop(columns=["__suffix__", "__pkl_sort_key__"], inplace=True, errors="ignore")
    return out



def _plot_event_selection_two_panel(events_df, title_prefix, save_html=None, save_svg=None, save_pdf=None):
    df = _prepare_event_selection_df(events_df)
    if len(df) == 0:
        return None

    n_total = int(len(df))
    n_sel = int(df["include"].sum())
    n_not = int(n_total - n_sel)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=(
            "Selected vs total events",
            "Selected by # spikes and event type",
        ),
        horizontal_spacing=0.16,
    )

    if n_total > 0:
        fig.add_trace(
            go.Pie(
                labels=["selected", "not selected"],
                values=[n_sel, n_not],
                marker=dict(colors=["#7b2cbf", "#8ecae6"]),
                sort=False,
                textinfo="percent+value",
                texttemplate="%{percent}<br>n=%{value}",
                textposition="inside",
                insidetextfont=dict(size=11),
                showlegend=False,
                hole=0.25,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Pie(
                labels=["no events"],
                values=[1],
                marker=dict(colors=["#cccccc"]),
                textinfo="label",
                showlegend=False,
                hole=0.25,
            ),
            row=1, col=1,
        )

    x_labels, sel_vals, nosel_vals = _event_selection_counts_for_subset(df, cap=SPIKE_COUNT_CAP)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=sel_vals,
            name="selected",
            marker=dict(color="#7b2cbf"),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=nosel_vals,
            name="not selected",
            marker=dict(color="#8ecae6"),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="event type + # spikes", tickangle=35, row=1, col=2)
    fig.update_yaxes(title_text="# events", row=1, col=2)
    fig.update_layout(
        template="simple_white",
        barmode="stack",
        width=1300,
        height=520,
        title=f"{title_prefix}<br><sup>selected={n_sel} / total={n_total}</sup>",
        legend=dict(orientation="h"),
    )
    _save_fig_triplet(fig, html_path=save_html, svg_path=save_svg, pdf_path=save_pdf, warn_prefix="Event-selection")
    return fig


def save_event_selection_cell_by_cell_figures(events_df, out_dir, n_cols=5):
    df = _prepare_event_selection_df(events_df)
    if len(df) == 0:
        print("[EVENT-CHOSS] skipped cell-by-cell (no events)")
        return []
    os.makedirs(str(out_dir), exist_ok=True)

    cell_list = sorted(df["cell_folder"].dropna().unique().tolist())
    n_cells = len(cell_list)
    if n_cells == 0:
        print("[EVENT-CHOSS] skipped cell-by-cell (no cells)")
        return []
    n_rows, n_cols = _grid_shape(n_cells, n_cols=n_cols)
    subplot_titles = [_short_cell_label(c) for c in cell_list]

    saved = []

    pie_specs = [[{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig_pie = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=pie_specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=max(0.03, 0.18 / max(1, n_rows)),
    )
    for i, cell_folder in enumerate(cell_list):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        sub = df[df["cell_folder"] == cell_folder]
        n_total = int(len(sub))
        n_sel = int(sub["include"].sum()) if n_total > 0 else 0
        n_not = int(n_total - n_sel)
        if n_total > 0:
            fig_pie.add_trace(
                go.Pie(
                    labels=["selected", "not selected"],
                    values=[n_sel, n_not],
                    marker=dict(colors=["#7b2cbf", "#8ecae6"]),
                    sort=False,
                    textinfo="percent+value",
                    texttemplate="%{percent}<br>n=%{value}",
                    textposition="inside",
                    insidetextfont=dict(size=9),
                    showlegend=(i == 0),
                    hole=0.30,
                ),
                row=r, col=c,
            )
        else:
            fig_pie.add_trace(
                go.Pie(
                    labels=["no events"],
                    values=[1],
                    marker=dict(colors=["#cccccc"]),
                    textinfo="none",
                    showlegend=False,
                    hole=0.30,
                ),
                row=r, col=c,
            )
    fig_pie.update_layout(
        template="simple_white",
        width=max(1500, 340 * n_cols),
        height=max(560, 300 * n_rows + 120),
        title="Cell-by-cell | Event selection ratio (selected vs not selected)",
        legend=dict(orientation="h"),
    )
    pie_html = os.path.join(out_dir, "cell_by_cell_eventChoss_pie.html")
    pie_svg = os.path.join(out_dir, "cell_by_cell_eventChoss_pie.svg")
    pie_pdf = os.path.join(out_dir, "cell_by_cell_eventChoss_pie.pdf")
    _save_fig_triplet(fig_pie, html_path=pie_html, svg_path=pie_svg, pdf_path=pie_pdf, warn_prefix="EventChoss cell-by-cell")
    saved.extend([pie_html, pie_svg, pie_pdf])

    fig_bar = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=max(0.03, 0.18 / max(1, n_rows)),
    )
    for i, cell_folder in enumerate(cell_list):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        sub = df[df["cell_folder"] == cell_folder]
        x_labels, sel_vals, nosel_vals = _event_selection_counts_for_subset(sub, cap=SPIKE_COUNT_CAP)
        fig_bar.add_trace(
            go.Bar(
                x=x_labels,
                y=sel_vals,
                name="selected",
                marker=dict(color="#7b2cbf"),
                showlegend=(i == 0),
            ),
            row=r, col=c,
        )
        fig_bar.add_trace(
            go.Bar(
                x=x_labels,
                y=nosel_vals,
                name="not selected",
                marker=dict(color="#8ecae6"),
                showlegend=(i == 0),
            ),
            row=r, col=c,
        )
        fig_bar.update_xaxes(title_text="S/C by # spikes", tickangle=35, row=r, col=c)
        if c == 1:
            fig_bar.update_yaxes(title_text="# events", row=r, col=c)

    fig_bar.update_layout(
        template="simple_white",
        barmode="stack",
        width=max(1500, 360 * n_cols),
        height=max(560, 300 * n_rows + 120),
        title="Cell-by-cell | Event selection by # spikes and event type",
        legend=dict(orientation="h"),
    )
    bar_html = os.path.join(out_dir, "cell_by_cell_eventChoss_stacked.html")
    bar_svg = os.path.join(out_dir, "cell_by_cell_eventChoss_stacked.svg")
    bar_pdf = os.path.join(out_dir, "cell_by_cell_eventChoss_stacked.pdf")
    _save_fig_triplet(fig_bar, html_path=bar_html, svg_path=bar_svg, pdf_path=bar_pdf, warn_prefix="EventChoss cell-by-cell")
    saved.extend([bar_html, bar_svg, bar_pdf])

    print(f"[EVENT-CHOSS] saved cell-by-cell figures to: {out_dir}")
    return saved


def save_event_selection_population_figures(events_df, out_dir):
    df = _prepare_event_selection_df(events_df)
    if len(df) == 0:
        print("[EVENT-CHOSS] skipped population (no events)")
        return []
    os.makedirs(str(out_dir), exist_ok=True)

    saved = []

    n_total = int(len(df))
    n_sel = int(df["include"].sum())
    n_not = int(n_total - n_sel)
    fig_pie = go.Figure()
    fig_pie.add_trace(
        go.Pie(
            labels=["selected", "not selected"],
            values=[n_sel, n_not],
            marker=dict(colors=["#7b2cbf", "#8ecae6"]),
            sort=False,
            textinfo="percent+value",
            texttemplate="%{percent}<br>n=%{value}",
            textposition="inside",
            insidetextfont=dict(size=12),
            hole=0.28,
        )
    )
    fig_pie.update_layout(
        template="simple_white",
        width=700,
        height=560,
        title=f"Population | Event selection ratio<br><sup>selected={n_sel} / total={n_total}</sup>",
    )
    pop_pie_html = os.path.join(out_dir, "population_eventChoss_pie.html")
    pop_pie_svg = os.path.join(out_dir, "population_eventChoss_pie.svg")
    pop_pie_pdf = os.path.join(out_dir, "population_eventChoss_pie.pdf")
    _save_fig_triplet(fig_pie, html_path=pop_pie_html, svg_path=pop_pie_svg, pdf_path=pop_pie_pdf, warn_prefix="EventChoss population")
    saved.extend([pop_pie_html, pop_pie_svg, pop_pie_pdf])

    x_labels, sel_vals, nosel_vals = _event_selection_counts_for_subset(df, cap=SPIKE_COUNT_CAP)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=x_labels, y=sel_vals, name="selected", marker=dict(color="#7b2cbf")))
    fig_bar.add_trace(go.Bar(x=x_labels, y=nosel_vals, name="not selected", marker=dict(color="#8ecae6")))
    fig_bar.update_layout(
        template="simple_white",
        barmode="stack",
        width=1100,
        height=560,
        title="Population | Event selection by # spikes and event type",
        xaxis=dict(title="event type + # spikes", tickangle=35),
        yaxis=dict(title="# events"),
        legend=dict(orientation="h"),
    )
    pop_bar_html = os.path.join(out_dir, "population_eventChoss_stacked.html")
    pop_bar_svg = os.path.join(out_dir, "population_eventChoss_stacked.svg")
    pop_bar_pdf = os.path.join(out_dir, "population_eventChoss_stacked.pdf")
    _save_fig_triplet(fig_bar, html_path=pop_bar_html, svg_path=pop_bar_svg, pdf_path=pop_bar_pdf, warn_prefix="EventChoss population")
    saved.extend([pop_bar_html, pop_bar_svg, pop_bar_pdf])

    print(f"[EVENT-CHOSS] saved population figures to: {out_dir}")
    return saved





# -------------------------------
# Run on all PyrLowFR cells
# -------------------------------
DB_PATH = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\PyrLowFR.csv"
POP_SUMMARY_OUT_DIR = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\Pyr\calciumRes"
SUMMARY_HTML = os.path.join(POP_SUMMARY_OUT_DIR, "PyrLowFR_event_metrics_summary.html")
SUMMARY_SVG = os.path.join(POP_SUMMARY_OUT_DIR, "PyrLowFR_event_metrics_summary.svg")
SUMMARY_CSV = os.path.join(POP_SUMMARY_OUT_DIR, "PyrLowFR_event_metrics_all_events.csv")
CELL_BY_CELL_OUT_DIR = POP_SUMMARY_OUT_DIR


def run_pyr_event_cal2_pipeline(db_path=None, summary_html=SUMMARY_HTML, summary_svg=SUMMARY_SVG, summary_csv=SUMMARY_CSV,
                                max_cells=None, cell_by_cell_out_dir=CELL_BY_CELL_OUT_DIR, cell_by_cell_n_cols=5):
    if db_path is None:
        db_path = DB_PATH
    pop_root = None
    for p in (summary_html, summary_svg, summary_csv):
        if p:
            pop_root = os.path.dirname(str(p))
            if pop_root:
                break
    if not pop_root:
        pop_root = POP_SUMMARY_OUT_DIR

    event_base_dir = os.path.join(str(pop_root), "event_base")
    cell_avg_dir = os.path.join(str(pop_root), "cell_avg")
    event_choss_dir = os.path.join(str(pop_root), "eventChoss")
    os.makedirs(event_base_dir, exist_ok=True)
    os.makedirs(cell_avg_dir, exist_ok=True)
    os.makedirs(event_choss_dir, exist_ok=True)

    summary_html_event = os.path.join(event_base_dir, os.path.basename(str(summary_html))) if summary_html else None
    summary_svg_event = os.path.join(event_base_dir, os.path.basename(str(summary_svg))) if summary_svg else None
    summary_csv_event = os.path.join(event_base_dir, os.path.basename(str(summary_csv))) if summary_csv else None

    if cell_by_cell_out_dir:
        os.makedirs(str(cell_by_cell_out_dir), exist_ok=True)

    db = pd.read_csv(db_path)
    if max_cells is not None:
        db = db.iloc[:int(max_cells)].copy()

    all_metrics = []
    all_event_selection = []

    for row_idx, row in db.iterrows():
        cell_folder = str(row["Link"])
        cal_sr = _safe_cal_sr(row.get("CALsr", 30.0), default=30.0)
        is_motor_cell = _is_motor_row(row)

        if not os.path.isdir(cell_folder):
            print(f"[SKIP] missing folder: {cell_folder}")
            continue

        pkl_list = _find_spike_pkls(cell_folder)
        if len(pkl_list) == 0:
            print(f"[SKIP] no spike pkl found: {cell_folder}")
            continue

        print("\n" + "=" * 110)
        print(f"Cell {row_idx + 1}/{len(db)} | {cell_folder} | CALsr={cal_sr} | pkls={len(pkl_list)}")

        cal_nb, cal_df = _compute_calcium_nb_and_df(
            cell_folder,
            neuropil_r=CAL_NEUROPIL_R,
            f0_percentile=CAL_F0_PERCENTILE,
            save_outputs=True,
        )
        if cal_df is not None and np.asarray(cal_df).size > 0:
            global_trace_cal_override = np.asarray(cal_df, dtype=float).ravel()
            print(f"[CAL] global trace: recomputed calTraceDF from raw+neuropil (r={CAL_NEUROPIL_R}, F0 p{CAL_F0_PERCENTILE:g})")
        else:
            global_trace_cal_override = None
            print("[CAL] warning: could not recompute global calTraceDF from raw+neuropil; fallback to pkl trace_cal when needed")

        cell_metrics = []
        cell_event_selection = []
        cell_outputs = []
        state_metrics = {}
        state_event_selection = {}

        for pkl_path in pkl_list:
            try:
                suffix_raw = _suffix_from_pkl_name(pkl_path)
                suffix_tag = _normalize_suffix_tag(suffix_raw)
                state_label = _motor_state_from_suffix(suffix_raw)
                seg_cal, seg_src = _load_segment_cal_trace_by_suffix(cell_folder, suffix_raw)
                if seg_cal is not None and np.asarray(seg_cal).size > 0:
                    trace_cal_override = np.asarray(seg_cal, dtype=float).ravel()
                    print(f"[CAL] {os.path.basename(pkl_path)} -> using segment calcium trace: {os.path.basename(seg_src)} (len={trace_cal_override.size})")
                else:
                    trace_cal_override = global_trace_cal_override
                    if trace_cal_override is not None and np.asarray(trace_cal_override).size > 0:
                        print(f"[CAL] {os.path.basename(pkl_path)} -> using global calTraceDF (len={trace_cal_override.size})")
                    else:
                        print(f"[CAL] {os.path.basename(pkl_path)} -> no segment/global override, using trace_cal from pkl")

                mdf, out_info, event_sel_df = _analyze_single_pkl(
                    cell_folder=cell_folder,
                    pkl_path=pkl_path,
                    cal_sr=cal_sr,
                    vol_sr=VOL_SR,
                    ratio_thr=TAIL_RATIO_THR,
                    trace_cal_override=trace_cal_override,
                )

                if event_sel_df is not None and len(event_sel_df) > 0:
                    event_sel_df = event_sel_df.copy()
                    event_sel_df["state_suffix"] = suffix_tag
                    event_sel_df["motor_state"] = state_label if state_label else "main"
                    cell_event_selection.append(event_sel_df)
                    all_event_selection.append(event_sel_df)
                    if state_label:
                        state_event_selection.setdefault(state_label, []).append(event_sel_df)

                if mdf is not None and len(mdf) > 0:
                    mdf = mdf.copy()
                    mdf["state_suffix"] = suffix_tag
                    mdf["motor_state"] = state_label if state_label else "main"
                    all_metrics.append(mdf)
                    cell_metrics.append(mdf)
                    if state_label:
                        state_metrics.setdefault(state_label, []).append(mdf)

                    per_pkl_html = os.path.join(cell_folder, f"event_metrics_4panel_by_{suffix_tag}.html")
                    per_pkl_svg = os.path.join(cell_folder, f"event_metrics_4panel_by_{suffix_tag}.svg")
                    _plot_summary(
                        mdf,
                        save_html=per_pkl_html,
                        save_svg=per_pkl_svg,
                        show_plot=False,
                        title_prefix=f"{os.path.basename(cell_folder)} | metrics for {os.path.basename(pkl_path)}",
                    )
                    per_pkl_decay_html = os.path.join(cell_folder, f"event_metrics_decay_time_by_{suffix_tag}.html")
                    per_pkl_decay_svg = os.path.join(cell_folder, f"event_metrics_decay_time_by_{suffix_tag}.svg")
                    per_pkl_decay_pdf = os.path.join(cell_folder, f"event_metrics_decay_time_by_{suffix_tag}.pdf")
                    _save_decay_time_individual_summary(
                        mdf,
                        save_html=per_pkl_decay_html,
                        save_svg=per_pkl_decay_svg,
                        save_pdf=per_pkl_decay_pdf,
                        title_prefix=f"{os.path.basename(cell_folder)} | metrics for {os.path.basename(pkl_path)}",
                    )
                    per_pkl_vauc_html = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_cal_peak_by_{suffix_tag}.html")
                    per_pkl_vauc_svg = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_cal_peak_by_{suffix_tag}.svg")
                    per_pkl_vauc_pdf = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_cal_peak_by_{suffix_tag}.pdf")
                    _plot_voltage_auc_vs_calcium_peak(
                        mdf,
                        save_html=per_pkl_vauc_html,
                        save_svg=per_pkl_vauc_svg,
                        save_pdf=per_pkl_vauc_pdf,
                        title_prefix=f"{os.path.basename(cell_folder)} | voltage AUC vs calcium peak ({os.path.basename(pkl_path)})",
                    )
                    print(f"[PKL-SUMMARY] saved: {per_pkl_html}")

                if out_info is not None:
                    cell_outputs.append(out_info)

            except Exception as e:
                print(f"[ERROR] {pkl_path}: {e}")

        if len(cell_outputs) > 0:
            main_html, main_svg, src_html = _ensure_main_overlay_for_cell(cell_folder, cell_outputs)
            src_txt = os.path.basename(src_html) if src_html else "n/a"
            print(f"[MAIN-OVERLAY] {os.path.basename(main_html)} + {os.path.basename(main_svg)} (source={src_txt})")
            main_v_html, main_v_svg, src_v_html = _ensure_main_voltage_overlay_for_cell(cell_folder, cell_outputs)
            src_v_txt = os.path.basename(src_v_html) if src_v_html else "n/a"
            print(f"[MAIN-V-OVERLAY] {os.path.basename(main_v_html)} + {os.path.basename(main_v_svg)} (source={src_v_txt})")
        else:
            main_html, main_svg = _save_placeholder_main_overlay(cell_folder)
            print(f"[MAIN-OVERLAY] placeholder saved: {main_html}")

        if len(cell_metrics) > 0:
            cell_df = pd.concat(cell_metrics, ignore_index=True)
            cell_summary_html = os.path.join(cell_folder, "event_metrics_4panel_main.html")
            cell_summary_svg = os.path.join(cell_folder, "event_metrics_4panel_main.svg")
            _plot_summary(
                cell_df,
                save_html=cell_summary_html,
                save_svg=cell_summary_svg,
                show_plot=False,
                title_prefix=f"{os.path.basename(cell_folder)} | chosen calcium events vs spike count",
            )
            cell_decay_html = os.path.join(cell_folder, "event_metrics_decay_time_main.html")
            cell_decay_svg = os.path.join(cell_folder, "event_metrics_decay_time_main.svg")
            cell_decay_pdf = os.path.join(cell_folder, "event_metrics_decay_time_main.pdf")
            _save_decay_time_individual_summary(
                cell_df,
                save_html=cell_decay_html,
                save_svg=cell_decay_svg,
                save_pdf=cell_decay_pdf,
                title_prefix=f"{os.path.basename(cell_folder)} | chosen calcium events vs spike count",
            )
            cell_vauc_html = os.path.join(cell_folder, "event_metrics_vol_auc_vs_cal_peak_main.html")
            cell_vauc_svg = os.path.join(cell_folder, "event_metrics_vol_auc_vs_cal_peak_main.svg")
            cell_vauc_pdf = os.path.join(cell_folder, "event_metrics_vol_auc_vs_cal_peak_main.pdf")
            _plot_voltage_auc_vs_calcium_peak(
                cell_df,
                save_html=cell_vauc_html,
                save_svg=cell_vauc_svg,
                save_pdf=cell_vauc_pdf,
                title_prefix=f"{os.path.basename(cell_folder)} | voltage AUC vs calcium peak",
            )
            cell_auc_resp_sos_html = os.path.join(cell_folder, "event_metrics_vol_auc_vs_calcium_response_summary_of_summaries_main.html")
            cell_auc_resp_sos_svg = os.path.join(cell_folder, "event_metrics_vol_auc_vs_calcium_response_summary_of_summaries_main.svg")
            cell_auc_resp_sos_pdf = os.path.join(cell_folder, "event_metrics_vol_auc_vs_calcium_response_summary_of_summaries_main.pdf")
            _plot_auc_vs_calcium_response_summaries(
                metrics_df=cell_df,
                cell_avg_df=_build_cell_average_metrics_by_vol_auc_bins(cell_df, n_bins=AUC_BIN_COUNT),
                save_html=cell_auc_resp_sos_html,
                save_svg=cell_auc_resp_sos_svg,
                save_pdf=cell_auc_resp_sos_pdf,
                title_prefix=f"{os.path.basename(cell_folder)} | calcium response vs voltage AUC (summary of summaries)",
            )
            if is_motor_cell and len(state_metrics) > 0:
                for state_label in sorted(state_metrics.keys()):
                    st_parts = state_metrics.get(state_label, [])
                    if len(st_parts) == 0:
                        continue
                    st_df = pd.concat(st_parts, ignore_index=True)
                    st_summary_html = os.path.join(cell_folder, f"event_metrics_4panel_main_{state_label}.html")
                    st_summary_svg = os.path.join(cell_folder, f"event_metrics_4panel_main_{state_label}.svg")
                    _plot_summary(
                        st_df,
                        save_html=st_summary_html,
                        save_svg=st_summary_svg,
                        show_plot=False,
                        title_prefix=f"{os.path.basename(cell_folder)} | chosen calcium events vs spike count ({state_label})",
                    )
                    st_decay_html = os.path.join(cell_folder, f"event_metrics_decay_time_main_{state_label}.html")
                    st_decay_svg = os.path.join(cell_folder, f"event_metrics_decay_time_main_{state_label}.svg")
                    _save_decay_time_individual_summary(
                        st_df,
                        save_html=st_decay_html,
                        save_svg=st_decay_svg,
                        save_pdf=None,
                        title_prefix=f"{os.path.basename(cell_folder)} | chosen calcium events vs spike count ({state_label})",
                    )
                    st_vauc_html = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_cal_peak_main_{state_label}.html")
                    st_vauc_svg = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_cal_peak_main_{state_label}.svg")
                    _plot_voltage_auc_vs_calcium_peak(
                        st_df,
                        save_html=st_vauc_html,
                        save_svg=st_vauc_svg,
                        save_pdf=None,
                        title_prefix=f"{os.path.basename(cell_folder)} | voltage AUC vs calcium peak ({state_label})",
                    )
                    st_auc_html = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_calcium_response_summary_of_summaries_main_{state_label}.html")
                    st_auc_svg = os.path.join(cell_folder, f"event_metrics_vol_auc_vs_calcium_response_summary_of_summaries_main_{state_label}.svg")
                    _plot_auc_vs_calcium_response_summaries(
                        metrics_df=st_df,
                        cell_avg_df=_build_cell_average_metrics_by_vol_auc_bins(st_df, n_bins=AUC_BIN_COUNT),
                        save_html=st_auc_html,
                        save_svg=st_auc_svg,
                        save_pdf=None,
                        title_prefix=f"{os.path.basename(cell_folder)} | calcium response vs voltage AUC ({state_label})",
                    )
            print(f"[CELL-SUMMARY] saved: {cell_summary_html}")
        else:
            print(f"[CELL-SUMMARY] skipped (no chosen events): {cell_folder}")

        if SAVE_EVENT_SELECTION_FIGURES and len(cell_event_selection) > 0:
            try:
                cell_evt_df = pd.concat(cell_event_selection, ignore_index=True)
                cell_evt_df = _filter_rows_to_overlay_source_per_cell(cell_evt_df)
                cell_evt_html = os.path.join(cell_folder, "evet_choosing_anlsys.html")
                cell_evt_svg = os.path.join(cell_folder, "evet_choosing_anlsys.svg")
                cell_evt_pdf = os.path.join(cell_folder, "evet_choosing_anlsys.pdf")
                _plot_event_selection_two_panel(
                    cell_evt_df,
                    title_prefix=f"{os.path.basename(cell_folder)} | event choosing analysis",
                    save_html=cell_evt_html,
                    save_svg=cell_evt_svg,
                    save_pdf=cell_evt_pdf,
                )
                print(f"[EVENT-CHOSS] saved cell figure: {cell_evt_html}")
            except Exception as e:
                print(f"[WARN] Cell event-selection figure failed ({cell_folder}): {e}")
        else:
            if SAVE_EVENT_SELECTION_FIGURES:
                print(f"[EVENT-CHOSS] skipped cell figure (no events): {cell_folder}")

    all_event_selection_df = pd.DataFrame()
    if len(all_event_selection) > 0:
        try:
            all_event_selection_df = pd.concat(all_event_selection, ignore_index=True)
        except Exception:
            all_event_selection_df = pd.DataFrame()

    if SAVE_EVENT_SELECTION_FIGURES and len(all_event_selection_df) > 0:
        try:
            all_event_selection_df = _filter_rows_to_overlay_source_per_cell(all_event_selection_df)
            event_sel_csv = os.path.join(event_choss_dir, "event_selection_all_events.csv")
            all_event_selection_df.to_csv(event_sel_csv, index=False)
            save_event_selection_cell_by_cell_figures(
                all_event_selection_df,
                out_dir=event_choss_dir,
                n_cols=cell_by_cell_n_cols,
            )
            save_event_selection_population_figures(
                all_event_selection_df,
                out_dir=event_choss_dir,
            )
            print(f"[EVENT-CHOSS] saved summary outputs to: {event_choss_dir}")
        except Exception as e:
            print(f"[WARN] Event-selection summary export failed: {e}")
    else:
        if SAVE_EVENT_SELECTION_FIGURES:
            print("[EVENT-CHOSS] skipped summaries (no event-selection rows)")

    if len(all_metrics) == 0:
        raise RuntimeError("No chosen events found across PyrLowFR cells.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    print(f"\nTotal chosen events: {len(metrics_df)}")

    try:
        if summary_csv_event:
            metrics_df.to_csv(summary_csv_event, index=False)
            print(f"[ALL-METRICS-CSV] saved: {summary_csv_event}")
    except Exception as e:
        print(f"[WARN] Could not save combined CSV: {e}")

    _plot_summary(
        metrics_df,
        save_html=summary_html_event,
        save_svg=summary_svg_event,
        show_plot=False,
        title_prefix="Chosen calcium events vs spike count (all PyrLowFR cells)",
        save_subplots=False,
    )
    summary_decay_html = _append_suffix_before_ext(summary_html_event, "_decay_time_individual") if summary_html_event else None
    summary_decay_svg = _append_suffix_before_ext(summary_svg_event, "_decay_time_individual") if summary_svg_event else None
    summary_decay_pdf = (os.path.splitext(summary_decay_html)[0] + ".pdf") if summary_decay_html else (
        (os.path.splitext(summary_decay_svg)[0] + ".pdf") if summary_decay_svg else None
    )
    _save_decay_time_individual_summary(
        metrics_df,
        save_html=summary_decay_html,
        save_svg=summary_decay_svg,
        save_pdf=summary_decay_pdf,
        title_prefix="Chosen calcium events vs spike count (all PyrLowFR cells)",
    )
    summary_vauc_html = _append_suffix_before_ext(summary_html_event, "_vol_auc_vs_cal_peak") if summary_html_event else None
    summary_vauc_svg = _append_suffix_before_ext(summary_svg_event, "_vol_auc_vs_cal_peak") if summary_svg_event else None
    summary_vauc_pdf = (os.path.splitext(summary_vauc_html)[0] + ".pdf") if summary_vauc_html else (
        (os.path.splitext(summary_vauc_svg)[0] + ".pdf") if summary_vauc_svg else None
    )
    _plot_voltage_auc_vs_calcium_peak(
        metrics_df,
        save_html=summary_vauc_html,
        save_svg=summary_vauc_svg,
        save_pdf=summary_vauc_pdf,
        title_prefix="Voltage AUC vs calcium peak (all PyrLowFR chosen events)",
    )
    # Backward-compatible copies to top summary folder for quick visibility.
    if pop_root:
        _copy_if_exists(summary_csv_event, os.path.join(pop_root, os.path.basename(str(summary_csv_event))) if summary_csv_event else None)
        _copy_if_exists(summary_html_event, os.path.join(pop_root, os.path.basename(str(summary_html_event))) if summary_html_event else None)
        _copy_if_exists(summary_svg_event, os.path.join(pop_root, os.path.basename(str(summary_svg_event))) if summary_svg_event else None)
        _copy_if_exists(summary_decay_html, os.path.join(pop_root, os.path.basename(str(summary_decay_html))) if summary_decay_html else None)
        _copy_if_exists(summary_decay_svg, os.path.join(pop_root, os.path.basename(str(summary_decay_svg))) if summary_decay_svg else None)
        _copy_if_exists(summary_decay_pdf, os.path.join(pop_root, os.path.basename(str(summary_decay_pdf))) if summary_decay_pdf else None)
        if summary_vauc_html:
            _copy_glob_to_dir(os.path.splitext(str(summary_vauc_html))[0] + "*", pop_root)
        if summary_vauc_svg:
            _copy_glob_to_dir(os.path.splitext(str(summary_vauc_svg))[0] + "*", pop_root)
        if summary_vauc_pdf:
            _copy_glob_to_dir(os.path.splitext(str(summary_vauc_pdf))[0] + "*", pop_root)
        # Also copy normalized summary outputs (main + subplots) to top folder.
        if summary_html_event:
            stem = os.path.splitext(str(summary_html_event))[0]
            _copy_glob_to_dir(stem + "_normalized_data*", pop_root)
        if summary_svg_event:
            stem = os.path.splitext(str(summary_svg_event))[0]
            _copy_glob_to_dir(stem + "_normalized_data*", pop_root)

    # Cell-average summary (each point is one cell mean per event-type and spike-count bin)
    cell_avg_df = pd.DataFrame()
    try:
        cell_avg_df = _build_cell_average_metrics(metrics_df)
        if len(cell_avg_df) > 0:
            summary_html_cellavg = os.path.join(
                cell_avg_dir,
                os.path.basename(_append_suffix_before_ext(summary_html_event, "_cellavg"))
            ) if summary_html_event else None
            summary_svg_cellavg = os.path.join(
                cell_avg_dir,
                os.path.basename(_append_suffix_before_ext(summary_svg_event, "_cellavg"))
            ) if summary_svg_event else None
            summary_csv_cellavg = os.path.join(
                cell_avg_dir,
                os.path.basename(_append_suffix_before_ext(summary_csv_event, "_cellavg"))
            ) if summary_csv_event else None

            if summary_csv_cellavg:
                cell_avg_df.to_csv(summary_csv_cellavg, index=False)
                print(f"[ALL-METRICS-CSV-CELLAVG] saved: {summary_csv_cellavg}")

            _plot_summary(
                cell_avg_df,
                save_html=summary_html_cellavg,
                save_svg=summary_svg_cellavg,
                show_plot=False,
                title_prefix="Cell-average calcium metrics vs spike count (all PyrLowFR cells)",
                save_subplots=False,
            )
            summary_decay_html_cellavg = _append_suffix_before_ext(summary_html_cellavg, "_decay_time_individual") if summary_html_cellavg else None
            summary_decay_svg_cellavg = _append_suffix_before_ext(summary_svg_cellavg, "_decay_time_individual") if summary_svg_cellavg else None
            summary_decay_pdf_cellavg = (os.path.splitext(summary_decay_html_cellavg)[0] + ".pdf") if summary_decay_html_cellavg else (
                (os.path.splitext(summary_decay_svg_cellavg)[0] + ".pdf") if summary_decay_svg_cellavg else None
            )
            _save_decay_time_individual_summary(
                cell_avg_df,
                save_html=summary_decay_html_cellavg,
                save_svg=summary_decay_svg_cellavg,
                save_pdf=summary_decay_pdf_cellavg,
                title_prefix="Cell-average calcium metrics vs spike count (all PyrLowFR cells)",
            )
            if pop_root:
                _copy_if_exists(summary_csv_cellavg, os.path.join(pop_root, os.path.basename(str(summary_csv_cellavg))) if summary_csv_cellavg else None)
                _copy_if_exists(summary_html_cellavg, os.path.join(pop_root, os.path.basename(str(summary_html_cellavg))) if summary_html_cellavg else None)
                _copy_if_exists(summary_svg_cellavg, os.path.join(pop_root, os.path.basename(str(summary_svg_cellavg))) if summary_svg_cellavg else None)
                _copy_if_exists(summary_decay_html_cellavg, os.path.join(pop_root, os.path.basename(str(summary_decay_html_cellavg))) if summary_decay_html_cellavg else None)
                _copy_if_exists(summary_decay_svg_cellavg, os.path.join(pop_root, os.path.basename(str(summary_decay_svg_cellavg))) if summary_decay_svg_cellavg else None)
                _copy_if_exists(summary_decay_pdf_cellavg, os.path.join(pop_root, os.path.basename(str(summary_decay_pdf_cellavg))) if summary_decay_pdf_cellavg else None)
                # Also copy normalized cell-average summary outputs (main + subplots) to top folder.
                if summary_html_cellavg:
                    stem = os.path.splitext(str(summary_html_cellavg))[0]
                    _copy_glob_to_dir(stem + "_normalized_data*", pop_root)
                if summary_svg_cellavg:
                    stem = os.path.splitext(str(summary_svg_cellavg))[0]
                    _copy_glob_to_dir(stem + "_normalized_data*", pop_root)
        else:
            print("[CELLAVG-SUMMARY] skipped (no valid aggregated points)")
    except Exception as e:
        print(f"[WARN] Cell-average summary export failed: {e}")

    # New population figure: summary of summaries (row1 events, row2 cell averages)
    try:
        sos_html = _append_suffix_before_ext(summary_html_event, "_summary_of_summaries") if summary_html_event else None
        sos_svg = _append_suffix_before_ext(summary_svg_event, "_summary_of_summaries") if summary_svg_event else None
        sos_pdf = (os.path.splitext(sos_html)[0] + ".pdf") if sos_html else (
            (os.path.splitext(sos_svg)[0] + ".pdf") if sos_svg else None
        )
        _plot_summary_of_summaries(
            metrics_df=metrics_df,
            cell_avg_df=cell_avg_df,
            save_html=sos_html,
            save_svg=sos_svg,
            save_pdf=sos_pdf,
            title_prefix="PyrLowFR summary of summaries",
        )
        if pop_root:
            if sos_html:
                _copy_glob_to_dir(os.path.splitext(str(sos_html))[0] + "*", pop_root)
            if sos_svg:
                _copy_glob_to_dir(os.path.splitext(str(sos_svg))[0] + "*", pop_root)
            if sos_pdf:
                _copy_glob_to_dir(os.path.splitext(str(sos_pdf))[0] + "*", pop_root)
    except Exception as e:
        print(f"[WARN] Summary-of-summaries export failed: {e}")

    # New population figure: calcium response metrics as a function of calcium AUC
    # (row1 events, row2 cell averages), saved under event_base.
    try:
        auc_sos_html = _append_suffix_before_ext(summary_html_event, "_vol_auc_vs_calcium_response_summary_of_summaries") if summary_html_event else None
        auc_sos_svg = _append_suffix_before_ext(summary_svg_event, "_vol_auc_vs_calcium_response_summary_of_summaries") if summary_svg_event else None
        auc_sos_pdf = (os.path.splitext(auc_sos_html)[0] + ".pdf") if auc_sos_html else (
            (os.path.splitext(auc_sos_svg)[0] + ".pdf") if auc_sos_svg else None
        )
        _plot_auc_vs_calcium_response_summaries(
            metrics_df=metrics_df,
            cell_avg_df=cell_avg_df,
            save_html=auc_sos_html,
            save_svg=auc_sos_svg,
            save_pdf=auc_sos_pdf,
            title_prefix="PyrLowFR calcium response vs voltage AUC (summary of summaries)",
        )
        if pop_root:
            if auc_sos_html:
                _copy_glob_to_dir(os.path.splitext(str(auc_sos_html))[0] + "*", pop_root)
            if auc_sos_svg:
                _copy_glob_to_dir(os.path.splitext(str(auc_sos_svg))[0] + "*", pop_root)
            if auc_sos_pdf:
                _copy_glob_to_dir(os.path.splitext(str(auc_sos_pdf))[0] + "*", pop_root)
    except Exception as e:
        print(f"[WARN] AUC-vs-calcium-response summary export failed: {e}")

    metrics_df_cell_by_cell = _filter_rows_to_overlay_source_per_cell(metrics_df)
    if len(metrics_df_cell_by_cell) != len(metrics_df):
        print(
            f"[CELL-BY-CELL] using overlay-consistent rows: {len(metrics_df_cell_by_cell)}/{len(metrics_df)} "
            "(main suffix per cell when available)"
        )
    try:
        save_cell_by_cell_figures(metrics_df_cell_by_cell, out_dir=cell_by_cell_out_dir, n_cols=cell_by_cell_n_cols)
    except Exception as e:
        print(f"[WARN] Cell-by-cell summary export failed: {e}")
    try:
        save_cell_by_cell_auc_vs_calcium_response_figures(
            metrics_df_cell_by_cell,
            out_dir=cell_by_cell_out_dir,
            n_cols=cell_by_cell_n_cols,
        )
    except Exception as e:
        print(f"[WARN] Cell-by-cell AUC-vs-response summary export failed: {e}")
    print(
        "Summary saved:\n"
        f"  Event-base HTML: {summary_html_event}\n"
        f"  Event-base SVG:  {summary_svg_event}\n"
        f"  Cell-avg dir:    {cell_avg_dir}\n"
        f"  EventChoss dir:  {event_choss_dir}"
    )
    return metrics_df


if __name__ == "__main__":
    run_pyr_event_cal2_pipeline()



