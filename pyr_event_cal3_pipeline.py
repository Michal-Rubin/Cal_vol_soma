import glob
import os
import pickle
import re
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from scipy import stats as spstats
except Exception:
    spstats = None
try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
except Exception:
    smf = None
    pairwise_tukeyhsd = None
    multipletests = None
try:
    import pyr_event_cal2_pipeline as pe2
except Exception:
    pe2 = None


DB_PATH = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\PyrLowFR.csv"
VOL_SR = 500.0
SIMPLE_ISI_MS = 30.0
SIMPLE_ISI_FR = int(round(VOL_SR * SIMPLE_ISI_MS / 1000.0))
SIMPLE_PRE_MS = 250.0
SIMPLE_POST_MS = 150.0
COMPLEX_PRE_MS = 400.0
COMPLEX_POST_MS = 150.0
COMPLEX_POST_FOR_COMPLEX_MS = 300.0
TAIL_RATIO_THR = 0.30
PRE_F0_MS = 200.0
LOCAL_BASELINE_PRE_MS = 210.0
START_EXCLUDE_MS = 200.0
GLOBAL_F0_WIN_S = 60.0
GLOBAL_F0_PCTL = 8
OVERLAY_SMOOTH_SIGMA_S = 0.01
PEAK_FIND_SMOOTH_SIGMA_S = 0.01
CALCIUM_THRESHOLD_SMOOTH_SIGMA_S = 0.06
CALCIUM_THRESHOLD_FALLBACK = 0.5
CALCIUM_THRESHOLD_STD_MULT = 2.0
PLATEAU_MIN_AUC_DUR_MS = 90.0

COLOR_NON_CHOSEN = "#9e9e9e"
COLOR_SINGLE = "black"
COLOR_SIMPLE_BURST = "black"
COLOR_COMPLEX = "red"
COLOR_PLATEAU = "#2ca02c"
# Plotly uses CSS/hex colors; matplotlib tab:red equivalent is #d62728.
VOLTAGE_TRACE_COLOR = "#d62728"   # tab:red
CALCIUM_TRACE_COLOR = "#4b0082"
GLOBAL_DFF_Y_COL = "peak_global_dff"
GLOBAL_DFF_Y_LABEL = "calcium response (global dF/F0)"
GLOBAL_DFF_CELLAVG_Y_LABEL = "cell-mean calcium peak (global dF/F0)"
CALCIUM_TRACE_DFF_MODE = "auto"  # "auto", True, or False

PLOT_TYPE_COLOR = {
    "simple": "black",
    "complex": "red",
    "plateau": "#2ca02c",
}
PLOT_AXIS_TICK_FONT = dict(size=14, family="Arial Black")
PLOT_AXIS_TITLE_FONT = dict(size=16, family="Arial Black")
POP_SUMMARY_DIR = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data_summery\2026\Pyr\calciumRes\eventcal3"

SQUARE_PANEL_PX = 520
TWO_PANEL_HEIGHT_PX = 620
SINGLE_PANEL_WIDTH_PX = SQUARE_PANEL_PX + 260
PANEL_GAP_PX = 70
PANEL_MARGIN_L_PX = 80
PANEL_MARGIN_R_PX = 80
PANEL_MARGIN_T_PX = 60
PANEL_MARGIN_B_PX = 70
CM_TO_PX = 37.7952755906
GRID_SUBPLOT_CM = 3.0
SUMMARY_ZSCORE_SUBPLOT_MM = 50.0
SUMMARY_ZSCORE_SUBPLOT_PX = int(round((SUMMARY_ZSCORE_SUBPLOT_MM / 10.0) * CM_TO_PX))

SHAPE_PRE_MS = 50.0
SHAPE_POST_MS = 500.0
SHAPE_SPIKE_CAP = 6
AUC_SPIKE_MASK_PRE_MS = 2.0
AUC_SPIKE_MASK_POST_MS = 4.0


def _safe_cal_sr(v, default=30.0):
    try:
        x = float(v)
        if np.isfinite(x) and x > 0:
            return x
    except Exception:
        pass
    return float(default)


def _as_int_sorted_unique(x):
    if x is None:
        return np.array([], dtype=int)
    a = np.asarray(x, dtype=int).ravel()
    if a.size == 0:
        return np.array([], dtype=int)
    return np.unique(a)


def _read_csv_1d(path):
    arr = pd.read_csv(path).to_numpy(dtype=float).ravel()
    return np.asarray(arr, dtype=float)


def _load_or_create_volpy_dff(cell_folder):
    csv_path = os.path.join(cell_folder, "volVolpyDF.csv")
    if os.path.isfile(csv_path):
        try:
            arr = _read_csv_1d(csv_path)
            if arr.size > 0:
                return np.asarray(arr, dtype=float).ravel(), "volVolpyDF.csv"
        except Exception:
            pass

    pkl_path = os.path.join(cell_folder, "output_data.pkl")
    if not os.path.isfile(pkl_path):
        return None, ""
    try:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        arr = np.asarray(payload.get("dFF", []), dtype=float).ravel()
        if arr.size == 0:
            return None, ""
        pd.DataFrame({"volVolpyDF": arr}).to_csv(csv_path, index=False)
        return arr, "output_data.pkl:dFF->volVolpyDF.csv"
    except Exception:
        return None, ""


def _read_bool_mask_csv(path, target_len):
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        col = "mask" if "mask" in df.columns else df.columns[0]
        vals = df[col].to_numpy()
        mask = np.asarray([str(v).strip().lower() in {"true", "1", "yes", "y"} for v in vals], dtype=bool)
        if mask.size != int(target_len):
            return None
        return mask
    except Exception:
        return None


def _apply_mask_to_trace(trace, mask):
    arr = np.asarray(trace, dtype=float).ravel().copy()
    if mask is None:
        return arr
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != arr.size:
        return arr
    arr[~mask] = np.nan
    return arr


def _filter_payload_spikes_by_mask(payload, vol_mask):
    if vol_mask is None:
        return payload
    mask = np.asarray(vol_mask, dtype=bool).ravel()
    out = copy.deepcopy(payload)
    for key in (
        "input_spikes",
        "vm_simple_spikes",
        "vm_complex_spikes",
        "vm_all_spikes",
        "vm_simple_spikes_before_raw_threshold",
        "vm_simple_candidate_pool_before_raw_threshold",
    ):
        if key not in out:
            continue
        arr = np.asarray(out.get(key, []), dtype=int).ravel()
        ok = (arr >= 0) & (arr < mask.size) & mask[arr]
        out[key] = np.unique(arr[ok].astype(int))
    return out


def _mean_fr_hz_from_payload(payload, trace_vol, vol_sr=VOL_SR):
    trace = np.asarray(trace_vol, dtype=float).ravel()
    valid_s = float(np.isfinite(trace).sum()) / float(vol_sr) if trace.size else 0.0
    if valid_s <= 0:
        return np.nan
    spikes = _as_int_sorted_unique(payload.get("vm_all_spikes", []))
    if spikes.size == 0:
        spike_parts = []
        for key in ("vm_simple_spikes", "vm_complex_spikes", "input_spikes"):
            arr = _as_int_sorted_unique(payload.get(key, []))
            if arr.size:
                spike_parts.append(arr)
        spikes = _as_int_sorted_unique(np.concatenate(spike_parts)) if spike_parts else np.asarray([], dtype=int)
    spikes = spikes[(spikes >= 0) & (spikes < trace.size)]
    if spikes.size and np.any(~np.isfinite(trace[spikes])):
        spikes = spikes[np.isfinite(trace[spikes])]
    return float(spikes.size) / valid_s


def _vol_to_cal_idx(v_idx, vol_sr, cal_sr, cal_len):
    if cal_len <= 0:
        return 0
    t = float(v_idx) / float(vol_sr)
    c = int(round(t * float(cal_sr)))
    return int(max(0, min(cal_len - 1, c)))


def _vol_to_cal_idx_before(v_idx, vol_sr, cal_sr, cal_len):
    if cal_len <= 0:
        return 0
    t = float(v_idx) / float(vol_sr)
    c = int(np.floor(t * float(cal_sr)))
    return int(max(0, min(cal_len - 1, c)))


def _group_by_isi(spikes, isi_frames):
    s = _as_int_sorted_unique(spikes)
    if s.size == 0:
        return []
    groups = [[int(s[0])]]
    for x in s[1:]:
        if int(x) - groups[-1][-1] < int(isi_frames):
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return [np.asarray(g, dtype=int) for g in groups]


@dataclass
class Event:
    event_type: str  # simple / complex / plateau
    spikes: np.ndarray
    start_frame: int
    end_frame: int
    source: str
    include: bool = False
    include_reason: str = ""
    peak_idx: int = -1
    peak_global_dff: float = np.nan
    amp_local_dff: float = np.nan
    amp_local_z: float = np.nan
    prev_event_type: str = ""

    @property
    def n_spikes(self):
        return int(self.spikes.size)

    @property
    def first_spike(self):
        return int(self.spikes[0]) if self.spikes.size else int(self.start_frame)

    @property
    def last_spike(self):
        return int(self.spikes[-1]) if self.spikes.size else int(self.end_frame)

    @property
    def class_name(self):
        if self.event_type == "plateau":
            return "plateau"
        if self.event_type == "complex":
            return "complex"
        return "single" if self.n_spikes == 1 else "simple_burst"


def _interval_overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def _interval_overlap_with_gap(a0, a1, b0, b1, gap_frames=0):
    gap = int(max(0, gap_frames))
    return not (int(a1) < int(b0) - gap or int(b1) < int(a0) - gap)


def _sorted_arrays_share_any(a, b):
    a = np.asarray(a, dtype=int).ravel()
    b = np.asarray(b, dtype=int).ravel()
    if a.size == 0 or b.size == 0:
        return False
    if int(a[-1]) < int(b[0]) or int(b[-1]) < int(a[0]):
        return False
    small, large = (a, b) if a.size <= b.size else (b, a)
    lo = int(max(small[0], large[0]))
    hi = int(min(small[-1], large[-1]))
    small = small[(small >= lo) & (small <= hi)]
    if small.size == 0:
        return False
    idx = np.searchsorted(large, small)
    ok = idx < large.size
    return bool(np.any(large[idx[ok]] == small[ok]))


def _extract_plateau_events(d):
    out = []
    pdct = d.get("vm_plateaus_dict", {})
    if not isinstance(pdct, dict):
        return out
    starts = _as_int_sorted_unique(pdct.get("starts", []))
    ends = _as_int_sorted_unique(pdct.get("ends", []))
    spike_indices = pdct.get("spike_indices", [])
    locs = _as_int_sorted_unique(pdct.get("locs", []))
    n = int(min(starts.size, ends.size))
    for i in range(n):
        s = int(starts[i])
        e = int(max(s, ends[i]))
        sp = np.array([], dtype=int)
        if isinstance(spike_indices, (list, tuple)) and i < len(spike_indices):
            sp = _as_int_sorted_unique(spike_indices[i])
        if sp.size == 0:
            sp = locs[(locs >= s) & (locs <= e)]
        if sp.size == 0:
            sp = np.array([s], dtype=int)
        out.append(Event("plateau", sp, int(sp[0]), int(sp[-1]), "plateau_window", prev_event_type="complex"))
    return out


def _extract_complex_events(d):
    complex_spikes = _as_int_sorted_unique(d.get("vm_complex_spikes", []))
    bdict = d.get("vm_burst_dict", {})
    out = []
    if isinstance(bdict, dict):
        starts = _as_int_sorted_unique(bdict.get("starts", []))
        ends = _as_int_sorted_unique(bdict.get("ends", []))
        n = int(min(starts.size, ends.size))
        for i in range(n):
            s = int(starts[i])
            e = int(max(s, ends[i]))
            sp = complex_spikes[(complex_spikes >= s) & (complex_spikes <= e)]
            if sp.size == 0:
                continue
            out.append(Event("complex", sp, int(sp[0]), int(sp[-1]), "complex_window"))
    if len(out) == 0 and complex_spikes.size > 0:
        for g in _group_by_isi(complex_spikes, SIMPLE_ISI_FR):
            out.append(Event("complex", g, int(g[0]), int(g[-1]), "complex_isi_fallback"))
    return out


def _extract_simple_events(d):
    simple_spikes = _as_int_sorted_unique(d.get("vm_simple_spikes", []))
    out = []
    for g in _group_by_isi(simple_spikes, SIMPLE_ISI_FR):
        out.append(Event("simple", g, int(g[0]), int(g[-1]), "simple_isi"))
    return out


def _merge_simple_into_complex(simple_events, complex_events):
    rem_simple = []
    comp = sorted(list(complex_events), key=lambda e: (e.first_spike, e.last_spike))
    for se in simple_events:
        merged = False
        for i, ce in enumerate(comp):
            if ce.last_spike < se.first_spike:
                continue
            if ce.first_spike > se.last_spike and (ce.first_spike - se.last_spike) >= SIMPLE_ISI_FR:
                break
            shared = _sorted_arrays_share_any(se.spikes, ce.spikes)
            near_to_complex_start = (se.last_spike <= ce.first_spike) and (
                ce.first_spike - se.last_spike < SIMPLE_ISI_FR
            )
            if shared or near_to_complex_start:
                new_sp = _as_int_sorted_unique(np.r_[ce.spikes, se.spikes])
                comp[i] = Event(
                    "complex",
                    new_sp,
                    int(new_sp[0]),
                    int(new_sp[-1]),
                    "complex_plus_simple_merge",
                )
                merged = True
                break
        if not merged:
            rem_simple.append(se)
    return rem_simple, comp


def _apply_plateau_reclass(events, plateau_events):
    out = list(events)
    for pe in plateau_events:
        overlap_idx = []
        for i, ev in enumerate(out):
            by_spike = _sorted_arrays_share_any(ev.spikes, pe.spikes)
            # A plateau that sits immediately beside a burst should still be one
            # event for AUC analysis; otherwise adjacent complex/plateau labels
            # can split the same depolarization into two events.
            by_time = _interval_overlap_with_gap(
                ev.first_spike,
                ev.last_spike,
                pe.start_frame,
                pe.end_frame,
                gap_frames=SIMPLE_ISI_FR,
            )
            if by_spike or by_time:
                overlap_idx.append(i)
        if len(overlap_idx) == 0:
            out.append(pe)
            continue
        all_sp = [pe.spikes] + [out[i].spikes for i in overlap_idx]
        merged_sp = _as_int_sorted_unique(np.concatenate(all_sp))
        prev_types = [str(out[i].event_type).strip().lower() for i in overlap_idx]
        if any(t == "complex" for t in prev_types):
            fallback_type = "complex"
        else:
            fallback_type = "simple"
        merged_ev = Event(
            "plateau",
            merged_sp,
            int(merged_sp[0]),
            int(merged_sp[-1]),
            "plateau_reclass_merge",
            prev_event_type=fallback_type,
        )
        keep = [ev for j, ev in enumerate(out) if j not in set(overlap_idx)]
        keep.append(merged_ev)
        out = keep
    out = sorted(out, key=lambda e: (e.first_spike, e.last_spike, e.event_type))
    return out


def build_events(d):
    simple_e = _extract_simple_events(d)
    complex_e = _extract_complex_events(d)
    plateau_e = _extract_plateau_events(d)
    simple_e, complex_e = _merge_simple_into_complex(simple_e, complex_e)
    base = complex_e + simple_e
    events = _apply_plateau_reclass(base, plateau_e)
    return events


def build_events_no_plateau(d):
    simple_e = _extract_simple_events(d)
    complex_e = _extract_complex_events(d)
    simple_e, complex_e = _merge_simple_into_complex(simple_e, complex_e)
    events = sorted((complex_e + simple_e), key=lambda e: (e.first_spike, e.last_spike, e.event_type))
    return events


def _with_filename_tag(name, tag):
    t = str(tag).strip()
    if t == "":
        return str(name)
    stem, ext = os.path.splitext(str(name))
    return f"{stem}_{t}{ext}"


def _calcium_trace_looks_like_dff(trace):
    x = np.asarray(trace, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    q01, q50, q99 = np.nanpercentile(x, [1, 50, 99])
    return bool(
        np.isfinite(q01)
        and np.isfinite(q50)
        and np.isfinite(q99)
        and q01 > -10.0
        and abs(q50) < 10.0
        and q99 < 20.0
    )


def _rolling_percentile_f0(trace, sr, win_s=60.0, p=20.0):
    x = np.asarray(trace, dtype=float).ravel()
    n = x.size
    if n == 0:
        return x.copy(), x.copy()
    mode = CALCIUM_TRACE_DFF_MODE
    if isinstance(mode, str):
        mode_norm = mode.strip().lower()
        use_as_dff = _calcium_trace_looks_like_dff(x) if mode_norm == "auto" else mode_norm in {"true", "dff", "already_dff"}
    else:
        use_as_dff = bool(mode)
    if use_as_dff:
        return x.copy(), np.full(n, np.nan, dtype=float)
    w = max(1, int(round(float(win_s) * float(sr))))
    w = min(w, n)
    half = w // 2
    f0 = np.full(n, np.nan, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        seg = x[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            f0[i] = np.percentile(seg, p)
    if np.any(~np.isfinite(f0)):
        good = np.where(np.isfinite(f0))[0]
        if good.size == 0:
            f0[:] = 1e-9
        else:
            bad = np.where(~np.isfinite(f0))[0]
            f0[bad] = np.interp(bad, good, f0[good])
    # Keep baseline positive for stable dF/F in traces that can dip below zero.
    pos = x[np.isfinite(x) & (x > 0)]
    if pos.size > 0:
        f0_floor = float(np.nanpercentile(pos, 5))
        f0_floor = max(f0_floor, 1e-6)
    else:
        f0_floor = 1e-6
    f0_safe = f0.copy()
    f0_safe[~np.isfinite(f0_safe)] = f0_floor
    f0_safe[f0_safe <= f0_floor] = f0_floor
    gdf = (x - f0_safe) / f0_safe
    return gdf, f0_safe


def _gaussian_smooth_1d(x, sigma_frames):
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        return a
    s = float(sigma_frames)
    if (not np.isfinite(s)) or s <= 0:
        return a.copy()
    rad = int(max(1, np.ceil(4.0 * s)))
    kx = np.arange(-rad, rad + 1, dtype=float)
    ker = np.exp(-0.5 * (kx / s) ** 2)
    ker /= np.sum(ker)
    valid = np.isfinite(a).astype(float)
    af = np.where(np.isfinite(a), a, 0.0)
    num = np.convolve(af, ker, mode="same")
    den = np.convolve(valid, ker, mode="same")
    out = np.full(a.shape, np.nan, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def _interp_nan_1d(x):
    a = np.asarray(x, dtype=float).ravel().copy()
    if a.size == 0:
        return a
    nans = np.isnan(a)
    if not np.any(nans):
        return a
    valid = ~nans
    if not np.any(valid):
        return a
    a[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(valid), a[valid])
    return a


def _candidate_peak_idx(ev, trace_for_peak, cal_sr, vol_sr):
    if len(trace_for_peak) == 0:
        return -1
    a, b = _candidate_peak_search_bounds(ev, len(trace_for_peak), cal_sr, vol_sr)
    if b < a:
        return -1
    seg = trace_for_peak[a : b + 1]
    if seg.size == 0:
        return -1
    if not np.any(np.isfinite(seg)):
        return -1
    rel = int(np.nanargmax(seg))
    return int(a + rel)


def _candidate_peak_search_bounds(ev, gdf_len, cal_sr, vol_sr):
    if int(gdf_len) <= 0:
        return 0, -1
    c_last = _vol_to_cal_idx(ev.last_spike, vol_sr, cal_sr, int(gdf_len))
    win_ms = 300.0 if ev.event_type in ("complex", "plateau") else 150.0
    n = max(1, int(round(float(cal_sr) * (float(win_ms) / 1000.0))))
    a = int(c_last)
    b = int(min(int(gdf_len) - 1, a + n))
    return a, b


def _highpass_onepole(trace, sr_hz, cutoff_hz=0.5):
    x = np.asarray(trace, dtype=float).ravel()
    if x.size == 0:
        return np.array([], dtype=float)
    y = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x)
    if not np.any(ok):
        return y
    dt = 1.0 / float(sr_hz) if float(sr_hz) > 0 else 1.0
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = rc / (rc + dt)
    idx = np.where(ok)[0]
    s = int(idx[0])
    y[s] = 0.0
    prev_x = float(x[s])
    prev_y = 0.0
    for i in range(s + 1, x.size):
        if not np.isfinite(x[i]):
            y[i] = np.nan
            continue
        xi = float(x[i])
        yi = float(alpha) * (prev_y + xi - prev_x)
        y[i] = yi
        prev_x = xi
        prev_y = yi
    return y


def _global_ztrace_p8(trace, sr_hz, return_stats=False):
    a = np.asarray(trace, dtype=float).ravel()
    if a.size == 0:
        if return_stats:
            return np.array([], dtype=float), {"center": np.nan, "sigma": np.nan, "baseline_p8": np.nan}
        return np.array([], dtype=float)
    out = np.full(a.shape, np.nan, dtype=float)
    ok = np.isfinite(a)
    if not np.any(ok):
        if return_stats:
            return out, {"center": np.nan, "sigma": np.nan, "baseline_p8": np.nan}
        return out
    x = a[ok]
    center = float(np.nanpercentile(x, 20))
    hp = _highpass_onepole(a, sr_hz, cutoff_hz=0.5)
    hp_ok = hp[np.isfinite(hp)]
    if hp_ok.size == 0:
        hp_ok = x
    scale = float(np.nanstd(hp_ok))
    if (not np.isfinite(scale)) or scale <= 1e-12:
        sd = float(np.nanstd(x))
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    out[ok] = (x - center) / scale
    z_ok = out[np.isfinite(out)]
    z_b8 = float(np.nanpercentile(z_ok, 8)) if z_ok.size > 0 else np.nan
    if return_stats:
        return out, {"center": float(center), "sigma": float(scale), "baseline_p8": float(z_b8)}
    return out


def _local_amp_from_fluor(ev, fluor, peak_idx, cal_sr, vol_sr, normalize=False):
    if peak_idx is None or int(peak_idx) < 0 or len(fluor) == 0:
        return 0.0
    c_first = _vol_to_cal_idx(ev.first_spike, vol_sr, cal_sr, len(fluor))
    pre_n = max(1, int(round(cal_sr * (LOCAL_BASELINE_PRE_MS / 1000.0))))
    a = max(0, c_first - pre_n)
    b = max(a, c_first - 1)
    if b >= a:
        pre = np.asarray(fluor[a : b + 1], dtype=float)
        pre = pre[np.isfinite(pre)]
        if pre.size > 0:
            b0 = float(np.nanmedian(pre))
        else:
            b0 = np.nan
    else:
        b0 = np.nan
    if not np.isfinite(b0):
        return 0.0
    fpk = float(fluor[int(max(0, min(len(fluor) - 1, peak_idx)))])
    if bool(normalize):
        if abs(b0) < 1e-9:
            return 0.0
        amp = float((fpk - b0) / b0)
    else:
        amp = float(fpk - b0)
    if not np.isfinite(amp):
        return 0.0
    return float(max(0.0, amp))


def _local_baseline_from_trace(ev, trace, cal_sr, vol_sr, pre_ms=None):
    arr = np.asarray(trace, dtype=float).ravel()
    if arr.size == 0:
        return np.nan
    c_first = _vol_to_cal_idx(ev.first_spike, vol_sr, cal_sr, len(arr))
    use_pre_ms = LOCAL_BASELINE_PRE_MS if pre_ms is None else float(pre_ms)
    pre_n = max(1, int(round(cal_sr * (use_pre_ms / 1000.0))))
    a = max(0, c_first - pre_n)
    b = max(a, c_first - 1)
    if b < a:
        return np.nan
    pre = np.asarray(arr[a : b + 1], dtype=float)
    pre = pre[np.isfinite(pre)]
    if pre.size == 0:
        return np.nan
    return float(np.nanmedian(pre))


def _calcium_threshold_value(gdf_for_threshold=None):
    if gdf_for_threshold is None:
        return float(CALCIUM_THRESHOLD_FALLBACK)
    a = np.asarray(gdf_for_threshold, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float(CALCIUM_THRESHOLD_FALLBACK)
    med = float(np.nanmedian(a))
    baseline = a[a <= med]
    if baseline.size < 10:
        baseline = a
    mu = float(np.nanmean(baseline))
    sd = float(np.nanstd(baseline))
    if (not np.isfinite(mu)) or (not np.isfinite(sd)):
        return float(CALCIUM_THRESHOLD_FALLBACK)
    return float(mu + float(CALCIUM_THRESHOLD_STD_MULT) * sd)


def _event_start_gdf_at_ap(ev, gdf, cal_sr, vol_sr):
    if len(gdf) == 0:
        return -1, np.nan
    c_first = _vol_to_cal_idx(ev.first_spike, vol_sr, cal_sr, len(gdf))
    c_check = int(max(0, min(len(gdf) - 1, c_first)))
    gprev = float(gdf[c_check]) if len(gdf) else np.nan
    return int(c_check), float(gprev)


def _compute_event_features(events, gdf, fluor, cal_sr, vol_sr, gdf_peak_for_idx=None, fluor_z=None):
    out = sorted(events, key=lambda e: (e.first_spike, e.last_spike))
    gpk = np.asarray(gdf_peak_for_idx if gdf_peak_for_idx is not None else gdf, dtype=float).ravel()
    if fluor_z is None:
        fz = _global_ztrace_p8(gdf, cal_sr, return_stats=False)
    else:
        fz = np.asarray(fluor_z, dtype=float).ravel()
    fz = np.asarray(fz, dtype=float).ravel()
    for ev in out:
        ev.peak_idx = _candidate_peak_idx(ev, gpk, cal_sr, vol_sr)
        peak_idx_z = _candidate_peak_idx(ev, fz, cal_sr, vol_sr)
        if ev.peak_idx is None or int(ev.peak_idx) < 0 or len(gdf) == 0:
            ev.peak_idx = -1
            ev.peak_global_dff = 0.0
        else:
            v = float(gdf[int(ev.peak_idx)])
            ev.peak_global_dff = float(v) if np.isfinite(v) else 0.0
        ev.amp_local_dff = _local_amp_from_fluor(ev, fluor, ev.peak_idx, cal_sr, vol_sr, normalize=True)
        ev.amp_local_z = _local_amp_from_fluor(ev, fz, peak_idx_z, cal_sr, vol_sr, normalize=False)
    return out


def _apply_calcium_threshold_only(out, gdf, cal_sr, vol_sr, threshold_value=None):
    thr = _calcium_threshold_value(gdf) if threshold_value is None else float(threshold_value)
    for ev in out:
        c_check, gprev = _event_start_gdf_at_ap(ev, gdf, cal_sr, vol_sr)
        if (not np.isfinite(gprev)) or not (gprev < thr):
            ev.include = False
            ev.include_reason = f"global_dff_before_ap_ge_{thr:.3f}"
    return out


def _apply_event_distance_rules(out, gdf, cal_sr, vol_sr, gdf_peak_for_idx=None):
    simple_pre_gap = int(round(vol_sr * (SIMPLE_PRE_MS / 1000.0)))
    simple_post_gap = int(round(vol_sr * (SIMPLE_POST_MS / 1000.0)))
    c_pre_gap = int(round(vol_sr * (COMPLEX_PRE_MS / 1000.0)))
    c_post_gap = int(round(vol_sr * (COMPLEX_POST_MS / 1000.0)))
    c_post_gap_for_complex = int(round(vol_sr * (COMPLEX_POST_FOR_COMPLEX_MS / 1000.0)))
    gpk = np.asarray(gdf_peak_for_idx if gdf_peak_for_idx is not None else gdf, dtype=float).ravel()

    for i, ev in enumerate(out):
        for j, other in enumerate(out):
            if i == j:
                continue
            if other.event_type == "simple":
                before = (other.last_spike < ev.first_spike) and (
                    other.last_spike >= ev.first_spike - simple_pre_gap
                )
                after = (other.first_spike > ev.last_spike) and (
                    other.first_spike <= ev.last_spike + simple_post_gap
                )
                if before or after:
                    ev.include = False
                    ev.include_reason = "near_simple_event_250pre_150post"
                    break
        if not ev.include:
            continue

        for j, other in enumerate(out):
            if i == j:
                continue
            if other.event_type in ("complex", "plateau"):
                post_gap = c_post_gap_for_complex if ev.event_type in ("complex", "plateau") else c_post_gap
                before = (other.last_spike < ev.first_spike) and (
                    other.last_spike >= ev.first_spike - c_pre_gap
                )
                after = (other.first_spike > ev.last_spike) and (
                    other.first_spike <= ev.last_spike + post_gap
                )
                if before or after:
                    ev.include = False
                    ev.include_reason = (
                        "near_complex_event_400pre_300post"
                        if ev.event_type in ("complex", "plateau")
                        else "near_complex_event_400pre_150post"
                    )
                    break
        if not ev.include:
            continue

        if ev.event_type in ("complex", "plateau"):
            prev_complex = None
            for k in range(i - 1, -1, -1):
                if out[k].event_type in ("complex", "plateau"):
                    prev_complex = out[k]
                    break
            if prev_complex is not None:
                prev_peak = int(prev_complex.peak_idx) if int(prev_complex.peak_idx) >= 0 else _candidate_peak_idx(
                    prev_complex, gpk, cal_sr, vol_sr
                )
                if prev_peak < 0:
                    continue
                next_start = len(gdf) - 1
                if i > 0:
                    next_start = _vol_to_cal_idx(ev.first_spike, vol_sr, cal_sr, len(gdf))
                end_cap = int(max(prev_peak, min(len(gdf) - 1, next_start - 1)))
                tail_seg = gdf[prev_peak : end_cap + 1] if end_cap >= prev_peak else np.array([], dtype=float)
                if tail_seg.size:
                    prev_tail_min = float(np.nanmin(tail_seg))
                    curr_pk = float(max(ev.peak_global_dff, 1e-9))
                    ratio = abs(prev_tail_min) / curr_pk
                    if ratio > TAIL_RATIO_THR:
                        ev.include = False
                        ev.include_reason = "tail_ratio_gt_0.3"
                        continue
    return out


def choose_events(
    events,
    gdf,
    fluor,
    cal_sr,
    vol_sr,
    selection_mode="event_distance",
    calcium_threshold_value=None,
):
    start_exclude_frames = int(round(float(vol_sr) * (float(START_EXCLUDE_MS) / 1000.0)))
    sigma_frames = float(PEAK_FIND_SMOOTH_SIGMA_S) * float(cal_sr)
    gdf_peak = _gaussian_smooth_1d(gdf, sigma_frames)
    out = _compute_event_features(events, gdf, fluor, cal_sr, vol_sr, gdf_peak_for_idx=gdf_peak)
    for ev in out:
        ev.include = True
        ev.include_reason = "selected"
        if int(ev.first_spike) < int(start_exclude_frames):
            ev.include = False
            ev.include_reason = f"event_start_lt_{int(START_EXCLUDE_MS)}ms_from_recording_start"
    mode = str(selection_mode).strip().lower()
    if mode == "calcium_threshold":
        thr_val = _calcium_threshold_value(gdf) if calcium_threshold_value is None else float(calcium_threshold_value)
        return _apply_calcium_threshold_only(out, gdf, cal_sr, vol_sr, threshold_value=thr_val)
    return _apply_event_distance_rules(out, gdf, cal_sr, vol_sr, gdf_peak_for_idx=gdf_peak)


def _color_for_event(ev):
    if not ev.include:
        return COLOR_NON_CHOSEN
    cname = ev.class_name
    if cname == "single":
        return COLOR_SINGLE
    if cname == "simple_burst":
        return COLOR_SIMPLE_BURST
    if cname == "complex":
        return COLOR_COMPLEX
    return COLOR_PLATEAU


def save_summary_plot_html(
    trace_vol,
    gdf,
    events,
    out_html,
    vol_sr,
    cal_sr,
    title,
    calcium_threshold=None,
    overlay_smooth_sigma_s=OVERLAY_SMOOTH_SIGMA_S,
    calcium_trace_label="global dF/F0",
    calcium_axis_title="Global dF/F0",
    calcium_value_label="value",
    extra_note=None,
    single_peak_labels=None,
    connect_masked_gaps=False,
):
    tv = np.arange(len(trace_vol), dtype=float) / float(vol_sr)
    tc = np.arange(len(gdf), dtype=float) / float(cal_sr)
    sigma_frames = float(overlay_smooth_sigma_s) * float(cal_sr)
    gdf_plot = _gaussian_smooth_1d(gdf, sigma_frames)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tv,
            y=trace_vol,
            mode="lines",
            name="voltage",
            line=dict(color=VOLTAGE_TRACE_COLOR, width=1),
            opacity=0.85,
            yaxis="y1",
            connectgaps=bool(connect_masked_gaps),
        )
    )
    all_sp = sorted({int(s) for ev in events for s in ev.spikes.tolist() if 0 <= int(s) < len(trace_vol)})
    y_by_spike = {s: float(trace_vol[s]) for s in all_sp}
    for ev in events:
        col = _color_for_event(ev)
        sp = [int(s) for s in ev.spikes if 0 <= int(s) < len(trace_vol)]
        if len(sp) == 0:
            continue
        xs = np.asarray(sp, dtype=float) / float(vol_sr)
        ys = np.asarray([y_by_spike[s] for s in sp], dtype=float)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(color=col, size=6),
                showlegend=False,
                yaxis="y1",
                hovertemplate="t=%{x:.3f}s<br>v=%{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=tc,
            y=gdf_plot,
            mode="lines",
            name=f"{calcium_trace_label} (smoothed, sigma={float(overlay_smooth_sigma_s):.3f}s)",
            line=dict(color=CALCIUM_TRACE_COLOR, width=1),
            opacity=0.85,
            yaxis="y2",
            connectgaps=bool(connect_masked_gaps),
        )
    )
    peak_x = []
    peak_y = []
    peak_txt = []
    for ev in events:
        if not bool(ev.include):
            continue
        p = int(ev.peak_idx) if ev.peak_idx is not None else -1
        if p < 0 or p >= len(gdf):
            continue
        wa, wb = _candidate_peak_search_bounds(ev, len(gdf), cal_sr, vol_sr)
        y_raw = float(gdf[p]) if np.isfinite(gdf[p]) else np.nan
        yv = float(gdf_plot[p])
        if not np.isfinite(yv):
            continue
        peak_x.append(float(p) / float(cal_sr))
        peak_y.append(yv)
        peak_txt.append(
            f"type={ev.event_type}<br>class={ev.class_name}<br>include={bool(ev.include)}"
            f"<br>peak_idx={p}"
            f"<br>search_window=[{int(wa)}:{int(wb)}]"
            f"<br>last_spike_cal_idx={int(_vol_to_cal_idx(ev.last_spike, vol_sr, cal_sr, len(gdf)))}"
            f"<br>{calcium_value_label}_raw@peak={y_raw:.3f}"
        )
    if len(peak_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_x,
                y=peak_y,
                mode="markers",
                name="calcium peaks",
                marker=dict(color="black", size=9, symbol="x"),
                yaxis="y2",
                text=peak_txt,
                hovertemplate=f"%{{text}}<br>t=%{{x:.3f}}s<br>{calcium_value_label}=%{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        width=1400,
        height=500,
        template="simple_white",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Voltage (a.u.)"),
        yaxis2=dict(title=calcium_axis_title, overlaying="y", side="right"),
    )
    if calcium_threshold is not None:
        thr = float(calcium_threshold)
        fig.add_shape(
            type="line",
            x0=float(tv[0]) if len(tv) else 0.0,
            x1=float(tv[-1]) if len(tv) else 1.0,
            y0=thr,
            y1=thr,
            xref="x",
            yref="y2",
            line=dict(color="#d62728", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=float(tv[-1]) if len(tv) else 1.0,
            y=thr,
            xref="x",
            yref="y2",
            text=f"threshold={thr:g}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#d62728"),
            bgcolor="rgba(255,255,255,0.7)",
        )
    if extra_note is not None and str(extra_note).strip() != "":
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            text=str(extra_note),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        )
    if single_peak_labels is not None:
        for it in single_peak_labels:
            try:
                xx = float(it.get("x", np.nan))
                yy = float(it.get("y", np.nan))
                txt = str(it.get("text", ""))
            except Exception:
                continue
            if (not np.isfinite(xx)) or (not np.isfinite(yy)) or txt.strip() == "":
                continue
            fig.add_annotation(
                x=xx,
                y=yy,
                xref="x",
                yref="y2",
                text=txt,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yshift=8,
                align="center",
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.7)",
            )
    fig.write_html(out_html)


def _safe_write_svg(fig, path):
    try:
        fig.write_image(path)
    except Exception as e:
        print(f"[WARN] could not save SVG: {path} ({e})")


def _remove_plotly_colorbars(fig):
    """Disable accidental marker/coloraxis colorbars before saved subplot export."""
    try:
        fig.update_layout(coloraxis_showscale=False, coloraxis2_showscale=False, coloraxis3_showscale=False)
    except Exception:
        pass
    for trace in getattr(fig, "data", []):
        try:
            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.showscale = False
                if hasattr(trace.marker, "colorbar"):
                    trace.marker.colorbar = None
        except Exception:
            pass
        try:
            if hasattr(trace, "showscale"):
                trace.showscale = False
        except Exception:
            pass
    return fig


def _apply_summary_zscore_subplot_export_style(fig):
    """50 mm x 50 mm, no colorbar, compact white standalone z-score subplot."""
    _remove_plotly_colorbars(fig)
    axis_font = dict(size=7, family="Arial, sans-serif")
    title_font = dict(size=8, family="Arial, sans-serif")
    fig.update_layout(
        width=int(SUMMARY_ZSCORE_SUBPLOT_PX),
        height=int(SUMMARY_ZSCORE_SUBPLOT_PX),
        margin=dict(l=32, r=3, t=3, b=24),
        font=dict(size=7, family="Arial, sans-serif"),
        showlegend=False,
        title="",
        template="simple_white",
    )
    fig.update_xaxes(
        automargin=False,
        tickfont=axis_font,
        title_font=title_font,
        title_standoff=2,
        tickangle=0,
        ticklen=2,
        tickwidth=0.7,
        linecolor="black",
        linewidth=0.8,
    )
    fig.update_yaxes(
        automargin=False,
        tickfont=axis_font,
        title_font=title_font,
        title_standoff=2,
        ticklen=2,
        tickwidth=0.7,
        linecolor="black",
        linewidth=0.8,
    )
    fig.update_annotations(font=dict(size=7, family="Arial, sans-serif"))
    for trace in getattr(fig, "data", []):
        try:
            if hasattr(trace, "marker") and trace.marker is not None and getattr(trace.marker, "size", None) is not None:
                trace.marker.size = min(float(trace.marker.size), 4.0)
                if getattr(trace.marker, "line", None) is not None and getattr(trace.marker.line, "width", None) is not None:
                    trace.marker.line.width = min(float(trace.marker.line.width), 0.7)
        except Exception:
            pass
        try:
            if hasattr(trace, "line") and trace.line is not None and getattr(trace.line, "width", None) is not None:
                trace.line.width = min(float(trace.line.width), 1.0)
        except Exception:
            pass
    return fig


def _append_suffix_before_ext(path, suffix):
    stem, ext = os.path.splitext(str(path))
    return f"{stem}{suffix}{ext}"


def _event_type_norm(t):
    tt = str(t).strip().lower()
    if tt in ("simple", "complex", "plateau"):
        return tt
    return "simple"


def _bold_axis_title(text):
    if text is None:
        return text
    return str(text)


def _apply_bold_axis_fonts(fig, tick_size=14, title_size=16):
    regular_family = "Arial, sans-serif"
    tick_font = dict(size=int(tick_size), family=regular_family)
    title_font = dict(size=int(title_size), family=regular_family)
    fig.update_xaxes(tickfont=tick_font, title_font=title_font)
    fig.update_yaxes(tickfont=tick_font, title_font=title_font)
    return fig


def _auc_length_type_mask(df, display_type):
    et = df["event_type"].astype(str).str.lower()
    if str(display_type).lower() == "complex":
        return et.isin(["complex", "plateau"])
    return et == "simple"


def _event_auc_bounds(
    ev,
    trace_vol,
    vol_sr,
    z_thr=1.5,
    all_spike_idx=None,
    spike_mask_pre_ms=AUC_SPIKE_MASK_PRE_MS,
    spike_mask_post_ms=AUC_SPIKE_MASK_POST_MS,
):
    n = int(len(trace_vol))
    if n <= 0:
        return 0, 0
    fs = int(max(0, min(n - 1, int(ev.first_spike))))
    ls = int(max(fs, min(n - 1, int(ev.last_spike))))

    v = np.asarray(trace_vol, dtype=float).ravel()
    v_stats = v
    if all_spike_idx is not None:
        sp = _as_int_sorted_unique(all_spike_idx)
        if sp.size > 0:
            trace_sub = v.copy()
            pre_fr = int(max(0, round(float(vol_sr) * (float(spike_mask_pre_ms) / 1000.0))))
            post_fr = int(max(0, round(float(vol_sr) * (float(spike_mask_post_ms) / 1000.0))))
            for s in sp:
                si = int(max(0, min(v.shape[0] - 1, int(s))))
                a = int(max(0, si - pre_fr))
                b = int(min(v.shape[0] - 1, si + post_fr))
                trace_sub[a : b + 1] = np.nan
            trace_sub = _interp_nan_1d(trace_sub)
            cand = np.asarray(trace_sub, dtype=float).ravel()
            cand = cand[np.isfinite(cand)]
            if cand.size >= 10:
                v_stats = cand
    mu = float(np.nanmean(v_stats))
    sd = float(np.nanstd(v_stats))
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return int(max(0, fs - 3)), int(min(n - 1, ls + 3))
    z = (v - mu) / sd

    left = z[: fs + 1]
    below_left = np.where(left < float(z_thr))[0]
    has_left = below_left.size > 0
    if has_left:
        start = int(below_left[-1] + 1)
        start = int(max(0, min(fs, start)))
    else:
        start = -1

    right = z[ls:]
    below_right = np.where(right < float(z_thr))[0]
    has_right = below_right.size > 0
    if has_right:
        end = int(ls + below_right[0] - 1)
        end = int(max(ls, min(n - 1, end)))
    else:
        end = -1

    if (not has_left) or (not has_right) or (end <= start):
        start = int(max(0, fs - 3))
        end = int(min(n - 1, ls + 3))
    return start, end


def _event_auc_value(ev, trace_vol, vol_sr, all_spike_idx=None):
    s, e = _event_auc_bounds(ev, trace_vol, vol_sr, z_thr=1.5, all_spike_idx=all_spike_idx)
    v = np.asarray(trace_vol, dtype=float).ravel()
    if v.size == 0:
        return np.nan, s, e
    seg = v[s : e + 1]
    if seg.size == 0 or not np.any(np.isfinite(seg)):
        return np.nan, s, e
    t = np.arange(seg.size, dtype=float) / float(vol_sr)
    auc = float(np.trapz(seg, x=t))
    return auc, s, e


def _apply_plateau_min_auc_duration_rule(events, trace_vol, vol_sr, min_dur_ms=PLATEAU_MIN_AUC_DUR_MS):
    if events is None or len(events) == 0:
        return events
    all_spikes = _as_int_sorted_unique(
        np.concatenate([np.asarray(ev.spikes, dtype=int).ravel() for ev in events])
    ) if len(events) > 0 else np.array([], dtype=int)
    min_frames = int(max(1, round(float(min_dur_ms) * float(vol_sr) / 1000.0)))
    out = []
    for ev in events:
        if str(ev.event_type).strip().lower() != "plateau":
            out.append(ev)
            continue
        s, e = _event_auc_bounds(ev, trace_vol, vol_sr, z_thr=1.5, all_spike_idx=all_spikes)
        dur_frames = int(max(0, int(e) - int(s) + 1))
        if dur_frames < min_frames:
            fb = str(getattr(ev, "prev_event_type", "")).strip().lower()
            if fb not in ("simple", "complex"):
                fb = "complex" if int(ev.n_spikes) > 1 else "simple"
            ev.event_type = fb
            ev.source = f"{ev.source}|plateau_relabel_lt_{int(min_dur_ms)}ms"
        out.append(ev)
    out = sorted(out, key=lambda e: (e.first_spike, e.last_spike, e.event_type))
    return out


def _make_cell_trace_id(cell_folder, suffix):
    return f"{cell_folder}::{suffix}"


def _make_cell_trace_label(cell_folder, suffix):
    base = os.path.basename(str(cell_folder).rstrip("\\/"))
    return f"{base} | {suffix}"


def _events_to_analysis_df(events, trace_vol, cell_folder, suffix, pipeline_name, brain_state):
    rows = []
    cid = _make_cell_trace_id(cell_folder, suffix)
    clabel = _make_cell_trace_label(cell_folder, suffix)
    all_spikes = _as_int_sorted_unique(np.concatenate([np.asarray(ev.spikes, dtype=int).ravel() for ev in events])) if len(events) > 0 else np.array([], dtype=int)
    for k, ev in enumerate(events):
        vauc, vs, ve = _event_auc_value(ev, trace_vol, VOL_SR, all_spike_idx=all_spikes)
        rows.append(
            {
                "selection_pipeline": pipeline_name,
                "cell_folder": cell_folder,
                "suffix": suffix,
                "cell_trace_id": cid,
                "cell_trace_label": clabel,
                "brainState": brain_state,
                "event_idx": k,
                "event_type": _event_type_norm(ev.event_type),
                "n_spikes": int(ev.n_spikes),
                "include": bool(ev.include),
                "include_reason": str(ev.include_reason),
                "peak_global_dff": float(ev.peak_global_dff),
                "amp_local_dff": float(ev.amp_local_dff),
                "amp_local_z": float(getattr(ev, "amp_local_z", np.nan)),
                "voltage_auc": float(vauc) if np.isfinite(vauc) else np.nan,
                "voltage_auc_start": int(vs),
                "voltage_auc_end": int(ve),
                "event_length_s": (float(max(0, int(ve) - int(vs) + 1)) / float(VOL_SR))
                if (np.isfinite(vs) and np.isfinite(ve) and int(ve) >= int(vs))
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _spike_bin(n_spikes):
    n = int(max(1, int(n_spikes)))
    return 6 if n >= 6 else n


def _spike_ticks():
    vals = [1, 2, 3, 4, 5, 6]
    txt = ["1", "2", "3", "4", "5", "6+"]
    return vals, txt


def _add_peak_vs_spike_subplot(
    fig,
    df,
    row,
    col,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    add_mean_lines=False,
    compact=False,
    mean_label_mode=None,
    add_regression=False,
    show_corr_stats=False,
    n_cols_for_stats=3,
):
    if df is None or len(df) == 0:
        return
    tick_vals, tick_txt = _spike_ticks()
    type_offset = {"simple": -0.18, "complex": 0.0, "plateau": 0.18}
    mean_stack_y = {"simple": 1.09, "complex": 1.06, "plateau": 1.03}
    mean_tag = {"simple": "S", "complex": "C", "plateau": "P"}
    for et in ("simple", "complex", "plateau"):
        colr = PLOT_TYPE_COLOR[et]
        sub = df[df["event_type"] == et].copy()
        if len(sub) == 0:
            continue
        sub["n_bin"] = sub["n_spikes"].astype(int).map(_spike_bin)
        for n in [1, 2, 3, 4, 5, 6]:
            y = sub.loc[sub["n_bin"] == int(n), str(y_col)].to_numpy(float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            x0 = float(n) + float(type_offset[et])
            fig.add_trace(
                go.Violin(
                    x=np.full(y.size, x0),
                    y=y,
                    points=False,
                    line=dict(color=colr, width=1),
                    fillcolor=colr,
                    opacity=0.18,
                    width=0.8,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            rng = np.random.default_rng(1234 + int(n) + (0 if et == "simple" else (100 if et == "complex" else 200)))
            xj = x0 + rng.uniform(-0.24, 0.24, size=y.size)
            fig.add_trace(
                go.Scatter(
                    x=xj,
                    y=y,
                    mode="markers",
                    marker=dict(symbol="circle-open", color=colr, size=8, line=dict(color=colr, width=1.8)),
                    showlegend=False,
                    hovertemplate=f"type={et}<br>n_spikes={'6+' if int(n)==6 else int(n)}<br>peak=%{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            if add_mean_lines:
                m = float(np.nanmean(y))
                # Draw mean lines after points so they stay visible as top layer.
                if et == "simple":
                    # white border under black mean line
                    fig.add_trace(
                        go.Scatter(
                            x=[x0 - 0.16, x0 + 0.16],
                            y=[m, m],
                            mode="lines",
                            line=dict(color="white", width=6),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    # black border under colored mean line
                    fig.add_trace(
                        go.Scatter(
                            x=[x0 - 0.16, x0 + 0.16],
                            y=[m, m],
                            mode="lines",
                            line=dict(color="black", width=6),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=[x0 - 0.16, x0 + 0.16],
                        y=[m, m],
                        mode="lines",
                        line=dict(color=("black" if et == "simple" else colr), width=3.5),
                        showlegend=False,
                        hovertemplate=f"type={et}<br>n_spikes={'6+' if int(n)==6 else int(n)}<br>mean={m:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                if str(mean_label_mode).lower() == "above":
                    x_ann = float(n)
                    xref_name = f"x{'' if (row==1 and col==1) else ((row-1)*2+col)}"
                    fig.add_annotation(
                        x=x_ann,
                        y=float(mean_stack_y.get(et, 1.03)),
                        xref=xref_name,
                        yref="paper",
                        text=f"{mean_tag.get(et, et[0].upper())}:{m:.2f}",
                        showarrow=False,
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=10, color=colr),
                    )
        if bool(add_regression) and et in ("simple", "complex"):
            xf = sub["n_bin"].to_numpy(float)
            yf = sub[str(y_col)].to_numpy(float)
            ok = np.isfinite(xf) & np.isfinite(yf)
            xf = xf[ok]
            yf = yf[ok]
            if xf.size >= 2 and np.unique(xf).size >= 2:
                lin = _fit_linear(xf, yf)
                if lin is not None:
                    p, _rmse, r2 = lin
                    xs = np.linspace(float(np.nanmin(xf)), float(np.nanmax(xf)), 100)
                    fig.add_trace(
                        go.Scatter(
                            x=xs + float(type_offset[et]),
                            y=np.polyval(p, xs),
                            mode="lines",
                            line=dict(color=("black" if et == "simple" else colr), width=2.4),
                            name=f"{et} regression R²={r2:.2f}",
                            legendgroup=et,
                            showlegend=False,
                            hovertemplate=f"type={et}<br>spike-count regression<br>R²={r2:.3f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
        if bool(show_corr_stats) and et in ("simple", "complex"):
            xf_stat = sub["n_bin"].to_numpy(float)
            yf_stat = sub[str(y_col)].to_numpy(float)
            r_stat, p_stat = _pearson_r_p_local(xf_stat, yf_stat)
            if np.isfinite(r_stat):
                tag = "S" if et == "simple" else "C"
                fig.add_annotation(
                    x=0.98,
                    y=(0.98 if et == "simple" else 0.86),
                    xref=_subplot_axis_domain_ref(row, col, n_cols=n_cols_for_stats)[0],
                    yref=_subplot_axis_domain_ref(row, col, n_cols=n_cols_for_stats)[1],
                    text=f"{tag}: r={r_stat:.2f}, {_p_text(p_stat)} {_p_to_stars(p_stat)}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    align="right",
                    font=dict(size=10, color=("black" if et == "simple" else colr)),
                    bgcolor="rgba(255,255,255,0.72)",
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=0.5,
                )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_txt,
        tickangle=(0 if compact else 35),
        title_text=("" if compact else _bold_axis_title("# spikes in event")),
        row=row,
        col=col,
    )
    fig.update_yaxes(title_text=("" if compact else _bold_axis_title(str(y_label))), row=row, col=col)


def _mean_summary_text_by_group(df, y_col):
    lines = []
    if df is None or len(df) == 0:
        return lines
    work = df.copy()
    work["n_bin"] = work["n_spikes"].astype(int).map(_spike_bin)
    for n in [1, 2, 3, 4, 5, 6]:
        parts = []
        for et in ("simple", "complex", "plateau"):
            sub = work[(work["event_type"] == et) & (work["n_bin"] == n)]
            y = sub[str(y_col)].to_numpy(float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            m = float(np.nanmean(y))
            tag = "S" if et == "simple" else ("C" if et == "complex" else "P")
            parts.append(f"{tag}:{m:.2f}")
        if parts:
            ntag = "6+" if n == 6 else str(n)
            lines.append(f"n={ntag}  " + "  ".join(parts))
    return lines


def _add_auc_subplot(
    fig,
    df,
    row,
    col,
    show_legend=False,
    compact=False,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    add_regression=False,
    show_corr_stats=False,
    n_cols_for_stats=3,
):
    stat_lines = []
    for et in ("simple", "complex"):
        colr = PLOT_TYPE_COLOR[et]
        sub = df[_auc_length_type_mask(df, et)].copy()
        if len(sub) == 0:
            continue
        x = sub["voltage_auc"].to_numpy(float)
        y = sub[str(y_col)].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol="circle-open", color=colr, size=8, line=dict(color=colr, width=1.8)),
                name=et,
                legendgroup=et,
                showlegend=show_legend,
                hovertemplate=f"type={et}<br>AUC=%{{x:.4f}}<br>calcium=%{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if bool(show_corr_stats):
            r_stat, p_stat = _pearson_r_p_local(x, y)
            if np.isfinite(r_stat):
                tag = "S" if et == "simple" else "C"
                stat_lines.append(f"{tag}: r={r_stat:.2f}, {_p_text(p_stat)} {_p_to_stars(p_stat)}")
        if bool(add_regression) and x.size >= 2 and np.unique(x).size >= 2:
            lin = _fit_linear(x, y)
            if lin is not None:
                p, _rmse, r2 = lin
                xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=np.polyval(p, xs),
                        mode="lines",
                        line=dict(color=colr, width=2),
                        name=f"{et} regression R²={r2:.2f}",
                        legendgroup=et,
                        showlegend=show_legend,
                        hovertemplate=f"type={et}<br>regression<br>R²={r2:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
    if bool(show_corr_stats):
        _add_corr_stats_annotation(fig, row, col, stat_lines, n_cols=n_cols_for_stats)
    fig.update_xaxes(title_text=("" if compact else _bold_axis_title("voltage AUC")), row=row, col=col)
    fig.update_yaxes(title_text=("" if compact else _bold_axis_title(str(y_label))), row=row, col=col)


def _add_event_length_subplot(
    fig,
    df,
    row,
    col,
    show_legend=False,
    compact=False,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    add_regression=False,
    show_corr_stats=False,
    n_cols_for_stats=3,
):
    stat_lines = []
    for et in ("simple", "complex"):
        colr = PLOT_TYPE_COLOR[et]
        sub = df[_auc_length_type_mask(df, et)].copy()
        if len(sub) == 0:
            continue
        x = sub["event_length_s"].to_numpy(float)
        y = sub[str(y_col)].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol="circle-open", color=colr, size=8, line=dict(color=colr, width=1.8)),
                name=et,
                legendgroup=et,
                showlegend=show_legend,
                hovertemplate=f"type={et}<br>length=%{{x:.4f}} s<br>calcium=%{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if bool(show_corr_stats):
            r_stat, p_stat = _pearson_r_p_local(x, y)
            if np.isfinite(r_stat):
                tag = "S" if et == "simple" else "C"
                stat_lines.append(f"{tag}: r={r_stat:.2f}, {_p_text(p_stat)} {_p_to_stars(p_stat)}")
        if bool(add_regression) and x.size >= 2 and np.unique(x).size >= 2:
            lin = _fit_linear(x, y)
            if lin is not None:
                p, _rmse, r2 = lin
                xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=np.polyval(p, xs),
                        mode="lines",
                        line=dict(color=colr, width=2),
                        name=f"{et} regression R²={r2:.2f}",
                        legendgroup=et,
                        showlegend=show_legend,
                        hovertemplate=f"type={et}<br>regression<br>R²={r2:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
    if bool(show_corr_stats):
        _add_corr_stats_annotation(fig, row, col, stat_lines, n_cols=n_cols_for_stats)
    fig.update_xaxes(title_text=("" if compact else _bold_axis_title("event length (s)")), row=row, col=col)
    fig.update_yaxes(title_text=("" if compact else _bold_axis_title(str(y_label))), row=row, col=col)


def _plot_cell_two_panel(
    chosen_df,
    title,
    save_html,
    save_svg,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    auc_df=None,
    fig_width_px=None,
    fig_height_px=None,
    add_regression=False,
    show_corr_stats=False,
):
    if auc_df is None:
        auc_df = chosen_df
    cell_panel_px = int(SQUARE_PANEL_PX)
    cell_gap_px = int(PANEL_GAP_PX)
    cell_margin_l = int(PANEL_MARGIN_L_PX)
    cell_margin_r = int(max(PANEL_MARGIN_R_PX, 140))
    cell_margin_t = int(max(40, PANEL_MARGIN_T_PX - 20))
    cell_margin_b = int(PANEL_MARGIN_B_PX)
    if fig_width_px is None:
        fig_width_px = int((3 * cell_panel_px) + (2 * cell_gap_px) + cell_margin_l + cell_margin_r)
    if fig_height_px is None:
        fig_height_px = int(cell_panel_px + cell_margin_t + cell_margin_b)
    compact_layout = bool(fig_width_px <= (11.0 * CM_TO_PX) or fig_height_px <= (3.5 * CM_TO_PX))
    plot_w = max(120, int(fig_width_px) - cell_margin_l - cell_margin_r)
    hspace = (float(cell_gap_px) / float(plot_w)) if plot_w > 0 else 0.1
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=(0.18 if compact_layout else min(0.16, hspace * 1.6)),
    )
    _add_peak_vs_spike_subplot(
        fig,
        chosen_df,
        1,
        1,
        y_col=str(y_col),
        y_label=str(y_label),
        add_mean_lines=True,
        compact=compact_layout,
        mean_label_mode=None,
        add_regression=add_regression,
        show_corr_stats=show_corr_stats,
        n_cols_for_stats=3,
    )
    _add_auc_subplot(
        fig, auc_df, 1, 2, show_legend=False, compact=compact_layout, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression,
        show_corr_stats=show_corr_stats, n_cols_for_stats=3
    )
    _add_event_length_subplot(
        fig, auc_df, 1, 3, show_legend=False, compact=compact_layout, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression,
        show_corr_stats=show_corr_stats, n_cols_for_stats=3
    )
    fig.update_layout(
        template="simple_white",
        # Keep pooled per-subplot width consistent with cellavg_amp_vs_spikecount_zscore panel size.
        width=int(fig_width_px),
        height=int(fig_height_px),
        title="",
        margin=dict(
            l=(190 if compact_layout else max(cell_margin_l, 170)),
            r=(36 if compact_layout else cell_margin_r),
            t=(12 if compact_layout else cell_margin_t),
            b=(30 if compact_layout else cell_margin_b),
        ),
        legend=dict(
            orientation=("h" if compact_layout else "v"),
            x=1.0,
            y=(1.02 if compact_layout else 1.0),
            xanchor="right",
            yanchor=("bottom" if compact_layout else "top"),
            font=dict(size=(7 if compact_layout else 12)),
        ),
        showlegend=False,
    )
    if compact_layout:
        fig.update_annotations(font=dict(size=16, family="Arial"))
        fig.update_xaxes(title_standoff=4)
        fig.update_yaxes(title_standoff=0)
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(title_text=_bold_axis_title("# spikes"), row=1, col=1)
        fig.update_xaxes(title_text=_bold_axis_title("AUC"), row=1, col=2)
        fig.update_xaxes(title_text=_bold_axis_title("length (s)"), row=1, col=3)
        fig.update_yaxes(title_text=_bold_axis_title("calcium (z)" if "z" in str(y_col).lower() else "calcium"), row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=3)
    _apply_bold_axis_fonts(fig, tick_size=(20 if compact_layout else 22), title_size=(22 if compact_layout else 24))
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)


def _plot_cell_single_panel(
    chosen_df,
    title,
    save_html,
    save_svg,
    mode,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    auc_df=None,
    add_regression=False,
    show_corr_stats=False,
):
    if auc_df is None:
        auc_df = chosen_df
    panel_px = int(SQUARE_PANEL_PX)
    margin_l = int(PANEL_MARGIN_L_PX)
    margin_r = 26
    margin_t = 30
    margin_b = int(PANEL_MARGIN_B_PX)
    fig = make_subplots(rows=1, cols=1)
    if mode == "spike":
        _add_peak_vs_spike_subplot(
            fig,
            chosen_df,
            1,
            1,
            y_col=str(y_col),
            y_label=str(y_label),
            add_mean_lines=True,
            mean_label_mode=None,
            add_regression=add_regression,
            show_corr_stats=show_corr_stats,
            n_cols_for_stats=1,
        )
    elif mode == "auc":
        _add_auc_subplot(fig, auc_df, 1, 1, show_legend=False, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression, show_corr_stats=show_corr_stats, n_cols_for_stats=1)
    else:
        _add_event_length_subplot(fig, auc_df, 1, 1, show_legend=False, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression, show_corr_stats=show_corr_stats, n_cols_for_stats=1)
    fig.update_layout(
        template="simple_white",
        width=int(panel_px + margin_l + margin_r),
        height=int(panel_px + margin_t + margin_b),
        title="",
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        legend=dict(
            orientation="h",
            x=1.0,
            y=1.02,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10),
        ),
    )
    fig.update_layout(showlegend=False)
    _apply_bold_axis_fonts(fig, tick_size=22, title_size=24)
    if str(y_col) == "amp_local_z":
        _apply_summary_zscore_subplot_export_style(fig)
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)


def _plot_all_cells_grid(
    all_df,
    mode,
    save_html,
    save_svg,
    all_trace_meta=None,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
):
    if all_trace_meta is None:
        all_trace_meta = []
    if (all_df is None or len(all_df) == 0) and len(all_trace_meta) == 0:
        return
    if len(all_trace_meta) > 0:
        ids = [t["cell_trace_id"] for t in all_trace_meta]
        label_by_id = {t["cell_trace_id"]: t["cell_trace_label"] for t in all_trace_meta}
    else:
        ids = sorted(all_df["cell_trace_id"].unique().tolist())
        label_by_id = {
            cid: str(all_df[all_df["cell_trace_id"] == cid]["cell_trace_label"].iloc[0]) for cid in ids
        }
    n = len(ids)
    ncols = 6
    nrows = int(np.ceil(n / float(ncols)))
    specs = [[{} for _ in range(ncols)] for _ in range(nrows)]
    titles = []
    for i in range(nrows * ncols):
        if i < n:
            cid = ids[i]
            label = str(label_by_id.get(cid, cid))
            titles.append(label)
        else:
            titles.append("")
    panel_px = int(round(float(GRID_SUBPLOT_CM) * CM_TO_PX))
    gap_px = 18
    margin_l = 55
    margin_r = 20
    margin_t = 55
    margin_b = 45

    plot_w = int(ncols * panel_px + (ncols - 1) * gap_px)
    plot_h = int(nrows * panel_px + max(0, nrows - 1) * gap_px)
    hspace = float(gap_px) / float(plot_w) if ncols > 1 else 0.0
    if nrows > 1:
        vspace = float(gap_px) / float(plot_h)
        vspace = min(vspace, (1.0 / float(nrows - 1)) - 1e-6)
        vspace = max(0.0, float(vspace))
    else:
        vspace = 0.0

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
        specs=specs,
        subplot_titles=titles,
    )
    for i, cid in enumerate(ids):
        r = i // ncols + 1
        c = i % ncols + 1
        sub = all_df[all_df["cell_trace_id"] == cid].copy()
        if mode == "spike":
            _add_peak_vs_spike_subplot(
                fig,
                sub,
                r,
                c,
                y_col=str(y_col),
                y_label=str(y_label),
                compact=True,
            )
        elif mode == "auc":
            _add_auc_subplot(
                fig,
                sub,
                r,
                c,
                show_legend=(i == 0),
                compact=True,
                y_col=str(y_col),
                y_label=str(y_label),
            )
        else:
            _add_event_length_subplot(
                fig,
                sub,
                r,
                c,
                show_legend=(i == 0),
                compact=True,
                y_col=str(y_col),
                y_label=str(y_label),
            )

    if mode == "spike":
        ttl = "All cells | calcium response vs spike count"
    elif mode == "auc":
        ttl = "All cells | calcium response vs voltage AUC"
    else:
        ttl = "All cells | calcium response vs event length"
    fig.update_layout(
        template="simple_white",
        width=plot_w + margin_l + margin_r,
        height=plot_h + margin_t + margin_b,
        title="",
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
    )
    fig.update_annotations(font=dict(size=18, family="Arial"))
    fig.update_yaxes(title_standoff=0)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    _apply_bold_axis_fonts(fig, tick_size=17, title_size=20)
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)

    # Save an exact subplot mapping that matches THIS figure generation call.
    # This avoids any mismatch from later reconstruction assumptions.
    try:
        map_rows = []
        for i, cid in enumerate(ids):
            r = i // ncols + 1
            c = i % ncols + 1
            sub = all_df[all_df["cell_trace_id"] == cid]
            row0 = sub.iloc[0] if len(sub) > 0 else None
            map_rows.append(
                {
                    "idx": int(i + 1),
                    "row": int(r),
                    "col": int(c),
                    "cell_trace_id": str(cid),
                    "cell_trace_label": str(label_by_id.get(cid, cid)),
                    "cell_path": ("" if row0 is None else str(row0.get("cell_path", ""))),
                    "suffix": ("" if row0 is None else str(row0.get("suffix", ""))),
                    "pipeline": ("" if row0 is None else str(row0.get("pipeline", ""))),
                    "brainState": ("" if row0 is None else str(row0.get("brainState", ""))),
                    "figure_mode": str(mode),
                    "ncols": int(ncols),
                    "nrows": int(nrows),
                    "figure_html": str(save_html),
                }
            )
        if len(map_rows) > 0:
            map_df = pd.DataFrame(map_rows)
            base, _ = os.path.splitext(str(save_html))
            map_csv = base + "_mapping.csv"
            map_tsv = base + "_mapping.tsv"
            map_df.to_csv(map_csv, index=False)
            map_df.to_csv(map_tsv, sep="\t", index=False)
    except Exception as _map_err:
        print(f"[warn] failed to save all-cells mapping for {save_html}: {_map_err}")



def _cohens_d_complex_minus_simple(simple_values, complex_values):
    simple_values = np.asarray(simple_values, dtype=float)
    complex_values = np.asarray(complex_values, dtype=float)
    simple_values = simple_values[np.isfinite(simple_values)]
    complex_values = complex_values[np.isfinite(complex_values)]
    if simple_values.size < 2 or complex_values.size < 2:
        return np.nan
    simple_var = float(np.var(simple_values, ddof=1))
    complex_var = float(np.var(complex_values, ddof=1))
    pooled_var = (
        ((simple_values.size - 1) * simple_var + (complex_values.size - 1) * complex_var)
        / float(simple_values.size + complex_values.size - 2)
    )
    if (not np.isfinite(pooled_var)) or pooled_var <= 0:
        return np.nan
    return float((np.mean(complex_values) - np.mean(simple_values)) / np.sqrt(pooled_var))


def _simple_complex_grouped_separation_metrics(df, x_col, y_col, panel_name):
    """Quantify simple/complex scatter separation without event-within-cell leakage.

    The primary metric is leave-one-cell(or segment)-out ROC-AUC from a logistic
    model using the two plotted coordinates. AUC=0.5 means no separability and
    AUC=1.0 means perfect separation. Cohen's d values are descriptive only.
    """
    out = {
        "panel": str(panel_name),
        "x_column": str(x_col),
        "y_column": str(y_col),
        "method": "leave-one-cell-out standardized logistic regression on plotted x/y; complex=positive class",
        "status": "not_run",
        "n_events": 0,
        "n_simple": 0,
        "n_complex": 0,
        "n_groups": 0,
        "n_evaluated_events": 0,
        "cv_roc_auc": np.nan,
        "cv_balanced_accuracy": np.nan,
        "cohens_d_x_complex_minus_simple": np.nan,
        "cohens_d_y_complex_minus_simple": np.nan,
    }
    if df is None or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        out["status"] = "missing_required_columns"
        return out

    work = df.copy()
    work["event_type"] = work["event_type"].astype(str).str.lower().str.strip()
    work = work[work["event_type"].isin(["simple", "complex"])].copy()
    work["x"] = pd.to_numeric(work[x_col], errors="coerce")
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work[np.isfinite(work["x"]) & np.isfinite(work["y"])].copy()
    if "cell_trace_id" in work.columns:
        group = work["cell_trace_id"].astype(str)
    elif "cell_path" in work.columns:
        group = work["cell_path"].astype(str)
    elif "recording_id" in work.columns:
        group = work["recording_id"].astype(str)
    else:
        group = pd.Series([f"event_{idx}" for idx in work.index], index=work.index)
    work["group"] = group

    simple = work[work["event_type"] == "simple"]
    complex_events = work[work["event_type"] == "complex"]
    out.update({
        "n_events": int(len(work)),
        "n_simple": int(len(simple)),
        "n_complex": int(len(complex_events)),
        "n_groups": int(work["group"].nunique()),
        "cohens_d_x_complex_minus_simple": _cohens_d_complex_minus_simple(simple["x"], complex_events["x"]),
        "cohens_d_y_complex_minus_simple": _cohens_d_complex_minus_simple(simple["y"], complex_events["y"]),
    })
    if len(simple) < 2 or len(complex_events) < 2 or work["group"].nunique() < 2:
        out["status"] = "insufficient_events_or_groups"
        return out

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x_all = work[["x", "y"]].to_numpy(float)
        y_all = (work["event_type"] == "complex").to_numpy(int)
        groups = work["group"].to_numpy(str)
        pred = np.full(len(work), np.nan, dtype=float)
        for held_out_group in pd.unique(groups):
            test = groups == held_out_group
            train = ~test
            if np.unique(y_all[train]).size < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0),
            )
            model.fit(x_all[train], y_all[train])
            pred[test] = model.predict_proba(x_all[test])[:, 1]
        valid = np.isfinite(pred)
        out["n_evaluated_events"] = int(np.sum(valid))
        if np.sum(valid) >= 2 and np.unique(y_all[valid]).size == 2:
            out["cv_roc_auc"] = float(roc_auc_score(y_all[valid], pred[valid]))
            out["cv_balanced_accuracy"] = float(balanced_accuracy_score(y_all[valid], (pred[valid] >= 0.5).astype(int)))
            out["status"] = "ok"
        else:
            out["status"] = "insufficient_valid_leave_one_group_out_predictions"
    except Exception as exc:
        out["status"] = f"failed:{type(exc).__name__}:{exc}"
    return out


def _save_pooled_zscore_single_panels(all_df, auc_df, save_svg, y_col, y_label):
    """Write the three pooled z-score panels as standalone SVG/HTML files."""
    base, _ = os.path.splitext(str(save_svg))
    panel_specs = [
        ("spike", "subplot01_spike_count"),
        ("auc", "subplot02_voltage_auc"),
        ("length", "subplot03_event_length"),
    ]
    paths = []
    for mode, suffix in panel_specs:
        svg_path = f"{base}_{suffix}.svg"
        html_path = f"{base}_{suffix}.html"
        _plot_cell_single_panel(
            all_df,
            title="",
            save_html=html_path,
            save_svg=svg_path,
            mode=mode,
            y_col=str(y_col),
            y_label=str(y_label),
            auc_df=auc_df,
        )
        paths.append({"panel": mode, "svg": svg_path, "html": html_path})
    return paths


def _save_pooled_zscore_simple_complex_separation(auc_df, save_svg, y_col):
    """Save grouped simple/complex scatter-separation metrics for AUC and duration."""
    base, _ = os.path.splitext(str(save_svg))
    metrics = pd.DataFrame([
        _simple_complex_grouped_separation_metrics(
            auc_df, "voltage_auc", str(y_col), "calcium_response_vs_voltage_auc"
        ),
        _simple_complex_grouped_separation_metrics(
            auc_df, "event_length_s", str(y_col), "calcium_response_vs_event_length"
        ),
    ])
    csv_path = f"{base}_simple_vs_complex_separation.csv"
    metrics.to_csv(csv_path, index=False)
    return metrics, csv_path


def _plot_population_two_panel(
    all_df,
    title,
    save_html,
    save_svg,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    auc_df=None,
    add_regression=False,
    show_corr_stats=False,
):
    if auc_df is None:
        auc_df = all_df
    ncols = 3
    panel_plot_w = max(120, int(SINGLE_PANEL_WIDTH_PX - PANEL_MARGIN_L_PX - PANEL_MARGIN_R_PX))
    plot_w = int(ncols * panel_plot_w + (ncols - 1) * PANEL_GAP_PX)
    hspace = float(PANEL_GAP_PX) / float(plot_w) if ncols > 1 else 0.0
    fig_w = int(plot_w + PANEL_MARGIN_L_PX + PANEL_MARGIN_R_PX)
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=min(0.16, hspace * 1.6),
    )
    _add_peak_vs_spike_subplot(
        fig,
        all_df,
        1,
        1,
        y_col=str(y_col),
        y_label=str(y_label),
        add_mean_lines=True,
        mean_label_mode=None,
        add_regression=add_regression,
        show_corr_stats=show_corr_stats,
        n_cols_for_stats=3,
    )
    _add_auc_subplot(fig, auc_df, 1, 2, show_legend=True, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression, show_corr_stats=show_corr_stats, n_cols_for_stats=3)
    _add_event_length_subplot(fig, auc_df, 1, 3, show_legend=False, y_col=str(y_col), y_label=str(y_label), add_regression=add_regression, show_corr_stats=show_corr_stats, n_cols_for_stats=3)
    fig.update_layout(
        template="simple_white",
        width=fig_w,
        height=TWO_PANEL_HEIGHT_PX,
        title="",
        margin=dict(l=PANEL_MARGIN_L_PX, r=PANEL_MARGIN_R_PX, t=PANEL_MARGIN_T_PX, b=PANEL_MARGIN_B_PX),
    )
    _apply_bold_axis_fonts(fig, tick_size=16, title_size=28)
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)

    # The z-score pooled figure also gets standalone exports for its three panels
    # and grouped simple-vs-complex separation metrics for the AUC/duration scatters.
    if str(y_col) == "amp_local_z":
        panel_paths = _save_pooled_zscore_single_panels(
            all_df, auc_df, save_svg=save_svg, y_col=str(y_col), y_label=str(y_label)
        )
        for panel_path in panel_paths:
            print(f"[saved pooled z-score subplot] {panel_path['svg']}")
        separation_df, separation_csv = _save_pooled_zscore_simple_complex_separation(
            auc_df, save_svg=save_svg, y_col=str(y_col)
        )
        print(f"[saved simple/complex separation metrics] {separation_csv}")
        print(separation_df.to_string(index=False))


def _fit_linear(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return None
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    rmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))
    sse = float(np.nansum((y - yhat) ** 2))
    sst = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    return p, rmse, r2


def _pearson_r_p_local(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan
    if spstats is not None:
        res = spstats.pearsonr(x, y)
        if hasattr(res, 'statistic'):
            return float(res.statistic), float(res.pvalue)
        return float(res[0]), float(res[1])
    return float(np.corrcoef(x, y)[0, 1]), np.nan


def _p_to_stars(p):
    if not np.isfinite(p):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def _p_text(p):
    if not np.isfinite(p):
        return 'p=n/a'
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}'


def _subplot_axis_domain_ref(row, col, n_cols=3):
    axis_num = int((int(row) - 1) * int(n_cols) + int(col))
    suffix = '' if axis_num == 1 else str(axis_num)
    return f'x{suffix} domain', f'y{suffix} domain'


def _add_corr_stats_annotation(fig, row, col, lines, n_cols=3):
    lines = [str(x) for x in lines if str(x)]
    if not lines:
        return
    xref, yref = _subplot_axis_domain_ref(row, col, n_cols=n_cols)
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref=xref,
        yref=yref,
        text='<br>'.join(lines),
        showarrow=False,
        xanchor='right',
        yanchor='top',
        align='right',
        font=dict(size=10, color='black'),
        bgcolor='rgba(255,255,255,0.72)',
        bordercolor='rgba(0,0,0,0.25)',
        borderwidth=0.5,
    )


def _fit_michaelis_menten(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3:
        return None
    x_pos = x[x > 0]
    if x_pos.size == 0:
        return None
    xmin = float(np.nanmin(x_pos))
    xmax = float(np.nanmax(x_pos))
    if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or xmax <= 0:
        return None
    km_grid = np.logspace(np.log10(max(xmin * 0.1, 1e-3)), np.log10(max(xmax * 10.0, 1e-2)), 220)
    best = None
    for km in km_grid:
        f = x / (km + x)
        A = np.column_stack([np.ones_like(f), f])  # y0 + vmax * f
        try:
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            continue
        y0 = float(beta[0])
        vmax = float(beta[1])
        yhat = y0 + vmax * f
        sse = float(np.nansum((y - yhat) ** 2))
        if (best is None) or (sse < best["sse"]):
            best = {"km": float(km), "y0": y0, "vmax": vmax, "yhat": yhat, "sse": sse}
    if best is None:
        return None
    yhat = np.asarray(best["yhat"], dtype=float)
    rmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))
    sse = float(np.nansum((y - yhat) ** 2))
    sst = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    return best, rmse, r2


def _normalize_01(x):
    a = np.asarray(x, dtype=float).ravel()
    out = np.full(a.shape, np.nan, dtype=float)
    ok = np.isfinite(a)
    if not np.any(ok):
        return out
    lo = float(np.nanmin(a[ok]))
    hi = float(np.nanmax(a[ok]))
    if hi - lo <= 1e-12:
        out[ok] = 0.5
        return out
    out[ok] = (a[ok] - lo) / (hi - lo)
    return out


def _plot_pooled_fit_panels(
    all_df,
    title,
    save_html,
    save_svg,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_Y_LABEL,
    auc_df=None,
):
    if all_df is None or len(all_df) == 0:
        return
    if auc_df is None:
        auc_df = all_df
    ncols = 2
    panel_plot_w = max(120, int(SINGLE_PANEL_WIDTH_PX - PANEL_MARGIN_L_PX - PANEL_MARGIN_R_PX))
    plot_w = int(ncols * panel_plot_w + (ncols - 1) * PANEL_GAP_PX)
    hspace = float(PANEL_GAP_PX) / float(plot_w) if ncols > 1 else 0.0
    fig_w = int(plot_w + PANEL_MARGIN_L_PX + PANEL_MARGIN_R_PX)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=hspace)

    # Subplot 1: per-event-type fit of calcium vs spike count (all events).
    work = all_df.copy()
    work["n_bin"] = work["n_spikes"].astype(int).map(_spike_bin)
    for et in ("simple", "complex", "plateau"):
        colr = PLOT_TYPE_COLOR[et]
        sub = work[work["event_type"] == et].copy()
        if len(sub) == 0:
            continue
        x = sub["n_bin"].to_numpy(float)
        y = sub[str(y_col)].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(symbol="circle-open", color=colr, size=7, line=dict(color=colr, width=1)),
                name=f"{et} all events",
                legendgroup=f"fit_{et}",
                showlegend=True,
                hovertemplate=f"type={et}<br>spikes=%{{x}}<br>calcium=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        lin = _fit_linear(x, y)
        if lin is not None:
            p, rmse, r2_lin = lin
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 120)
            ys = np.polyval(p, xs)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=colr, width=2),
                    name=f"{et} linear RMSE={rmse:.3f}, R²={r2_lin:.3f}",
                    legendgroup=f"fit_{et}",
                    showlegend=True,
                    hovertemplate=f"type={et}<br>linear fit<br>RMSE={rmse:.3f}<br>R²={r2_lin:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        mm = _fit_michaelis_menten(x, y)
        if mm is not None:
            pars, rmse_mm, r2_mm = mm
            xs2 = np.linspace(np.nanmin(x), np.nanmax(x), 120)
            ys2 = float(pars["y0"]) + float(pars["vmax"]) * (xs2 / (float(pars["km"]) + xs2))
            fig.add_trace(
                go.Scatter(
                    x=xs2,
                    y=ys2,
                    mode="lines",
                    line=dict(color=colr, width=2, dash="dash"),
                    name=f"{et} Michaelis-Menten RMSE={rmse_mm:.3f}, R²={r2_mm:.3f}",
                    legendgroup=f"fit_{et}",
                    showlegend=True,
                    hovertemplate=(
                        f"type={et}<br>nonlinear: Michaelis-Menten"
                        f"<br>y0={float(pars['y0']):.3f}, Vmax={float(pars['vmax']):.3f}, Km={float(pars['km']):.3f}"
                        f"<br>RMSE={rmse_mm:.3f}<br>R²={r2_mm:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    fig.update_xaxes(
        title_text="# spikes in event",
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5, 6],
        ticktext=["1", "2", "3", "4", "5", "6+"],
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text=str(y_label), row=1, col=1)

    # Subplot 2: AUC and event length use auc_df so plateau-aware events stay visible there.
    cdf = (
        work.groupby("event_type", as_index=False)[["voltage_auc", "event_length_s", str(y_col)]]
        .mean()
        .rename(columns={str(y_col): "ymean"})
    )
    cdf = cdf[cdf["event_type"].isin(["simple", "complex", "plateau"])].copy()
    auc_work = auc_df.copy()
    auc_cdf = (
        auc_work.groupby("event_type", as_index=False)[["voltage_auc", "event_length_s", str(y_col)]]
        .mean()
        .rename(columns={str(y_col): "ymean"})
    )
    auc_cdf = auc_cdf[auc_cdf["event_type"].isin(["simple", "complex", "plateau"])].copy()
    # AUC: use all data points
    x_auc_all = _normalize_01(auc_work["voltage_auc"].to_numpy(float))
    y_all = auc_work[str(y_col)].to_numpy(float)
    ok_auc_all = np.isfinite(x_auc_all) & np.isfinite(y_all)
    if np.any(ok_auc_all):
        fig.add_trace(
            go.Scatter(
                x=x_auc_all[ok_auc_all],
                y=y_all[ok_auc_all],
                mode="markers",
                marker=dict(symbol="circle-open", color="rgba(0,0,0,0.22)", size=5, line=dict(color="rgba(0,0,0,0.22)", width=1)),
                name="AUC all events",
                showlegend=True,
                hovertemplate="AUC all data (norm)<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        lin_auc = _fit_linear(x_auc_all[ok_auc_all], y_all[ok_auc_all])
        if lin_auc is not None:
            p_auc, rmse_auc, r2_auc = lin_auc
            xs = np.linspace(float(np.nanmin(x_auc_all[ok_auc_all])), float(np.nanmax(x_auc_all[ok_auc_all])), 200)
            ys = np.polyval(p_auc, xs)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name=f"AUC fit (all) RMSE={rmse_auc:.3f}, R²={r2_auc:.3f}",
                    showlegend=True,
                    hovertemplate=f"AUC fit on all events (solid)<br>RMSE={rmse_auc:.3f}<br>R²={r2_auc:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Length: keep centroid-based fit
    if len(auc_cdf) > 0:
        x_auc_cent = _normalize_01(auc_cdf["voltage_auc"].to_numpy(float))
        y = auc_cdf["ymean"].to_numpy(float)

        ok_auc_cent = np.isfinite(x_auc_cent) & np.isfinite(y)

        if np.any(ok_auc_cent):
            fig.add_trace(
                go.Scatter(
                    x=x_auc_cent[ok_auc_cent],
                    y=y[ok_auc_cent],
                    mode="markers+text",
                    text=auc_cdf.loc[ok_auc_cent, "event_type"].astype(str).tolist(),
                    textposition="top center",
                    marker=dict(symbol="diamond-open", color="black", size=8, line=dict(color="black", width=1)),
                    name="AUC centroids",
                    showlegend=True,
                    hovertemplate="AUC centroid (norm)<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    if len(auc_cdf) > 0:
        x_len = _normalize_01(auc_cdf["event_length_s"].to_numpy(float)) if "event_length_s" in auc_cdf.columns else np.array([])
        y = auc_cdf["ymean"].to_numpy(float)

        ok_len = np.isfinite(x_len) & np.isfinite(y)

        if np.any(ok_len):
            fig.add_trace(
                go.Scatter(
                    x=x_len[ok_len],
                    y=y[ok_len],
                    mode="markers",
                    marker=dict(symbol="square-open", color="gray", size=8, line=dict(color="gray", width=1)),
                    name="Length centroids",
                    showlegend=True,
                    hovertemplate="Length centroid (norm)<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            lin_len = _fit_linear(x_len[ok_len], y[ok_len])
            if lin_len is not None:
                p_len, rmse_len, r2_len = lin_len
                xs = np.linspace(float(np.nanmin(x_len[ok_len])), float(np.nanmax(x_len[ok_len])), 120)
                ys = np.polyval(p_len, xs)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="gray", width=2),
                        name=f"Length fit RMSE={rmse_len:.3f}, R²={r2_len:.3f}",
                        showlegend=True,
                        hovertemplate=f"Length fit<br>RMSE={rmse_len:.3f}<br>R²={r2_len:.3f}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

    fig.update_xaxes(title_text="normalized centroid predictor (0..1)", row=1, col=2)
    fig.update_yaxes(title_text=str(y_label), row=1, col=2)
    fig.update_layout(
        template="simple_white",
        title="",
        width=fig_w,
        height=TWO_PANEL_HEIGHT_PX,
        margin=dict(l=PANEL_MARGIN_L_PX, r=PANEL_MARGIN_R_PX, t=PANEL_MARGIN_T_PX, b=PANEL_MARGIN_B_PX),
    )
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)


def _build_cellavg_amp_vs_spikes_df(all_df, y_col=GLOBAL_DFF_Y_COL):
    if all_df is None or len(all_df) == 0:
        return pd.DataFrame()
    tmp = all_df.copy()
    tmp["n_bin"] = tmp["n_spikes"].astype(int).map(_spike_bin)
    if "brainState" in tmp.columns:
        is_motor_cell = tmp["brainState"].astype(str).str.lower().str.contains("motor", na=False)
    else:
        is_motor_cell = pd.Series(False, index=tmp.index)
    if "cell_folder" in tmp.columns:
        cell_folder_for_group = tmp["cell_folder"].astype(str)
    elif "cell_trace_id" in tmp.columns:
        cell_folder_for_group = tmp["cell_trace_id"].astype(str)
    else:
        cell_folder_for_group = pd.Series([f"cell_{i}" for i in range(len(tmp))], index=tmp.index)
    if "cell_trace_id" in tmp.columns:
        cell_trace_id_for_group = tmp["cell_trace_id"].astype(str)
    else:
        cell_trace_id_for_group = cell_folder_for_group
    if "cell_trace_label" in tmp.columns:
        cell_trace_label_for_group = tmp["cell_trace_label"].astype(str)
    else:
        cell_trace_label_for_group = cell_folder_for_group.map(lambda p: os.path.basename(str(p).rstrip("\\/")) or str(p))
    cell_label_for_group = cell_folder_for_group.map(lambda p: os.path.basename(str(p).rstrip("\\/")) or str(p))
    tmp["cellavg_group_id"] = np.where(is_motor_cell, cell_folder_for_group, cell_trace_id_for_group)
    tmp["cellavg_group_label"] = np.where(
        is_motor_cell,
        cell_label_for_group + " | motor/rest chunks collapsed",
        cell_trace_label_for_group,
    )
    g = (
        tmp.groupby(["cellavg_group_id", "cellavg_group_label", "event_type", "n_bin"], as_index=False)[str(y_col)]
        .mean()
        .rename(columns={str(y_col): "cell_mean_amp"})
    )
    g["cell_mean_amp"] = pd.to_numeric(g["cell_mean_amp"], errors="coerce")
    g["n_bin"] = pd.to_numeric(g["n_bin"], errors="coerce")
    g = g[np.isfinite(g["cell_mean_amp"]) & np.isfinite(g["n_bin"])].copy()
    g["n_bin"] = g["n_bin"].astype(int)
    g["event_type"] = g["event_type"].astype(str)
    return g


def _plot_cellavg_amp_vs_spikes(
    all_df,
    title,
    save_html,
    save_svg,
    y_col=GLOBAL_DFF_Y_COL,
    y_label=GLOBAL_DFF_CELLAVG_Y_LABEL,
    show_fit_annotation=True,
    show_legend=True,
    show_corr_stats=False,
):
    g = _build_cellavg_amp_vs_spikes_df(all_df, y_col=y_col)
    if g is None or len(g) == 0:
        return
    fig = go.Figure()
    tick_vals, tick_txt = _spike_ticks()
    match_single_panel_style = str(y_col) == "amp_local_z"
    type_offset = {"simple": -0.13, "complex": 0.13, "plateau": 0.31} if match_single_panel_style else {"simple": -0.18, "complex": 0.0, "plateau": 0.18}
    ann_tag = {"simple": "S", "complex": "C", "plateau": "P"}
    ann = []
    fit_ann_y = {"simple": 0.99, "complex": 0.93}
    for et in ("simple", "complex", "plateau"):
        colr = PLOT_TYPE_COLOR[et]
        sub = g[g["event_type"] == et].copy()
        if len(sub) == 0:
            continue
        for nb in [1, 2, 3, 4, 5, 6]:
            subn = sub[sub["n_bin"].astype(int) == int(nb)].copy()
            if len(subn) == 0:
                continue
            x0 = float(nb) + float(type_offset[et])
            y = subn["cell_mean_amp"].to_numpy(float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            fig.add_trace(
                go.Violin(
                    x=np.full(y.size, x0),
                    y=y,
                    points=False,
                    line=dict(color=colr, width=1),
                    fillcolor=colr,
                    opacity=(0.18 if match_single_panel_style else 0.15),
                    width=(0.52 if match_single_panel_style else 0.34),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            x = []
            yy = []
            txt = []
            for _, r in subn.iterrows():
                rng = np.random.default_rng(abs(hash((str(r["cellavg_group_id"]), et, int(nb)))) % (2**32))
                x.append(x0 + float(rng.uniform(-0.11, 0.11) if match_single_panel_style else rng.uniform(-0.08, 0.08)))
                yy.append(float(r["cell_mean_amp"]))
                txt.append(f"{r['cellavg_group_label']}<br>type={et}<br>n={'6+' if nb==6 else nb}")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=yy,
                    mode="markers",
                    marker=dict(symbol="circle-open", color=colr, size=7, line=dict(color=colr, width=1)),
                    name=et,
                    legendgroup=et,
                    showlegend=(bool(show_legend) and nb == 1),
                    hovertemplate="%{text}<br>cell mean amp=%{y:.3f}<extra></extra>",
                    text=txt,
                )
            )
            m = float(np.nanmean(y))
            if et == "simple":
                # Match pooled plot style: white edge under a black center mean line.
                fig.add_trace(
                    go.Scatter(
                        x=[x0 - (0.16 if match_single_panel_style else 0.11), x0 + (0.16 if match_single_panel_style else 0.11)],
                        y=[m, m],
                        mode="lines",
                        line=dict(color="white", width=6),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            else:
                # Add black border for complex/plateau mean lines.
                fig.add_trace(
                    go.Scatter(
                        x=[x0 - (0.16 if match_single_panel_style else 0.11), x0 + (0.16 if match_single_panel_style else 0.11)],
                        y=[m, m],
                        mode="lines",
                        line=dict(color="black", width=5.5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[x0 - (0.16 if match_single_panel_style else 0.11), x0 + (0.16 if match_single_panel_style else 0.11)],
                    y=[m, m],
                    mode="lines",
                    line=dict(color=("black" if et == "simple" else colr), width=2.8),
                    showlegend=False,
                    hovertemplate=f"type={et}<br>n={'6+' if nb==6 else nb}<br>mean={m:.3f}<extra></extra>",
                )
            )
        if et in ("simple", "complex"):
            xf = sub["n_bin"].to_numpy(float)
            yf = sub["cell_mean_amp"].to_numpy(float)
            ok = np.isfinite(xf) & np.isfinite(yf)
            xf = xf[ok]
            yf = yf[ok]
            if xf.size >= 2 and np.unique(xf).size >= 2:
                try:
                    coef = np.polyfit(xf, yf, 1)
                    x_line = np.linspace(float(np.nanmin(xf)), float(np.nanmax(xf)), 100)
                    y_line = np.polyval(coef, x_line)
                    y_hat = np.polyval(coef, xf)
                    sse = float(np.nansum((yf - y_hat) ** 2))
                    sst = float(np.nansum((yf - np.nanmean(yf)) ** 2))
                    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
                    fig.add_trace(
                        go.Scatter(
                            x=x_line + float(type_offset.get(et, 0.0)),
                            y=y_line,
                            mode="lines",
                            line=dict(color=("black" if et == "simple" else colr), width=2.4),
                            name=f"{et} cell-mean fit",
                            legendgroup=et,
                            showlegend=bool(show_legend),
                            hovertemplate=(
                                f"{et} fit<br>"
                                f"slope={float(coef[0]):.3f}<br>"
                                f"intercept={float(coef[1]):.3f}<br>"
                                f"R²={r2:.3f}<extra></extra>"
                            ),
                        )
                    )
                    r_stat, p_stat = _pearson_r_p_local(xf, yf)
                    stat_text = f", r={r_stat:.2f}, {_p_text(p_stat)} {_p_to_stars(p_stat)}" if np.isfinite(r_stat) else ""
                    ann.append(
                        dict(
                            x=0.01,
                            y=float(fit_ann_y[et]),
                            xref="paper",
                            yref="paper",
                            text=(
                                f"{ann_tag[et]} fit: y={float(coef[0]):.2f}x+{float(coef[1]):.2f}, "
                                f"R2={r2:.2f}{stat_text}"
                            ),
                            showarrow=False,
                            xanchor="left",
                            yanchor="top",
                            font=dict(size=9, color=colr),
                        )
                    )
                except Exception:
                    pass
    fig.update_layout(
        template="simple_white",
        width=(int(SQUARE_PANEL_PX + PANEL_MARGIN_L_PX + 26) if match_single_panel_style else SINGLE_PANEL_WIDTH_PX),
        height=(int(SQUARE_PANEL_PX + 30 + PANEL_MARGIN_B_PX) if match_single_panel_style else TWO_PANEL_HEIGHT_PX),
        title="",
        margin=(
            dict(l=PANEL_MARGIN_L_PX, r=26, t=30, b=PANEL_MARGIN_B_PX)
            if match_single_panel_style
            else dict(l=PANEL_MARGIN_L_PX, r=PANEL_MARGIN_R_PX, t=PANEL_MARGIN_T_PX, b=PANEL_MARGIN_B_PX)
        ),
        annotations=(ann if (bool(show_fit_annotation) or bool(show_corr_stats)) else []),
        showlegend=bool(show_legend),
        xaxis=dict(
            title="# spikes in event",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_txt,
            tickangle=35,
        ),
        yaxis=dict(title=str(y_label)),
    )
    if match_single_panel_style:
        _apply_bold_axis_fonts(fig, tick_size=22, title_size=24)
        _apply_summary_zscore_subplot_export_style(fig)
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)


def _save_cellavg_amp_vs_spikecount_lmm_tukey(
    all_df,
    out_dir,
    filename_prefix="cellavg_amp_vs_spikecount_zscore",
    y_col="amp_local_z",
):
    os.makedirs(out_dir, exist_ok=True)
    cellavg_df = _build_cellavg_amp_vs_spikes_df(all_df, y_col=y_col)
    base_csv = os.path.join(out_dir, f"{filename_prefix}_lmm_input_cellavg.csv")
    fixed_csv = os.path.join(out_dir, f"{filename_prefix}_lmm_fixed_effects.csv")
    anova_csv = os.path.join(out_dir, f"{filename_prefix}_lmm_wald_terms.csv")
    lmm_pairwise_csv = os.path.join(out_dir, f"{filename_prefix}_lmm_pairwise_contrasts.csv")
    tukey_csv = os.path.join(out_dir, f"{filename_prefix}_tukey_hsd.csv")
    summary_csv = os.path.join(out_dir, f"{filename_prefix}_lmm_tukey_summary.csv")
    out_paths = {
        "input_csv": base_csv,
        "fixed_effects_csv": fixed_csv,
        "wald_terms_csv": anova_csv,
        "lmm_pairwise_contrasts_csv": lmm_pairwise_csv,
        "tukey_csv": tukey_csv,
        "summary_csv": summary_csv,
    }

    if cellavg_df is None or len(cellavg_df) == 0:
        pd.DataFrame([{"status": "skipped_empty_input"}]).to_csv(summary_csv, index=False)
        return out_paths

    work = cellavg_df.copy()
    work = work[work["event_type"].isin(["simple", "complex", "plateau"])].copy()
    work["cell_mean_amp"] = pd.to_numeric(work["cell_mean_amp"], errors="coerce")
    work["n_bin"] = pd.to_numeric(work["n_bin"], errors="coerce")
    work = work[np.isfinite(work["cell_mean_amp"]) & np.isfinite(work["n_bin"])].copy()
    observed_spike_counts = [str(int(v)) for v in sorted(work["n_bin"].dropna().unique())]
    observed_event_types = [v for v in ["simple", "complex", "plateau"] if v in set(work["event_type"].astype(str))]
    work["spike_count"] = pd.Categorical(work["n_bin"].astype(int).astype(str), categories=observed_spike_counts, ordered=True)
    work["event_type"] = pd.Categorical(work["event_type"].astype(str), categories=observed_event_types, ordered=False)
    work["event_spike_group"] = work["event_type"].astype(str) + "_n" + work["spike_count"].astype(str)
    work.to_csv(base_csv, index=False)

    summary_rows = [{
        "status": "started",
        "n_rows": int(len(work)),
        "n_cells": int(work["cellavg_group_id"].nunique()) if "cellavg_group_id" in work.columns else 0,
        "n_groups": int(work["event_spike_group"].nunique()) if "event_spike_group" in work.columns else 0,
        "model": "cell_mean_amp ~ C(event_type) * C(spike_count) + (1|cellavg_group_id)",
        "posthoc": "Tukey HSD on event_type:spike_count groups",
    }]

    if len(work) < 4 or work["cellavg_group_id"].nunique() < 2 or work["event_spike_group"].nunique() < 2:
        summary_rows[0]["status"] = "skipped_not_enough_data"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        return out_paths

    if smf is None:
        summary_rows[0]["status"] = "skipped_statsmodels_unavailable"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        return out_paths

    lmm_fit = None
    lmm_error = None
    formula_attempts = [
        ("requested_event_type_x_spike_count", "cell_mean_amp ~ C(event_type) * C(spike_count)"),
        ("observed_event_spike_group_fallback", "cell_mean_amp ~ C(event_spike_group)"),
    ]
    for formula_name, formula in formula_attempts:
        model = smf.mixedlm(
            formula,
            data=work,
            groups=work["cellavg_group_id"].astype(str),
        )
        for fit_method in ("lbfgs", "powell"):
            try:
                lmm_fit = model.fit(reml=False, method=fit_method, maxiter=1000, disp=False)
                summary_rows[0]["status"] = "ok" if formula_name == formula_attempts[0][0] else "ok_observed_interaction_fallback"
                summary_rows[0]["lmm_formula_used"] = formula
                summary_rows[0]["lmm_formula_note"] = formula_name
                summary_rows[0]["lmm_fit_method"] = fit_method
                break
            except Exception as exc:
                lmm_error = repr(exc)
        if lmm_fit is not None:
            break

    if lmm_fit is not None:
        fit = lmm_fit
        params = pd.Series(fit.params)
        bse = pd.Series(fit.bse).reindex(params.index)
        tvalues = pd.Series(fit.tvalues).reindex(params.index)
        pvalues = pd.Series(fit.pvalues).reindex(params.index)
        fe = pd.DataFrame({
            "term": params.index.astype(str),
            "estimate": params.to_numpy(float),
            "std_error": bse.to_numpy(float),
            "z_value": tvalues.to_numpy(float),
            "p_value": pvalues.to_numpy(float),
        })
        fe.to_csv(fixed_csv, index=False)
        try:
            wald = fit.wald_test_terms(skip_single=False)
            wald_df = wald.table.reset_index().rename(columns={"index": "term"})
            wald_df.to_csv(anova_csv, index=False)
        except Exception as exc:
            pd.DataFrame([{"status": "wald_terms_failed", "error": repr(exc)}]).to_csv(anova_csv, index=False)
        try:
            from patsy import build_design_matrices

            group_df = (
                work[["event_type", "spike_count", "event_spike_group"]]
                .drop_duplicates()
                .sort_values(["event_type", "spike_count", "event_spike_group"])
                .reset_index(drop=True)
            )
            design_info = fit.model.data.design_info
            x_df = build_design_matrices([design_info], group_df, return_type="dataframe")[0]
            fe_params = pd.Series(fit.fe_params)
            x_df = x_df.reindex(columns=fe_params.index, fill_value=0.0)
            x_mat = x_df.to_numpy(dtype=float)
            cov = fit.cov_params()
            if hasattr(cov, "loc"):
                cov_fe = cov.loc[fe_params.index, fe_params.index].to_numpy(dtype=float)
            else:
                cov_fe = np.asarray(cov, dtype=float)[: len(fe_params), : len(fe_params)]
            means = x_mat @ fe_params.to_numpy(dtype=float)
            mean_se = np.sqrt(np.maximum(np.diag(x_mat @ cov_fe @ x_mat.T), 0.0))

            pair_rows = []
            for i in range(len(group_df)):
                for j in range(i + 1, len(group_df)):
                    contrast = x_mat[j, :] - x_mat[i, :]
                    diff = float(means[j] - means[i])
                    se = float(np.sqrt(max(float(contrast @ cov_fe @ contrast.T), 0.0)))
                    z_value = float(diff / se) if se > 0 else np.nan
                    p_raw = float(2.0 * spstats.norm.sf(abs(z_value))) if spstats is not None and np.isfinite(z_value) else np.nan
                    p_tukey = np.nan
                    if spstats is not None and hasattr(spstats, "studentized_range") and np.isfinite(z_value):
                        p_tukey = float(spstats.studentized_range.sf(np.sqrt(2.0) * abs(z_value), len(group_df), np.inf))
                    pair_rows.append({
                        "group1": str(group_df.loc[i, "event_spike_group"]),
                        "group2": str(group_df.loc[j, "event_spike_group"]),
                        "estimated_mean_group1": float(means[i]),
                        "estimated_mean_group2": float(means[j]),
                        "se_mean_group1": float(mean_se[i]),
                        "se_mean_group2": float(mean_se[j]),
                        "meandiff_lmm_group2_minus_group1": diff,
                        "se_diff": se,
                        "z_value": z_value,
                        "p_raw": p_raw,
                        "p_tukey_approx": p_tukey,
                        "lmm_formula_used": str(summary_rows[0].get("lmm_formula_used", "")),
                    })
            pairwise_df = pd.DataFrame(pair_rows)
            if len(pairwise_df) > 0 and multipletests is not None and "p_raw" in pairwise_df:
                pvals = pd.to_numeric(pairwise_df["p_raw"], errors="coerce").to_numpy(dtype=float)
                ok_p = np.isfinite(pvals)
                for method, out_col, reject_col in [
                    ("holm", "p_holm", "reject_holm_0.05"),
                    ("fdr_bh", "p_fdr_bh", "reject_fdr_bh_0.05"),
                    ("bonferroni", "p_bonferroni", "reject_bonferroni_0.05"),
                ]:
                    corrected = np.full_like(pvals, np.nan, dtype=float)
                    rejected = np.full(pvals.shape, False, dtype=bool)
                    if np.any(ok_p):
                        rej, p_corr, _, _ = multipletests(pvals[ok_p], alpha=0.05, method=method)
                        corrected[ok_p] = p_corr
                        rejected[ok_p] = rej
                    pairwise_df[out_col] = corrected
                    pairwise_df[reject_col] = rejected
            if "p_tukey_approx" in pairwise_df:
                pairwise_df["reject_tukey_approx_0.05"] = pd.to_numeric(pairwise_df["p_tukey_approx"], errors="coerce") < 0.05
            pairwise_df.to_csv(lmm_pairwise_csv, index=False)
            summary_rows[0]["lmm_pairwise_contrasts"] = "saved"
        except Exception as exc:
            pd.DataFrame([{"status": "lmm_pairwise_contrasts_failed", "error": repr(exc)}]).to_csv(lmm_pairwise_csv, index=False)
            summary_rows[0]["lmm_pairwise_contrasts"] = "failed"
        summary_rows[0]["lmm_converged"] = bool(getattr(fit, "converged", False))
        summary_rows[0]["log_likelihood"] = float(getattr(fit, "llf", np.nan))
    else:
        summary_rows[0]["status"] = "lmm_failed_tukey_only"
        summary_rows[0]["error"] = lmm_error
        pd.DataFrame([{"status": "lmm_failed", "error": lmm_error}]).to_csv(fixed_csv, index=False)
        pd.DataFrame([{"status": "lmm_failed", "error": lmm_error}]).to_csv(anova_csv, index=False)
        pd.DataFrame([{"status": "skipped_lmm_not_available", "error": lmm_error}]).to_csv(lmm_pairwise_csv, index=False)

    if pairwise_tukeyhsd is None:
        pd.DataFrame([{"status": "skipped_pairwise_tukeyhsd_unavailable"}]).to_csv(tukey_csv, index=False)
    else:
        try:
            tuk = pairwise_tukeyhsd(
                endog=work["cell_mean_amp"].astype(float).to_numpy(),
                groups=work["event_spike_group"].astype(str).to_numpy(),
                alpha=0.05,
            )
            tukey_df = pd.DataFrame(
                tuk.summary().data[1:],
                columns=[str(c).strip().replace(" ", "_").lower() for c in tuk.summary().data[0]],
            )
            tukey_df.to_csv(tukey_csv, index=False)
        except Exception as exc:
            pd.DataFrame([{"status": "tukey_failed", "error": repr(exc)}]).to_csv(tukey_csv, index=False)

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return out_paths


def _shape_rel_frames(vol_sr=VOL_SR, pre_ms=SHAPE_PRE_MS, post_ms=SHAPE_POST_MS):
    pre_f = int(round(float(pre_ms) * float(vol_sr) / 1000.0))
    post_f = int(round(float(post_ms) * float(vol_sr) / 1000.0))
    rel = np.arange(-pre_f, post_f + 1, dtype=int)
    t_ms = (rel.astype(float) / float(vol_sr)) * 1000.0
    return rel, t_ms


def _robust_z_p20_mad(x):
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        return np.array([], dtype=float)
    p20 = float(np.nanpercentile(a, 20))
    med = float(np.nanmedian(a))
    mad = float(np.nanmedian(np.abs(a - med)))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale <= 1e-12:
        sd = float(np.nanstd(a))
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return (a - p20) / scale


def _shape_init_groups():
    return {}


def _shape_add_event(groups, ev, trace_vol, trace_fluor, cal_sr, vol_sr, cell_trace_id, trace_cal_dff=None):
    if not bool(ev.include):
        return
    et = _event_type_norm(ev.event_type)
    nb = _spike_bin(ev.n_spikes)
    key = (et, int(nb))
    if key not in groups:
        groups[key] = {
            "vol_dff": [],
            "cal_dff": [],
            "cal_t_ms": [],
            "vol_z": [],
            "cal_z": [],
            "cell_ids": [],
        }

    v = np.asarray(trace_vol, dtype=float).ravel()
    c = np.asarray(trace_fluor, dtype=float).ravel()
    if v.size == 0 or c.size == 0:
        return
    c_dff = np.asarray(trace_cal_dff if trace_cal_dff is not None else c, dtype=float).ravel()
    v_z = _robust_z_p20_mad(v)
    c_z = _robust_z_p20_mad(c)

    rel_f, _ = _shape_rel_frames(vol_sr=vol_sr, pre_ms=SHAPE_PRE_MS, post_ms=SHAPE_POST_MS)
    wv_dff = np.full(rel_f.size, np.nan, dtype=float)
    wc_dff = np.full(rel_f.size, np.nan, dtype=float)
    wv_z = np.full(rel_f.size, np.nan, dtype=float)
    wc_z = np.full(rel_f.size, np.nan, dtype=float)
    fs = int(ev.first_spike)
    for i, rf in enumerate(rel_f):
        vi = int(fs + rf)
        if 0 <= vi < v_z.size:
            wv_dff[i] = float(v[vi])
            wv_z[i] = float(v_z[vi])
            ci = _vol_to_cal_idx(vi, vol_sr, cal_sr, c.size)
            if 0 <= ci < c_dff.size:
                wc_dff[i] = float(c_dff[ci])
            if 0 <= ci < c_z.size:
                wc_z[i] = float(c_z[ci])

    fs_s = float(fs) / float(vol_sr)
    cal_t_s = np.arange(c.size, dtype=float) / float(cal_sr)
    cal_t_ms = (cal_t_s - fs_s) * 1000.0
    cal_window_mask = (cal_t_ms >= -float(SHAPE_PRE_MS)) & (cal_t_ms <= float(SHAPE_POST_MS))
    cal_t_native_ms = cal_t_ms[cal_window_mask]
    wc_dff_native = c_dff[cal_window_mask] if c_dff.size == c.size else np.array([], dtype=float)
    wc_z_native = c_z[cal_window_mask] if c_z.size == c.size else np.array([], dtype=float)

    groups[key]["vol_dff"].append(wv_dff)
    groups[key]["cal_dff"].append(wc_dff)
    groups[key]["cal_t_ms"].append(cal_t_native_ms)
    groups[key].setdefault("cal_dff_native", []).append(wc_dff_native)
    groups[key].setdefault("cal_z_native", []).append(wc_z_native)
    groups[key]["vol_z"].append(wv_z)
    groups[key]["cal_z"].append(wc_z)
    groups[key]["cell_ids"].append(str(cell_trace_id))


def _shape_merge_groups(dst, src):
    out = dict(dst)
    for k, v in src.items():
        if k not in out:
            out[k] = {"vol_dff": [], "cal_dff": [], "cal_t_ms": [], "cal_dff_native": [], "cal_z_native": [], "vol_z": [], "cal_z": [], "cell_ids": []}
        out[k]["vol_dff"].extend(v.get("vol_dff", []))
        out[k]["cal_dff"].extend(v.get("cal_dff", []))
        out[k].setdefault("cal_t_ms", []).extend(v.get("cal_t_ms", []))
        out[k].setdefault("cal_dff_native", []).extend(v.get("cal_dff_native", []))
        out[k].setdefault("cal_z_native", []).extend(v.get("cal_z_native", []))
        out[k]["vol_z"].extend(v.get("vol_z", []))
        out[k]["cal_z"].extend(v.get("cal_z", []))
        out[k]["cell_ids"].extend(v.get("cell_ids", []))
    return out


def _shape_plot_groups(groups, title, save_html, save_svg, mode="dff"):
    if groups is None or len(groups) == 0:
        return
    rel_f, t_ms = _shape_rel_frames(vol_sr=VOL_SR, pre_ms=SHAPE_PRE_MS, post_ms=SHAPE_POST_MS)
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("dff", "z"):
        mode_norm = "dff"
    cal_key = "cal_dff" if mode_norm == "dff" else "cal_z"
    cal_native_key = "cal_dff_native" if mode_norm == "dff" else "cal_z_native"
    vol_key = "vol_dff" if mode_norm == "dff" else "vol_z"
    cal_name = "cal mean (global dF/F0)" if mode_norm == "dff" else "cal mean (z)"
    vol_name = "vol mean (a.u.)" if mode_norm == "dff" else "vol mean (z)"
    subplot_title_size = 13 if mode_norm == "z" else 20
    axis_tick_size = 15 if mode_norm == "z" else 17
    axis_title_size = 14 if mode_norm == "z" else 20
    row_spacing = 0.20 if mode_norm == "z" else 0.12
    row_height = 310 if mode_norm == "z" else 260

    bins = [1, 2, 3, 4, 5, 6]
    types = ["simple", "complex"]
    n_rows = len(types)
    n_cols = len(bins)
    subtitles = []
    for et in types:
        for b in bins:
            entry = groups.get((et, b), None)
            n_evt = 0
            if entry is not None:
                arr = np.asarray(entry.get(cal_key, []), dtype=float)
                n_evt = int(arr.shape[0]) if arr.ndim == 2 else 0
            subtitles.append(f"# spikes = {'6+' if b == 6 else b} | N={n_evt}")
    specs = [[{"secondary_y": True} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=0.04,
        vertical_spacing=row_spacing,
        subplot_titles=subtitles,
        specs=specs,
    )

    # Axis caps shared across all subplots:
    # y_min=-0.1 and y_max=1.2*max(mean-trace peak across all event-type/spike-count groups).
    vol_peak_global = np.nan
    cal_peak_global = np.nan
    for et in types:
        for b in bins:
            entry = groups.get((et, b), None)
            if entry is None:
                continue
            vol_arr = np.asarray(entry.get(vol_key, []), dtype=float)
            cal_arr = np.asarray(entry.get(cal_key, []), dtype=float)
            if vol_arr.ndim == 2 and vol_arr.shape[0] > 0:
                vm = np.nanmean(vol_arr, axis=0)
                if np.any(np.isfinite(vm)):
                    vpk = float(np.nanmax(vm))
                    vol_peak_global = vpk if not np.isfinite(vol_peak_global) else max(vol_peak_global, vpk)
            if cal_arr.ndim == 2 and cal_arr.shape[0] > 0:
                cm = np.nanmean(cal_arr, axis=0)
                if np.any(np.isfinite(cm)):
                    cpk = float(np.nanmax(cm))
                    cal_peak_global = cpk if not np.isfinite(cal_peak_global) else max(cal_peak_global, cpk)
    if (not np.isfinite(vol_peak_global)) or vol_peak_global <= 0:
        vol_peak_global = 1.0
    if (not np.isfinite(cal_peak_global)) or cal_peak_global <= 0:
        cal_peak_global = 1.0
    vol_ymax = 1.2 * float(vol_peak_global)
    cal_ymax = 1.2 * float(cal_peak_global)
    vol_ymin = -0.1
    cal_ymin = -0.1

    legend_added = set()
    for r, et in enumerate(types, start=1):
        for cidx, b in enumerate(bins, start=1):
            key = (et, b)
            entry = groups.get(key, None)
            if entry is None or len(entry.get(cal_key, [])) == 0:
                fig.add_vline(x=0.0, line_dash="dash", line_color="black", opacity=0.5, row=r, col=cidx)
                continue
            vol_arr = np.asarray(entry.get(vol_key, []), dtype=float)
            cal_arr = np.asarray(entry.get(cal_key, []), dtype=float)
            cal_t_native_list = entry.get("cal_t_ms", [])
            cal_native_list = entry.get(cal_native_key, [])
            n_evt = int(cal_arr.shape[0]) if cal_arr.ndim == 2 else 0
            n_show = min(n_evt, 120)
            if n_show > 0:
                pick = np.linspace(0, n_evt - 1, n_show).astype(int)
                for ii in pick:
                    x_light = t_ms
                    y_light = cal_arr[ii]
                    if ii < len(cal_t_native_list) and ii < len(cal_native_list):
                        x_native = np.asarray(cal_t_native_list[ii], dtype=float).ravel()
                        y_native = np.asarray(cal_native_list[ii], dtype=float).ravel()
                        if x_native.size == y_native.size and x_native.size > 0:
                            x_light = x_native
                            y_light = y_native
                    fig.add_trace(
                        go.Scatter(
                            x=x_light,
                            y=y_light,
                            mode="lines",
                            line=dict(color="#c9a0ff", width=1),
                            opacity=0.18,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=r,
                        col=cidx,
                        secondary_y=False,
                    )
            fig.add_vline(x=0.0, line_dash="dash", line_color="black", opacity=0.5, row=r, col=cidx)
            if vol_arr.ndim == 2 and vol_arr.shape[0] > 0:
                vol_mean = np.nanmean(vol_arr, axis=0)
                vol_label = "voltage simple spikes" if str(et).lower() == "simple" else "voltage complex spikes"
                vol_mean_color = "black" if str(et).lower() == "simple" else "red"
                fig.add_trace(
                    go.Scatter(
                        x=t_ms,
                        y=vol_mean,
                        mode="lines",
                        line=dict(color=vol_mean_color, width=2.0),
                        name=vol_label,
                        legendgroup=vol_label,
                        showlegend=(vol_label not in legend_added),
                        hovertemplate=f"{vol_label}<br>time=%{{x:.1f}} ms<br>value=%{{y:.3f}}<extra></extra>",
                    ),
                    row=r,
                    col=cidx,
                    secondary_y=True,
                )
                legend_added.add(vol_label)
            if cal_arr.ndim == 2 and cal_arr.shape[0] > 0:
                cal_mean = np.nanmean(cal_arr, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=t_ms,
                        y=cal_mean,
                        mode="lines",
                        line=dict(color=CALCIUM_TRACE_COLOR, width=2.5),
                        name="calcium",
                        legendgroup="calcium",
                        showlegend=("calcium" not in legend_added),
                        hovertemplate=f"calcium<br>time=%{{x:.1f}} ms<br>value=%{{y:.3f}}<extra></extra>",
                    ),
                    row=r,
                    col=cidx,
                    secondary_y=False,
                )
                legend_added.add("calcium")

    for r in range(1, n_rows + 1):
        for cidx in range(1, n_cols + 1):
            fig.update_xaxes(title_text=(_bold_axis_title("time (ms)") if r == n_rows else ""), row=r, col=cidx)
            fig.update_yaxes(
                title_text=(_bold_axis_title("calcium (global dF/F0)" if mode_norm == "dff" else "calcium (z)") if cidx == 1 else ""),
                range=[cal_ymin, cal_ymax],
                row=r,
                col=cidx,
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text=(_bold_axis_title("voltage (a.u.)" if mode_norm == "dff" else "voltage (z)") if cidx == n_cols else ""),
                range=[vol_ymin, vol_ymax],
                row=r,
                col=cidx,
                secondary_y=True,
            )

    fig.update_layout(
        template="simple_white",
        width=320 * n_cols,
        height=row_height * n_rows,
        title=("" if mode_norm == "z" else title),
    )
    fig.update_annotations(font=dict(size=subplot_title_size, family="Arial"))
    fig.update_yaxes(title_standoff=0)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    _apply_bold_axis_fonts(fig, tick_size=axis_tick_size, title_size=axis_title_size)
    fig.write_html(save_html)
    _safe_write_svg(fig, save_svg)


def _find_plus_plateau_pkls(cell_folder):
    # Priority per suffix:
    #   eventpost cleanup > highplateau cleanup > after_peak cleanup > plus_plateau
    patterns = [
        (0, re.compile(r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_eventpostpeak\.pkl$", re.IGNORECASE)),
        (0, re.compile(r"^event_spike_overlay__plus_plateau(?P<suf>[mr]\d+)?_rm_complex_eventpostpeak\.pkl$", re.IGNORECASE)),
        (1, re.compile(r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$", re.IGNORECASE)),
        (2, re.compile(r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_after_peak\.pkl$", re.IGNORECASE)),
        (3, re.compile(r"^spike_detection_refined_new_plus_plateau(?P<suf>[mr]\d+)?\.pkl$", re.IGNORECASE)),
    ]

    best_by_suffix = {}
    pkl_candidates = []
    pkl_candidates.extend(glob.glob(os.path.join(cell_folder, "spike_detection_refined_new*.pkl")))
    pkl_candidates.extend(glob.glob(os.path.join(cell_folder, "event_spike_overlay__plus_plateau*_rm_complex_eventpostpeak.pkl")))
    for p in pkl_candidates:
        name = os.path.basename(p)
        for priority, pat in patterns:
            m = pat.match(name)
            if not m:
                continue
            suffix = (m.group("suf") or "main").lower()
            old = best_by_suffix.get(suffix)
            if old is None or priority < old[0]:
                best_by_suffix[suffix] = (priority, p)
            break

    def _sort_key(item):
        suffix, (_priority, path) = item
        rank = {"main": -1, "m0": 0, "r0": 1, "m1": 2, "r1": 3}
        return (rank.get(suffix, 99), suffix, os.path.basename(path).lower())

    return [path for _suffix, (_priority, path) in sorted(best_by_suffix.items(), key=_sort_key)]


def _suffix_from_pkl(path):
    name = os.path.basename(path)
    patterns = [
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_eventpostpeak\.pkl$",
        r"^event_spike_overlay__plus_plateau(?P<suf>[mr]\d+)?_rm_complex_eventpostpeak\.pkl$",
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_highplateau\.pkl$",
        r"^spike_detection_refined_new(?P<suf>[mr]\d+)?_rm_complex_after_peak\.pkl$",
        r"^spike_detection_refined_new_plus_plateau(?P<suf>[mr]\d+)?\.pkl$",
    ]
    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            s = m.group("suf")
            return "main" if not s else s.lower()
    return "main"


def _is_motor_row(row):
    vals = [str(row.get("brainState", "")).lower(), str(row.get("motor", "")).lower()]
    return any("motor" in v for v in vals)


def _brain_state_output_folder(row):
    vals = [
        str(row.get("brainState", "")).strip().lower(),
        str(row.get("state", "")).strip().lower(),
        str(row.get("population", "")).strip().lower(),
        str(row.get("motor", "")).strip().lower(),
    ]
    if any("motor" in v for v in vals):
        return "motor"
    if any(("anst" in v) or ("anest" in v) for v in vals):
        return "anst"
    if any("awake" in v for v in vals):
        return "awake"
    return None


def _load_segment_fluor(cell_folder, suffix, trace_cal_pkl):
    cands = []
    m = re.fullmatch(r"([mr])(\d+)", str(suffix).lower())
    if m:
        grp, idx = m.group(1), m.group(2)
        if grp == "m":
            cands += [
                f"calTraceMot{idx}df.csv",
                f"calTraceMot{idx}df..csv",
                f"calTraceM{idx}df.csv",
                f"calTraceM{idx}df..csv",
                f"calTrace{suffix}df.csv",
                f"calTrace{suffix}df..csv",
                f"calTraceMot{idx}.csv",
                f"calTraceM{idx}.csv",
                f"calTraceResm{idx}.csv",
                f"calTrace{suffix}.csv",
            ]
        else:
            cands += [
                f"calTraceRest{idx}df.csv",
                f"calTraceRest{idx}df..csv",
                f"calTraceRes{idx}df.csv",
                f"calTraceRes{idx}df..csv",
                f"calTraceR{idx}df.csv",
                f"calTraceR{idx}df..csv",
                f"calTrace{suffix}df.csv",
                f"calTrace{suffix}df..csv",
                f"calTraceRest{idx}.csv",
                f"calTraceRes{idx}.csv",
                f"calTraceResr{idx}.csv",
                f"calTraceR{idx}.csv",
                f"calTrace{suffix}.csv",
            ]
    cands += ["calTraceNBdf.csv", "calTraceNBdf..csv", "calTraceNB.csv", "calTrace.csv"]
    for fn in cands:
        p = os.path.join(cell_folder, fn)
        if os.path.isfile(p):
            try:
                arr = _read_csv_1d(p)
                if arr.size == len(trace_cal_pkl):
                    return arr, fn
            except Exception:
                pass
    return np.asarray(trace_cal_pkl, dtype=float).ravel(), "trace_cal_from_pkl"


def _resize_1d_to_len(x, target_len):
    a = np.asarray(x, dtype=float).ravel()
    n = int(target_len)
    if n <= 0:
        return np.array([], dtype=float)
    if a.size == n:
        return a
    if a.size <= 1:
        val = float(a[0]) if a.size == 1 else 0.0
        return np.full(n, val, dtype=float)
    xi = np.linspace(0.0, 1.0, num=a.size)
    xo = np.linspace(0.0, 1.0, num=n)
    return np.interp(xo, xi, a).astype(float)


def _load_motor_nb_segments_by_changepoint(
    cell_folder,
    cal_sr,
    target_len_by_suffix,
    target_vol_len_by_suffix=None,
    vol_sr=VOL_SR,
):
    """
    Build motor segments from calTraceNB using changepoint.csv in this fixed order:
    1) m0, 2) r0, 3) m1, 4) r1
    """
    nb_path = os.path.join(cell_folder, "calTraceNB.csv")
    cp_path = os.path.join(cell_folder, "changepoint.csv")
    if not os.path.isfile(nb_path):
        return {}
    try:
        nb = _read_csv_1d(nb_path)
        cp_vals = []
        if os.path.isfile(cp_path):
            cp = pd.read_csv(cp_path)
            if "trace" in cp.columns:
                cp_vals = [int(v) for v in cp["trace"].dropna().astype(int).tolist()]
        if len(cp_vals) < 4 and isinstance(target_vol_len_by_suffix, dict):
            order = ["m0", "r0", "m1", "r1"]
            if all(k in target_vol_len_by_suffix for k in order):
                l0 = int(target_vol_len_by_suffix["m0"])
                l1 = int(target_vol_len_by_suffix["r0"])
                l2 = int(target_vol_len_by_suffix["m1"])
                l3 = int(target_vol_len_by_suffix["r1"])
                cp_vals = [l0 - 1, l0 + l1 - 1, l0 + l1 + l2 - 1, l0 + l1 + l2 + l3 - 1]
        if len(cp_vals) < 4:
            return {}
    except Exception:
        return {}

    # Use first 4 change points and required ordering.
    cp_vals = cp_vals[:4]
    order = ["m0", "r0", "m1", "r1"]
    vol_starts = [0, cp_vals[0] + 1, cp_vals[1] + 1, cp_vals[2] + 1]
    cal_starts_anchor = [int(round(float(vs) * float(cal_sr) / float(vol_sr))) for vs in vol_starts]

    out = {}
    prev_end = -1
    n_nb = int(nb.size)
    for i, suf in enumerate(order):
        if suf not in target_len_by_suffix:
            continue
        tgt = int(target_len_by_suffix[suf])
        if tgt <= 0:
            continue
        s0 = int(max(0, min(n_nb - 1, cal_starts_anchor[i])))
        s = int(max(s0, prev_end + 1))
        e = int(s + tgt - 1)
        if e >= n_nb:
            e = n_nb - 1
            s = int(max(0, e - tgt + 1))
        seg = nb[s : e + 1]
        if seg.size != tgt:
            seg = _resize_1d_to_len(seg, tgt)
        out[suf] = (np.asarray(seg, dtype=float).ravel(), f"calTraceNB.csv|changepoint.csv|{suf}")
        prev_end = int(min(n_nb - 1, s + tgt - 1))
    return out


def _load_motor_voltage_segments_by_changepoint(cell_folder, target_len_by_suffix):
    cands = ["volVolpyDF.csv", "VolTraceDFReal.csv", "volTraceDF.csv", "volTrace.csv", "Vm.csv"]
    full = None
    src_name = ""
    volpy_arr, volpy_src = _load_or_create_volpy_dff(cell_folder)
    if volpy_arr is not None and volpy_arr.size > 0:
        full = np.asarray(volpy_arr, dtype=float).ravel()
        src_name = volpy_src
    for fn in cands:
        if full is not None:
            break
        p = os.path.join(cell_folder, fn)
        if not os.path.isfile(p):
            continue
        try:
            arr = _read_csv_1d(p)
            if arr.size > 0:
                full = np.asarray(arr, dtype=float).ravel()
                src_name = fn
                break
        except Exception:
            pass
    if full is None or full.size == 0:
        return {}

    full_mask = _read_bool_mask_csv(os.path.join(cell_folder, "volMask.csv"), full.size)
    full_masked = _apply_mask_to_trace(full, full_mask) if full_mask is not None else full.copy()
    cp_path = os.path.join(cell_folder, "changepoint.csv")
    cp_vals = []
    try:
        if os.path.isfile(cp_path):
            cp = pd.read_csv(cp_path)
            if "trace" in cp.columns:
                cp_vals = [int(v) for v in cp["trace"].dropna().astype(int).tolist()]
    except Exception:
        cp_vals = []

    order = ["m0", "r0", "m1", "r1"]
    if len(cp_vals) < 4 and isinstance(target_len_by_suffix, dict):
        if all(k in target_len_by_suffix for k in order):
            l0 = int(target_len_by_suffix["m0"])
            l1 = int(target_len_by_suffix["r0"])
            l2 = int(target_len_by_suffix["m1"])
            l3 = int(target_len_by_suffix["r1"])
            cp_vals = [l0 - 1, l0 + l1 - 1, l0 + l1 + l2 - 1, l0 + l1 + l2 + l3 - 1]
    if len(cp_vals) < 4:
        return {}

    cp_vals = cp_vals[:4]
    starts = [0, cp_vals[0] + 1, cp_vals[1] + 1, cp_vals[2] + 1]
    out = {}
    n_full = int(full_masked.size)
    for i, suf in enumerate(order):
        if suf not in target_len_by_suffix:
            continue
        tgt = int(target_len_by_suffix[suf])
        if tgt <= 0:
            continue
        s = int(max(0, min(n_full - 1, starts[i])))
        e = int(min(n_full - 1, s + tgt - 1))
        seg = np.asarray(full_masked[s:e + 1], dtype=float).ravel()
        if seg.size != tgt:
            seg = _resize_1d_to_len(seg, tgt)
        seg_mask = None
        if full_mask is not None:
            seg_mask = np.asarray(full_mask[s:e + 1], dtype=bool).ravel()
            if seg_mask.size != tgt:
                if seg_mask.size > tgt:
                    seg_mask = seg_mask[:tgt]
                else:
                    seg_mask = np.pad(seg_mask, (0, tgt - seg_mask.size), constant_values=False)
        out[suf] = {
            "trace": seg,
            "mask": seg_mask,
            "src": f"{src_name}|volMask.csv|changepoint.csv|{suf}",
            "already_masked": True,
        }
    return out


def _load_segment_voltage(cell_folder, suffix, trace_vol_pkl, motor_segments=None):
    trace_vol_pkl = np.asarray(trace_vol_pkl, dtype=float).ravel()
    if trace_vol_pkl.size > 0:
        return trace_vol_pkl, "trace_vol_from_pkl", None, False
    if isinstance(motor_segments, dict) and suffix in motor_segments:
        rec = motor_segments[suffix]
        return rec["trace"], rec["src"], rec.get("mask"), bool(rec.get("already_masked", False))
    cands = []
    m = re.fullmatch(r"([mr])(\d+)", str(suffix).lower())
    if m:
        grp, idx = m.group(1), m.group(2)
        if grp == "m":
            cands += [f"volTraceMot{idx}.csv", f"volTraceM{idx}.csv", f"volTrace{suffix}.csv"]
        else:
            cands += [f"volTraceRest{idx}.csv", f"volTraceRes{idx}.csv", f"volTraceR{idx}.csv", f"volTrace{suffix}.csv"]
    cands += ["volVolpyDF.csv", "VolTraceDFReal.csv", "volTraceDF.csv", "volTrace.csv", "Vm.csv"]
    volpy_arr, volpy_src = _load_or_create_volpy_dff(cell_folder)
    if volpy_arr is not None and volpy_arr.size == len(trace_vol_pkl):
        return volpy_arr, volpy_src, None, False
    for fn in cands:
        p = os.path.join(cell_folder, fn)
        if os.path.isfile(p):
            try:
                arr = _read_csv_1d(p)
                if arr.size == len(trace_vol_pkl):
                    return arr, fn, None, False
            except Exception:
                pass
    return trace_vol_pkl, "trace_vol_from_pkl", None, False



def _load_full_voltage_for_merged_motor(cell_folder):
    volpy_arr, volpy_src = _load_or_create_volpy_dff(cell_folder)
    if volpy_arr is not None and volpy_arr.size > 0:
        return np.asarray(volpy_arr, dtype=float).ravel(), volpy_src
    for fn in ("volVolpyDF.csv", "VolTraceDFReal.csv", "volTraceDF.csv", "volTrace.csv", "Vm.csv"):
        p = os.path.join(cell_folder, fn)
        if os.path.isfile(p):
            arr = _read_csv_1d(p)
            if arr.size:
                return np.asarray(arr, dtype=float).ravel(), fn
    return np.asarray([], dtype=float), "missing_full_voltage"


def _load_full_calcium_for_merged_motor(cell_folder):
    for fn in ("calTraceNBdf.csv", "calTraceNBdf..csv", "calTraceNB.csv", "calTrace.csv"):
        p = os.path.join(cell_folder, fn)
        if os.path.isfile(p):
            arr = _read_csv_1d(p)
            if arr.size:
                return np.asarray(arr, dtype=float).ravel(), fn
    return np.asarray([], dtype=float), "missing_full_calcium"


def _offset_int_array(values, offset):
    arr = np.asarray(values if values is not None else [], dtype=float).ravel()
    arr = arr[np.isfinite(arr)].astype(np.int64)
    return (arr + int(offset)).astype(np.int64) if arr.size else np.asarray([], dtype=np.int64)


def _offset_event_dict(ed, offset):
    if not isinstance(ed, dict):
        return {}
    out = copy.deepcopy(ed)
    for key in ("starts", "ends", "locs"):
        if key in out:
            out[key] = _offset_int_array(out.get(key, []), offset)
    if isinstance(out.get("spike_indices", None), list):
        out["spike_indices"] = [_offset_int_array(x, offset) for x in out["spike_indices"]]
    return out


def _concat_event_dicts(dicts):
    out = {}
    for key in ("starts", "ends", "locs"):
        parts = [np.asarray(d.get(key, []), dtype=np.int64).ravel() for d in dicts if isinstance(d, dict) and len(d.get(key, []))]
        out[key] = np.concatenate(parts).astype(np.int64) if parts else np.asarray([], dtype=np.int64)
    spike_lists = []
    for d in dicts:
        if isinstance(d, dict) and isinstance(d.get("spike_indices", None), list):
            spike_lists.extend(d.get("spike_indices", []))
    out["spike_indices"] = spike_lists
    return out


def _offset_metric_list(metrics, offset):
    out = []
    if not isinstance(metrics, list):
        return out
    for m in metrics:
        if not isinstance(m, dict):
            out.append(copy.deepcopy(m))
            continue
        mm = copy.deepcopy(m)
        for key in ("start", "end", "loc", "peak_idx"):
            if key in mm and mm[key] is not None:
                try:
                    mm[key] = int(mm[key]) + int(offset)
                except Exception:
                    pass
        for key in ("spikes", "spike_indices", "indices"):
            if key in mm:
                mm[key] = _offset_int_array(mm[key], offset).tolist()
        out.append(mm)
    return out


def _motor_suffix_sort_key(suffix):
    m = re.fullmatch(r"([mr])(\d+)", str(suffix).strip().lower())
    if not m:
        return (10**9, str(suffix))
    idx = int(m.group(2))
    state_rank = 0 if m.group(1) == "m" else 1
    return (idx * 2 + state_rank, str(suffix).lower())


def _merged_motor_offsets_by_changepoint(cell_folder, suffixes):
    cp_path = os.path.join(cell_folder, "changepoint.csv")
    if not os.path.isfile(cp_path):
        return {}
    try:
        cp = pd.read_csv(cp_path)
        if "trace" not in cp.columns:
            return {}
        cp_vals = [int(v) for v in cp["trace"].dropna().astype(int).tolist()]
    except Exception:
        return {}
    starts = [0] + [int(v) for v in cp_vals]
    out = {}
    for suf in suffixes:
        order_idx = int(_motor_suffix_sort_key(suf)[0])
        if 0 <= order_idx < len(starts):
            out[str(suf).lower()] = int(starts[order_idx])
    return out


def _merge_motor_chunk_payloads(cell_folder, pkl_paths):
    rows = []
    for pkl_path in sorted(pkl_paths, key=lambda p: _motor_suffix_sort_key(_suffix_from_pkl(p))):
        suf = _suffix_from_pkl(pkl_path)
        if not re.fullmatch(r"[mr]\d+", str(suf).lower()):
            continue
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        trace_vol = np.asarray(d.get("trace_vol", []), dtype=float).ravel()
        rows.append((suf, pkl_path, d, int(trace_vol.size)))
    if not rows:
        return None, None, None, None

    full_vol = np.asarray([], dtype=float)
    vol_src = "missing_full_voltage"
    for pkl_path in pkl_paths:
        if _suffix_from_pkl(pkl_path) != "main":
            continue
        try:
            with open(pkl_path, "rb") as f:
                main_d = pickle.load(f)
            main_vol = np.asarray(main_d.get("trace_vol", []), dtype=float).ravel()
            if main_vol.size > 0:
                full_vol = main_vol
                vol_src = "trace_vol_from_main_pkl"
                break
        except Exception:
            pass
    if full_vol.size == 0:
        full_vol, vol_src = _load_full_voltage_for_merged_motor(cell_folder)
    full_cal, cal_src = _load_full_calcium_for_merged_motor(cell_folder)
    merged = copy.deepcopy(rows[0][2])
    if full_vol.size == 0:
        full_vol = np.concatenate([np.asarray(r[2].get("trace_vol", []), dtype=float).ravel() for r in rows])
        vol_src = "concat_trace_vol_from_chunk_pkls"
    if full_cal.size == 0:
        full_cal = np.concatenate([np.asarray(r[2].get("trace_cal", []), dtype=float).ravel() for r in rows])
        cal_src = "concat_trace_cal_from_chunk_pkls"
    merged["trace_vol"] = full_vol
    merged["trace_cal"] = full_cal

    offsets = _merged_motor_offsets_by_changepoint(cell_folder, [suf for suf, _p, _d, _n in rows])
    off = 0
    for suf, _p, _d, n_vol in rows:
        offsets.setdefault(suf, int(off))
        off += int(n_vol)

    for key in ("vm_all_spikes", "vm_simple_spikes", "vm_complex_spikes", "input_spikes"):
        parts = [_offset_int_array(d.get(key, []), offsets[suf]) for suf, _p, d, _n in rows]
        merged[key] = np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.asarray([], dtype=np.int64)
    merged["vm_burst_dict"] = _concat_event_dicts([_offset_event_dict(d.get("vm_burst_dict", {}), offsets[suf]) for suf, _p, d, _n in rows])
    merged["vm_plateaus_dict"] = _concat_event_dicts([_offset_event_dict(d.get("vm_plateaus_dict", {}), offsets[suf]) for suf, _p, d, _n in rows])
    metrics = []
    for suf, _p, d, _n in rows:
        metrics.extend(_offset_metric_list(d.get("vm_burst_metrics", d.get("burst_metrics", [])), offsets[suf]))
    merged["vm_burst_metrics"] = metrics
    merged["burst_metrics"] = metrics
    merged["merged_motor_chunks"] = {
        "enabled": True,
        "order": [suf for suf, _p, _d, _n in rows],
        "offsets": offsets,
        "source_pkls": [os.path.basename(p) for _suf, p, _d, _n in rows],
        "vol_source": vol_src,
        "cal_source": cal_src,
    }
    if "detection_params" not in merged or not isinstance(merged.get("detection_params"), dict):
        merged["detection_params"] = {}
    merged["detection_params"]["motor_mode"] = "merged"
    merged_label = "+".join([suf for suf, _p, _d, _n in rows])
    return os.path.join(cell_folder, f"<merged:{merged_label}>"), merged, vol_src, cal_src

def _write_pyr_event_cal3_population_summary_bundle(
    p_df,
    p_df_auc,
    pipeline_name,
    out_dir,
    filename_tag,
    title_prefix,
    shape_groups=None,
    event_selection_rows=None,
):
    p_chosen = p_df[p_df["include"].astype(bool)].copy()
    p_chosen_auc = p_df_auc[p_df_auc["include"].astype(bool)].copy()
    trace_meta = (
        p_df[["cell_trace_id", "cell_trace_label"]]
        .drop_duplicates()
        .sort_values(["cell_trace_label", "cell_trace_id"])
        .to_dict("records")
    )
    if len(p_chosen) == 0 and len(trace_meta) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)

    _plot_all_cells_grid(
        p_chosen,
        mode="spike",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
    )
    _plot_all_cells_grid(
        p_chosen_auc,
        mode="auc",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
    )
    _plot_all_cells_grid(
        p_chosen_auc,
        mode="length",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
    )
    _plot_all_cells_grid(
        p_chosen,
        mode="spike",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike_zscore.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
    )
    _plot_all_cells_grid(
        p_chosen_auc,
        mode="auc",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc_zscore.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
    )
    _plot_all_cells_grid(
        p_chosen_auc,
        mode="length",
        save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length_zscore.svg", filename_tag)),
        all_trace_meta=trace_meta,
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
    )
    if len(p_chosen) == 0:
        return

    _plot_population_two_panel(
        p_chosen,
        title=f"{title_prefix} | {pipeline_name}",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel.svg", filename_tag)),
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
        auc_df=p_chosen_auc,
    )
    _plot_population_two_panel(
        p_chosen,
        title=f"{title_prefix} | {pipeline_name}",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_regression.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_regression.svg", filename_tag)),
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
        auc_df=p_chosen_auc,
        add_regression=True,
    )
    _plot_pooled_fit_panels(
        p_chosen,
        title=f"{title_prefix} fits | {pipeline_name}",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels.svg", filename_tag)),
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_Y_LABEL,
        auc_df=p_chosen_auc,
    )
    _plot_population_two_panel(
        p_chosen,
        title=f"{title_prefix} | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
        auc_df=p_chosen_auc,
    )
    _plot_population_two_panel(
        p_chosen,
        title=f"{title_prefix} | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
        auc_df=p_chosen_auc,
        add_regression=True,
    )
    _plot_population_two_panel(
        p_chosen,
        title=f"{title_prefix} | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression_stats.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression_stats.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
        auc_df=p_chosen_auc,
        add_regression=True,
        show_corr_stats=True,
    )
    _plot_pooled_fit_panels(
        p_chosen,
        title=f"{title_prefix} fits | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels_zscore.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="calcium response (z-score)",
        auc_df=p_chosen_auc,
    )
    _plot_cellavg_amp_vs_spikes(
        p_chosen,
        title=f"Cell-average calcium amplitude vs spike count | {title_prefix} | {pipeline_name}",
        save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount.svg", filename_tag)),
        y_col=GLOBAL_DFF_Y_COL,
        y_label=GLOBAL_DFF_CELLAVG_Y_LABEL,
    )
    _plot_cellavg_amp_vs_spikes(
        p_chosen,
        title=f"Cell-average calcium amplitude vs spike count | {title_prefix} | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="cell-mean calcium amplitude (z-score)",
        show_fit_annotation=False,
        show_legend=False,
    )
    _plot_cellavg_amp_vs_spikes(
        p_chosen,
        title=f"Cell-average calcium amplitude vs spike count | {title_prefix} | {pipeline_name} | z-score",
        save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore_stats.html", filename_tag)),
        save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore_stats.svg", filename_tag)),
        y_col="amp_local_z",
        y_label="cell-mean calcium amplitude (z-score)",
        show_fit_annotation=True,
        show_legend=False,
        show_corr_stats=True,
    )
    _save_cellavg_amp_vs_spikecount_lmm_tukey(
        p_chosen,
        out_dir=out_dir,
        filename_prefix=_with_filename_tag("cellavg_amp_vs_spikecount_zscore", filename_tag),
        y_col="amp_local_z",
    )

    if shape_groups and len(shape_groups) > 0:
        _shape_plot_groups(
            shape_groups,
            title=f"Event-shape summary | {title_prefix} | {pipeline_name} | dF/F",
            save_html=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all.svg", filename_tag)),
            mode="dff",
        )
        _shape_plot_groups(
            shape_groups,
            title=f"Event-shape summary | {title_prefix} | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all_zscore.svg", filename_tag)),
            mode="z",
        )

    if pe2 is not None and event_selection_rows:
        try:
            evdf = pd.concat(event_selection_rows, ignore_index=True).copy()
            evdf["cell_folder"] = evdf["cell_folder"].astype(str) + "::" + evdf["suffix"].astype(str)
            evt_dir = os.path.join(out_dir, "eventChoss")
            os.makedirs(evt_dir, exist_ok=True)
            pe2._plot_event_selection_two_panel(
                evdf,
                title_prefix=f"Pyr event choosing analysis | {title_prefix} | {pipeline_name}",
                save_html=os.path.join(evt_dir, _with_filename_tag("event_choosing_analysis.html", filename_tag)),
                save_svg=os.path.join(evt_dir, _with_filename_tag("event_choosing_analysis.svg", filename_tag)),
                save_pdf=None,
            )
            pe2.save_event_selection_cell_by_cell_figures(evdf, out_dir=evt_dir, n_cols=6)
            pe2.save_event_selection_population_figures(evdf, out_dir=evt_dir)
        except Exception as e:
            print(f"[WARN] event choosing summary failed ({title_prefix}, {pipeline_name}): {e}")


def run_pyr_event_cal3(db_path=DB_PATH, max_cells=None, include_plateau=True, filename_tag="", motor_mode="chunked"):
    motor_mode = str(motor_mode).strip().lower()
    if motor_mode not in {"chunked", "full", "merged"}:
        raise ValueError("motor_mode must be 'chunked', 'full', or 'merged'")
    db = pd.read_csv(db_path)
    if max_cells is not None:
        db = db.iloc[: int(max_cells)].copy()
    all_rows = []
    all_rows_by_pipeline = {"event_distance": [], "calcium_threshold": []}
    analysis_rows_by_pipeline = {"event_distance": [], "calcium_threshold": []}
    analysis_rows_auc_by_pipeline = {"event_distance": [], "calcium_threshold": []}
    shape_groups_by_pipeline = {"event_distance": _shape_init_groups(), "calcium_threshold": _shape_init_groups()}
    event_selection_rows_by_pipeline = {"event_distance": [], "calcium_threshold": []}
    state_analysis_rows_by_pipeline = {"event_distance": {}, "calcium_threshold": {}}
    state_analysis_rows_auc_by_pipeline = {"event_distance": {}, "calcium_threshold": {}}
    state_shape_groups_by_pipeline = {"event_distance": {}, "calcium_threshold": {}}
    state_event_selection_rows_by_pipeline = {"event_distance": {}, "calcium_threshold": {}}
    for i, row in db.iterrows():
        cell_folder = str(row["Link"])
        cal_sr = _safe_cal_sr(row.get("CALsr", 30.0), default=30.0)
        if not os.path.isdir(cell_folder):
            print(f"[SKIP] missing folder: {cell_folder}")
            continue
        pkl_list = _find_plus_plateau_pkls(cell_folder)
        if len(pkl_list) == 0:
            print(f"[SKIP] no corrected/plus-plateau spike pkl: {cell_folder}")
            continue
        is_motor = _is_motor_row(row)
        brain_state_folder = _brain_state_output_folder(row)
        if is_motor:
            if motor_mode in {"chunked", "merged"}:
                keep = []
                for p in pkl_list:
                    s = _suffix_from_pkl(p)
                    if s in ("m0", "r0", "m1", "r1"):
                        keep.append(p)
                rank = {"m0": 0, "r0": 1, "m1": 2, "r1": 3}
                pkl_list = sorted(keep, key=lambda p: rank.get(_suffix_from_pkl(p), 99))
            else:
                pkl_list = [p for p in pkl_list if _suffix_from_pkl(p) == "main"]
                if len(pkl_list) == 0:
                    print(f"[SKIP] motor_mode='full' but no main/full corrected pkl: {cell_folder}")
                    continue
        if is_motor and motor_mode == "merged":
            merged_path, merged_d, merged_vol_src, merged_cal_src = _merge_motor_chunk_payloads(cell_folder, pkl_list)
            if merged_d is None:
                print(f"[SKIP] motor_mode='merged' but no m0/r0/m1/r1 corrected pkls: {cell_folder}")
                continue
            pkl_records = [(merged_path, merged_d, "merged", merged_vol_src, merged_cal_src)]
        else:
            pkl_records = [(p, None, _suffix_from_pkl(p), None, None) for p in pkl_list]
        print(f"\nCell {i+1}/{len(db)} | {cell_folder} | CALsr={cal_sr} | traces={len(pkl_records)}")

        motor_vol_segments = None
        if is_motor and motor_mode == "chunked":
            target_vol_len_by_suffix = {}
            for _pkl_path in pkl_list:
                _suffix = _suffix_from_pkl(_pkl_path)
                try:
                    with open(_pkl_path, "rb") as _f:
                        _d = pickle.load(_f)
                    _trace_vol = np.asarray(_d.get("trace_vol", []), dtype=float).ravel()
                    if _trace_vol.size > 0:
                        target_vol_len_by_suffix[_suffix] = int(_trace_vol.size)
                except Exception:
                    pass
            if len(target_vol_len_by_suffix) > 0:
                motor_vol_segments = _load_motor_voltage_segments_by_changepoint(cell_folder, target_vol_len_by_suffix)

        for pkl_path, d_preloaded, suffix, forced_vol_src, forced_cal_src in pkl_records:
            if d_preloaded is None:
                with open(pkl_path, "rb") as f:
                    d = pickle.load(f)
            else:
                d = d_preloaded
            trace_vol_pkl = np.asarray(d.get("trace_vol", []), dtype=float).ravel()
            trace_cal_pkl = np.asarray(d.get("trace_cal", []), dtype=float).ravel()
            if trace_vol_pkl.size == 0 or trace_cal_pkl.size == 0:
                print(f"  [SKIP] empty trace in {os.path.basename(pkl_path)}")
                continue
            if forced_vol_src is not None:
                trace_vol = trace_vol_pkl
                vol_src = forced_vol_src
                vol_mask_override = None
                vol_already_masked = False
            else:
                trace_vol, vol_src, vol_mask_override, vol_already_masked = _load_segment_voltage(
                    cell_folder,
                    suffix,
                    trace_vol_pkl,
                    motor_segments=motor_vol_segments,
                )
            if forced_cal_src is not None:
                trace_fluor = trace_cal_pkl
                cal_src = forced_cal_src
            else:
                trace_fluor, cal_src = _load_segment_fluor(cell_folder, suffix, trace_cal_pkl)
            vol_mask = vol_mask_override if vol_mask_override is not None else _read_bool_mask_csv(os.path.join(cell_folder, "volMask.csv"), len(trace_vol))
            cal_mask = _read_bool_mask_csv(os.path.join(cell_folder, "calMask.csv"), len(trace_fluor))
            if vol_mask is not None:
                if not vol_already_masked:
                    trace_vol = _apply_mask_to_trace(trace_vol, vol_mask)
                d = _filter_payload_spikes_by_mask(d, vol_mask)
            if cal_mask is not None:
                trace_fluor = _apply_mask_to_trace(trace_fluor, cal_mask)
            mean_fr_hz_for_trace = _mean_fr_hz_from_payload(d, trace_vol, VOL_SR)
            gdf, gf0 = _rolling_percentile_f0(trace_fluor, cal_sr, win_s=GLOBAL_F0_WIN_S, p=GLOBAL_F0_PCTL)
            # Plateau classification is used only for dedicated AUC figures.
            # Spike-count-to-calcium-response, event-shape, event tables, and overlays
            # use the simple/complex event definition without plateau relabeling.
            events_base = build_events_no_plateau(d)
            events_base_overlay = build_events(d) if bool(include_plateau) else build_events_no_plateau(d)
            events_base_auc = build_events(d) if bool(include_plateau) else build_events_no_plateau(d)
            if bool(include_plateau):
                events_base_auc = _apply_plateau_min_auc_duration_rule(
                    events_base_auc,
                    trace_vol,
                    VOL_SR,
                    min_dur_ms=PLATEAU_MIN_AUC_DUR_MS,
                )

            out_base = os.path.join(cell_folder, "event_cal3")
            out_dirs = {
                "event_distance": os.path.join(out_base, "event_distance"),
                "calcium_threshold": os.path.join(out_base, "calcium_threshold"),
            }
            os.makedirs(out_base, exist_ok=True)
            for od in out_dirs.values():
                os.makedirs(od, exist_ok=True)

            pipeline_defs = [
                ("event_distance", "event_distance"),
                ("calcium_threshold", "calcium_threshold"),
            ]
            for pipeline_name, selection_mode in pipeline_defs:
                thr_val = None
                gdf_thr_for_pipeline = None
                if str(selection_mode).strip().lower() == "calcium_threshold":
                    gdf_thr_for_pipeline = np.asarray(gdf, dtype=float).ravel()
                    thr_val = _calcium_threshold_value(gdf_thr_for_pipeline)
                events = choose_events(
                    copy.deepcopy(events_base),
                    gdf,
                    trace_fluor,
                    cal_sr,
                    VOL_SR,
                    selection_mode=selection_mode,
                    calcium_threshold_value=thr_val,
                )
                events_auc = choose_events(
                    copy.deepcopy(events_base_auc),
                    gdf,
                    trace_fluor,
                    cal_sr,
                    VOL_SR,
                    selection_mode=selection_mode,
                    calcium_threshold_value=thr_val,
                )
                events_overlay = choose_events(
                    copy.deepcopy(events_base_overlay),
                    gdf,
                    trace_fluor,
                    cal_sr,
                    VOL_SR,
                    selection_mode=selection_mode,
                    calcium_threshold_value=thr_val,
                )
                # Safety guard: for calcium-threshold pipeline, enforce start global dF/F < threshold at AP frame.
                if str(selection_mode).strip().lower() == "calcium_threshold":
                    gdf_thr = gdf_thr_for_pipeline
                    if gdf_thr is None:
                        gdf_thr = np.asarray(gdf, dtype=float).ravel()
                    if thr_val is None:
                        thr_val = _calcium_threshold_value(gdf_thr)
                    for ev in events:
                        _, gstart = _event_start_gdf_at_ap(ev, gdf_thr, cal_sr, VOL_SR)
                        if (not np.isfinite(gstart)) or (gstart >= thr_val):
                            ev.include = False
                            ev.include_reason = f"global_dff_before_ap_ge_{thr_val:.3f}"
                    for ev in events_auc:
                        _, gstart = _event_start_gdf_at_ap(ev, gdf_thr, cal_sr, VOL_SR)
                        if (not np.isfinite(gstart)) or (gstart >= thr_val):
                            ev.include = False
                            ev.include_reason = f"global_dff_before_ap_ge_{thr_val:.3f}"
                    for ev in events_overlay:
                        _, gstart = _event_start_gdf_at_ap(ev, gdf_thr, cal_sr, VOL_SR)
                        if (not np.isfinite(gstart)) or (gstart >= thr_val):
                            ev.include = False
                            ev.include_reason = f"global_dff_before_ap_ge_{thr_val:.3f}"
                out_dir = out_dirs[pipeline_name]
                html_name = _with_filename_tag(f"event_overlay_cal3_{suffix}.html", filename_tag)
                html_path = os.path.join(out_dir, html_name)
                ttl = (
                    f"{os.path.basename(cell_folder)} | {suffix} | {pipeline_name} | "
                    "chosen events + global dF/F0"
                )
                thr = float(thr_val) if (pipeline_name == "calcium_threshold" and thr_val is not None) else None
                save_summary_plot_html(
                    trace_vol,
                    gdf,
                    events_overlay,
                    html_path,
                    VOL_SR,
                    cal_sr,
                    ttl,
                    calcium_threshold=thr,
                    connect_masked_gaps=(str(suffix).lower() == "merged"),
                )
                gz, zstats = _global_ztrace_p8(gdf, cal_sr, return_stats=True)
                z_html_name = _with_filename_tag(f"event_overlay_cal3_zscore_{suffix}.html", filename_tag)
                z_html_path = os.path.join(out_dir, z_html_name)
                z_ttl = (
                    f"{os.path.basename(cell_folder)} | {suffix} | {pipeline_name} | "
                    "chosen events + global calcium z-score"
                )
                z_note = (
                    f"center={zstats.get('center', np.nan):.4g}<br>"
                    f"sigma={zstats.get('sigma', np.nan):.4g}<br>"
                    f"baseline_p8={zstats.get('baseline_p8', np.nan):.4g}"
                )
                z_single_labels = []
                for ev in events_overlay:
                    if (not bool(ev.include)) or (str(ev.class_name) != "single"):
                        continue
                    pz = _candidate_peak_idx(ev, gz, cal_sr, VOL_SR)
                    if pz < 0 or pz >= len(gz):
                        continue
                    zpk = float(gz[pz])
                    zb = _local_baseline_from_trace(ev, gz, cal_sr, VOL_SR)
                    if (not np.isfinite(zpk)) or (not np.isfinite(zb)):
                        continue
                    z_single_labels.append(
                        {
                            "x": float(pz) / float(cal_sr),
                            "y": zpk,
                            "text": f"p={zpk:.2f}<br>b={zb:.2f}",
                        }
                    )
                save_summary_plot_html(
                    trace_vol,
                    gz,
                    events_overlay,
                    z_html_path,
                    VOL_SR,
                    cal_sr,
                    z_ttl,
                    calcium_threshold=None,
                    calcium_trace_label="global z-score",
                    calcium_axis_title="Global calcium z-score",
                    calcium_value_label="z-score",
                    extra_note=z_note,
                    single_peak_labels=z_single_labels,
                    connect_masked_gaps=(str(suffix).lower() == "merged"),
                )

                rows = []
                gdf_thr_for_csv = gdf_thr_for_pipeline
                if gdf_thr_for_csv is None:
                    gdf_thr_for_csv = _gaussian_smooth_1d(gdf, float(CALCIUM_THRESHOLD_SMOOTH_SIGMA_S) * float(cal_sr))
                for k, ev in enumerate(events):
                    start_cal_idx, start_gdf_raw = _event_start_gdf_at_ap(ev, gdf, cal_sr, VOL_SR)
                    start_cal_idx_s, start_gdf_smooth = _event_start_gdf_at_ap(
                        ev,
                        gdf_thr_for_csv,
                        cal_sr,
                        VOL_SR,
                    )
                    rows.append(
                        {
                            "selection_pipeline": pipeline_name,
                            "cell_folder": cell_folder,
                            "suffix": suffix,
                            "brainState": row.get("brainState", ""),
                            "mean_fr_hz": float(mean_fr_hz_for_trace) if np.isfinite(mean_fr_hz_for_trace) else np.nan,
                            "event_idx": k,
                            "event_type": ev.event_type,
                            "event_class": ev.class_name,
                            "n_spikes": ev.n_spikes,
                            "first_spike": ev.first_spike,
                            "last_spike": ev.last_spike,
                            "include": bool(ev.include),
                            "include_reason": ev.include_reason,
                            "calcium_threshold_value": float(thr_val) if thr_val is not None else np.nan,
                            "start_cal_idx": int(start_cal_idx),
                            "start_global_dff_raw": float(start_gdf_raw) if np.isfinite(start_gdf_raw) else np.nan,
                            "start_cal_idx_smooth": int(start_cal_idx_s),
                            "start_global_dff_smooth": float(start_gdf_smooth) if np.isfinite(start_gdf_smooth) else np.nan,
                            "peak_cal_idx": int(ev.peak_idx),
                            "peak_global_dff": float(ev.peak_global_dff),
                            "amp_local_dff": float(ev.amp_local_dff),
                            "amp_local_z": float(getattr(ev, "amp_local_z", np.nan)),
                            "vol_source": vol_src,
                            "cal_source": cal_src,
                            "pkl_name": os.path.basename(pkl_path),
                        }
                    )
                seg_df = pd.DataFrame(rows)
                csv_name = _with_filename_tag(f"event_table_cal3_{suffix}.csv", filename_tag)
                seg_df.to_csv(os.path.join(out_dir, csv_name), index=False)
                pd.DataFrame({"global_f0": gf0, "global_dff": gdf}).to_csv(
                    os.path.join(out_dir, _with_filename_tag(f"global_dff_cal3_{suffix}.csv", filename_tag)), index=False
                )
                all_rows.append(seg_df)
                all_rows_by_pipeline[pipeline_name].append(seg_df)
                event_selection_rows_by_pipeline[pipeline_name].append(seg_df.copy())
                if brain_state_folder is not None:
                    state_event_selection_rows_by_pipeline[pipeline_name].setdefault(brain_state_folder, []).append(seg_df.copy())
                an_df = _events_to_analysis_df(
                    events=events,
                    trace_vol=trace_vol,
                    cell_folder=cell_folder,
                    suffix=suffix,
                    pipeline_name=pipeline_name,
                    brain_state=row.get("brainState", ""),
                )
                analysis_rows_by_pipeline[pipeline_name].append(an_df)
                if brain_state_folder is not None:
                    state_analysis_rows_by_pipeline[pipeline_name].setdefault(brain_state_folder, []).append(an_df)
                an_df_auc = _events_to_analysis_df(
                    events=events_auc,
                    trace_vol=trace_vol,
                    cell_folder=cell_folder,
                    suffix=suffix,
                    pipeline_name=pipeline_name,
                    brain_state=row.get("brainState", ""),
                )
                analysis_rows_auc_by_pipeline[pipeline_name].append(an_df_auc)
                if brain_state_folder is not None:
                    state_analysis_rows_auc_by_pipeline[pipeline_name].setdefault(brain_state_folder, []).append(an_df_auc)

                chosen_df = an_df[an_df["include"].astype(bool)].copy()
                chosen_auc_df = an_df_auc[an_df_auc["include"].astype(bool)].copy()
                if len(chosen_df) > 0:
                    two_html = os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_{suffix}.html", filename_tag))
                    two_svg = os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_{suffix}.svg", filename_tag))
                    two_title = f"{os.path.basename(cell_folder)} | {suffix} | {pipeline_name}"
                    _plot_cell_two_panel(
                        chosen_df,
                        two_title,
                        two_html,
                        two_svg,
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_two_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_regression_{suffix}.svg", filename_tag)),
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_two_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_{suffix}.svg", filename_tag)),
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_two_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_regression_{suffix}.svg", filename_tag)),
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_two_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_regression_stats_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_regression_stats_{suffix}.svg", filename_tag)),
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                        show_corr_stats=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_spike_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_spike_{suffix}.svg", filename_tag)),
                        mode="spike",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_auc_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_auc_{suffix}.svg", filename_tag)),
                        mode="auc",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_spike_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_spike_regression_{suffix}.svg", filename_tag)),
                        mode="spike",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_auc_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_auc_regression_{suffix}.svg", filename_tag)),
                        mode="auc",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_length_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_length_{suffix}.svg", filename_tag)),
                        mode="length",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_length_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_length_regression_{suffix}.svg", filename_tag)),
                        mode="length",
                        y_col=GLOBAL_DFF_Y_COL,
                        y_label=GLOBAL_DFF_Y_LABEL,
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_{suffix}.svg", filename_tag)),
                        mode="spike",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_regression_{suffix}.svg", filename_tag)),
                        mode="spike",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_regression_stats_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_spike_regression_stats_{suffix}.svg", filename_tag)),
                        mode="spike",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                        show_corr_stats=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_{suffix}.svg", filename_tag)),
                        mode="auc",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_regression_{suffix}.svg", filename_tag)),
                        mode="auc",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_regression_stats_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_auc_regression_stats_{suffix}.svg", filename_tag)),
                        mode="auc",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                        show_corr_stats=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_{suffix}.svg", filename_tag)),
                        mode="length",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_regression_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_regression_{suffix}.svg", filename_tag)),
                        mode="length",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                    )
                    _plot_cell_single_panel(
                        chosen_df,
                        two_title,
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_regression_stats_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_cal3_cell_summary_zscore_length_regression_stats_{suffix}.svg", filename_tag)),
                        mode="length",
                        y_col="amp_local_z",
                        y_label="calcium response (z score)",
                        auc_df=chosen_auc_df,
                        add_regression=True,
                        show_corr_stats=True,
                    )

                cell_shape_groups = _shape_init_groups()
                ctid = _make_cell_trace_id(cell_folder, suffix)
                for ev in events:
                    _shape_add_event(
                        groups=cell_shape_groups,
                        ev=ev,
                        trace_vol=trace_vol,
                        trace_fluor=trace_fluor,
                        cal_sr=cal_sr,
                        vol_sr=VOL_SR,
                        cell_trace_id=ctid,
                        trace_cal_dff=gdf,
                    )
                shape_groups_by_pipeline[pipeline_name] = _shape_merge_groups(shape_groups_by_pipeline[pipeline_name], cell_shape_groups)
                if brain_state_folder is not None:
                    current_state_groups = state_shape_groups_by_pipeline[pipeline_name].get(brain_state_folder, _shape_init_groups())
                    state_shape_groups_by_pipeline[pipeline_name][brain_state_folder] = _shape_merge_groups(current_state_groups, cell_shape_groups)
                if len(cell_shape_groups) > 0:
                    shp_html = os.path.join(out_dir, _with_filename_tag(f"event_shape_summary_{suffix}.html", filename_tag))
                    shp_svg = os.path.join(out_dir, _with_filename_tag(f"event_shape_summary_{suffix}.svg", filename_tag))
                    shp_title = f"{os.path.basename(cell_folder)} | {suffix} | {pipeline_name} | event shape"
                    _shape_plot_groups(cell_shape_groups, shp_title + " | dF/F", shp_html, shp_svg, mode="dff")
                    _shape_plot_groups(
                        cell_shape_groups,
                        shp_title + " | z-score",
                        os.path.join(out_dir, _with_filename_tag(f"event_shape_summary_zscore_{suffix}.html", filename_tag)),
                        os.path.join(out_dir, _with_filename_tag(f"event_shape_summary_zscore_{suffix}.svg", filename_tag)),
                        mode="z",
                    )

                print(
                    f"  [{suffix}] {pipeline_name}: events={len(seg_df)} chosen={int(seg_df['include'].sum())} "
                    f"| plot={os.path.join('event_cal3', pipeline_name, html_name)} "
                    f"| zplot={os.path.join('event_cal3', pipeline_name, z_html_name)} | cal={cal_src} | vol={vol_src}"
                )

    if len(all_rows) == 0:
        return pd.DataFrame()

    os.makedirs(POP_SUMMARY_DIR, exist_ok=True)
    for pipeline_name in ("event_distance", "calcium_threshold"):
        if len(analysis_rows_by_pipeline[pipeline_name]) == 0:
            continue
        p_df = pd.concat(analysis_rows_by_pipeline[pipeline_name], ignore_index=True)
        p_chosen = p_df[p_df["include"].astype(bool)].copy()
        p_df_auc = pd.concat(analysis_rows_auc_by_pipeline[pipeline_name], ignore_index=True) if len(analysis_rows_auc_by_pipeline[pipeline_name]) else p_df.copy()
        p_chosen_auc = p_df_auc[p_df_auc["include"].astype(bool)].copy()
        trace_meta = (
            p_df[["cell_trace_id", "cell_trace_label"]]
            .drop_duplicates()
            .sort_values(["cell_trace_label", "cell_trace_id"])
            .to_dict("records")
        )
        if len(p_chosen) == 0 and len(trace_meta) == 0:
            continue
        out_dir = os.path.join(POP_SUMMARY_DIR, pipeline_name)
        os.makedirs(out_dir, exist_ok=True)

        _plot_all_cells_grid(
            p_chosen,
            mode="spike",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
        )
        _plot_all_cells_grid(
            p_chosen_auc,
            mode="auc",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
        )
        _plot_all_cells_grid(
            p_chosen_auc,
            mode="length",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
        )
        _plot_all_cells_grid(
            p_chosen,
            mode="spike",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_peak_vs_spike_zscore.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
        )
        _plot_all_cells_grid(
            p_chosen_auc,
            mode="auc",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_auc_zscore.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
        )
        _plot_all_cells_grid(
            p_chosen_auc,
            mode="length",
            save_html=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("all_cells_calcium_vs_event_length_zscore.svg", filename_tag)),
            all_trace_meta=trace_meta,
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
        )
        if len(p_chosen) == 0:
            continue
        _plot_population_two_panel(
            p_chosen,
            title=f"All cells pooled | {pipeline_name}",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel.svg", filename_tag)),
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
            auc_df=p_chosen_auc,
        )
        _plot_population_two_panel(
            p_chosen,
            title=f"All cells pooled | {pipeline_name}",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_regression.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_regression.svg", filename_tag)),
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
            auc_df=p_chosen_auc,
            add_regression=True,
        )
        _plot_pooled_fit_panels(
            p_chosen,
            title=f"All cells pooled fits | {pipeline_name}",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels.svg", filename_tag)),
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_Y_LABEL,
            auc_df=p_chosen_auc,
        )
        _plot_population_two_panel(
            p_chosen,
            title=f"All cells pooled | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
            auc_df=p_chosen_auc,
        )
        _plot_population_two_panel(
            p_chosen,
            title=f"All cells pooled | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
            auc_df=p_chosen_auc,
            add_regression=True,
        )
        _plot_population_two_panel(
            p_chosen,
            title=f"All cells pooled | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression_stats.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_two_panel_zscore_regression_stats.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
            auc_df=p_chosen_auc,
            add_regression=True,
            show_corr_stats=True,
        )
        _plot_pooled_fit_panels(
            p_chosen,
            title=f"All cells pooled fits | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("pooled_fit_panels_zscore.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="calcium response (z-score)",
            auc_df=p_chosen_auc,
        )
        _plot_cellavg_amp_vs_spikes(
            p_chosen,
            title=f"Cell-average calcium amplitude vs spike count | {pipeline_name}",
            save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount.svg", filename_tag)),
            y_col=GLOBAL_DFF_Y_COL,
            y_label=GLOBAL_DFF_CELLAVG_Y_LABEL,
        )
        _plot_cellavg_amp_vs_spikes(
            p_chosen,
            title=f"Cell-average calcium amplitude vs spike count | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="cell-mean calcium amplitude (z-score)",
            show_fit_annotation=False,
            show_legend=False,
        )
        _plot_cellavg_amp_vs_spikes(
            p_chosen,
            title=f"Cell-average calcium amplitude vs spike count | {pipeline_name} | z-score",
            save_html=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore_stats.html", filename_tag)),
            save_svg=os.path.join(out_dir, _with_filename_tag("cellavg_amp_vs_spikecount_zscore_stats.svg", filename_tag)),
            y_col="amp_local_z",
            y_label="cell-mean calcium amplitude (z-score)",
            show_fit_annotation=True,
            show_legend=False,
            show_corr_stats=True,
        )
        _save_cellavg_amp_vs_spikecount_lmm_tukey(
            p_chosen,
            out_dir=out_dir,
            filename_prefix=_with_filename_tag("cellavg_amp_vs_spikecount_zscore", filename_tag),
            y_col="amp_local_z",
        )

        # Event-shape summary (all chosen events pooled) per selection method
        shp_groups = shape_groups_by_pipeline.get(pipeline_name, {})
        if shp_groups and len(shp_groups) > 0:
            _shape_plot_groups(
                shp_groups,
                title=f"Event-shape summary | {pipeline_name} | dF/F",
                save_html=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all.html", filename_tag)),
                save_svg=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all.svg", filename_tag)),
                mode="dff",
            )
            _shape_plot_groups(
                shp_groups,
                title=f"Event-shape summary | {pipeline_name} | z-score",
                save_html=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all_zscore.html", filename_tag)),
                save_svg=os.path.join(out_dir, _with_filename_tag("event_shape_summary_all_zscore.svg", filename_tag)),
                mode="z",
            )

        # Event-choosing summaries (same style as pyr_event_cal2_pipeline)
        if pe2 is not None and len(event_selection_rows_by_pipeline[pipeline_name]) > 0:
            try:
                evdf = pd.concat(event_selection_rows_by_pipeline[pipeline_name], ignore_index=True)
                evdf = evdf.copy()
                evdf["cell_folder"] = evdf["cell_folder"].astype(str) + "::" + evdf["suffix"].astype(str)
                evt_dir = os.path.join(out_dir, "eventChoss")
                os.makedirs(evt_dir, exist_ok=True)
                pe2._plot_event_selection_two_panel(
                    evdf,
                    title_prefix=f"Pyr event choosing analysis | {pipeline_name}",
                    save_html=os.path.join(evt_dir, _with_filename_tag("event_choosing_analysis.html", filename_tag)),
                    save_svg=os.path.join(evt_dir, _with_filename_tag("event_choosing_analysis.svg", filename_tag)),
                    save_pdf=None,
                )
                pe2.save_event_selection_cell_by_cell_figures(evdf, out_dir=evt_dir, n_cols=6)
                pe2.save_event_selection_population_figures(evdf, out_dir=evt_dir)
            except Exception as e:
                print(f"[WARN] event choosing summary failed ({pipeline_name}): {e}")

    for state_folder in ("anst", "motor", "awake"):
        for pipeline_name in ("event_distance", "calcium_threshold"):
            state_rows = state_analysis_rows_by_pipeline[pipeline_name].get(state_folder, [])
            if len(state_rows) == 0:
                continue
            state_df = pd.concat(state_rows, ignore_index=True)
            state_auc_rows = state_analysis_rows_auc_by_pipeline[pipeline_name].get(state_folder, [])
            state_df_auc = pd.concat(state_auc_rows, ignore_index=True) if len(state_auc_rows) else state_df.copy()
            state_out_dir = os.path.join(POP_SUMMARY_DIR, state_folder, pipeline_name)
            _write_pyr_event_cal3_population_summary_bundle(
                p_df=state_df,
                p_df_auc=state_df_auc,
                pipeline_name=pipeline_name,
                out_dir=state_out_dir,
                filename_tag=filename_tag,
                title_prefix=f"{state_folder} cells pooled",
                shape_groups=state_shape_groups_by_pipeline[pipeline_name].get(state_folder, _shape_init_groups()),
                event_selection_rows=state_event_selection_rows_by_pipeline[pipeline_name].get(state_folder, []),
            )

    out_df = pd.concat(all_rows, ignore_index=True)
    one_spike_complex = out_df[
        (out_df["event_type"].astype(str).str.lower() == "complex")
        & (pd.to_numeric(out_df["n_spikes"], errors="coerce") == 1)
    ].copy()
    if len(one_spike_complex) > 0:
        report_cols = [
            "cell_folder",
            "suffix",
            "selection_pipeline",
            "event_idx",
            "first_spike",
            "pkl_name",
        ]
        report_cols = [c for c in report_cols if c in one_spike_complex.columns]
        uniq = (
            one_spike_complex[["cell_folder", "suffix", "pkl_name"]]
            .drop_duplicates()
            .sort_values(["cell_folder", "suffix", "pkl_name"])
        )
        print("\n[CHECK] cells with complex events containing exactly 1 spike:")
        for _, rr in uniq.iterrows():
            print(f"  {rr['cell_folder']} | suffix={rr['suffix']} | pkl={rr['pkl_name']}")
        print(f"[CHECK] total one-spike complex event rows: {len(one_spike_complex)}")
        print(one_spike_complex[report_cols].to_string(index=False))
    else:
        print("\n[CHECK] no complex events with exactly 1 spike found")
    print(f"\n[ALL] processed {len(out_df)} events (population CSV saving disabled)")
    return out_df


def run_pyr_event_cal3_no_plateau(db_path=DB_PATH, max_cells=None, motor_mode="chunked"):
    return run_pyr_event_cal3(
        db_path=db_path,
        max_cells=max_cells,
        include_plateau=False,
        filename_tag="no_plateau",
        motor_mode=motor_mode,
    )


if __name__ == "__main__":
    run_pyr_event_cal3()
