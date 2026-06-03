# SINGLE_CELL_SVG_OVERLAY_V2
# One saved SVG: voltage + calcium overlay with voltage-detected spikes.
# Size: 21 cm wide x 7 cm high.
# Voltage is red, calcium is dark purple. Normal axes are hidden.
# Spike PKL selection prefers *_rm_complex_highplateau.pkl when available.

import os
import re
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# Convert text to vector paths in SVG exports. This avoids PowerPoint/Edge font issues.
plt.rcParams['svg.fonttype'] = 'path'


def _read_trace_csv_1d(csv_path):
    arr = pd.read_csv(csv_path).to_numpy(dtype=float).ravel()
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def _find_existing_file(base_path, candidates):
    base_path = Path(base_path)
    for name in candidates:
        p = base_path / name
        if p.is_file():
            return p
    return None


def _cm_to_in(cm):
    return float(cm) / 2.54


def _add_vertical_scale_bar(ax, value, label, color, side='left', y_frac=0.12, label_fontsize=5):
    value = float(value)
    if (not np.isfinite(value)) or value <= 0:
        return

    y0_lim, y1_lim = ax.get_ylim()
    yr = float(y1_lim - y0_lim)
    if (not np.isfinite(yr)) or yr <= 0:
        return

    y0 = y0_lim + float(y_frac) * yr
    y1 = y0 + value
    if y1 > y1_lim:
        y1 = y1_lim - 0.04 * yr
        y0 = y1 - value
    if y0 < y0_lim:
        y0 = y0_lim + 0.04 * yr
        y1 = y0 + value

    x = -0.012 if side == 'left' else 1.012
    ha = 'right' if side == 'left' else 'left'
    text_x = x - 0.006 if side == 'left' else x + 0.006
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.plot([x, x], [y0, y1], transform=trans, color=color, lw=1.4, clip_on=False)
    if label:
        ax.text(text_x, (y0 + y1) / 2.0, label, transform=trans,
                ha=ha, va='center', rotation=0, fontsize=label_fontsize,
                color=color, linespacing=0.9, clip_on=False)


def _unique_sorted_in_bounds(x, n):
    if x is None:
        return np.array([], dtype=int)
    try:
        a = np.asarray(x, dtype=int).ravel()
    except Exception:
        # Support ragged inputs like list[ndarray], list[list[int]], etc.
        vals = []
        try:
            it = list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]
            for elem in it:
                try:
                    ee = np.asarray(elem, dtype=float).ravel()
                    if ee.size:
                        vals.extend(ee.tolist())
                except Exception:
                    try:
                        vals.append(float(elem))
                    except Exception:
                        pass
            a = np.asarray(vals, dtype=int).ravel()
        except Exception:
            return np.array([], dtype=int)
    if a.size == 0:
        return np.array([], dtype=int)
    a = a[(a >= 0) & (a < int(n))]
    return np.unique(a) if a.size else np.array([], dtype=int)


def _enforce_refractory(idx, min_frames=2):
    idx = np.asarray(idx, dtype=int).ravel()
    if idx.size == 0:
        return idx
    keep = [int(idx[0])]
    for x in idx[1:]:
        if int(x) - keep[-1] >= int(min_frames):
            keep.append(int(x))
    return np.asarray(keep, dtype=int)


def _extract_spikes_from_pkl_payload(payload, n_vol):
    if payload is None:
        return np.array([], dtype=int)
    if isinstance(payload, (list, tuple, np.ndarray)):
        return _unique_sorted_in_bounds(payload, n_vol)
    if not isinstance(payload, dict):
        return np.array([], dtype=int)

    # Prefer corrected/all-spike keys, then merge class-specific spike keys if needed.
    for key in ('spike_indices', 'vm_all_spikes', 'all_spikes', 'spikes', 'spike_idx', 'spikeID'):
        if key in payload:
            sp = _unique_sorted_in_bounds(payload.get(key), n_vol)
            if sp.size:
                return sp

    parts = []
    for key in ('vm_simple_spike', 'vm_complex_spike', 'vm_plateau_spike', 'simple_spikes', 'complex_spikes', 'plateau_spikes'):
        sp = _unique_sorted_in_bounds(payload.get(key), n_vol)
        if sp.size:
            parts.append(sp)
    if parts:
        return np.unique(np.concatenate(parts).astype(int))
    return np.array([], dtype=int)


def _extract_spike_types_from_pkl_payload(payload, n_vol):
    """Return per-type spike indices from known payload keys."""
    if not isinstance(payload, dict):
        return {
            "simple": np.array([], dtype=int),
            "complex": np.array([], dtype=int),
            "plateau": np.array([], dtype=int),
        }

    def _first_existing(keys):
        for k in keys:
            if k in payload:
                return payload.get(k)
        return None

    simple = _unique_sorted_in_bounds(
        _first_existing(["vm_simple_spike", "vm_simple_spikes", "simple_spikes"]), n_vol
    )
    complex_ = _unique_sorted_in_bounds(
        _first_existing(["vm_complex_spike", "vm_complex_spikes", "complex_spikes"]), n_vol
    )
    plateau = _unique_sorted_in_bounds(
        _first_existing(["vm_plateau_spike", "vm_plateau_spikes", "plateau_spikes"]), n_vol
    )

    # Some PKLs store plateau spike indices inside vm_plateaus_dict['spike_indices'].
    if plateau.size == 0 and isinstance(payload.get("vm_plateaus_dict"), dict):
        pidx = payload["vm_plateaus_dict"].get("spike_indices", None)
        plateau = _unique_sorted_in_bounds(pidx, n_vol)

    # Also derive plateau spikes from plateau start/end windows, using available spike train.
    # This is robust when spike_indices is missing/incomplete.
    if isinstance(payload.get("vm_plateaus_dict"), dict):
        pdict = payload["vm_plateaus_dict"]
        p_starts = _unique_sorted_in_bounds(pdict.get("starts", None), n_vol)
        p_ends = _unique_sorted_in_bounds(pdict.get("ends", None), n_vol)
        if p_starts.size and p_ends.size:
            m = min(p_starts.size, p_ends.size)
            p_starts = p_starts[:m]
            p_ends = p_ends[:m]

            all_sp = _unique_sorted_in_bounds(
                _first_existing(["vm_all_spikes", "all_spikes"]), n_vol
            )
            if all_sp.size == 0:
                all_sp = np.unique(np.concatenate([simple, complex_])).astype(int) if (simple.size or complex_.size) else np.array([], dtype=int)

            if all_sp.size:
                pwin = []
                for s, e in zip(p_starts, p_ends):
                    if e < s:
                        s, e = e, s
                    hit = all_sp[(all_sp >= int(s)) & (all_sp <= int(e))]
                    if hit.size:
                        pwin.append(hit)
                if pwin:
                    plateau_from_windows = np.unique(np.concatenate(pwin)).astype(int)
                    plateau = np.unique(np.concatenate([plateau, plateau_from_windows])).astype(int) if plateau.size else plateau_from_windows

    # Keep types mutually exclusive for plotting.
    if plateau.size:
        complex_ = complex_[~np.isin(complex_, plateau)]
        simple = simple[~np.isin(simple, plateau)]
    if complex_.size:
        simple = simple[~np.isin(simple, complex_)]

    return {"simple": simple, "complex": complex_, "plateau": plateau}


def _fallback_detect_spikes_from_voltage(vol, min_dist_frames=2):
    vol = np.asarray(vol, dtype=float).ravel()
    if vol.size < 3:
        return np.array([], dtype=int)
    y = vol.copy()
    finite = np.isfinite(y)
    if not finite.any():
        return np.array([], dtype=int)
    med = np.nanmedian(y[finite])
    mad = np.nanmedian(np.abs(y[finite] - med))
    sd = 1.4826 * mad if mad > 0 else np.nanstd(y[finite])
    thr = med + max(3.0 * sd, 0.15 * (np.nanmax(y[finite]) - np.nanmin(y[finite])))
    peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thr))[0] + 1
    return _enforce_refractory(peaks, min_frames=min_dist_frames)


def _estimate_segment_offset_in_full_trace(full_vol, seg_vol):
    full = np.asarray(full_vol, dtype=float).ravel()
    seg = np.asarray(seg_vol, dtype=float).ravel()
    if full.size == 0 or seg.size == 0 or seg.size > full.size:
        return None

    step = max(1, int(round(seg.size / 3000)))
    template = seg[::step]
    template = template - np.nanmean(template)
    tnorm = np.linalg.norm(template)
    if not np.isfinite(tnorm) or tnorm <= 0:
        return None

    best_off = None
    best_score = -np.inf
    max_off = full.size - seg.size
    probe_step = max(1, int(round(full.size / 2000)))
    for off in range(0, max_off + 1, probe_step):
        cand = full[off:off + seg.size:step]
        if cand.size != template.size:
            continue
        cand = cand - np.nanmean(cand)
        cnorm = np.linalg.norm(cand)
        if not np.isfinite(cnorm) or cnorm <= 0:
            continue
        score = float(np.dot(template, cand) / (tnorm * cnorm))
        if score > best_score:
            best_score = score
            best_off = off
    return best_off


def _align_spikes_to_full_trace_if_needed(spike_idx, payload, full_vol, pkl_path=None, vol_sr=500.0):
    sp = _unique_sorted_in_bounds(spike_idx, len(full_vol))
    if sp.size == 0 or not isinstance(payload, dict):
        return sp, 0

    pkl_vol = None
    for key in ('vm', 'voltage', 'vol', 'trace_raw', 'raw_voltage'):
        if key in payload:
            try:
                pkl_vol = np.asarray(payload[key], dtype=float).ravel()
            except Exception:
                pkl_vol = None
            if pkl_vol is not None and pkl_vol.size > 0:
                break

    # If payload voltage is a segment and spike indices are segment-local, shift to full trace.
    if pkl_vol is not None and pkl_vol.size < len(full_vol) and sp.max(initial=0) < pkl_vol.size:
        off = _estimate_segment_offset_in_full_trace(full_vol, pkl_vol)
        if off is not None and off > 0:
            name = Path(pkl_path).name if pkl_path is not None else 'pkl'
            print(f'[ALIGN] {name}: shifted spikes by +{off} frames ({off / float(vol_sr):.3f}s) to full-trace timeline')
            return _unique_sorted_in_bounds(sp + int(off), len(full_vol)), int(off)
    return sp, 0


def _spike_pkl_priority(path, preferred_name=None):
    name = Path(path).name.lower()
    preferred = str(preferred_name or '').lower()
    if preferred.startswith('final_spikes') and name.startswith('final_spikes'):
        return 0
    if name.startswith('spike_detection_refined_new') and 'rm_complex_highplateau' in name:
        return 1
    if name.startswith('event_spike_overlay__plus_plateau') and 'rm_complex_highplateau' in name:
        return 2
    if preferred and name == preferred:
        return 3
    if name.startswith('spike_detection_refined_new'):
        return 4
    if name.startswith('final_correct_spike_detection'):
        return 5
    if name.startswith('final_spikes'):
        return 6
    return 9


def _candidate_spike_pkls(cell_path, preferred_name=None):
    cell_path = Path(cell_path)
    cand = []
    if preferred_name:
        exact = cell_path / str(preferred_name)
        if exact.is_file():
            cand.append(exact)
    cand.extend(Path(x) for x in glob.glob(str(cell_path / 'spike_detection_refined_new*_rm_complex_highplateau.pkl')))
    cand.extend(Path(x) for x in glob.glob(str(cell_path / 'event_spike_overlay__plus_plateau*_rm_complex_highplateau.pkl')))
    cand.extend(Path(x) for x in glob.glob(str(cell_path / 'spike_detection_refined_new*.pkl')))
    cand.extend(Path(x) for x in glob.glob(str(cell_path / 'final_correct_spike_detection*.pkl')))
    cand.extend(Path(x) for x in glob.glob(str(cell_path / 'final_spikes*.pkl')))

    seen, out = set(), []
    for c in cand:
        c = Path(c)
        k = str(c).lower()
        if k not in seen and c.is_file():
            seen.add(k)
            out.append(c)
    return out


def _find_spike_pkl(cell_path, preferred_name='spike_detection_refined_new.pkl'):
    cands = _candidate_spike_pkls(cell_path, preferred_name=preferred_name)
    if not cands:
        return None

    def _key(path):
        name = Path(path).name.lower()
        m = re.search(r'(?:m|r)?(\d+)(?=.*\.pkl$)', name)
        num = int(m.group(1)) if m else -1
        return (_spike_pkl_priority(path, preferred_name), -num, name)

    return sorted(cands, key=_key)[0]


def _choose_best_pkl_for_window(cell_path, preferred_name, full_vol, i0, i1, vol_sr=500.0):
    cands = _candidate_spike_pkls(cell_path, preferred_name=preferred_name)
    if not cands:
        return None

    preferred_l = str(preferred_name or '').lower()
    state_match = re.search(r'([mr]\d+)', preferred_l)
    if state_match:
        state = state_match.group(1)
        state_cands = [c for c in cands if state in c.name.lower()]
        if state_cands:
            cands = state_cands

    scored = []
    for c in cands:
        try:
            with open(c, 'rb') as f:
                payload = pickle.load(f)
            sp = _extract_spikes_from_pkl_payload(payload, len(full_vol))
            sp, _ = _align_spikes_to_full_trace_if_needed(sp, payload, full_vol, pkl_path=c, vol_sr=vol_sr)
            n_in = int(np.sum((sp >= int(i0)) & (sp < int(i1))))
            scored.append((_spike_pkl_priority(c, preferred_name), -n_in, c.name.lower(), c, n_in))
        except Exception:
            continue

    if not scored:
        return _find_spike_pkl(cell_path, preferred_name=preferred_name)

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best = scored[0][3]
    best_n = scored[0][4]
    print(f'[PKL-SELECT] chose {best.name} for window ({i0 / vol_sr:.3f}-{i1 / vol_sr:.3f}s), spikes_in_window={best_n}')
    return best


def _gaussian_smooth_1d(x, sigma_samples):
    x = np.asarray(x, dtype=float).ravel()
    sigma_samples = float(sigma_samples)
    if x.size == 0 or (not np.isfinite(sigma_samples)) or sigma_samples <= 0:
        return x.copy()
    radius = max(1, int(np.ceil(3.0 * sigma_samples)))
    k = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (k / sigma_samples) ** 2)
    w /= np.sum(w)
    return np.convolve(x, w, mode='same')


def _safe_axis_limits(y, pad_frac=0.50, fallback=(0.0, 1.0)):
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return [float(fallback[0]), float(fallback[1])]
    lo = float(np.min(y))
    hi = float(np.max(y))
    span = hi - lo
    if not np.isfinite(lo) or not np.isfinite(hi):
        return [float(fallback[0]), float(fallback[1])]
    pad = max(1e-3, (float(pad_frac) * span) if span > 0 else 0.05 * abs(hi) + 1e-3)
    return [lo - pad, hi + pad]


def plot_10s_window_with_spikes_and_overlay(
    cell_path,
    cal_sr,
    start_s,
    duration_s=10.0,
    vol_sr=500.0,
    save_stem='cal_vol_single_trace_with_spikes',
    spike_pkl_name='spike_detection_refined_new.pkl',
    cal_smooth_sigma_s=0.03,
    fig_width_cm=21.0,
    fig_height_cm=7.0,
    vol_color='#d62728',
    cal_color='#4b0082',
    spike_color='black',
    color_spikes_by_type=False,
    auto_choose_pkl_for_window=True,
):
    cell_path = Path(cell_path)
    if not cell_path.is_dir():
        raise FileNotFoundError(f'Folder not found: {cell_path}')

    cal_file = _find_existing_file(cell_path, ['calDF.csv', 'calTraceDF.csv'])
    vol_file = _find_existing_file(cell_path, ['volDF.csv', 'volTraceDF.csv'])
    if cal_file is None:
        raise FileNotFoundError(f'No calcium file found in {cell_path} (expected calDF.csv or calTraceDF.csv)')
    if vol_file is None:
        raise FileNotFoundError(f'No voltage file found in {cell_path} (expected volDF.csv or volTraceDF.csv)')

    cal = _read_trace_csv_1d(cal_file)
    vol = _read_trace_csv_1d(vol_file)
    cal_sr = float(cal_sr)
    vol_sr = float(vol_sr)
    start_s = float(start_s)
    duration_s = float(duration_s)
    end_s = start_s + duration_s

    i0 = max(0, int(round(start_s * vol_sr)))
    i1 = min(len(vol), int(round(end_s * vol_sr)))
    if i1 <= i0:
        raise ValueError('Selected window is outside voltage trace range.')

    ci0 = max(0, int(round(start_s * cal_sr)))
    ci1 = min(len(cal), int(round(end_s * cal_sr)))
    if ci1 <= ci0:
        raise ValueError('Selected window is outside calcium trace range.')

    pkl_path = None
    if bool(auto_choose_pkl_for_window):
        pkl_path = _choose_best_pkl_for_window(
            cell_path=cell_path,
            preferred_name=spike_pkl_name,
            full_vol=vol,
            i0=i0,
            i1=i1,
            vol_sr=vol_sr,
        )
    if pkl_path is None:
        pkl_path = _find_spike_pkl(cell_path, preferred_name=spike_pkl_name)

    spike_idx = np.array([], dtype=int)
    spike_types = {"simple": np.array([], dtype=int), "complex": np.array([], dtype=int), "plateau": np.array([], dtype=int)}
    pkl_payload = None
    spike_align_off = 0
    if pkl_path is not None:
        try:
            with open(pkl_path, 'rb') as f:
                pkl_payload = pickle.load(f)
            spike_idx = _extract_spikes_from_pkl_payload(pkl_payload, len(vol))
            spike_idx, spike_align_off = _align_spikes_to_full_trace_if_needed(
                spike_idx, payload=pkl_payload, full_vol=vol, pkl_path=pkl_path, vol_sr=vol_sr
            )
            spike_types = _extract_spike_types_from_pkl_payload(pkl_payload, len(vol))
            if int(spike_align_off) != 0:
                for k in ("simple", "complex", "plateau"):
                    if spike_types[k].size:
                        spike_types[k] = _unique_sorted_in_bounds(spike_types[k] + int(spike_align_off), len(vol))
        except Exception as e:
            print(f'[WARN] failed loading spikes from {pkl_path}: {e}')
            spike_idx = np.array([], dtype=int)
            spike_types = {"simple": np.array([], dtype=int), "complex": np.array([], dtype=int), "plateau": np.array([], dtype=int)}
    if spike_idx.size == 0:
        spike_idx = _fallback_detect_spikes_from_voltage(vol, min_dist_frames=2)

    vol_w = vol[i0:i1]
    cal_w = cal[ci0:ci1]
    cal_sigma_samples = max(0.0, float(cal_smooth_sigma_s) * cal_sr)
    cal_w_smooth = _gaussian_smooth_1d(cal_w, cal_sigma_samples)

    t_vol = np.arange(i0, i1, dtype=float) / vol_sr - start_s
    t_cal = np.arange(ci0, ci1, dtype=float) / cal_sr - start_s
    spk_w = spike_idx[(spike_idx >= i0) & (spike_idx < i1)]
    spk_t = spk_w.astype(float) / vol_sr - start_s
    spk_y = vol[spk_w] if spk_w.size else np.array([], dtype=float)

    spk_types_w = {}
    for _k in ("simple", "complex", "plateau"):
        arr = np.asarray(spike_types.get(_k, np.array([], dtype=int)), dtype=int).ravel()
        arr = arr[(arr >= i0) & (arr < i1)]
        spk_types_w[_k] = arr
    if bool(color_spikes_by_type):
        print(
            "Spikes by type in window:",
            f"simple={int(spk_types_w['simple'].size)}",
            f"complex={int(spk_types_w['complex'].size)}",
            f"plateau={int(spk_types_w['plateau'].size)}",
        )

    v_rng = _safe_axis_limits(vol_w, pad_frac=0.55)
    c_rng = _safe_axis_limits(cal_w_smooth, pad_frac=0.55)

    fig, ax = plt.subplots(figsize=(_cm_to_in(fig_width_cm), _cm_to_in(fig_height_cm)))
    ax2 = ax.twinx()

    ax.plot(t_vol, vol_w, color=vol_color, lw=0.7, alpha=0.9)
    ax2.plot(t_cal, cal_w_smooth, color=cal_color, lw=0.7, alpha=0.95)
    if spk_w.size > 0:
        if bool(color_spikes_by_type) and (
            spk_types_w["simple"].size or spk_types_w["complex"].size or spk_types_w["plateau"].size
        ):
            type_colors = {
                "simple": "black",   # simple
                "complex": "red",    # complex non-plateau
                "plateau": "#c04dff", # plateau (brighter purple)
            }
            for _k in ("simple", "complex", "plateau"):
                arr = spk_types_w[_k]
                if arr.size == 0:
                    continue
                _t = arr.astype(float) / vol_sr - start_s
                _y = vol[arr]
                ax.scatter(
                    _t, _y, s=8, color=type_colors[_k], marker='o',
                    linewidths=0, zorder=5, clip_on=False
                )
        else:
            ax.scatter(spk_t, spk_y, s=8, color=spike_color, marker='o', linewidths=0, zorder=5, clip_on=False)

    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(v_rng)
    ax2.set_ylim(c_rng)

    for a in (ax, ax2):
        a.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                      labelleft=False, labelright=False, labelbottom=False)
        for spine in a.spines.values():
            spine.set_visible(False)

    _add_vertical_scale_bar(ax, 1.00, '100%\ndf/f', vol_color, side='left', label_fontsize=5)
    _add_vertical_scale_bar(ax2, 0.20, '20%\ndf/f', cal_color, side='right', label_fontsize=5)

    # x scale bar + label in reserved figure space below the trace.
    left_margin = 0.16
    right_margin = 0.88
    bar_len = min(0.5, duration_s)
    bar_frac = bar_len / duration_s if duration_s > 0 else 0.05
    x0_fig = left_margin
    x1_fig = left_margin + (right_margin - left_margin) * min(1.0, bar_frac)
    fig.add_artist(plt.Line2D([x0_fig, x1_fig], [0.16, 0.16],
                              transform=fig.transFigure, color='black', lw=1.8, zorder=1000))
    fig.text((x0_fig + x1_fig) / 2.0, 0.095, '0.5 sec',
             ha='center', va='center', fontsize=10, fontweight='bold', color='black', zorder=1001)

    fig.subplots_adjust(left=left_margin, right=right_margin, top=0.88, bottom=0.24)

    svg_path = cell_path / f'{save_stem}.svg'
    fig.savefig(svg_path, format='svg', transparent=True, bbox_inches=None)
    plt.close(fig)

    pkl_name = pkl_path.name if pkl_path is not None else 'fallback_voltage_detection'
    print('Spike source:', pkl_name)
    print('Spikes in window:', int(spk_w.size))
    print('Saved SVG :', svg_path)
    return svg_path


# ===== Example usage =====
RUN_FOR_SST = False  # True -> use SST-style final_spikes.pkl
CELL_PATH = r'Z:\Adam-Lab-Shared\Data\Michal_Rubin\rugc42\Wh\21-10-2025-MOTOR\fov7\cell0'
CAL_SR = 29.97
START_S = 45.5
WINDOW_S = 9.0
SPIKE_PKL_NAME = 'final_spikes.pkl' if RUN_FOR_SST else 'spike_detection_refined_new.pkl'

svg_path = plot_10s_window_with_spikes_and_overlay(
    cell_path=CELL_PATH,
    cal_sr=CAL_SR,
    start_s=START_S,
    duration_s=WINDOW_S,
    cal_smooth_sigma_s=0.06,
    vol_sr=500.0,
    save_stem='cal_vol_single_trace_with_spikes',
    spike_pkl_name=('final_spikes.pkl' if RUN_FOR_SST else 'spike_detection_refined_new_rm_complex_highplateau.pkl'),
    vol_color=('#1f77b4' if RUN_FOR_SST else '#d62728'),
    color_spikes_by_type=(False if RUN_FOR_SST else True),
    auto_choose_pkl_for_window=False,
)
