# NEW_CELL: SST_CORR_ANLSYS_FIVE_PANEL_SUMMARY
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from scipy import stats as spstats
from statsmodels.stats.oneway import anova_oneway
import statsmodels.formula.api as smf

# Requested minimal smoothing target for BOTH FR and calcium
SIGMA_BASIC_TARGET = 0.00

# If you want to save the figure, keep this folder; set to None to skip saving.
OUT_DIR = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\SST\BasicCorr"
OUT_STEM_ZERO = f"sst_corr_summary_zero_lag_sigma{SIGMA_BASIC_TARGET:.2f}"
OUT_STEM_LAGFIX = f"sst_corr_summary_fixedlag_sigma{SIGMA_BASIC_TARGET:.2f}"
OUT_STEM_POP = f"sst_corr_summary_by_population_sigma{SIGMA_BASIC_TARGET:.2f}"
OUT_STEM_EXPLAIN = f"sst_corr_explained_variance_sigma{SIGMA_BASIC_TARGET:.2f}"
OUT_STEM_MIXED = f"sst_corr_mixedlm_population_sigma{SIGMA_BASIC_TARGET:.2f}"

_PAT = re.compile(r"corrDict_frSm(?P<fr>[0-9.]+)_calSm(?P<cal>[0-9.]+)\.pkl$", re.IGNORECASE)


def _to_float_nan(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _infer_population_from_folder(folder):
    s = str(folder).strip().lower()
    if "motor" in s:
        return "motor"
    if ("awake" in s) or ("awke" in s):
        return "awake"
    if ("anst" in s) or ("ans" in s) or ("anest" in s) or ("anaest" in s):
        return "anesthetized"
    return "unknown"


def _derive_cell_id_from_folder(folder):
    """
    Build a stable cell ID used as random effect group:
    tries to capture .../fovX/cellY across sessions/states.
    """
    s = str(folder).replace("\\", "/")
    m = re.search(r"(fov\d+)/(cell\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}/{m.group(2).lower()}"
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2].lower()}/{parts[-1].lower()}"
    return str(folder).lower()


def _extract_fr_array(d):
    """Best-effort FR array extraction from corrDict payload."""
    candidates = [
        "fr_on_ca", "fr_on_cal", "fr_smooth", "fr", "fr_real", "frReal", "volFR", "FR",
    ]
    for k in candidates:
        if k in d:
            try:
                arr = np.asarray(d[k], dtype=float).ravel()
                if arr.size > 0:
                    return arr
            except Exception:
                pass
    return np.array([], dtype=float)


def _extract_fs_ca(d):
    try:
        params = d.get("params", {}) if isinstance(d, dict) else {}
        fs = _to_float_nan(params.get("fs_ca", np.nan))
        if np.isfinite(fs) and fs > 0:
            return float(fs)
    except Exception:
        pass

    try:
        t = np.asarray(d.get("t_ca", []), dtype=float).ravel()
        if t.size >= 2:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                med_dt = float(np.median(dt))
                if med_dt > 0:
                    return float(1.0 / med_dt)
    except Exception:
        pass

    return np.nan


def _pair_by_lag_samples(x, y, lag_samples):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 3:
        return np.array([], float), np.array([], float)
    x = x[:n]
    y = y[:n]

    lag = int(lag_samples)
    if lag >= 0:
        xa = x[: n - lag] if lag < n else np.array([], float)
        ya = y[lag:] if lag < n else np.array([], float)
    else:
        k = -lag
        xa = x[k:] if k < n else np.array([], float)
        ya = y[: n - k] if k < n else np.array([], float)

    ok = np.isfinite(xa) & np.isfinite(ya)
    return xa[ok], ya[ok]


def _corr_from_arrays(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or y.size < 3:
        return np.nan
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _lag_scan_best_r(x, y, fs, min_lag_s=0.0, max_lag_s=0.5, min_overlap_s=1.0, mode="max_pos"):
    """
    Lag scan restricted to calcium lagging voltage only.
    Positive lag means calcium is shifted earlier by lag to match voltage/FR.
    Allowed lag range: [min_lag_s, max_lag_s] (default 0..0.5 s).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 5 or (not np.isfinite(fs)) or fs <= 0:
        return np.nan, np.nan
    x = x[:n]
    y = y[:n]

    min_lag = int(round(max(0.0, float(min_lag_s)) * float(fs)))
    max_lag = int(round(max(0.0, float(max_lag_s)) * float(fs)))
    if max_lag < min_lag:
        max_lag = min_lag
    lags = np.arange(min_lag, max_lag + 1, dtype=int)

    min_overlap = max(int(round(float(min_overlap_s) * float(fs))), 3)

    r_vals = []
    lag_vals = []
    for lag in lags:
        xv, yv = _pair_by_lag_samples(x, y, lag)
        if xv.size < min_overlap or yv.size < min_overlap:
            continue
        r = _corr_from_arrays(xv, yv)
        if np.isfinite(r):
            r_vals.append(r)
            lag_vals.append(lag)

    if len(r_vals) == 0:
        return np.nan, np.nan

    r_vals = np.asarray(r_vals, float)
    lag_vals = np.asarray(lag_vals, int)

    if mode == "max_abs":
        idx = int(np.nanargmax(np.abs(r_vals)))
    elif mode == "max_neg":
        idx = int(np.nanargmin(r_vals))
    else:
        idx = int(np.nanargmax(r_vals))

    return float(r_vals[idx]), float(lag_vals[idx]) / float(fs)


def _corr_with_fixed_lag(x, y, fs, lag_s):
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(lag_s):
        return np.nan
    lag_samples = int(round(float(lag_s) * float(fs)))
    xv, yv = _pair_by_lag_samples(x, y, lag_samples)
    return _corr_from_arrays(xv, yv)


def _load_corrdict_full_records(folder):
    files = glob.glob(os.path.join(folder, "corrDict_frSm*_calSm*.pkl"))
    out = []
    for p in files:
        m = _PAT.search(os.path.basename(p))
        if not m:
            continue
        fr_sig = _to_float_nan(m.group("fr"))
        cal_sig = _to_float_nan(m.group("cal"))
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue

        rr0 = _to_float_nan(d.get("pearson_r", np.nan))
        fr_arr = _extract_fr_array(d)
        x = np.asarray(d.get("fr_on_ca", []), dtype=float).ravel()
        y = np.asarray(d.get("ca_used", []), dtype=float).ravel()
        fs = _extract_fs_ca(d)

        out.append({
            "path": p,
            "folder": folder,
            "fr_sig": fr_sig,
            "cal_sig": cal_sig,
            "pearson_r_0lag": rr0,
            "mean_fr_hz": float(np.nanmean(fr_arr)) if fr_arr.size else np.nan,
            "std_fr_hz": float(np.nanstd(fr_arr)) if fr_arr.size else np.nan,
            "fr_on_ca": x,
            "ca_used": y,
            "fs_ca": fs,
        })
    return out


def _pick_record_near_sigma(records, fr_target, cal_target):
    finite = [r for r in records if np.isfinite(r["fr_sig"]) and np.isfinite(r["cal_sig"])]
    if len(finite) == 0:
        return None

    def d2(r):
        return (r["fr_sig"] - fr_target) ** 2 + (r["cal_sig"] - cal_target) ** 2

    return min(finite, key=d2)


def _pick_basic_record(records, sigma_target=0.10):
    if len(records) == 0:
        return None
    fr_vals = np.array(sorted({r["fr_sig"] for r in records if np.isfinite(r["fr_sig"])}), dtype=float)
    cal_vals = np.array(sorted({r["cal_sig"] for r in records if np.isfinite(r["cal_sig"])}), dtype=float)
    if fr_vals.size == 0 or cal_vals.size == 0:
        return None
    fr_used = float(fr_vals[np.argmin(np.abs(fr_vals - float(sigma_target)))])
    cal_used = float(cal_vals[np.argmin(np.abs(cal_vals - float(sigma_target)))])

    exact = [r for r in records if np.isfinite(r["fr_sig"]) and np.isfinite(r["cal_sig"]) and abs(r["fr_sig"] - fr_used) < 1e-12 and abs(r["cal_sig"] - cal_used) < 1e-12]
    if len(exact) > 0:
        return exact[0]

    return _pick_record_near_sigma(records, fr_used, cal_used)


def _choose_fixed_lag_from_nonsmoothed(records):
    base = _pick_record_near_sigma(records, 0.0, 0.0)
    if base is None:
        return np.nan
    x = base.get("fr_on_ca", np.array([], float))
    y = base.get("ca_used", np.array([], float))
    fs = _to_float_nan(base.get("fs_ca", np.nan))
    if x.size < 5 or y.size < 5 or not np.isfinite(fs) or fs <= 0:
        return np.nan

    # same lag for all smoothing per cell, chosen from NON-smoothed trace
    _best_r, best_lag_s = _lag_scan_best_r(x, y, fs=fs, min_lag_s=0.0, max_lag_s=0.5, min_overlap_s=1.0, mode="max_pos")
    return best_lag_s


def _add_linear_fit_with_stats(fig, x, y, row, col, line_color="rgba(0,0,0,0.7)"):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 3:
        return
    xv = x[ok]
    yv = y[ok]

    lr = linregress(xv, yv)
    slope = float(lr.slope)
    intercept = float(lr.intercept)
    r = float(lr.rvalue)
    p = float(lr.pvalue)

    xmin = float(np.nanmin(xv))
    xmax = float(np.nanmax(xv))
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return
    if xmax <= xmin:
        xx = np.array([xmin, xmin + 1.0])
    else:
        xx = np.linspace(xmin, xmax, 120)
    yy = slope * xx + intercept

    fig.add_trace(
        go.Scatter(
            x=xx,
            y=yy,
            mode="lines",
            line=dict(color=line_color, width=2),
            showlegend=False,
            hovertemplate="linear fit<extra></extra>",
        ),
        row=row,
        col=col,
    )

    ymin = float(np.nanmin(yv))
    ymax = float(np.nanmax(yv))
    dx = xmax - xmin
    dy = ymax - ymin
    xt = xmin + (0.98 * dx if dx > 0 else 0.0)
    yt = ymin + (0.04 * dy if dy > 0 else 0.0)
    txt = f"r={r:.3f}<br>p={p:.2e}<br>m={slope:.3f}<br>b={intercept:.3f}"

    fig.add_trace(
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


def _build_summary_dataframe(path_list, sigma_target=0.10):
    rows = []
    for folder in path_list:
        recs = _load_corrdict_full_records(str(folder))
        if len(recs) == 0:
            continue

        basic = _pick_basic_record(recs, sigma_target=sigma_target)

        # Zero-lag optimal over smoothing
        zero_candidates = [r for r in recs if np.isfinite(r.get("pearson_r_0lag", np.nan))]
        opt0 = max(zero_candidates, key=lambda r: r["pearson_r_0lag"]) if len(zero_candidates) else None

        # Fixed-lag optimal over smoothing (lag chosen once from non-smoothed trace)
        lag_s_fixed = _choose_fixed_lag_from_nonsmoothed(recs)
        for r in recs:
            rr_fix = _corr_with_fixed_lag(r.get("fr_on_ca", np.array([], float)), r.get("ca_used", np.array([], float)), r.get("fs_ca", np.nan), lag_s_fixed)
            r["pearson_r_lagfixed"] = rr_fix

        lag_candidates = [r for r in recs if np.isfinite(r.get("pearson_r_lagfixed", np.nan))]
        optL = max(lag_candidates, key=lambda r: r["pearson_r_lagfixed"]) if len(lag_candidates) else None

        basic0 = np.nan if basic is None else _to_float_nan(basic.get("pearson_r_0lag"))
        basicL = np.nan
        if basic is not None:
            basicL = _to_float_nan(basic.get("pearson_r_lagfixed"))

        rows.append({
            "folder": str(folder),
            "cell_id": _derive_cell_id_from_folder(folder),
            "population": _infer_population_from_folder(folder),
            "basic_mean_fr_hz": (np.nan if basic is None else _to_float_nan(basic.get("mean_fr_hz"))),
            "basic_std_fr_hz": (np.nan if basic is None else _to_float_nan(basic.get("std_fr_hz"))),
            "basic_fr_sigma_used": (np.nan if basic is None else _to_float_nan(basic.get("fr_sig"))),
            "basic_cal_sigma_used": (np.nan if basic is None else _to_float_nan(basic.get("cal_sig"))),

            "basic_corr_r_zero": basic0,
            "optimal_corr_r_zero": (np.nan if opt0 is None else _to_float_nan(opt0.get("pearson_r_0lag"))),
            "optimal_fr_sigma_zero": (np.nan if opt0 is None else _to_float_nan(opt0.get("fr_sig"))),
            "optimal_cal_sigma_zero": (np.nan if opt0 is None else _to_float_nan(opt0.get("cal_sig"))),

            "fixed_lag_s": _to_float_nan(lag_s_fixed),
            "basic_corr_r_lagfixed": basicL,
            "optimal_corr_r_lagfixed": (np.nan if optL is None else _to_float_nan(optL.get("pearson_r_lagfixed"))),
            "optimal_fr_sigma_lagfixed": (np.nan if optL is None else _to_float_nan(optL.get("fr_sig"))),
            "optimal_cal_sigma_lagfixed": (np.nan if optL is None else _to_float_nan(optL.get("cal_sig"))),
        })

    return pd.DataFrame(rows)


def _build_summary_figure(df, corr_col_basic, corr_col_opt, opt_fr_sig_col, opt_cal_sig_col, title_suffix, include_lag_extras=False):
    # Force near-square panels by deriving figure size from fixed panel size.
    panel_px = 420
    hspace = 0.07
    vspace = 0.10
    fig_w = int(3 * panel_px + 260)
    fig_h_4 = int(4 * panel_px + 320)
    fig_h_3 = int(3 * panel_px + 320)

    if bool(include_lag_extras):
        fig = make_subplots(
            rows=4,
            cols=3,
            row_heights=[1.0, 1.0, 1.0, 1.0],
            column_widths=[1.0, 1.0, 1.0],
            horizontal_spacing=hspace,
            vertical_spacing=vspace,
            subplot_titles=(
                "1) Basic correlation histogram (fr/cal sigma~0)",
                "2) Optimal correlation histogram (best smooth)",
                "3) Basic correlation vs mean FR",
                "4) Basic correlation vs FR STD",
                "5) Optimal sigmas vs correlation",
                "6) Optimal calcium sigma vs FR STD",
                "7) Optimal correlation vs mean FR",
                "8) Optimal correlation vs FR STD",
                "",
                "9) Chosen lag histogram (fixed lag)",
                "10) Basic correlation vs chosen lag",
                "",
            ),
        )
    else:
        fig = make_subplots(
            rows=3,
            cols=3,
            row_heights=[1.0, 1.0, 1.0],
            column_widths=[1.0, 1.0, 1.0],
            horizontal_spacing=hspace,
            vertical_spacing=vspace,
            subplot_titles=(
                "1) Basic correlation histogram (fr/cal sigma~0)",
                "2) Optimal correlation histogram (best smooth)",
                "3) Basic correlation vs mean FR",
                "4) Basic correlation vs FR STD",
                "5) Optimal sigmas vs correlation",
                "6) Optimal calcium sigma vs FR STD",
                "7) Optimal correlation vs mean FR",
                "8) Optimal correlation vs FR STD",
                "",
            ),
        )

    # 1) Histogram basic corr
    x1 = df[corr_col_basic].to_numpy(float)
    fig.add_trace(
        go.Histogram(x=x1[np.isfinite(x1)], marker=dict(color="rgba(0,0,0,0.55)", line=dict(color="rgba(0,0,0,1)", width=1)), showlegend=False),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Pearson r", row=1, col=1)
    fig.update_yaxes(title_text="# cells", row=1, col=1)

    # 2) Histogram optimal corr
    x2 = df[corr_col_opt].to_numpy(float)
    fig.add_trace(
        go.Histogram(x=x2[np.isfinite(x2)], marker=dict(color="rgba(0,0,0,0.55)", line=dict(color="rgba(0,0,0,1)", width=1)), showlegend=False),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Pearson r", row=1, col=2)
    fig.update_yaxes(title_text="# cells", row=1, col=2)

    # 3) basic corr vs mean FR
    mask3 = np.isfinite(df[corr_col_basic].to_numpy(float)) & np.isfinite(df["basic_mean_fr_hz"].to_numpy(float))
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask3, "basic_mean_fr_hz"],
            y=df.loc[mask3, corr_col_basic],
            mode="markers",
            marker=dict(color="black", size=7, opacity=0.8),
            text=df.loc[mask3, "folder"],
            hovertemplate="%{text}<br>mean FR=%{x:.3f} Hz<br>r=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="mean FR (Hz)", row=1, col=3)
    fig.update_yaxes(title_text="basic correlation (r)", row=1, col=3)
    _add_linear_fit_with_stats(fig, df.loc[mask3, "basic_mean_fr_hz"].to_numpy(float), df.loc[mask3, corr_col_basic].to_numpy(float), row=1, col=3)

    # 4) basic corr vs FR STD
    mask4 = np.isfinite(df[corr_col_basic].to_numpy(float)) & np.isfinite(df["basic_std_fr_hz"].to_numpy(float))
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask4, "basic_std_fr_hz"],
            y=df.loc[mask4, corr_col_basic],
            mode="markers",
            marker=dict(color="black", size=7, opacity=0.8),
            text=df.loc[mask4, "folder"],
            hovertemplate="%{text}<br>FR std=%{x:.3f} Hz<br>r=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="FR STD (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="basic correlation (r)", row=2, col=1)
    _add_linear_fit_with_stats(fig, df.loc[mask4, "basic_std_fr_hz"].to_numpy(float), df.loc[mask4, corr_col_basic].to_numpy(float), row=2, col=1)

    # 5) optimal sigmas vs optimal correlation (flipped axes)
    mask5a = np.isfinite(df[corr_col_opt].to_numpy(float)) & np.isfinite(df[opt_cal_sig_col].to_numpy(float))
    mask5b = np.isfinite(df[corr_col_opt].to_numpy(float)) & np.isfinite(df[opt_fr_sig_col].to_numpy(float))

    fig.add_trace(
        go.Scatter(
            x=df.loc[mask5a, opt_cal_sig_col],
            y=df.loc[mask5a, corr_col_opt],
            mode="markers",
            marker=dict(color="purple", size=7, opacity=0.8),
            name="optimal cal sigma",
            text=df.loc[mask5a, "folder"],
            hovertemplate="%{text}<br>optimal cal sigma=%{x:.3f} s<br>optimal r=%{y:.3f}<extra></extra>",
            showlegend=True,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask5b, opt_fr_sig_col],
            y=df.loc[mask5b, corr_col_opt],
            mode="markers",
            marker=dict(color="red", size=7, opacity=0.8),
            name="optimal voltage(FR) sigma",
            text=df.loc[mask5b, "folder"],
            hovertemplate="%{text}<br>optimal voltage sigma=%{x:.3f} s<br>optimal r=%{y:.3f}<extra></extra>",
            showlegend=True,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="sigma (s)", row=2, col=2)
    fig.update_yaxes(title_text="optimal correlation (r)", row=2, col=2)

    # 6) FR STD vs optimal sigmas (flipped axes; includes cal+FR sigma)
    mask6a = np.isfinite(df[opt_cal_sig_col].to_numpy(float)) & np.isfinite(df["basic_std_fr_hz"].to_numpy(float))
    mask6b = np.isfinite(df[opt_fr_sig_col].to_numpy(float)) & np.isfinite(df["basic_std_fr_hz"].to_numpy(float))
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask6a, opt_cal_sig_col],
            y=df.loc[mask6a, "basic_std_fr_hz"],
            mode="markers",
            marker=dict(color="purple", size=7, opacity=0.8),
            text=df.loc[mask6a, "folder"],
            hovertemplate="%{text}<br>optimal cal sigma=%{x:.3f} s<br>FR std=%{y:.3f} Hz<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask6b, opt_fr_sig_col],
            y=df.loc[mask6b, "basic_std_fr_hz"],
            mode="markers",
            marker=dict(color="red", size=7, opacity=0.8),
            text=df.loc[mask6b, "folder"],
            hovertemplate="%{text}<br>optimal voltage sigma=%{x:.3f} s<br>FR std=%{y:.3f} Hz<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.update_xaxes(title_text="sigma (s)", row=2, col=3)
    fig.update_yaxes(title_text="FR STD (Hz)", row=2, col=3)

    # 7) optimal correlation vs mean FR
    mask7 = np.isfinite(df[corr_col_opt].to_numpy(float)) & np.isfinite(df["basic_mean_fr_hz"].to_numpy(float))
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask7, "basic_mean_fr_hz"],
            y=df.loc[mask7, corr_col_opt],
            mode="markers",
            marker=dict(color="black", size=7, opacity=0.8),
            text=df.loc[mask7, "folder"],
            hovertemplate="%{text}<br>mean FR=%{x:.3f} Hz<br>optimal r=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_xaxes(title_text="mean FR (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="optimal correlation (r)", row=3, col=1)
    _add_linear_fit_with_stats(fig, df.loc[mask7, "basic_mean_fr_hz"].to_numpy(float), df.loc[mask7, corr_col_opt].to_numpy(float), row=3, col=1)

    # 8) optimal correlation vs FR STD
    mask8 = np.isfinite(df[corr_col_opt].to_numpy(float)) & np.isfinite(df["basic_std_fr_hz"].to_numpy(float))
    fig.add_trace(
        go.Scatter(
            x=df.loc[mask8, "basic_std_fr_hz"],
            y=df.loc[mask8, corr_col_opt],
            mode="markers",
            marker=dict(color="black", size=7, opacity=0.8),
            text=df.loc[mask8, "folder"],
            hovertemplate="%{text}<br>FR std=%{x:.3f} Hz<br>optimal r=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    fig.update_xaxes(title_text="FR STD (Hz)", row=3, col=2)
    fig.update_yaxes(title_text="optimal correlation (r)", row=3, col=2)
    _add_linear_fit_with_stats(fig, df.loc[mask8, "basic_std_fr_hz"].to_numpy(float), df.loc[mask8, corr_col_opt].to_numpy(float), row=3, col=2)

    if bool(include_lag_extras):
        # 9) Histogram of chosen fixed lag
        lag_ms = 1000.0 * pd.to_numeric(df.get("fixed_lag_s", np.nan), errors="coerce").to_numpy(float)
        fig.add_trace(
            go.Histogram(x=lag_ms[np.isfinite(lag_ms)], marker=dict(color="rgba(0,0,0,0.55)", line=dict(color="rgba(0,0,0,1)", width=1)), showlegend=False),
            row=4,
            col=1,
        )
        fig.update_xaxes(title_text="chosen lag (ms)", row=4, col=1)
        fig.update_yaxes(title_text="# cells", row=4, col=1)

        # 10) Basic correlation as function of chosen lag correction
        xlag = 1000.0 * pd.to_numeric(df.get("fixed_lag_s", np.nan), errors="coerce").to_numpy(float)
        ybasic = pd.to_numeric(df.get("basic_corr_r_zero", np.nan), errors="coerce").to_numpy(float)
        m10 = np.isfinite(xlag) & np.isfinite(ybasic)
        fig.add_trace(
            go.Scatter(
                x=xlag[m10],
                y=ybasic[m10],
                mode="markers",
                marker=dict(color="black", size=7, opacity=0.8),
                text=df.loc[m10, "folder"],
                hovertemplate="%{text}<br>chosen lag=%{x:.1f} ms<br>basic r=%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=4,
            col=2,
        )
        fig.update_xaxes(title_text="chosen lag (ms)", row=4, col=2)
        fig.update_yaxes(title_text="basic correlation (r, no smoothing)", row=4, col=2)
        _add_linear_fit_with_stats(fig, xlag[m10], ybasic[m10], row=4, col=2)

        fig.update_xaxes(visible=False, row=3, col=3)
        fig.update_yaxes(visible=False, row=3, col=3)
        fig.update_xaxes(visible=False, row=4, col=3)
        fig.update_yaxes(visible=False, row=4, col=3)

        fig.update_layout(
            template="plotly_white",
            width=fig_w,
            height=fig_h_4,
            title=f"SST correlation summary | {title_suffix} | basic sigma target={SIGMA_BASIC_TARGET:.2f}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.55),
        )
    else:
        # keep panel 9 empty
        fig.update_xaxes(visible=False, row=3, col=3)
        fig.update_yaxes(visible=False, row=3, col=3)

        fig.update_layout(
            template="plotly_white",
            width=fig_w,
            height=fig_h_3,
            title=f"SST correlation summary | {title_suffix} | basic sigma target={SIGMA_BASIC_TARGET:.2f}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.55),
        )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    return fig


def _build_summary_figure_by_population(df, corr_col_basic, corr_col_opt, opt_fr_sig_col, opt_cal_sig_col, title_suffix, include_lag_extras=True):
    panel_px = 420
    hspace = 0.07
    vspace = 0.10
    fig_w = int(3 * panel_px + 260)
    fig_h_4 = int(4 * panel_px + 320)
    fig_h_3 = int(3 * panel_px + 320)

    pop_order = ["anesthetized", "awake", "motor"]
    pop_color = {
        "anesthetized": "rgba(31,119,180,0.30)",
        "awake": "rgba(44,160,44,0.30)",
        "motor": "rgba(255,127,14,0.30)",
    }
    pop_line = {
        "anesthetized": "rgba(31,119,180,0.95)",
        "awake": "rgba(44,160,44,0.95)",
        "motor": "rgba(255,127,14,0.95)",
    }

    if bool(include_lag_extras):
        fig = make_subplots(
            rows=4,
            cols=3,
            row_heights=[1.0, 1.0, 1.0, 1.0],
            column_widths=[1.0, 1.0, 1.0],
            horizontal_spacing=hspace,
            vertical_spacing=vspace,
            subplot_titles=(
                "1) Basic correlation box+points by population",
                "2) Optimal correlation box+points by population",
                "3) Basic correlation vs mean FR",
                "4) Basic correlation vs FR STD",
                "5) Optimal sigmas vs correlation",
                "6) Optimal sigmas vs FR STD",
                "7) Optimal correlation vs mean FR",
                "8) Optimal correlation vs FR STD",
                "",
                "9) Chosen lag box+points by population",
                "10) Basic correlation vs chosen lag",
                "",
            ),
        )
    else:
        fig = make_subplots(
            rows=3,
            cols=3,
            row_heights=[1.0, 1.0, 1.0],
            column_widths=[1.0, 1.0, 1.0],
            horizontal_spacing=hspace,
            vertical_spacing=vspace,
            subplot_titles=(
                "1) Basic correlation box+points by population",
                "2) Optimal correlation box+points by population",
                "3) Basic correlation vs mean FR",
                "4) Basic correlation vs FR STD",
                "5) Optimal sigmas vs correlation",
                "6) Optimal sigmas vs FR STD",
                "7) Optimal correlation vs mean FR",
                "8) Optimal correlation vs FR STD",
                "",
            ),
        )

    df2 = df.copy()
    if "population" not in df2.columns:
        df2["population"] = "unknown"
    df2["population"] = df2["population"].astype(str).str.lower().str.strip()
    if "cell_id" not in df2.columns:
        df2["cell_id"] = df2["folder"].astype(str)
    df2["cell_id"] = df2["cell_id"].astype(str).str.lower().str.strip()

    # For group-comparison boxplots:
    # r per recording -> Fisher z -> average z per cell per state -> back to r by tanh(z)
    box_basic_df = _build_cell_state_fisher_table(df2, corr_col_basic)
    box_opt_df = _build_cell_state_fisher_table(df2, corr_col_opt)

    welch_basic = _welch_anova_on_groups_z(box_basic_df)
    welch_opt = _welch_anova_on_groups_z(box_opt_df)
    gh_basic = _games_howell_on_groups_z(box_basic_df, pop_order=pop_order)
    gh_opt = _games_howell_on_groups_z(box_opt_df, pop_order=pop_order)

    def _add_pop_box_from_fisher(box_df, row, col, show_legend=False):
        for pop in pop_order:
            m = (box_df["population"].to_numpy(str) == pop)
            vv = pd.to_numeric(box_df.loc[m, "r_from_mean_z"], errors="coerce").to_numpy(float)
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=vv,
                    x=[pop] * int(vv.size),
                    name=pop,
                    legendgroup=pop,
                    showlegend=bool(show_legend),
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.46,
                    width=0.52,
                    boxmean=True,
                    marker=dict(
                        color=pop_line[pop],
                        size=6,
                        opacity=0.78,
                        line=dict(color="rgba(0,0,0,0.45)", width=0.5),
                    ),
                    line=dict(color=pop_line[pop], width=1.6),
                    fillcolor=pop_color[pop],
                    opacity=1.0,
                ),
                row=row,
                col=col,
            )

    def _add_pop_box_raw(colname, row, col, show_legend=False):
        for pop in pop_order:
            m = (df2["population"].to_numpy(str) == pop)
            vv = pd.to_numeric(df2.loc[m, colname], errors="coerce").to_numpy(float)
            vv = vv[np.isfinite(vv)]
            if vv.size == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=vv,
                    x=[pop] * int(vv.size),
                    name=pop,
                    legendgroup=pop,
                    showlegend=bool(show_legend),
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.46,
                    width=0.52,
                    boxmean=True,
                    marker=dict(
                        color=pop_line[pop],
                        size=6,
                        opacity=0.78,
                        line=dict(color="rgba(0,0,0,0.45)", width=0.5),
                    ),
                    line=dict(color=pop_line[pop], width=1.6),
                    fillcolor=pop_color[pop],
                    opacity=1.0,
                ),
                row=row,
                col=col,
            )

    def _add_pop_scatter(xcol, ycol, row, col, symbols=None):
        for pop in pop_order:
            m = (df2["population"].to_numpy(str) == pop)
            xx = pd.to_numeric(df2.loc[m, xcol], errors="coerce").to_numpy(float)
            yy = pd.to_numeric(df2.loc[m, ycol], errors="coerce").to_numpy(float)
            ok = np.isfinite(xx) & np.isfinite(yy)
            if np.sum(ok) == 0:
                continue
            marker = dict(color=pop_line[pop], size=7, opacity=0.82)
            if symbols is not None:
                marker["symbol"] = symbols
            fig.add_trace(
                go.Scatter(
                    x=xx[ok],
                    y=yy[ok],
                    mode="markers",
                    marker=marker,
                    text=df2.loc[m, "folder"].astype(str).to_numpy()[ok],
                    hovertemplate="%{text}<br>x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
                    name=pop,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    _add_pop_box_from_fisher(box_basic_df, 1, 1, show_legend=True)
    _add_pop_box_from_fisher(box_opt_df, 1, 2, show_legend=False)
    fig.update_xaxes(title_text="population", row=1, col=1)
    fig.update_yaxes(title_text="Pearson r", row=1, col=1)
    fig.update_xaxes(title_text="population", row=1, col=2)
    fig.update_yaxes(title_text="Pearson r", row=1, col=2)
    _add_sig_brackets_to_boxplot(fig, 1, 1, gh_basic, box_basic_df["r_from_mean_z"].to_numpy(float))
    _add_sig_brackets_to_boxplot(fig, 1, 2, gh_opt, box_opt_df["r_from_mean_z"].to_numpy(float))

    _add_pop_scatter("basic_mean_fr_hz", corr_col_basic, 1, 3)
    fig.update_xaxes(title_text="mean FR (Hz)", row=1, col=3)
    fig.update_yaxes(title_text="basic correlation (r)", row=1, col=3)
    _add_linear_fit_with_stats(fig, pd.to_numeric(df2.get("basic_mean_fr_hz", np.nan), errors="coerce").to_numpy(float), pd.to_numeric(df2.get(corr_col_basic, np.nan), errors="coerce").to_numpy(float), row=1, col=3)

    _add_pop_scatter("basic_std_fr_hz", corr_col_basic, 2, 1)
    fig.update_xaxes(title_text="FR STD (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="basic correlation (r)", row=2, col=1)
    _add_linear_fit_with_stats(fig, pd.to_numeric(df2.get("basic_std_fr_hz", np.nan), errors="coerce").to_numpy(float), pd.to_numeric(df2.get(corr_col_basic, np.nan), errors="coerce").to_numpy(float), row=2, col=1)

    # panel 5: sigma vs correlation, colors=population, symbol indicates sigma-type
    for pop in pop_order:
        m = (df2["population"].to_numpy(str) == pop)
        x1 = pd.to_numeric(df2.loc[m, opt_cal_sig_col], errors="coerce").to_numpy(float)
        y1 = pd.to_numeric(df2.loc[m, corr_col_opt], errors="coerce").to_numpy(float)
        o1 = np.isfinite(x1) & np.isfinite(y1)
        if np.sum(o1) > 0:
            fig.add_trace(go.Scatter(x=x1[o1], y=y1[o1], mode="markers", marker=dict(color=pop_line[pop], size=7, opacity=0.82, symbol="circle"), name=pop, showlegend=False), row=2, col=2)
        x2 = pd.to_numeric(df2.loc[m, opt_fr_sig_col], errors="coerce").to_numpy(float)
        y2 = pd.to_numeric(df2.loc[m, corr_col_opt], errors="coerce").to_numpy(float)
        o2 = np.isfinite(x2) & np.isfinite(y2)
        if np.sum(o2) > 0:
            fig.add_trace(go.Scatter(x=x2[o2], y=y2[o2], mode="markers", marker=dict(color=pop_line[pop], size=8, opacity=0.82, symbol="diamond"), name=pop, showlegend=False), row=2, col=2)
    fig.update_xaxes(title_text="sigma (s)", row=2, col=2)
    fig.update_yaxes(title_text="optimal correlation (r)", row=2, col=2)
    _add_linear_fit_with_stats(
        fig,
        pd.to_numeric(df2.get(opt_cal_sig_col, np.nan), errors="coerce").to_numpy(float),
        pd.to_numeric(df2.get(corr_col_opt, np.nan), errors="coerce").to_numpy(float),
        row=2,
        col=2,
        line_color="rgba(128,0,128,0.65)",
    )
    _add_linear_fit_with_stats(
        fig,
        pd.to_numeric(df2.get(opt_fr_sig_col, np.nan), errors="coerce").to_numpy(float),
        pd.to_numeric(df2.get(corr_col_opt, np.nan), errors="coerce").to_numpy(float),
        row=2,
        col=2,
        line_color="rgba(220,20,60,0.65)",
    )

    # panel 6: sigma vs FR STD, colors=population, symbol indicates sigma-type
    for pop in pop_order:
        m = (df2["population"].to_numpy(str) == pop)
        ystd = pd.to_numeric(df2.loc[m, "basic_std_fr_hz"], errors="coerce").to_numpy(float)
        x1 = pd.to_numeric(df2.loc[m, opt_cal_sig_col], errors="coerce").to_numpy(float)
        o1 = np.isfinite(x1) & np.isfinite(ystd)
        if np.sum(o1) > 0:
            fig.add_trace(go.Scatter(x=x1[o1], y=ystd[o1], mode="markers", marker=dict(color=pop_line[pop], size=7, opacity=0.82, symbol="circle"), name=pop, showlegend=False), row=2, col=3)
        x2 = pd.to_numeric(df2.loc[m, opt_fr_sig_col], errors="coerce").to_numpy(float)
        o2 = np.isfinite(x2) & np.isfinite(ystd)
        if np.sum(o2) > 0:
            fig.add_trace(go.Scatter(x=x2[o2], y=ystd[o2], mode="markers", marker=dict(color=pop_line[pop], size=8, opacity=0.82, symbol="diamond"), name=pop, showlegend=False), row=2, col=3)
    fig.update_xaxes(title_text="sigma (s)", row=2, col=3)
    fig.update_yaxes(title_text="FR STD (Hz)", row=2, col=3)
    _add_linear_fit_with_stats(
        fig,
        pd.to_numeric(df2.get(opt_cal_sig_col, np.nan), errors="coerce").to_numpy(float),
        pd.to_numeric(df2.get("basic_std_fr_hz", np.nan), errors="coerce").to_numpy(float),
        row=2,
        col=3,
        line_color="rgba(128,0,128,0.65)",
    )
    _add_linear_fit_with_stats(
        fig,
        pd.to_numeric(df2.get(opt_fr_sig_col, np.nan), errors="coerce").to_numpy(float),
        pd.to_numeric(df2.get("basic_std_fr_hz", np.nan), errors="coerce").to_numpy(float),
        row=2,
        col=3,
        line_color="rgba(220,20,60,0.65)",
    )

    _add_pop_scatter("basic_mean_fr_hz", corr_col_opt, 3, 1)
    fig.update_xaxes(title_text="mean FR (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="optimal correlation (r)", row=3, col=1)
    _add_linear_fit_with_stats(fig, pd.to_numeric(df2.get("basic_mean_fr_hz", np.nan), errors="coerce").to_numpy(float), pd.to_numeric(df2.get(corr_col_opt, np.nan), errors="coerce").to_numpy(float), row=3, col=1)

    _add_pop_scatter("basic_std_fr_hz", corr_col_opt, 3, 2)
    fig.update_xaxes(title_text="FR STD (Hz)", row=3, col=2)
    fig.update_yaxes(title_text="optimal correlation (r)", row=3, col=2)
    _add_linear_fit_with_stats(fig, pd.to_numeric(df2.get("basic_std_fr_hz", np.nan), errors="coerce").to_numpy(float), pd.to_numeric(df2.get(corr_col_opt, np.nan), errors="coerce").to_numpy(float), row=3, col=2)

    if bool(include_lag_extras):
        _add_pop_box_raw("fixed_lag_s", 4, 1, show_legend=False)
        fig.update_xaxes(title_text="population", row=4, col=1)
        fig.update_yaxes(title_text="chosen lag (s)", row=4, col=1)

        _add_pop_scatter("fixed_lag_s", "basic_corr_r_zero", 4, 2)
        fig.update_xaxes(title_text="chosen lag (s)", row=4, col=2)
        fig.update_yaxes(title_text="basic correlation (r, no smoothing)", row=4, col=2)
        _add_linear_fit_with_stats(fig, pd.to_numeric(df2.get("fixed_lag_s", np.nan), errors="coerce").to_numpy(float), pd.to_numeric(df2.get("basic_corr_r_zero", np.nan), errors="coerce").to_numpy(float), row=4, col=2)

        fig.update_xaxes(visible=False, row=3, col=3)
        fig.update_yaxes(visible=False, row=3, col=3)
        fig.update_xaxes(visible=False, row=4, col=3)
        fig.update_yaxes(visible=False, row=4, col=3)

        fig.update_layout(
            template="plotly_white",
            width=fig_w,
            height=fig_h_4,
            title=f"SST correlation summary by population | {title_suffix} | basic sigma target={SIGMA_BASIC_TARGET:.2f}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        )
    else:
        fig.update_xaxes(visible=False, row=3, col=3)
        fig.update_yaxes(visible=False, row=3, col=3)
        fig.update_layout(
            template="plotly_white",
            width=fig_w,
            height=fig_h_3,
            title=f"SST correlation summary by population | {title_suffix} | basic sigma target={SIGMA_BASIC_TARGET:.2f}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    return fig


def _fisher_z_from_r(r):
    rr = np.asarray(r, dtype=float)
    rr = np.clip(rr, -0.999999, 0.999999)
    return np.arctanh(rr)


def _ols_r2(y, X):
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.size == 0 or X.size == 0 or X.shape[0] != y.size:
        return np.nan, 0
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    yv = y[ok]
    Xv = X[ok]
    n = int(yv.size)
    if n < max(6, Xv.shape[1] + 2):
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
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2), n


def _build_partial_r2_table(df, target_col, opt_fr_col, opt_cal_col):
    d = df.copy()
    if "population" not in d.columns:
        d["population"] = "unknown"
    d["population"] = d["population"].astype(str).str.lower().str.strip()

    # Use Fisher-z of correlation as response for more stable linear modeling.
    y_raw = pd.to_numeric(d.get(target_col, np.nan), errors="coerce").to_numpy(float)
    y = _fisher_z_from_r(y_raw)

    x_mean = pd.to_numeric(d.get("basic_mean_fr_hz", np.nan), errors="coerce").to_numpy(float)
    x_std = pd.to_numeric(d.get("basic_std_fr_hz", np.nan), errors="coerce").to_numpy(float)
    x_opt_fr = pd.to_numeric(d.get(opt_fr_col, np.nan), errors="coerce").to_numpy(float)
    x_opt_cal = pd.to_numeric(d.get(opt_cal_col, np.nan), errors="coerce").to_numpy(float)

    pop_dum = pd.get_dummies(d["population"], prefix="pop", drop_first=True)
    pop_cols = list(pop_dum.columns)
    pop_mat = pop_dum.to_numpy(dtype=float) if len(pop_cols) else np.empty((len(d), 0), dtype=float)

    groups = {
        "mean_FR": x_mean.reshape(-1, 1),
        "FR_STD": x_std.reshape(-1, 1),
        "opt_FR_sigma": x_opt_fr.reshape(-1, 1),
        "opt_Ca_sigma": x_opt_cal.reshape(-1, 1),
        "population": pop_mat,
    }

    # Full design matrix
    X_parts = [groups[k] for k in ["mean_FR", "FR_STD", "opt_FR_sigma", "opt_Ca_sigma", "population"] if groups[k].size > 0]
    X_full = np.column_stack(X_parts) if len(X_parts) else np.empty((len(d), 0), dtype=float)
    full_r2, n_used = _ols_r2(y, X_full)

    rows = []
    for gname in ["mean_FR", "FR_STD", "opt_FR_sigma", "opt_Ca_sigma", "population"]:
        part = groups[gname]
        if part.size == 0:
            continue
        X_reduced_parts = [groups[k] for k in ["mean_FR", "FR_STD", "opt_FR_sigma", "opt_Ca_sigma", "population"] if (k != gname and groups[k].size > 0)]
        X_red = np.column_stack(X_reduced_parts) if len(X_reduced_parts) else np.empty((len(d), 0), dtype=float)
        red_r2, _ = _ols_r2(y, X_red)
        delta = np.nan
        share = np.nan
        if np.isfinite(full_r2) and np.isfinite(red_r2):
            delta = float(full_r2 - red_r2)
            if abs(full_r2) > 1e-12:
                share = float(delta / full_r2)
        rows.append(
            {
                "target": str(target_col),
                "predictor_group": str(gname),
                "n_used": int(n_used),
                "full_r2": float(full_r2) if np.isfinite(full_r2) else np.nan,
                "reduced_r2": float(red_r2) if np.isfinite(red_r2) else np.nan,
                "delta_r2": float(delta) if np.isfinite(delta) else np.nan,
                "delta_r2_share": float(share) if np.isfinite(share) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_partial_r2_summary(ev_df, title):
    d = ev_df.copy()
    if len(d) == 0:
        return go.Figure()

    target_order = ["optimal_corr_r_zero", "optimal_corr_r_lagfixed"]
    label_map = {
        "optimal_corr_r_zero": "zero-lag target",
        "optimal_corr_r_lagfixed": "fixed-lag target",
    }
    color_map = {
        "optimal_corr_r_zero": "#4C78A8",
        "optimal_corr_r_lagfixed": "#F58518",
    }

    xcats = ["mean_FR", "FR_STD", "opt_FR_sigma", "opt_Ca_sigma", "population"]
    fig = go.Figure()
    for t in target_order:
        sub = d[d["target"].astype(str) == t].copy()
        if len(sub) == 0:
            continue
        yvals = []
        for x in xcats:
            rr = pd.to_numeric(sub.loc[sub["predictor_group"] == x, "delta_r2"], errors="coerce").to_numpy(float)
            yvals.append(float(rr[0]) if rr.size else np.nan)
        fig.add_trace(
            go.Bar(
                x=xcats,
                y=yvals,
                name=label_map.get(t, t),
                marker=dict(color=color_map.get(t, "#777777")),
            )
        )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=560,
        barmode="group",
        title=title,
        xaxis_title="Predictor group removed from full model",
        yaxis_title="Drop in R? (full - reduced)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False)
    return fig


def _p_to_stars(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _build_cell_state_fisher_table(df, r_col):
    d = df.copy()
    if "cell_id" not in d.columns:
        d["cell_id"] = d["folder"].astype(str)
    if "population" not in d.columns:
        d["population"] = "unknown"
    d["population"] = d["population"].astype(str).str.lower().str.strip()
    d["cell_id"] = d["cell_id"].astype(str).str.lower().str.strip()
    r = pd.to_numeric(d.get(r_col, np.nan), errors="coerce").to_numpy(float)
    ok = np.isfinite(r)
    d = d.loc[ok, ["cell_id", "population"]].copy()
    d["r"] = r[ok]
    d["z"] = _fisher_z_from_r(d["r"].to_numpy(float))

    # average Fisher-z per cell per brain-state, then map back to r for plotting
    g = d.groupby(["cell_id", "population"], as_index=False)["z"].mean()
    g["r_from_mean_z"] = np.tanh(pd.to_numeric(g["z"], errors="coerce").to_numpy(float))
    return g


def _welch_anova_on_groups_z(cell_state_z_df):
    tmp = cell_state_z_df.copy()
    groups = {}
    for pop, sub in tmp.groupby("population"):
        vv = pd.to_numeric(sub["z"], errors="coerce").to_numpy(float)
        vv = vv[np.isfinite(vv)]
        if vv.size >= 2:
            groups[str(pop)] = vv
    if len(groups) < 2:
        return {"status": "failed", "reason": "need>=2 groups with n>=2", "result": None, "groups": groups}
    try:
        res = anova_oneway(list(groups.values()), use_var="unequal")
        return {"status": "ok", "result": res, "groups": groups}
    except Exception as e:
        return {"status": "failed", "reason": str(e), "result": None, "groups": groups}


def _games_howell_on_groups_z(cell_state_z_df, pop_order=None):
    tmp = cell_state_z_df.copy()
    groups = {}
    for pop, sub in tmp.groupby("population"):
        vv = pd.to_numeric(sub["z"], errors="coerce").to_numpy(float)
        vv = vv[np.isfinite(vv)]
        if vv.size >= 2:
            groups[str(pop)] = vv
    pops = [p for p in (pop_order or sorted(groups.keys())) if p in groups]
    k = len(groups)
    out = []
    if k < 2:
        return pd.DataFrame(out)

    for i in range(len(pops)):
        for j in range(i + 1, len(pops)):
            p1 = pops[i]
            p2 = pops[j]
            x = np.asarray(groups[p1], dtype=float)
            y = np.asarray(groups[p2], dtype=float)
            n1, n2 = x.size, y.size
            if n1 < 2 or n2 < 2:
                continue
            m1, m2 = float(np.mean(x)), float(np.mean(y))
            v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
            se2 = (v1 / n1) + (v2 / n2)
            if not np.isfinite(se2) or se2 <= 0:
                continue
            t = abs(m1 - m2) / np.sqrt(se2)
            # Games-Howell p uses studentized range: q = sqrt(2)*|t|
            q = np.sqrt(2.0) * float(t)
            denom = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
            if denom <= 0 or (not np.isfinite(denom)):
                df_ij = np.nan
            else:
                df_ij = (se2 ** 2) / denom
            pval = np.nan
            if np.isfinite(df_ij) and df_ij > 0:
                try:
                    pval = float(spstats.studentized_range.sf(q, k, df_ij))
                except Exception:
                    pval = np.nan
            out.append(
                {
                    "group1": p1,
                    "group2": p2,
                    "n1": int(n1),
                    "n2": int(n2),
                    "mean_z_1": m1,
                    "mean_z_2": m2,
                    "diff_z": float(m1 - m2),
                    "q_stat": float(q),
                    "df": float(df_ij) if np.isfinite(df_ij) else np.nan,
                    "p_value": float(pval) if np.isfinite(pval) else np.nan,
                    "stars": _p_to_stars(pval),
                }
            )
    return pd.DataFrame(out)


def _add_sig_brackets_to_boxplot(fig, row, col, pairs_df, y_values_r, pad_frac=0.06, show_nonsig=True):
    if pairs_df is None or len(pairs_df) == 0:
        return
    ann = pairs_df.copy()
    ann = ann[np.isfinite(pd.to_numeric(ann.get("p_value", np.nan), errors="coerce"))]
    if not bool(show_nonsig):
        ann = ann[pd.to_numeric(ann["p_value"], errors="coerce") < 0.05]
    if len(ann) == 0:
        return

    yy = np.asarray(y_values_r, dtype=float)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return
    y_min = float(np.min(yy))
    y_max = float(np.max(yy))
    span = max(1e-6, y_max - y_min)
    step = pad_frac * span
    base = y_max + 0.15 * span

    x_order = {"anesthetized": 0, "awake": 1, "motor": 2}
    ann = ann.sort_values("p_value", ascending=True)
    used = 0
    for _, rr in ann.iterrows():
        g1 = str(rr["group1"])
        g2 = str(rr["group2"])
        if g1 not in x_order or g2 not in x_order:
            continue
        pval = pd.to_numeric(rr.get("p_value", np.nan), errors="coerce")
        is_sig = bool(np.isfinite(pval) and (pval < 0.05))
        stars = str(rr.get("stars", ""))
        y = base + used * step
        used += 1
        line_color = "black" if is_sig else "rgba(80,80,80,0.55)"
        text_color = "black" if is_sig else "rgba(80,80,80,0.75)"
        fig.add_trace(
            go.Scatter(
                x=[g1, g1, g2, g2],
                y=[y, y + 0.35 * step, y + 0.35 * step, y],
                mode="lines",
                line=dict(color=line_color, width=1.2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[g1 if x_order[g1] <= x_order[g2] else g2],
                y=[y + 0.43 * step],
                mode="text",
                text=[stars],
                textfont=dict(size=12, color=text_color),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _collect_population_box_stats(df, corr_col_basic, corr_col_opt):
    d = df.copy()
    if "population" not in d.columns:
        d["population"] = "unknown"
    if "cell_id" not in d.columns:
        d["cell_id"] = d["folder"].astype(str)
    d["population"] = d["population"].astype(str).str.lower().str.strip()
    d["cell_id"] = d["cell_id"].astype(str).str.lower().str.strip()

    out_rows = []
    gh_frames = []
    for metric_name, col in [("basic_corr", corr_col_basic), ("optimal_corr", corr_col_opt)]:
        bz = _build_cell_state_fisher_table(d, col)
        wa = _welch_anova_on_groups_z(bz)
        if wa.get("status") == "ok":
            res = wa["result"]
            out_rows.append(
                {
                    "metric": metric_name,
                    "test": "Welch_ANOVA_on_Fisher_z",
                    "statistic_F": float(getattr(res, "statistic", np.nan)),
                    "df_num": float(getattr(res, "df_num", np.nan)),
                    "df_denom": float(getattr(res, "df_denom", np.nan)),
                    "p_value": float(getattr(res, "pvalue", np.nan)),
                    "status": "ok",
                    "info": "",
                }
            )
        else:
            out_rows.append(
                {
                    "metric": metric_name,
                    "test": "Welch_ANOVA_on_Fisher_z",
                    "statistic_F": np.nan,
                    "df_num": np.nan,
                    "df_denom": np.nan,
                    "p_value": np.nan,
                    "status": str(wa.get("status", "failed")),
                    "info": str(wa.get("reason", "")),
                }
            )

        gh = _games_howell_on_groups_z(bz, pop_order=["anesthetized", "awake", "motor"])
        if len(gh):
            gh = gh.copy()
            gh.insert(0, "metric", metric_name)
            gh.insert(1, "test", "Games_Howell_on_Fisher_z")
            gh_frames.append(gh)

    welch_df = pd.DataFrame(out_rows)
    gh_df = pd.concat(gh_frames, axis=0, ignore_index=True) if len(gh_frames) else pd.DataFrame()
    return welch_df, gh_df

def _prepare_mixed_df(df):
    d = df.copy()
    if "population" not in d.columns:
        d["population"] = "unknown"
    if "cell_id" not in d.columns:
        d["cell_id"] = d["folder"].astype(str)
    d["population"] = d["population"].astype(str).str.lower().str.strip()
    d["cell_id"] = d["cell_id"].astype(str).str.lower().str.strip()
    d["z_basic"] = _fisher_z_from_r(pd.to_numeric(d.get("basic_corr_r_lagfixed", np.nan), errors="coerce").to_numpy(float))
    d["z_opt"] = _fisher_z_from_r(pd.to_numeric(d.get("optimal_corr_r_lagfixed", np.nan), errors="coerce").to_numpy(float))
    d["basic_mean_fr_hz"] = pd.to_numeric(d.get("basic_mean_fr_hz", np.nan), errors="coerce")
    d["basic_std_fr_hz"] = pd.to_numeric(d.get("basic_std_fr_hz", np.nan), errors="coerce")
    return d


def _fit_mixedlm_safe(formula, data, group_col="cell_id"):
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        return None, {"status": "failed", "reason": "too_few_rows", "n": int(len(d))}
    if d[group_col].nunique() < 2:
        return None, {"status": "failed", "reason": "too_few_groups", "n_groups": int(d[group_col].nunique())}
    try:
        mdl = smf.mixedlm(formula, data=d, groups=d[group_col], re_formula="1")
        fit = mdl.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        return fit, {"status": "ok", "n": int(len(d)), "n_groups": int(d[group_col].nunique())}
    except Exception as e:
        return None, {"status": "failed", "reason": str(e)}


def _cluster_bootstrap_fixed_effects(formula, data, group_col="cell_id", n_boot=800, seed=0):
    rng = np.random.default_rng(int(seed))
    d0 = data.copy()
    d0 = d0.replace([np.inf, -np.inf], np.nan).dropna()
    groups = d0[group_col].dropna().astype(str).unique().tolist()
    if len(groups) < 2:
        return pd.DataFrame()

    coefs = []
    for b in range(int(n_boot)):
        draw = rng.choice(groups, size=len(groups), replace=True)
        parts = []
        for j, gid in enumerate(draw):
            chunk = d0[d0[group_col].astype(str) == str(gid)].copy()
            if len(chunk) == 0:
                continue
            chunk[group_col] = f"{gid}__boot{b}_{j}"
            parts.append(chunk)
        if len(parts) == 0:
            continue
        db = pd.concat(parts, axis=0, ignore_index=True)
        fit, info = _fit_mixedlm_safe(formula, db, group_col=group_col)
        if fit is None:
            continue
        row = {"boot_iter": int(b)}
        for k, v in fit.fe_params.items():
            row[str(k)] = float(v)
        coefs.append(row)
    return pd.DataFrame(coefs)


def _mixedlm_and_bootstrap_summary(df, model_specs, group_col="cell_id", n_boot=800, seed=0):
    rows = []
    boot_rows = []
    for spec in model_specs:
        model_name = str(spec["name"])
        formula = str(spec["formula"])
        cols_needed = list(spec.get("required_cols", [])) + [group_col, "population"]
        d = df.copy()
        for c in cols_needed:
            if c not in d.columns:
                d[c] = np.nan
        d = d[cols_needed].copy()
        fit, info = _fit_mixedlm_safe(formula, d, group_col=group_col)
        if fit is None:
            rows.append(
                {
                    "model_name": model_name,
                    "formula": formula,
                    "term": np.nan,
                    "coef": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "ci_low_95": np.nan,
                    "ci_high_95": np.nan,
                    "status": info.get("status", "failed"),
                    "info": str(info.get("reason", "")),
                    "n_rows": int(info.get("n", 0)),
                    "n_groups": int(info.get("n_groups", 0)) if "n_groups" in info else np.nan,
                }
            )
            continue

        ci = fit.conf_int()
        for term in fit.fe_params.index:
            rows.append(
                {
                    "model_name": model_name,
                    "formula": formula,
                    "term": str(term),
                    "coef": float(fit.fe_params.get(term, np.nan)),
                    "se": float(fit.bse_fe.get(term, np.nan)),
                    "z": float(fit.tvalues.get(term, np.nan)),
                    "p_value": float(fit.pvalues.get(term, np.nan)),
                    "ci_low_95": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                    "ci_high_95": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                    "status": "ok",
                    "info": "",
                    "n_rows": int(info.get("n", 0)),
                    "n_groups": int(info.get("n_groups", 0)),
                }
            )

        boot_df = _cluster_bootstrap_fixed_effects(formula, d, group_col=group_col, n_boot=n_boot, seed=seed)
        if len(boot_df):
            param_cols = [c for c in boot_df.columns if c != "boot_iter"]
            for term in param_cols:
                vals = pd.to_numeric(boot_df[term], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                boot_rows.append(
                    {
                        "model_name": model_name,
                        "formula": formula,
                        "term": str(term),
                        "boot_n": int(vals.size),
                        "boot_mean": float(np.mean(vals)),
                        "boot_ci_low_95": float(np.percentile(vals, 2.5)),
                        "boot_ci_high_95": float(np.percentile(vals, 97.5)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def _plot_population_repeated_connections(df):
    d = df.copy()
    if "population" not in d.columns:
        d["population"] = "unknown"
    if "cell_id" not in d.columns:
        d["cell_id"] = d["folder"].astype(str)
    d["population"] = d["population"].astype(str).str.lower().str.strip()
    d["cell_id"] = d["cell_id"].astype(str).str.lower().str.strip()

    pop_order = ["anesthetized", "awake", "motor"]
    pop_x = {p: i for i, p in enumerate(pop_order)}
    pop_color = {
        "anesthetized": "rgba(31,119,180,0.92)",
        "awake": "rgba(44,160,44,0.92)",
        "motor": "rgba(255,127,14,0.92)",
    }

    d["x_pop"] = d["population"].map(pop_x)
    d = d[np.isfinite(pd.to_numeric(d["x_pop"], errors="coerce"))].copy()
    d["x_pop"] = d["x_pop"].astype(float)
    d["basic_corr_r_lagfixed"] = pd.to_numeric(d.get("basic_corr_r_lagfixed", np.nan), errors="coerce")
    d["optimal_corr_r_lagfixed"] = pd.to_numeric(d.get("optimal_corr_r_lagfixed", np.nan), errors="coerce")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=(
            "Basic lag-corrected correlation by population (repeated-cell links)",
            "Optimal lag-corrected correlation by population (repeated-cell links)",
        ),
    )

    rng = np.random.default_rng(0)
    d["x_jit"] = d["x_pop"] + rng.uniform(-0.08, 0.08, size=len(d))

    # Repeated-cell connection lines (shown once in legend)
    shown_link_legend = False
    for cid, g in d.groupby("cell_id"):
        g2 = g.sort_values("x_pop")
        if g2["population"].nunique() < 2:
            continue
        x = g2["x_pop"].to_numpy(float)
        yb = g2["basic_corr_r_lagfixed"].to_numpy(float)
        yo = g2["optimal_corr_r_lagfixed"].to_numpy(float)
        okb = np.isfinite(x) & np.isfinite(yb)
        oko = np.isfinite(x) & np.isfinite(yo)
        if np.sum(okb) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=x[okb],
                    y=yb[okb],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.20)", width=1.2),
                    name="same cell across states",
                    showlegend=(not shown_link_legend),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            shown_link_legend = True
        if np.sum(oko) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=x[oko],
                    y=yo[oko],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.20)", width=1.2),
                    name="same cell across states",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )

    # Points by population
    for pop in pop_order:
        gp = d[d["population"] == pop]
        if len(gp) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=gp["x_jit"],
                y=gp["basic_corr_r_lagfixed"],
                mode="markers",
                marker=dict(color=pop_color[pop], size=7, line=dict(color="rgba(0,0,0,0.55)", width=0.6)),
                name=pop,
                legendgroup=pop,
                showlegend=True,
                text=gp["folder"],
                hovertemplate="%{text}<br>population=" + pop + "<br>basic r=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=gp["x_jit"],
                y=gp["optimal_corr_r_lagfixed"],
                mode="markers",
                marker=dict(color=pop_color[pop], size=7, line=dict(color="rgba(0,0,0,0.55)", width=0.6)),
                name=pop,
                legendgroup=pop,
                showlegend=False,
                text=gp["folder"],
                hovertemplate="%{text}<br>population=" + pop + "<br>optimal r=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    for c in [1, 2]:
        fig.update_xaxes(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=pop_order,
            title_text="population",
            row=1,
            col=c,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.16)",
            zeroline=False,
        )
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.16)", zeroline=False, row=1, col=c)

    fig.update_yaxes(title_text="Pearson r", row=1, col=1)
    fig.update_yaxes(title_text="Pearson r", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        width=1380,
        height=600,
        title=f"SST population correlation with repeated-cell links | basic sigma target={SIGMA_BASIC_TARGET:.2f}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    return fig


# Build unified table
sst_corr_summary_df = _build_summary_dataframe(pathPyr, sigma_target=SIGMA_BASIC_TARGET)
print("Cells summarized:", len(sst_corr_summary_df))

# Figure 1: original behavior (zero-lag)
fig_zero = _build_summary_figure(
    sst_corr_summary_df,
    corr_col_basic="basic_corr_r_zero",
    corr_col_opt="optimal_corr_r_zero",
    opt_fr_sig_col="optimal_fr_sigma_zero",
    opt_cal_sig_col="optimal_cal_sigma_zero",
    title_suffix="zero-lag correlations",
    include_lag_extras=False,
)
fig_zero.show()

# Figure 2: fixed-lag behavior (lag chosen once from non-smoothed trace per cell)
fig_lag = _build_summary_figure(
    sst_corr_summary_df,
    corr_col_basic="basic_corr_r_lagfixed",
    corr_col_opt="optimal_corr_r_lagfixed",
    opt_fr_sig_col="optimal_fr_sigma_lagfixed",
    opt_cal_sig_col="optimal_cal_sigma_lagfixed",
    title_suffix="fixed-lag corrected correlations",
    include_lag_extras=True,
)
fig_lag.show()

# Figure 3: summary by population (brain-state separated)
fig_pop = _build_summary_figure_by_population(
    sst_corr_summary_df,
    corr_col_basic="basic_corr_r_lagfixed",
    corr_col_opt="optimal_corr_r_lagfixed",
    opt_fr_sig_col="optimal_fr_sigma_lagfixed",
    opt_cal_sig_col="optimal_cal_sigma_lagfixed",
    title_suffix="fixed-lag corrected correlations",
    include_lag_extras=True,
)
fig_pop.show()
pop_welch_df, pop_gh_df = _collect_population_box_stats(
    sst_corr_summary_df,
    corr_col_basic="basic_corr_r_lagfixed",
    corr_col_opt="optimal_corr_r_lagfixed",
)
print("Population Welch-ANOVA rows:", len(pop_welch_df), "| Games-Howell rows:", len(pop_gh_df))
if len(pop_welch_df):
    print(pop_welch_df.to_string(index=False))
if len(pop_gh_df):
    print(pop_gh_df[["metric", "group1", "group2", "p_value", "stars"]].to_string(index=False))

# Figure 4: explained variance of correlation differences (drop-one partial R^2)
ev_zero = _build_partial_r2_table(
    sst_corr_summary_df,
    target_col="optimal_corr_r_zero",
    opt_fr_col="optimal_fr_sigma_zero",
    opt_cal_col="optimal_cal_sigma_zero",
)
ev_lag = _build_partial_r2_table(
    sst_corr_summary_df,
    target_col="optimal_corr_r_lagfixed",
    opt_fr_col="optimal_fr_sigma_lagfixed",
    opt_cal_col="optimal_cal_sigma_lagfixed",
)
ev_all = pd.concat([ev_zero, ev_lag], axis=0, ignore_index=True)
fig_explain = _plot_partial_r2_summary(
    ev_all,
    title=f"SST explained variance of correlation | sigma target={SIGMA_BASIC_TARGET:.2f}",
)
fig_explain.show()

# Figure 5 + tables: Fisher-z mixed-effects (population fixed, cell random) + bootstrap CIs
mixed_df = _prepare_mixed_df(sst_corr_summary_df)
mixed_model_specs = [
    {
        "name": "basic_corr_population_effect",
        "formula": "z_basic ~ C(population)",
        "required_cols": ["z_basic"],
    },
    {
        "name": "optimal_corr_population_effect",
        "formula": "z_opt ~ C(population)",
        "required_cols": ["z_opt"],
    },
    {
        "name": "basic_corr_vs_meanFR_by_population",
        "formula": "z_basic ~ basic_mean_fr_hz * C(population)",
        "required_cols": ["z_basic", "basic_mean_fr_hz"],
    },
    {
        "name": "basic_corr_vs_stdFR_by_population",
        "formula": "z_basic ~ basic_std_fr_hz * C(population)",
        "required_cols": ["z_basic", "basic_std_fr_hz"],
    },
]
mixed_res_df, mixed_boot_df = _mixedlm_and_bootstrap_summary(
    mixed_df,
    model_specs=mixed_model_specs,
    group_col="cell_id",
    n_boot=600,
    seed=0,
)
fig_mixed_pop = _plot_population_repeated_connections(sst_corr_summary_df)
fig_mixed_pop.show()

# Save outputs
if OUT_DIR is not None:
    os.makedirs(OUT_DIR, exist_ok=True)

    zero_html = os.path.join(OUT_DIR, OUT_STEM_ZERO + ".html")
    zero_svg = os.path.join(OUT_DIR, OUT_STEM_ZERO + ".svg")
    fig_zero.write_html(zero_html, include_plotlyjs="cdn")
    try:
        fig_zero.write_image(zero_svg)
    except Exception as e:
        print("Zero-lag SVG save skipped:", e)

    pop_html = os.path.join(OUT_DIR, OUT_STEM_POP + ".html")
    pop_svg = os.path.join(OUT_DIR, OUT_STEM_POP + ".svg")
    pop_welch_csv = os.path.join(OUT_DIR, OUT_STEM_POP + "_welch_anova.csv")
    pop_gh_csv = os.path.join(OUT_DIR, OUT_STEM_POP + "_games_howell.csv")
    fig_pop.write_html(pop_html, include_plotlyjs="cdn")
    try:
        fig_pop.write_image(pop_svg)
    except Exception as e:
        print("Population SVG save skipped:", e)
    pop_welch_df.to_csv(pop_welch_csv, index=False)
    pop_gh_df.to_csv(pop_gh_csv, index=False)

    lag_html = os.path.join(OUT_DIR, OUT_STEM_LAGFIX + ".html")
    lag_svg = os.path.join(OUT_DIR, OUT_STEM_LAGFIX + ".svg")
    fig_lag.write_html(lag_html, include_plotlyjs="cdn")
    try:
        fig_lag.write_image(lag_svg)
    except Exception as e:
        print("Lag-fixed SVG save skipped:", e)

    explain_html = os.path.join(OUT_DIR, OUT_STEM_EXPLAIN + ".html")
    explain_svg = os.path.join(OUT_DIR, OUT_STEM_EXPLAIN + ".svg")
    explain_csv = os.path.join(OUT_DIR, OUT_STEM_EXPLAIN + "_table.csv")
    fig_explain.write_html(explain_html, include_plotlyjs="cdn")
    try:
        fig_explain.write_image(explain_svg)
    except Exception as e:
        print("Explained-variance SVG save skipped:", e)
    ev_all.to_csv(explain_csv, index=False)

    mixed_html = os.path.join(OUT_DIR, OUT_STEM_MIXED + ".html")
    mixed_svg = os.path.join(OUT_DIR, OUT_STEM_MIXED + ".svg")
    mixed_csv = os.path.join(OUT_DIR, OUT_STEM_MIXED + "_table.csv")
    mixed_boot_csv = os.path.join(OUT_DIR, OUT_STEM_MIXED + "_bootstrap.csv")
    fig_mixed_pop.write_html(mixed_html, include_plotlyjs="cdn")
    try:
        fig_mixed_pop.write_image(mixed_svg)
    except Exception as e:
        print("Mixed-pop SVG save skipped:", e)
    mixed_res_df.to_csv(mixed_csv, index=False)
    mixed_boot_df.to_csv(mixed_boot_csv, index=False)

    out_csv = os.path.join(OUT_DIR, f"sst_corr_summary_dual_zero_vs_fixedlag_sigma{SIGMA_BASIC_TARGET:.2f}_table.csv")
    sst_corr_summary_df.to_csv(out_csv, index=False)

    print("Saved:")
    print(" ", zero_html)
    print(" ", zero_svg)
    print(" ", lag_html)
    print(" ", lag_svg)
    print(" ", pop_html)
    print(" ", pop_svg)
    print(" ", pop_welch_csv)
    print(" ", pop_gh_csv)
    print(" ", explain_html)
    print(" ", explain_svg)
    print(" ", explain_csv)
    print(" ", mixed_html)
    print(" ", mixed_svg)
    print(" ", mixed_csv)
    print(" ", mixed_boot_csv)
    print(" ", out_csv)


