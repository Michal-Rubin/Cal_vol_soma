# NEW_CELL: SST_CLUSTER_PROPERTY_SUMMARY
# Cluster-level property summary (box + overlaid scatter for each cell)
# Uses lag-fixed correlation metrics from sst_corr_summary_df.

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as spstats
from statsmodels.stats.oneway import anova_oneway

# Requires prior execution of NEW_CELL: SST_CORR_ANLSYS_FIVE_PANEL_SUMMARY
# so that sst_corr_summary_df exists.
if 'sst_corr_summary_df' not in globals() or not isinstance(sst_corr_summary_df, pd.DataFrame):
    raise RuntimeError('Run NEW_CELL: SST_CORR_ANLSYS_FIVE_PANEL_SUMMARY first (sst_corr_summary_df is missing).')

# --- cluster mapping from hierarchical-clustering output ---
CLUSTER_CSV_PATH = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\SST\clustering\final\sst_calcium_trace_clusters_clean.csv"
CLUSTER_COL = "hier_cluster"  # explicit: use hierarchical clustering labels

if not os.path.isfile(CLUSTER_CSV_PATH):
    raise RuntimeError(f"Hierarchical cluster CSV not found: {CLUSTER_CSV_PATH}")

clu = pd.read_csv(CLUSTER_CSV_PATH)
if CLUSTER_COL not in clu.columns:
    raise RuntimeError(f"Column '{CLUSTER_COL}' not found in {CLUSTER_CSV_PATH}")

path_col = None
for cand in ["folder", "Link", "trace_path"]:
    if cand in clu.columns:
        path_col = cand
        break
if path_col is None:
    raise RuntimeError("No path column found in clustering CSV (expected one of: folder, Link, trace_path)")

def _norm_path(p):
    s = str(p).strip().replace("/", "\\").rstrip("\\")
    return s.lower()

md2 = clu[[path_col, CLUSTER_COL]].copy()
md2 = md2.rename(columns={path_col: "folder", CLUSTER_COL: "cluster"})
md2["folder_key"] = md2["folder"].map(_norm_path)
md2["cluster"] = pd.to_numeric(md2["cluster"], errors="coerce")
md2 = md2[np.isfinite(md2["cluster"])].copy()
md2["cluster"] = md2["cluster"].astype(int).astype(str)
md2 = md2.drop_duplicates(subset=["folder_key"], keep="first")

# merge with existing correlation summary
cdf = sst_corr_summary_df.copy()
cdf["folder"] = cdf["folder"].astype(str).str.strip()
cdf["folder_key"] = cdf["folder"].map(_norm_path)
cdf = cdf.merge(md2[["folder_key", "cluster"]], on="folder_key", how="left")
cdf = cdf[np.isfinite(pd.to_numeric(cdf["cluster"], errors="coerce"))].copy()
cdf["cluster"] = pd.to_numeric(cdf["cluster"], errors="coerce").astype(int).astype(str)


def _fisher_z_from_r(r):
    rr = np.asarray(r, dtype=float)
    rr = np.clip(rr, -0.999999, 0.999999)
    return np.arctanh(rr)


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


def _welch_and_games_howell(vals_by_group):
    # vals_by_group: dict[str, np.ndarray]
    clean = {}
    for g, v in vals_by_group.items():
        vv = np.asarray(v, dtype=float).ravel()
        vv = vv[np.isfinite(vv)]
        if vv.size >= 2:
            clean[str(g)] = vv

    welch_row = {
        "test": "Welch_ANOVA",
        "statistic_F": np.nan,
        "df_num": np.nan,
        "df_denom": np.nan,
        "p_value": np.nan,
        "status": "failed",
        "info": "",
    }
    gh_rows = []

    if len(clean) < 2:
        welch_row["info"] = "need>=2 groups with n>=2"
        return welch_row, pd.DataFrame(gh_rows)

    try:
        res = anova_oneway(list(clean.values()), use_var="unequal")
        welch_row.update(
            {
                "statistic_F": float(getattr(res, "statistic", np.nan)),
                "df_num": float(getattr(res, "df_num", np.nan)),
                "df_denom": float(getattr(res, "df_denom", np.nan)),
                "p_value": float(getattr(res, "pvalue", np.nan)),
                "status": "ok",
            }
        )
    except Exception as e:
        welch_row["info"] = str(e)

    groups = sorted(clean.keys(), key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else str(x))
    k = len(groups)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            x = clean[g1]
            y = clean[g2]
            n1, n2 = x.size, y.size
            m1, m2 = float(np.mean(x)), float(np.mean(y))
            v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
            se2 = (v1 / n1) + (v2 / n2)
            if not np.isfinite(se2) or se2 <= 0:
                continue
            t = abs(m1 - m2) / np.sqrt(se2)
            q = np.sqrt(2.0) * float(t)
            denom = ((v1 / n1) ** 2) / max(n1 - 1, 1) + ((v2 / n2) ** 2) / max(n2 - 1, 1)
            df_ij = (se2 ** 2) / denom if np.isfinite(denom) and denom > 0 else np.nan
            pval = np.nan
            if np.isfinite(df_ij) and df_ij > 0:
                try:
                    pval = float(spstats.studentized_range.sf(q, k, df_ij))
                except Exception:
                    pval = np.nan
            gh_rows.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "n1": int(n1),
                    "n2": int(n2),
                    "mean1": m1,
                    "mean2": m2,
                    "diff": float(m1 - m2),
                    "q_stat": float(q),
                    "df": float(df_ij) if np.isfinite(df_ij) else np.nan,
                    "p_value": float(pval) if np.isfinite(pval) else np.nan,
                    "stars": _p_to_stars(pval),
                }
            )
    return welch_row, pd.DataFrame(gh_rows)


def _add_sig_brackets(fig, row, col, pairs_df, y_values, show_nonsig=True):
    if pairs_df is None or len(pairs_df) == 0:
        return
    p = pairs_df.copy()
    p = p[np.isfinite(pd.to_numeric(p.get("p_value", np.nan), errors="coerce"))]
    if not bool(show_nonsig):
        p = p[pd.to_numeric(p["p_value"], errors="coerce") < 0.05]
    if len(p) == 0:
        return

    yy = np.asarray(y_values, dtype=float)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return
    y_max = float(np.max(yy))
    y_min = float(np.min(yy))
    span = max(1e-6, y_max - y_min)
    step = 0.06 * span
    base = y_max + 0.15 * span
    p = p.sort_values("p_value", ascending=True)
    used = 0
    for _, rr in p.iterrows():
        g1 = str(rr["group1"])
        g2 = str(rr["group2"])
        y = base + used * step
        used += 1
        pv = pd.to_numeric(rr.get("p_value", np.nan), errors="coerce")
        is_sig = bool(np.isfinite(pv) and pv < 0.05)
        lc = "black" if is_sig else "rgba(80,80,80,0.55)"
        tc = "black" if is_sig else "rgba(80,80,80,0.75)"
        fig.add_trace(
            go.Scatter(
                x=[g1, g1, g2, g2],
                y=[y, y + 0.35 * step, y + 0.35 * step, y],
                mode="lines",
                line=dict(color=lc, width=1.2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[g1],
                y=[y + 0.43 * step],
                mode="text",
                text=[str(rr.get("stars", "n.s."))],
                textfont=dict(size=12, color=tc),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

# metrics for requested panels
METRICS = [
    ('basic_corr_r_lagfixed', 'Basic correlation (lag-fixed)'),
    ('optimal_corr_r_lagfixed', 'Optimal correlation (lag-fixed)'),
    ('basic_mean_fr_hz', 'Mean FR (Hz)'),
    ('basic_std_fr_hz', 'FR STD (Hz)'),
]

# cluster order: known labels first (if numeric-like then numeric order), else alphabetical
clusters = sorted([x for x in cdf['cluster'].dropna().unique().tolist()])

def _cluster_sort_key(x):
    s = str(x)
    try:
        return (0, float(s))
    except Exception:
        return (1, s.lower())
clusters = sorted(clusters, key=_cluster_sort_key)

fig = make_subplots(
    rows=2,
    cols=2,
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
    subplot_titles=[m[1] for m in METRICS],
)

rng = np.random.default_rng(0)
welch_rows = []
gh_frames = []

for panel_idx, (col, title_txt) in enumerate(METRICS, start=1):
    r = 1 if panel_idx <= 2 else 2
    c = 1 if panel_idx % 2 == 1 else 2
    vals_by_group = {}
    all_vals = []

    for cl in clusters:
        sub = cdf[cdf['cluster'].astype(str) == str(cl)].copy()
        vals = pd.to_numeric(sub[col], errors='coerce').to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals_by_group[str(cl)] = vals
        all_vals.append(vals)

        fig.add_trace(
            go.Box(
                x=[str(cl)] * int(vals.size),
                y=vals,
                name=str(cl),
                boxpoints='all',
                jitter=0.42,
                pointpos=0,
                marker=dict(size=6, opacity=0.78, line=dict(color='rgba(0,0,0,0.45)', width=0.6)),
                line=dict(width=1.6),
                showlegend=False,
                boxmean=True,
            ),
            row=r,
            col=c,
        )

    # stats: for correlation metrics test on Fisher-z, for FR metrics test on raw values
    vals_for_test = vals_by_group
    if "corr" in str(col).lower():
        vals_for_test = {k: _fisher_z_from_r(v) for k, v in vals_by_group.items()}
    welch_row, gh_df = _welch_and_games_howell(vals_for_test)
    welch_row["metric_col"] = col
    welch_row["metric_title"] = title_txt
    welch_rows.append(welch_row)
    if len(gh_df):
        gh_df = gh_df.copy()
        gh_df.insert(0, "metric_col", col)
        gh_df.insert(1, "metric_title", title_txt)
        gh_frames.append(gh_df)

    if len(all_vals):
        _add_sig_brackets(
            fig=fig,
            row=r,
            col=c,
            pairs_df=gh_df if len(gh_df) else pd.DataFrame(),
            y_values=np.concatenate(all_vals),
            show_nonsig=True,
        )

    fig.update_xaxes(title_text='cluster', row=r, col=c)
    fig.update_yaxes(title_text=title_txt, row=r, col=c)

fig.update_layout(
    template='plotly_white',
    width=1400,
    height=980,
    title=f'SST cluster property summary | cluster_col={CLUSTER_COL} | n_cells={len(cdf)}',
)
fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.16)', zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.16)', zeroline=False)
fig.show()

# Save
OUT_DIR_CLUSTER = OUT_DIR if ('OUT_DIR' in globals() and OUT_DIR is not None) else r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\data summery\2026\SST\BasicCorr"
os.makedirs(OUT_DIR_CLUSTER, exist_ok=True)
OUT_STEM_CLUSTER = f'sst_cluster_property_summary_sigma{SIGMA_BASIC_TARGET:.2f}' if 'SIGMA_BASIC_TARGET' in globals() else 'sst_cluster_property_summary'

p_html = os.path.join(OUT_DIR_CLUSTER, OUT_STEM_CLUSTER + '.html')
p_svg = os.path.join(OUT_DIR_CLUSTER, OUT_STEM_CLUSTER + '.svg')
p_csv = os.path.join(OUT_DIR_CLUSTER, OUT_STEM_CLUSTER + '_table.csv')
p_welch = os.path.join(OUT_DIR_CLUSTER, OUT_STEM_CLUSTER + '_welch_anova.csv')
p_gh = os.path.join(OUT_DIR_CLUSTER, OUT_STEM_CLUSTER + '_games_howell.csv')

fig.write_html(p_html, include_plotlyjs='cdn')
try:
    fig.write_image(p_svg)
except Exception as e:
    print('SVG save skipped:', e)

cols_keep = ['folder', 'cluster', 'population', 'basic_corr_r_lagfixed', 'optimal_corr_r_lagfixed', 'basic_mean_fr_hz', 'basic_std_fr_hz']
for cc in cols_keep:
    if cc not in cdf.columns:
        cdf[cc] = np.nan
cdf[cols_keep].to_csv(p_csv, index=False)
pd.DataFrame(welch_rows).to_csv(p_welch, index=False)
pd.concat(gh_frames, axis=0, ignore_index=True).to_csv(p_gh, index=False) if len(gh_frames) else pd.DataFrame().to_csv(p_gh, index=False)

print('Saved:')
print(' ', p_html)
print(' ', p_svg)
print(' ', p_csv)
print(' ', p_welch)
print(' ', p_gh)
