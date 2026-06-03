# NEW_CELL: SST_CLUSTER_PROPERTY_SUMMARY
# Cluster-level property summary (box + overlaid scatter for each cell)
# Uses lag-fixed correlation metrics from sst_corr_summary_df.

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Requires prior execution of NEW_CELL: SST_CORR_ANLSYS_FIVE_PANEL_SUMMARY
# so that sst_corr_summary_df exists.
if 'sst_corr_summary_df' not in globals() or not isinstance(sst_corr_summary_df, pd.DataFrame):
    raise RuntimeError('Run NEW_CELL: SST_CORR_ANLSYS_FIVE_PANEL_SUMMARY first (sst_corr_summary_df is missing).')

# --- cluster mapping from metadata ---
DB_PATH = r"Z:\Adam-Lab-Shared\Data\Michal_Rubin\Dendrites\SST_Final.csv"
CLUSTER_COL_CANDIDATES = ['subGroup', 'cluster', 'Cluster', 'clust', 'cellCluster', 'group']

md = pd.read_csv(DB_PATH)
if 'Link' not in md.columns:
    raise RuntimeError("SST_Final.csv missing 'Link' column for path mapping")

cluster_col = None
for c in CLUSTER_COL_CANDIDATES:
    if c in md.columns:
        cluster_col = c
        break
if cluster_col is None:
    raise RuntimeError(f'No cluster column found in metadata. Tried: {CLUSTER_COL_CANDIDATES}')

md2 = md[['Link', cluster_col]].copy()
md2 = md2.rename(columns={'Link': 'folder', cluster_col: 'cluster'})
md2['folder'] = md2['folder'].astype(str).str.strip()
md2['cluster'] = md2['cluster'].astype(str).str.strip().replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'})

# merge with existing correlation summary
cdf = sst_corr_summary_df.copy()
cdf['folder'] = cdf['folder'].astype(str).str.strip()
cdf = cdf.merge(md2, on='folder', how='left')
cdf['cluster'] = cdf['cluster'].astype(str).str.strip().replace({'': 'unknown', 'nan': 'unknown', 'None': 'unknown'})

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

for panel_idx, (col, title_txt) in enumerate(METRICS, start=1):
    r = 1 if panel_idx <= 2 else 2
    c = 1 if panel_idx % 2 == 1 else 2

    for cl in clusters:
        sub = cdf[cdf['cluster'].astype(str) == str(cl)].copy()
        vals = pd.to_numeric(sub[col], errors='coerce').to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

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

    fig.update_xaxes(title_text='cluster', row=r, col=c)
    fig.update_yaxes(title_text=title_txt, row=r, col=c)

fig.update_layout(
    template='plotly_white',
    width=1400,
    height=980,
    title=f'SST cluster property summary | cluster_col={cluster_col} | n_cells={len(cdf)}',
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

print('Saved:')
print(' ', p_html)
print(' ', p_svg)
print(' ', p_csv)
