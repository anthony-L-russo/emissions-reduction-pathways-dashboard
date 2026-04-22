import io
import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
from config import CONFIG
from utils.utils import get_iso3_flag_table

SECTOR_COLORS = {
    'power':                  '#56979F',
    'transportation':         '#FBBA14',
    'fossil-fuel-operations': '#FF6F42',
    'agriculture':            '#E8516C',
    'manufacturing':          '#9554FF',
    'buildings':              '#03A0E3',
    'waste':                  '#BBD421',
    'mineral-extraction':     '#4380F5',
    'fluorinated-gases':      '#B6B4B4',
}

SECTOR_LABELS = {
    'power':                  'Power',
    'transportation':         'Transportation',
    'fossil-fuel-operations': 'Fossil Fuel Operations',
    'agriculture':            'Agriculture',
    'manufacturing':          'Manufacturing',
    'buildings':              'Buildings',
    'waste':                  'Waste',
    'mineral-extraction':     'Mineral Extraction',
    'fluorinated-gases':      'Fluorinated Gases',
}

SECTOR_LABELS_SHORT = {
    'power':                  'Power',
    'transportation':         'Transport',
    'fossil-fuel-operations': 'Fossil Fuel',
    'agriculture':            'Agriculture',
    'manufacturing':          'Manufacturing',
    'buildings':              'Buildings',
    'waste':                  'Waste',
    'mineral-extraction':     'Mining',
    'fluorinated-gases':      'F-Gases',
}

GEOGRAPHIC_CONTINENTS = (
    'Africa', 'Antarctica', 'Asia', 'Europe',
    'North America', 'Oceania', 'South America',
)

AVAILABLE_YEARS = [2021, 2022, 2023, 2024, 2025]


# ── DuckDB caching ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_df(sql):
    _con = duckdb.connect()
    result = _con.execute(sql).df()
    _con.close()
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetchone(sql):
    _con = duckdb.connect()
    result = _con.execute(sql).fetchone()
    _con.close()
    return result


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _shade_hex(hex_color, amount):
    n = int(hex_color.lstrip('#'), 16)
    r = min(255, max(0, (n >> 16) + amount))
    g = min(255, max(0, ((n >> 8) & 0xFF) + amount))
    b = min(255, max(0, (n & 0xFF) + amount))
    return f'#{r:02x}{g:02x}{b:02x}'


def _fmt_mt(mt):
    if abs(mt) >= 1000:
        return f"{mt / 1000:.1f}k"
    elif abs(mt) >= 1:
        return f"{mt:.1f}"
    elif abs(mt) >= 0.001:
        return f"{mt:.3f}"
    return "~0"


def _fmt_ef(ef):
    """Format an emissions factor value."""
    if ef is None or (isinstance(ef, float) and pd.isna(ef)):
        return "—"
    if ef >= 1000:
        return f"{ef:,.3f}"
    if ef >= 1:
        return f"{ef:.3f}"
    elif ef >= 0.0001:
        return f"{ef:.4f}"
    return f"{ef:.2e}"


def _pct_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '<span style="color:#999">—</span>'
    color = '#2e7d32' if val < 0 else '#c62828'
    sign = '+' if val > 0 else ''
    return f'<span style="color:{color};font-weight:600">{sign}{val:.1f}%</span>'


def _rank_badge_html(rank, total=None):
    if rank is None:
        return '<span style="color:#999">—</span>'
    rank = int(rank)
    cls = "ct-rank-top3" if rank <= 3 else ("ct-rank-top10" if rank <= 10 else "ct-rank-other")
    total_str = f" of {int(total)}" if total is not None else ""
    return f'<span class="ct-rank-badge {cls}">#{rank}{total_str}</span>'


def _df_to_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ── CSS ───────────────────────────────────────────────────────────────────────

_TABLE_CSS = """
<style>
.ct-table-scroll { overflow-x: auto; }
table.ct-metrics { width: 100%; border-collapse: collapse; font-size: 13px; }
table.ct-metrics thead tr th {
    background: #f5f6f8; padding: 3px 8px; white-space: nowrap;
    border-bottom: 2px solid #dde1ea; text-align: center;
    font-weight: 600; color: #555d72; font-size: 12px;
}
table.ct-metrics thead tr th:first-child { text-align: left; min-width: 160px; }
table.ct-metrics tbody tr td {
    padding: 3px 8px; text-align: center;
    border-bottom: 1px solid #f0f2f5; color: #3a4158; white-space: nowrap;
}
table.ct-metrics tbody tr td:first-child {
    text-align: left; font-weight: 600; color: #555d72;
    font-size: 12px; border-right: 1px solid #eaecf0;
}
table.ct-metrics tbody tr:hover td { background: #f7f9fb; }
.ct-col-header { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.ct-col-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.ct-rank-badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px; font-weight: 700; }
.ct-rank-top3  { background: #fff3e0; color: #e65100; }
.ct-rank-top10 { background: #e8f5e9; color: #2e7d32; }
.ct-rank-other { background: #f3f4f6; color: #555d72; }
.ct-pct-cell { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.ct-pct-bar-bg { width: 52px; height: 4px; background: #eaecf0; border-radius: 2px; overflow: hidden; }
.ct-pct-bar-fill { height: 100%; border-radius: 2px; opacity: 0.75; }
tr.ct-divider td {
    padding: 2px 8px; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.5px; color: #b0b8c4;
    background: #f9fafc; border-bottom: 1px solid #eaecf0;
}
.ct-summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }
.ct-summary-card { border-radius: 8px; padding: 12px 10px; text-align: center; border: 1px solid #eaecf0; background: #f8f9fb; }
.ct-summary-label { font-size: 13px; font-weight: 700; color: #8892a4; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 6px; }
.ct-summary-value { font-size: 22px; font-weight: 800; line-height: 1.15; }
</style>
"""


# ── Table builder ─────────────────────────────────────────────────────────────

def _build_sector_table(
    col_keys, col_labels, col_colors,
    country_vals, global_vals, continent_vals,
    pct_global, pct_continent,
    global_rank_map, global_count_map,
    continent_rank_map, continent_count_map,
    chg_vals, yoy_vals,
    from_year, to_year, continent_name,
    iso3=None, country_flag=None,
    ef_country=None, ef_continent=None, ef_global=None, ef_units=None,
    ef_pct_map=None,
    highlighted_key=None,
):
    """Transposed table: sectors/subsectors as columns.

    ef_* dicts are optional {grp_key -> float}; shown only when provided.
    ef_pct_map: optional {grp_key -> (percentile_float, n_assets_int)}.
    highlighted_key: bold that column header when a subsector is active.
    """

    def _header_cell(label, color, bold=False):
        weight = "font-weight:800;" if bold else ""
        return (
            f'<th style="{weight}">'
            f'<div class="ct-col-header">'
            f'<div class="ct-col-dot" style="background:{color}"></div>'
            f'{label}</div></th>'
        )

    header = '<th></th>' + ''.join(
        _header_cell(label, color, bold=(highlighted_key and k == highlighted_key))
        for k, label, color in zip(col_keys, col_labels, col_colors)
    )

    def row(label, cells, italic=False):
        style = ' style="font-style:italic;color:#777"' if italic else ''
        return (
            f'<tr><td{style}>{label}</td>'
            + ''.join(f'<td>{c}</td>' for c in cells)
            + '</tr>'
        )

    def divider(text):
        return f'<tr class="ct-divider"><td colspan="{len(col_keys) + 1}">{text}</td></tr>'

    country_label = f'{country_flag} {iso3} (MtCO₂e)' if (country_flag and iso3) else '🏳 Country (MtCO₂e)'
    rows_html = (
        row(country_label,
            [f'<strong>{_fmt_mt(country_vals.get(k, 0) / 1e6)}</strong>' for k in col_keys])
        + row(f'🌐 {continent_name} (MtCO₂e)',
              [f'<strong>{_fmt_mt(continent_vals.get(k, 0) / 1e6)}</strong>' for k in col_keys])
        + row('🌍 Global (MtCO₂e)',
              [f'<strong>{_fmt_mt(global_vals.get(k, 0) / 1e6)}</strong>' for k in col_keys])
        + row('% of Global', [
            (lambda pct, color: (
                f'<div class="ct-pct-cell"><span>{pct:.1f}%</span>'
                f'<div class="ct-pct-bar-bg">'
                f'<div class="ct-pct-bar-fill" style="width:{int(min(52, pct / 30 * 52))}px;background:{color}"></div>'
                f'</div></div>'
            ))(pct_global.get(k, 0), col_colors[i])
            for i, k in enumerate(col_keys)
        ])
        + row('% of ' + continent_name, [
            (lambda pct, color: (
                f'<div class="ct-pct-cell"><span>{pct:.1f}%</span>'
                f'<div class="ct-pct-bar-bg">'
                f'<div class="ct-pct-bar-fill" style="width:{int(min(52, pct / 50 * 52))}px;background:{color}"></div>'
                f'</div></div>'
            ))(pct_continent.get(k, 0), col_colors[i])
            for i, k in enumerate(col_keys)
        ])
        + divider('Rankings &amp; Trends')
        + row(f'🌐 {continent_name} Ranking',
              [_rank_badge_html(continent_rank_map.get(k), continent_count_map.get(k)) for k in col_keys])
        + row('🌍 Global Ranking',
              [_rank_badge_html(global_rank_map.get(k), global_count_map.get(k)) for k in col_keys])
        + row(f'📈 Change {from_year}→{to_year}', [_pct_str(chg_vals.get(k)) for k in col_keys])
        + row(f'📆 YoY {to_year - 1}→{to_year}',
              [_pct_str(yoy_vals.get(k)) for k in col_keys]
              if to_year > 2021 else
              ['<span style="color:#999">—</span>'] * len(col_keys))
    )

    # Optional EF section
    if ef_country is not None:
        country_ef_label = f'{country_flag} {iso3} EF' if (country_flag and iso3) else '🏳 Country EF'
        rows_html += divider('Average Emissions Factor Benchmarking')
        rows_html += row(country_ef_label,
                         [_fmt_ef(ef_country.get(k)) for k in col_keys])
        rows_html += row(f'🌐 {continent_name} EF',
                         [_fmt_ef(ef_continent.get(k) if ef_continent else None) for k in col_keys])
        rows_html += row('🌍 Global EF',
                         [_fmt_ef(ef_global.get(k) if ef_global else None) for k in col_keys])
        if ef_pct_map:
            def _ef_pct_cell(k):
                data = ef_pct_map.get(k)
                if not data:
                    return '<span style="color:#999">—</span>'
                pct, n = data
                return (
                    f'<span style="font-size:13px">{pct}th pctile<br>'
                    f'<span style="color:#999;font-size:13px">({n:,} assets)</span></span>'
                )
            rows_html += row('📊 EF Percentile (global)', [_ef_pct_cell(k) for k in col_keys])
        rows_html += row('📏 Unit',
                         [f'<span style="font-size:13px;color:#888">{ef_units.get(k, "—") if ef_units else "—"}</span>'
                          for k in col_keys],
                         italic=True)

    return (
        _TABLE_CSS
        + '<div class="ct-table-scroll"><table class="ct-metrics">'
        + f'<thead><tr>{header}</tr></thead>'
        + f'<tbody>{rows_html}</tbody>'
        + '</table></div>'
    )




# ── Main tab function ─────────────────────────────────────────────────────────

def show_country_trends():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    stats_path = CONFIG['country_subsector_stats_path']
    asset_path = CONFIG['asset_emissions_country_subsector_path']
    demographic_path = CONFIG['demographic_path']

    # ── Country list ──────────────────────────────────────────────────────────
    countries_df = _cached_df(f"""
        SELECT DISTINCT country_name, iso3_country
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr'
          AND country_name IS NOT NULL
          AND iso3_country IS NOT NULL
          AND length(iso3_country) = 3
        ORDER BY country_name
    """)

    country_options = countries_df['country_name'].tolist()
    country_to_iso3 = dict(zip(countries_df['country_name'], countries_df['iso3_country']))

    # ── Controls: all on one row ──────────────────────────────────────────────
    c_country, c_sector, c_subsector, c_from, c_to = st.columns([1.6, 1.4, 1.4, 0.9, 0.9])

    with c_country:
        default_idx = next(
            (i for i, n in enumerate(country_options) if 'United States' in n), 0
        )
        selected_country = st.selectbox(
            "Country", country_options, index=default_idx, key="ct_country"
        )

    iso3 = country_to_iso3.get(selected_country, '')

    with c_sector:
        sector_options = ["All Sectors"] + [SECTOR_LABELS[s] for s in SECTOR_COLORS]
        selected_sector_label = st.selectbox("Sector", sector_options, key="ct_sector")

    selected_sector = (
        next((k for k, v in SECTOR_LABELS.items() if v == selected_sector_label), None)
        if selected_sector_label != "All Sectors" else None
    )

    # Subsector options keyed by sector so they reset on sector change
    subsector_key = f"ct_subsector_{selected_sector or 'all'}"
    if selected_sector:
        sub_rows = _cached_df(f"""
            SELECT DISTINCT subsector
            FROM '{stats_path}'
            WHERE gas = 'co2e_100yr'
              AND iso3_country = '{iso3}'
              AND sector = '{selected_sector}'
              AND "2025_total_emissions" > 0
            ORDER BY subsector
        """)
        subsector_options = ["All Subsectors"] + sub_rows['subsector'].tolist()
        subsector_disabled = False
    else:
        subsector_options = ["All Subsectors"]
        subsector_disabled = True

    with c_subsector:
        selected_subsector_label = st.selectbox(
            "Subsector", subsector_options,
            disabled=subsector_disabled, key=subsector_key
        )

    selected_subsector = (
        selected_subsector_label
        if selected_subsector_label and selected_subsector_label != "All Subsectors"
        else None
    )

    with c_from:
        from_year = st.selectbox(
            "From year", AVAILABLE_YEARS, index=0, key="ct_from_year"
        )

    with c_to:
        to_options = [y for y in AVAILABLE_YEARS if y > from_year]
        to_year = st.selectbox(
            "To year", to_options if to_options else [AVAILABLE_YEARS[-1]],
            index=len(to_options) - 1 if to_options else 0,
            key=f"ct_to_year_{from_year}",
        )

    # ── Country continent ─────────────────────────────────────────────────────
    continent_row = _cached_fetchone(f"""
        SELECT continent FROM '{stats_path}'
        WHERE gas = 'co2e_100yr'
          AND iso3_country = '{iso3}'
          AND continent IN {str(GEOGRAPHIC_CONTINENTS)}
        LIMIT 1
    """)
    country_continent = continent_row[0] if continent_row else "Unknown"

    # ── Grouping ──────────────────────────────────────────────────────────────
    if selected_sector:
        group_col = 'subsector'
        where_extra = f"AND sector = '{selected_sector}'"
    else:
        group_col = 'sector'
        where_extra = ''

    from_col = f'"{from_year}_total_emissions"'
    to_col   = f'"{to_year}_total_emissions"'
    prev_col = f'"{to_year - 1}_total_emissions"' if to_year > 2021 else f'"{to_year}_total_emissions"'

    # ── Core queries ──────────────────────────────────────────────────────────
    country_df = _cached_df(f"""
        SELECT {group_col} AS grp,
               SUM("2021_total_emissions") AS y2021,
               SUM("2022_total_emissions") AS y2022,
               SUM("2023_total_emissions") AS y2023,
               SUM("2024_total_emissions") AS y2024,
               SUM("2025_total_emissions") AS y2025,
               SUM({from_col}) AS y_from,
               SUM({to_col})   AS y_to,
               SUM({prev_col}) AS y_prev
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr'
          AND iso3_country = '{iso3}'
          {where_extra}
        GROUP BY {group_col}
        HAVING SUM("2025_total_emissions") > 0
        ORDER BY y_to DESC
    """)
    country_df = country_df.head(10)
    col_keys = country_df['grp'].tolist()

    # When a subsector is selected, narrow everything to just that one
    if selected_subsector and selected_subsector in col_keys:
        col_keys = [selected_subsector]

    global_df = _cached_df(f"""
        SELECT {group_col} AS grp,
               SUM({to_col})   AS global_y_to,
               SUM({from_col}) AS global_y_from,
               SUM({prev_col}) AS global_y_prev
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr' AND length(iso3_country) = 3
          {where_extra}
        GROUP BY {group_col}
    """)
    global_y_to_map  = dict(zip(global_df['grp'], global_df['global_y_to']))
    global_y_from_map = dict(zip(global_df['grp'], global_df['global_y_from']))
    global_y_prev_map = dict(zip(global_df['grp'], global_df['global_y_prev']))

    continent_df = _cached_df(f"""
        SELECT {group_col} AS grp,
               SUM({to_col})   AS cont_y_to,
               SUM({from_col}) AS cont_y_from,
               SUM({prev_col}) AS cont_y_prev
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr'
          AND continent = '{country_continent}'
          AND length(iso3_country) = 3
          {where_extra}
        GROUP BY {group_col}
    """)
    cont_y_to_map  = dict(zip(continent_df['grp'], continent_df['cont_y_to']))
    cont_y_from_map = dict(zip(continent_df['grp'], continent_df['cont_y_from']))
    cont_y_prev_map = dict(zip(continent_df['grp'], continent_df['cont_y_prev']))

    # Rankings
    global_rank_df = _cached_df(f"""
        WITH totals AS (
            SELECT iso3_country, {group_col} AS grp,
                   SUM({to_col}) AS total
            FROM '{stats_path}'
            WHERE gas = 'co2e_100yr' AND iso3_country IS NOT NULL
              AND length(iso3_country) = 3 {where_extra}
            GROUP BY iso3_country, {group_col}
        )
        SELECT iso3_country, grp,
               RANK() OVER (PARTITION BY grp ORDER BY total DESC) AS rank
        FROM totals WHERE total > 0
    """)
    global_rank_map = (
        global_rank_df[global_rank_df['iso3_country'] == iso3]
        .set_index('grp')['rank'].to_dict()
    )

    cont_rank_df = _cached_df(f"""
        WITH totals AS (
            SELECT iso3_country, {group_col} AS grp,
                   SUM({to_col}) AS total
            FROM '{stats_path}'
            WHERE gas = 'co2e_100yr'
              AND continent = '{country_continent}'
              AND iso3_country IS NOT NULL AND length(iso3_country) = 3
              {where_extra}
            GROUP BY iso3_country, {group_col}
        )
        SELECT iso3_country, grp,
               RANK() OVER (PARTITION BY grp ORDER BY total DESC) AS rank
        FROM totals WHERE total > 0
    """)
    cont_rank_map = (
        cont_rank_df[cont_rank_df['iso3_country'] == iso3]
        .set_index('grp')['rank'].to_dict()
    )

    global_count_df = _cached_df(f"""
        SELECT {group_col} AS grp, COUNT(DISTINCT iso3_country) AS n
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr' AND length(iso3_country) = 3
          AND {to_col} > 0 {where_extra}
        GROUP BY {group_col}
    """)
    global_count_map = dict(zip(global_count_df['grp'], global_count_df['n']))

    cont_count_df = _cached_df(f"""
        SELECT {group_col} AS grp, COUNT(DISTINCT iso3_country) AS n
        FROM '{stats_path}'
        WHERE gas = 'co2e_100yr'
          AND continent = '{country_continent}'
          AND length(iso3_country) = 3
          AND {to_col} > 0 {where_extra}
        GROUP BY {group_col}
    """)
    cont_count_map = dict(zip(cont_count_df['grp'], cont_count_df['n']))

    # ── Emissions factor (only when sector is selected) ───────────────────────
    ef_country_map = ef_continent_map = ef_global_map = ef_unit_map = None
    ef_pct_map = {}
    if selected_sector:
        # Country EF (latest available month)
        ef_country_df = _cached_df(f"""
            WITH latest AS (
                SELECT original_inventory_sector, MAX(start_time) AS max_t
                FROM '{asset_path}'
                WHERE iso3_country = '{iso3}' AND sector = '{selected_sector}'
                  AND gas = 'co2e_100yr'
                  AND weighted_average_emissions_factor > 0
                  AND weighted_average_emissions_factor IS NOT NULL
                GROUP BY original_inventory_sector
            )
            SELECT m.original_inventory_sector AS grp,
                   m.weighted_average_emissions_factor AS ef,
                   m.activity_units
            FROM '{asset_path}' m
            JOIN latest l
              ON m.original_inventory_sector = l.original_inventory_sector
             AND m.start_time = l.max_t
            WHERE m.iso3_country = '{iso3}' AND m.sector = '{selected_sector}'
              AND m.gas = 'co2e_100yr'
              AND m.weighted_average_emissions_factor > 0
        """)
        ef_country_map = dict(zip(ef_country_df['grp'], ef_country_df['ef']))
        # EF unit = "t CO₂e / {activity_units}"
        ef_unit_map = {
            grp: f"t CO₂e / {unit}" if unit else "—"
            for grp, unit in zip(ef_country_df['grp'], ef_country_df['activity_units'])
        }

        # Continent activity-weighted avg EF (NULLIF handles zero-activity rows)
        ef_cont_df = _cached_df(f"""
            WITH latest AS (
                SELECT original_inventory_sector, MAX(start_time) AS max_t
                FROM '{asset_path}'
                WHERE sector = '{selected_sector}' AND gas = 'co2e_100yr'
                  AND continent = '{country_continent}' AND length(iso3_country) = 3
                  AND weighted_average_emissions_factor > 0
                GROUP BY original_inventory_sector
            )
            SELECT m.original_inventory_sector AS grp,
                   SUM(m.weighted_average_emissions_factor * NULLIF(m.activity, 0))
                     / NULLIF(SUM(NULLIF(m.activity, 0)), 0) AS ef
            FROM '{asset_path}' m
            JOIN latest l
              ON m.original_inventory_sector = l.original_inventory_sector
             AND m.start_time = l.max_t
            WHERE m.sector = '{selected_sector}' AND m.gas = 'co2e_100yr'
              AND m.continent = '{country_continent}' AND length(m.iso3_country) = 3
              AND m.weighted_average_emissions_factor > 0
            GROUP BY m.original_inventory_sector
        """)
        ef_continent_map = dict(zip(ef_cont_df['grp'], ef_cont_df['ef']))

        # Global activity-weighted avg EF (NULLIF handles zero-activity rows)
        ef_glob_df = _cached_df(f"""
            WITH latest AS (
                SELECT original_inventory_sector, MAX(start_time) AS max_t
                FROM '{asset_path}'
                WHERE sector = '{selected_sector}' AND gas = 'co2e_100yr'
                  AND length(iso3_country) = 3
                  AND weighted_average_emissions_factor > 0
                GROUP BY original_inventory_sector
            )
            SELECT m.original_inventory_sector AS grp,
                   SUM(m.weighted_average_emissions_factor * NULLIF(m.activity, 0))
                     / NULLIF(SUM(NULLIF(m.activity, 0)), 0) AS ef
            FROM '{asset_path}' m
            JOIN latest l
              ON m.original_inventory_sector = l.original_inventory_sector
             AND m.start_time = l.max_t
            WHERE m.sector = '{selected_sector}' AND m.gas = 'co2e_100yr'
              AND length(m.iso3_country) = 3
              AND m.weighted_average_emissions_factor > 0
            GROUP BY m.original_inventory_sector
        """)
        ef_global_map = dict(zip(ef_glob_df['grp'], ef_glob_df['ef']))

        # Fill unit from global data when country row is missing it
        for grp in ef_global_map:
            if grp not in ef_unit_map:
                unit_row = _cached_fetchone(f"""
                    SELECT activity_units FROM '{asset_path}'
                    WHERE sector = '{selected_sector}'
                      AND original_inventory_sector = '{grp}'
                      AND gas = 'co2e_100yr'
                    LIMIT 1
                """)
                if unit_row:
                    ef_unit_map[grp] = f"t CO₂e / {unit_row[0]}" if unit_row[0] else "—"

        # EF percentile: country's EF vs all individual assets globally
        # Fetch all asset EFs for the sector at latest year, compute in pandas
        annual_path = CONFIG['annual_asset_path']
        asset_ef_df = _cached_df(f"""
            SELECT subsector, average_emissions_factor
            FROM '{annual_path}'
            WHERE sector = '{selected_sector}'
              AND average_emissions_factor > 0
              AND average_emissions_factor IS NOT NULL
              AND year = (
                  SELECT MAX(year) FROM '{annual_path}'
                  WHERE sector = '{selected_sector}' AND year IS NOT NULL
              )
        """)

        ef_pct_map = {}
        for grp in col_keys:
            country_ef_val = ef_country_map.get(grp)
            if country_ef_val is None:
                continue
            sub_efs = asset_ef_df.loc[
                asset_ef_df['subsector'] == grp, 'average_emissions_factor'
            ]
            if len(sub_efs) > 0:
                pct = (sub_efs <= country_ef_val).sum() / len(sub_efs) * 100
                ef_pct_map[grp] = (round(pct, 1), len(sub_efs))

    # ── Derived values ────────────────────────────────────────────────────────
    country_y_to_map   = dict(zip(country_df['grp'], country_df['y_to']))
    country_y_from_map = dict(zip(country_df['grp'], country_df['y_from']))
    country_y_prev_map = dict(zip(country_df['grp'], country_df['y_prev']))

    pct_global = {
        k: (country_y_to_map[k] / global_y_to_map[k] * 100)
        if global_y_to_map.get(k, 0) > 0 else 0
        for k in col_keys
    }
    pct_continent = {
        k: (country_y_to_map[k] / cont_y_to_map[k] * 100)
        if cont_y_to_map.get(k, 0) > 0 else 0
        for k in col_keys
    }
    chg_vals = {
        k: ((country_y_to_map[k] - country_y_from_map[k]) / abs(country_y_from_map[k]) * 100)
        if country_y_from_map.get(k, 0) != 0 else None
        for k in col_keys
    }
    yoy_vals = {
        k: ((country_y_to_map[k] - country_y_prev_map[k]) / abs(country_y_prev_map[k]) * 100)
        if to_year > 2021 and country_y_prev_map.get(k, 0) != 0 else None
        for k in col_keys
    }

    # ── Country flag ─────────────────────────────────────────────────────────
    _flag_df = get_iso3_flag_table()
    _iso3_flag_dict = dict(zip(_flag_df['iso3'], _flag_df['flag']))
    country_flag = _iso3_flag_dict.get(iso3, '🏳')

    # ── Column styling ────────────────────────────────────────────────────────
    if selected_sector:
        base = SECTOR_COLORS.get(selected_sector, '#888888')
        n = len(col_keys)
        col_colors = [_shade_hex(base, int((i - n // 2) * 20)) for i in range(n)]
        col_labels = [k.replace('-', ' ').title() for k in col_keys]
    else:
        col_colors = [SECTOR_COLORS.get(k, '#888888') for k in col_keys]
        col_labels = [SECTOR_LABELS_SHORT.get(k, k.replace('-', ' ').title()) for k in col_keys]

    # ── Shared data for charts ────────────────────────────────────────────────
    pie_vals_mt = [country_y_to_map.get(k, 0) / 1e6 for k in col_keys]
    total_mt = sum(pie_vals_mt)
    trend_years = list(range(from_year, to_year + 1))
    yr_col_map = {2021: 'y2021', 2022: 'y2022', 2023: 'y2023', 2024: 'y2024', 2025: 'y2025'}

    # ── Pre-compute download data ─────────────────────────────────────────────
    pie_dl_df = pd.DataFrame({
        'label': col_labels,
        'key': col_keys,
        f'emissions_mt_{to_year}': pie_vals_mt,
        'pct_of_total': [(v / total_mt * 100) if total_mt else 0 for v in pie_vals_mt],
    })

    trend_traces = []
    trend_dl_rows = []
    for k, label, color in zip(col_keys, col_labels, col_colors):
        row_data = country_df[country_df['grp'] == k]
        if row_data.empty:
            continue
        r = row_data.iloc[0]
        yoy_pts, abs_pts, valid_years = [], [], []
        for yr in trend_years:
            curr_val = r.get(yr_col_map.get(yr, ''), None)
            if yr == from_year:
                yoy_pts.append(0.0)
                abs_pts.append(curr_val / 1e6 if curr_val is not None else None)
                valid_years.append(yr)
            else:
                prev_val = r.get(yr_col_map.get(yr - 1, ''), None)
                yoy_pts.append(
                    (curr_val - prev_val) / abs(prev_val) * 100
                    if curr_val is not None and prev_val and prev_val != 0 else None
                )
                abs_pts.append(curr_val / 1e6 if curr_val is not None else None)
                valid_years.append(yr)
        line_width = 3 if (selected_subsector and k == selected_subsector) else 2
        trend_traces.append((k, label, color, yoy_pts, line_width))
        for yr, yoy, abs_val in zip(valid_years, yoy_pts, abs_pts):
            trend_dl_rows.append({
                'year': yr, 'key': k, 'label': label,
                'yoy_pct_change': round(yoy, 4) if yoy is not None else None,
                'emissions_mt': round(abs_val, 4) if abs_val is not None else None,
            })
    trend_dl_df = pd.DataFrame(trend_dl_rows) if trend_dl_rows else None

    # ── Layout: left col (pie + trend + table) | right col (reductions) ─────
    col_left, col_right = st.columns([66, 34])

    # ── Pie + Trend side by side inside col_left ─────────────────────────────
    with col_left:
        # Header row: title + download inline for each chart (level-2 cols, no nesting)
        hdr_pie, dl_pie, hdr_trend, dl_trend = st.columns([42, 8, 42, 8])
        pie_subtitle = (
            f"{SECTOR_LABELS.get(selected_sector, '')} Subsectors"
            if selected_sector else "Emissions by Sector"
        )
        trend_subtitle = (
            f"{SECTOR_LABELS.get(selected_sector, '')} · YoY % Change"
            if selected_sector else "All Sectors · YoY % Change"
        )
        with hdr_pie:
            st.markdown(
                f"**{pie_subtitle}**  "
                f"<span style='font-size:13px;color:#888'>{to_year}</span>",
                unsafe_allow_html=True,
            )
        with dl_pie:
            st.download_button(
                "⬇", data=_df_to_csv(pie_dl_df),
                file_name=f"country_pie_{selected_country}_{to_year}.csv",
                mime="text/csv", key="ct_dl_pie",
            )
        with hdr_trend:
            st.markdown(f"**{trend_subtitle}**", unsafe_allow_html=True)
        with dl_trend:
            if trend_dl_df is not None:
                st.download_button(
                    "⬇", data=_df_to_csv(trend_dl_df),
                    file_name=f"country_trend_{selected_country}_{from_year}_{to_year}.csv",
                    mime="text/csv", key="ct_dl_trend",
                )

        # Chart row
        sub_pie, sub_trend = st.columns(2)

        with sub_pie:
            fig_pie = go.Figure(go.Pie(
                labels=col_labels,
                values=pie_vals_mt,
                marker=dict(colors=col_colors, line=dict(color='white', width=2)),
                hole=0.55,
                domain=dict(x=[0, 0.58]),
                textinfo='percent',
                textposition='inside',
                insidetextorientation='auto',
                textfont=dict(size=12),
                hovertemplate='%{label}<br>%{value:.3f} MtCO₂e (%{percent})<extra></extra>',
            ))
            if selected_subsector and selected_subsector in col_keys:
                pulls = [0.0] * len(col_keys)
                pulls[col_keys.index(selected_subsector)] = 0.08
                fig_pie.data[0].pull = pulls
            fig_pie.add_annotation(
                text=f"<b>{_fmt_mt(total_mt)}</b><br>Mt total",
                x=0.29, y=0.5, showarrow=False,
                font=dict(size=13), align='center',
            )
            fig_pie.update_layout(
                font=dict(size=13),
                showlegend=True,
                legend=dict(
                    orientation='v', x=0.62, y=0.5,
                    xanchor='left', yanchor='middle',
                    font=dict(size=12), bgcolor='rgba(0,0,0,0)',
                    itemsizing='constant',
                ),
                margin=dict(t=5, b=5, l=5, r=5),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="ct_pie")

        with sub_trend:
            if trend_years:
                fig_trend = go.Figure()
                for k, label, color, yoy_pts, line_width in trend_traces:
                    fig_trend.add_trace(go.Scatter(
                        x=trend_years, y=yoy_pts, name=label,
                        line=dict(color=color, width=line_width),
                        mode='lines+markers',
                        marker=dict(size=5 if (selected_subsector and k == selected_subsector) else 4),
                        hovertemplate='%{x}: %{y:.1f}% (YoY)<extra>' + label + '</extra>',
                    ))
                fig_trend.add_hline(y=0, line_dash='dash', line_color='#ccc', line_width=1)
                fig_trend.update_layout(
                    font=dict(size=13),
                    showlegend=False,
                    margin=dict(t=5, b=5, l=50, r=5),
                    height=260,
                    yaxis=dict(ticksuffix='%', gridcolor='#f0f2f5', zeroline=False,
                               tickfont=dict(size=13)),
                    xaxis=dict(gridcolor='rgba(0,0,0,0)', tickformat='d', tickvals=trend_years,
                               tickfont=dict(size=13)),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="ct_trend")

    # ── Reduction Strategies (right col, full height) ────────────────────────
    with col_right:
        ers_year = CONFIG['ers_baseline_year']
        reductions_label = (
            selected_subsector.replace('-', ' ').title() if selected_subsector
            else (SECTOR_LABELS.get(selected_sector, '') if selected_sector else '')
        )
        st.markdown(
            f"**Reduction Strategies**  "
            f"<span style='font-size:13px;color:#888'>"
            f"{reductions_label + ' · ' if reductions_label else ''}"
            f"{selected_country} · {ers_year}</span>",
            unsafe_allow_html=True,
        )

        sector_filter = f"AND sector = '{selected_sector}'" if selected_sector else ''
        subsector_filter = f"AND subsector = '{selected_subsector}'" if selected_subsector else ''
        reductions_df = _cached_df(f"""
            WITH best AS (
                SELECT asset_id, sector, subsector, strategy_name, strategy_description,
                       total_emissions_reduced_per_year, emissions_quantity,
                       asset_difficulty_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY asset_id
                           ORDER BY asset_difficulty_score ASC
                       ) AS rn
                FROM '{CONFIG['annual_asset_path']}'
                WHERE iso3_country = '{iso3}'
                  AND year = {ers_year}
                  AND strategy_name IS NOT NULL
                  AND reduction_q_type = 'asset'
                  AND total_emissions_reduced_per_year > 0
                  {sector_filter}
                  {subsector_filter}
            )
            SELECT
                strategy_name,
                sector,
                subsector,
                MAX(strategy_description) AS description,
                COUNT(DISTINCT asset_id) AS n_assets,
                SUM(total_emissions_reduced_per_year) AS reductions,
                SUM(emissions_quantity) AS inventory
            FROM best
            WHERE rn = 1
            GROUP BY strategy_name, sector, subsector
            ORDER BY reductions DESC
        """)

        if reductions_df.empty:
            st.info("No reduction strategies found for the selected filters.")
        else:
            total_inv = reductions_df['inventory'].sum()
            total_red = reductions_df['reductions'].sum()
            total_pct = (total_red / total_inv * 100) if total_inv > 0 else 0
            accent = SECTOR_COLORS.get(selected_sector, '#4380F5') if selected_sector else '#4380F5'
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px'>"
                f"<div class='ct-summary-card' style='border-top:3px solid #555d72'>"
                f"<div class='ct-summary-label'>Inventory</div>"
                f"<div class='ct-summary-value' style='color:#555d72'>{_fmt_mt(total_inv / 1e6)} Mt</div>"
                f"</div>"
                f"<div class='ct-summary-card' style='border-top:3px solid #2e7d32'>"
                f"<div class='ct-summary-label'>Reducible</div>"
                f"<div class='ct-summary-value' style='color:#2e7d32'>{_fmt_mt(total_red / 1e6)} Mt</div>"
                f"</div>"
                f"<div class='ct-summary-card' style='border-top:3px solid {accent}'>"
                f"<div class='ct-summary-label'>Strategies</div>"
                f"<div class='ct-summary-value' style='color:{accent}'>{len(reductions_df)}</div>"
                f"</div>"
                f"<div class='ct-summary-card' style='border-top:3px solid #2e7d32'>"
                f"<div class='ct-summary-label'>Potential</div>"
                f"<div class='ct-summary-value' style='color:#2e7d32'>{total_pct:.1f}%</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            items_html = """<style>
details.ct-strat summary::-webkit-details-marker{display:none}
details.ct-strat summary::after{
    content:'▶';font-size:11px;color:#aaa;margin-left:8px;
    display:inline-block;transition:transform 0.2s;flex-shrink:0;
}
details.ct-strat[open] summary::after{transform:rotate(90deg);color:#555;}
</style>"""
            for _, strat in reductions_df.iterrows():
                inv_mt  = strat['inventory'] / 1e6
                red_mt  = strat['reductions'] / 1e6
                pct_red = (strat['reductions'] / strat['inventory'] * 100
                           if strat['inventory'] > 0 else 0)
                n_assets  = int(strat['n_assets'])
                sec_key   = strat['sector']
                sub_key   = strat['subsector']
                color     = SECTOR_COLORS.get(sec_key, '#888888')
                sec_label = SECTOR_LABELS.get(sec_key, sec_key.replace('-', ' ').title())
                sub_label = sub_key.replace('-', ' ').title()
                desc      = str(strat['description']).strip()
                desc_html = (
                    f"<p style='font-size:13px;color:#555;margin:6px 0 10px'>{desc}</p>"
                    if desc and desc.lower() not in ('nan', 'none', '') else ''
                )
                items_html += (
                    f"<details class='ct-strat' style='"
                    f"border-left:4px solid {color};border-radius:0 8px 8px 0;"
                    f"background:#f8f9fb;margin-bottom:6px;padding:0'>"
                    f"<summary style='padding:9px 13px;cursor:pointer;font-size:13px;"
                    f"font-weight:600;display:flex;align-items:center;gap:8px'>"
                    f"<span>{strat['strategy_name']}</span>"
                    f"<span style='white-space:nowrap;color:#2e7d32;font-weight:700;margin-left:auto'>"
                    f"{_fmt_mt(red_mt)} Mt &nbsp;−{pct_red:.1f}%</span>"
                    f"</summary>"
                    f"<div style='padding:6px 13px 12px'>"
                    f"<div style='margin-bottom:8px'>"
                    f"<span style='background:{color}22;color:{color};border:1px solid {color}55;"
                    f"border-radius:10px;padding:2px 10px;font-size:12px;font-weight:700'>"
                    f"{sec_label}</span>"
                    f"<span style='margin-left:7px;font-size:13px;color:#777'>{sub_label}</span>"
                    f"</div>"
                    f"{desc_html}"
                    f"<div style='display:flex;gap:20px;font-size:13px;color:#555d72'>"
                    f"<div><span style='font-weight:700;color:#3a4158'>Inventory</span>"
                    f"<br>{_fmt_mt(inv_mt)} Mt</div>"
                    f"<div><span style='font-weight:700;color:#2e7d32'>Reduction</span>"
                    f"<br>{_fmt_mt(red_mt)} Mt"
                    f" <span style='color:#2e7d32'>−{pct_red:.1f}%</span></div>"
                    f"<div><span style='font-weight:700;color:#3a4158'>Assets</span>"
                    f"<br>{n_assets:,}</div>"
                    f"</div></div></details>"
                )
            with st.container(height=520, border=False):
                st.markdown(items_html, unsafe_allow_html=True)

    # ── Sector Breakdown table (inside col_left, below charts) ───────────────
    with col_left:
      st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    table_heading = (
        f"**{selected_country}"
        + (f" · {SECTOR_LABELS.get(selected_sector, selected_sector)}" if selected_sector else "")
        + " · Sector Breakdown**  "
        + f"<span style='font-size:13px;color:#888'>"
        + f"Global, {country_continent} &amp; Country · {to_year}</span>"
    )
    table_rows = []
    ef_rows_exist = ef_country_map is not None
    for k, label in zip(col_keys, col_labels):
        row_d = {
            'key': k, 'label': label,
            f'global_emissions_mt_{to_year}': round(global_y_to_map.get(k, 0) / 1e6, 4),
            f'{country_continent}_emissions_mt_{to_year}': round(cont_y_to_map.get(k, 0) / 1e6, 4),
            f'country_emissions_mt_{to_year}': round(country_y_to_map.get(k, 0) / 1e6, 4),
            'pct_of_global': round(pct_global.get(k, 0), 2),
            f'pct_of_{country_continent.replace(" ", "_")}': round(pct_continent.get(k, 0), 2),
            'global_rank': global_rank_map.get(k),
            'global_n_countries': global_count_map.get(k),
            f'{country_continent.replace(" ", "_")}_rank': cont_rank_map.get(k),
            f'{country_continent.replace(" ", "_")}_n_countries': cont_count_map.get(k),
            f'change_pct_{from_year}_to_{to_year}': round(chg_vals.get(k), 2) if chg_vals.get(k) is not None else None,
            f'yoy_pct_{to_year - 1}_to_{to_year}': round(yoy_vals.get(k), 2) if yoy_vals.get(k) is not None else None,
        }
        if ef_rows_exist:
            row_d['country_ef'] = ef_country_map.get(k)
            row_d[f'{country_continent.replace(" ", "_")}_avg_ef'] = ef_continent_map.get(k) if ef_continent_map else None
            row_d['global_avg_ef'] = ef_global_map.get(k) if ef_global_map else None
            row_d['ef_unit'] = ef_unit_map.get(k) if ef_unit_map else None
        table_rows.append(row_d)
    table_dl_df = pd.DataFrame(table_rows)

    with col_left:
        tbl_title_col, tbl_dl_col = st.columns([8, 1])
        with tbl_title_col:
            st.markdown(table_heading, unsafe_allow_html=True)
        with tbl_dl_col:
            st.download_button(
                "⬇ Data", data=_df_to_csv(table_dl_df),
                file_name=f"country_table_{selected_country}_{to_year}.csv",
                mime="text/csv", key="ct_dl_table",
            )
        st.markdown(
            _build_sector_table(
                col_keys=col_keys, col_labels=col_labels, col_colors=col_colors,
                country_vals=country_y_to_map, global_vals=global_y_to_map,
                continent_vals=cont_y_to_map,
                pct_global=pct_global, pct_continent=pct_continent,
                global_rank_map=global_rank_map, global_count_map=global_count_map,
                continent_rank_map=cont_rank_map, continent_count_map=cont_count_map,
                chg_vals=chg_vals, yoy_vals=yoy_vals,
                from_year=from_year, to_year=to_year,
                continent_name=country_continent,
                iso3=iso3, country_flag=country_flag,
                ef_country=ef_country_map, ef_continent=ef_continent_map,
                ef_global=ef_global_map, ef_units=ef_unit_map,
                ef_pct_map=ef_pct_map, highlighted_key=selected_subsector,
            ),
            unsafe_allow_html=True,
        )
