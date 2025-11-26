import streamlit as st
import streamlit.components.v1 as components
import duckdb
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
import base64
from calendar import month_name
import calendar
from collections import defaultdict
from config import CONFIG
from utils.utils import *
from utils.queries import *
import logging


def show_reduction_heatmap():

    logging.getLogger("streamlit.dataframe_util").setLevel(logging.ERROR)

    st.markdown(
        """
        <style>
        /* Hide the sidebar completely */
        section[data-testid="stSidebar"] {
            display: none;
        }
        /* Hide the sidebar collapse/expand arrow */
        [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # configure data paths and region options for querying
    annual_asset_path = CONFIG['annual_asset_path']
    city_path = CONFIG['city_path']
    gadm_0_path = CONFIG['gadm_0_path']
    gadm_1_path = CONFIG['gadm_1_path']
    gadm_2_path = CONFIG['gadm_2_path']
    country_subsector_totals_path = CONFIG['country_subsector_totals_path']
    percentile_path = CONFIG['percentile_path']
    region_options = CONFIG['region_options']
    gadm_0_path = CONFIG['gadm_0_path']

    con = duckdb.connect()

    country_rows = con.execute(
            f"SELECT DISTINCT country_name, iso3_country FROM '{gadm_0_path}' WHERE country_name IS NOT NULL order by country_name"
        ).fetchall()

    country_map = {row[0]: row[1] for row in country_rows}
    
    unique_countries = list(country_map.keys())

    st.markdown("<br>", unsafe_allow_html=True)


    country_dropdown, state_province_dropdown  = st.columns([2,2])

    with country_dropdown:
        
        selected_region = st.selectbox(
            "Country", 
            ["Global"] + ["G20"] + unique_countries, 
            key="selected_region_HM",
            #on_change=mark_hm_recompute
        )

        region_condition = map_region_condition(selected_region, country_map)

    country_selected_bool = selected_region not in ["Global","G20"]

    state_selected_bool = False
    with state_province_dropdown:
        if not country_selected_bool:
            state_province_options = ['Select a Country to Enable']
            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=True,
                key="state_province_selector_HM",
                index=0,
                #on_change=mark_hm_recompute
            )

            state_selected_bool = False

        else:
            col_value = region_condition['column_value']

            state_province_options = ['-- Select State / Province --'] + sorted(
                row[0] for row in con.execute(
                    f"SELECT DISTINCT gadm_1_corrected_name FROM '{gadm_1_path}' WHERE iso3_country = '{col_value}' and gadm_1_name is not null"
                ).fetchall()
            )

            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=False,
                key="state_province_selector_HM",
                index=0,
                # on_change=reset_city
            )

    
    g20_bool = False
    if selected_region == 'G20':
        g20_bool = True

    if selected_state_province not in ['-- Select State / Province --', 'Select a Country to Enable']:
        state_selected_bool = True
        
    selected_metric = st.radio(
        "Select metric",
        ['reductions', 'emissions_quantity'],
        horizontal=True,
        key="selection_metric_HM"
    )

    heatmap_sql = create_heatmap_sql(country_selected_bool=country_selected_bool,
                                     state_selected_bool=state_selected_bool,
                                     g20_bool=g20_bool,
                                     region_condition=region_condition,
                                     selected_state_province=selected_state_province,
                                     annual_asset_path=annual_asset_path,
                                     gadm_1_path=gadm_1_path,
                                     gadm_2_path=gadm_2_path,
                                     selected_metric=selected_metric)
    
    # print(heatmap_sql['sector_summary'])
    
    sector_df = con.execute(heatmap_sql['sector_summary']).df()
    table_df = con.execute(heatmap_sql['table_summary']).df()

    sector_df = sector_df.loc[:, ~sector_df.columns.duplicated()].copy()
    table_df  = table_df.loc[:, ~table_df.columns.duplicated()].copy()

    sector_df = sector_df.reset_index(drop=True)
    table_df = table_df.reset_index(drop=True)

    # sector_df.insert(0, "Region", ["Total"])
    if "country_name" in table_df.columns:
        table_df.rename(columns={"country_name": "Region"}, inplace=True)

    # --- Add one empty spacer row between them ---
    spacer = pd.DataFrame(columns=sector_df.columns)
    for col in sector_df.columns:
        if np.issubdtype(sector_df[col].dtype, np.number):
            spacer.at[0, col] = np.nan
        else:
            spacer.at[0, col] = ""

    # --- Combine everything ---
    combined_df = pd.concat([sector_df, spacer, table_df], ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()].copy()

    combined_df = combined_df.drop(columns=[col for col in combined_df.columns if str(col).strip() == ""], errors="ignore")

    


    # --- Define excluded and numeric columns ---
    excluded_cols = [
        "Region",
        "total_exc_forestry",
        "forestry_and_land_use",
        "total_reduction_potential",
        "asset_count"
    ]
    numeric_cols = [c for c in combined_df.select_dtypes(include=["number"]).columns if c not in excluded_cols]

    # --- Colormap ---
    # universal_red_cmap = LinearSegmentedColormap.from_list(
    #     "universal_orange",
    #     [
    #         (0.0, "#FFFFFF"),   # white
    #         (0.25, "#FFE5D0"),  # light peachy-orange
    #         (0.55, "#F9A66C"),  # vibrant warm orange
    #         (0.85, "#EF6C00"),  # vivid deep orange
    #         (1.0, "#C75B12"),   # burnt orange / toned-down dark
    #     ]
    # )

    universal_red_cmap = LinearSegmentedColormap.from_list(
        "universal_orange",
        [
            (0.00, "#FFFFFF"),   # white
            (0.50, "#F9A66C"),   # medium orange
            (1.00, "#E07B3D"),   # **softer terracotta orange**
        ]
    )


    def mixed_gradient(df, color_cols, dark_mode=False, low_thresh=0.05):
        colors = pd.DataFrame("", index=df.index, columns=df.columns)
        text_color = "white" if dark_mode else "black"

        # Protect against single-row DataFrames (no variance)
        if len(df) == 1:
            # normalize across that single row
            row = df[color_cols].iloc[0].astype(float)
            vmin, vmax = row.min(), row.max()
            scale = (vmax - vmin) or 1.0
            row_scaled = (row - vmin) / scale

            for col in color_cols:
                v = row_scaled[col]
                if np.isfinite(v) and v > low_thresh:
                    r, g, b, _ = universal_red_cmap(v)
                    r, g, b = [int(c * 255) for c in (r, g, b)]
                    bg = f"rgb({r},{g},{b})"
                else:
                    bg = "white" if not dark_mode else "#1E1E1E"

                colors.at[df.index[0], col] = f"background-color: {bg}; color: {text_color}; font-weight: bold;"
            return colors

        # Multi-row logic (normal case)
        mins = df[color_cols].min()
        maxs = df[color_cols].max()
        rngs = (maxs - mins).replace(0, 1.0)
        scaled = (df[color_cols] - mins) / rngs

        for col in color_cols:
            for idx in df.index:
                v = scaled.at[idx, col]
                if np.isfinite(v) and v > low_thresh:
                    r, g, b, _ = universal_red_cmap(v)
                    r, g, b = [int(c * 255) for c in (r, g, b)]
                    bg = f"rgb({r},{g},{b})"
                else:
                    bg = "white" if not dark_mode else "#1E1E1E"

                colors.at[idx, col] = f"background-color: {bg}; color: {text_color};"

        return colors


    # --- Apply highlight and gradient ---
    def bold_first_row(row):
        if row.name == 0:
            return ['font-weight: bold;' for _ in row]
        else:
            return ['' for _ in row]



    # --- Convert to numeric for all non-Region columns ---
    for col in combined_df.columns:
        if col != "Region":
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # --- Define which columns get color vs only formatting ---
    color_cols = [
        "agriculture",
        "buildings",
        "fluorinated_gases",
        "fossil_fuel_operations",
        "manufacturing",
        "mineral_extraction",
        "power",
        "transportation",
        "waste"
    ]

    # --- Identify numeric columns for formatting ---
    numeric_cols = [
        c for c in combined_df.columns
        if c != "Region" and np.issubdtype(combined_df[c].dtype, np.number)
    ]

    # --- Blank out the spacer row properly ---
    spacer_idx = combined_df.index[combined_df["Region"] == ""].tolist()
    if spacer_idx:
        r = spacer_idx[0]
        for c in combined_df.columns:
            combined_df.loc[r, c] = "" if c != "Region" else ""  # all blanks

    # --- Define a safe formatter function ---
    def safe_format(x):
        # Handle missing or None/NaN explicitly
        if x is None or x == "None" or (isinstance(x, float) and np.isnan(x)):
            return ""
        try:
            return f"{float(x):,.0f}"
        except (ValueError, TypeError):
            return ""
        
    def get_max_column_widths(df1, df2, font_char_width=6, padding=12):
        """
        Estimate consistent column widths across two DataFrames based on the longest
        string length (including headers).
        """
        widths = {}
        all_cols = [c for c in df1.columns if c in df2.columns]

        for col in all_cols:
            max_len = max(
                df1[col].astype(str).map(len).max(),
                df2[col].astype(str).map(len).max(),
                len(str(col))
            )
            widths[col] = max_len * font_char_width + padding

        return widths


    # --- Compute column widths ---
    col_widths = get_max_column_widths(sector_df, table_df)
    col_widths["Region"] = 90


    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### **Regional Reduction Opportunities by Sector**")

    # --- Display first table ---
    st.dataframe(
        sector_df.style
            .apply(lambda df: mixed_gradient(df, color_cols=color_cols), axis=None)
            .format(subset=numeric_cols, formatter=safe_format)
            .applymap(lambda v: "font-weight: bold;"),
        use_container_width=False,
        hide_index=True,
        column_config={
            col: st.column_config.Column(width=int(col_widths.get(col, 100)))
            for col in sector_df.columns
        },
    )

    # --- Display second table ---
    st.dataframe(
        table_df.style
            .apply(lambda df: mixed_gradient(df, color_cols=color_cols), axis=None)
            .format(subset=numeric_cols, formatter=safe_format)
            .hide(axis="columns"),
        use_container_width=False,
        hide_index=True,
        column_config={
            col: st.column_config.Column(width=int(col_widths.get(col, 100)))
            for col in table_df.columns
        },
        height=800
    )
