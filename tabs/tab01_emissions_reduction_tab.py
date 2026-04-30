import streamlit as st
import streamlit.components.v1 as components
import duckdb
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from calendar import month_name
import calendar
from collections import defaultdict
from config import CONFIG
from utils.utils import *
from utils.queries import *


def show_emissions_reduction_plan():

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
    country_subsector_stats_path = CONFIG['country_subsector_stats_path']
    percentile_path = CONFIG['percentile_path']
    region_options = CONFIG['region_options']
    gadm_0_path = CONFIG['gadm_0_path']
    ers_baseline_year = CONFIG['ers_baseline_year']

    con = duckdb.connect()

    country_rows = con.execute(
            f"SELECT DISTINCT country_name, iso3_country FROM '{gadm_0_path}' WHERE country_name IS NOT NULL order by country_name"
        ).fetchall()

    country_map = {row[0]: row[1] for row in country_rows}
    unique_countries = list(country_map.keys())

    selected_year = ers_baseline_year
    
    # st.markdown("<br>", unsafe_allow_html=True)

    year_col, radio_col, download_col = st.columns([1.1, 10, 1])

    # with radio_col:
    #     reduction_method = st.radio(
    #         "Select emissions reduction method:",
    #         [
    #             "Climate TRACE Solutions",
    #             "Percentile Benchmarking"
    #         ],
    #         horizontal=True,
    #         index=0,
    #         help=(
    #             "**Climate TRACE Solutions**: Uses asset-specific emissions reduction strategies "
    #             "developed by Climate TRACE sector leads.\n\n"
    #             "**Percentile Benchmarking**: Compares each asset’s emissions factor against " 
    #             "similar assets within the same subsector. Results are shown as global "
    #             "or country-level percentiles, helping identify how efficient or polluting an asset is relative "
    #             "to its peers, and how much room there is for improvement."
    #         ),
    #         key="reduction_method_RO",
    #         on_change=mark_ro_recompute
    #     )

    reduction_method = "Climate TRACE Solutions"

    # st.markdown("<br>", unsafe_allow_html=True)

    if reduction_method == "Climate TRACE Solutions":
        use_ct_ers = True
    else:
        use_ct_ers = False

    # --------- DROPDOWN ROW 1 ----------

    country_dropdown, state_province_dropdown, county_district_dropdown, city_dropdown, forestry_toggle, download_col  = st.columns([2,2.3,2.3,2.3,1.75,1])

    with country_dropdown:
        
        selected_region = st.selectbox(
            "Region/Country", 
            region_options + unique_countries, 
            key="selected_region_RO",
            on_change=mark_ro_recompute
        )

        region_condition = map_region_condition(selected_region, country_map)

    if region_condition is not None:
        col = region_condition['column_name']
        val = region_condition['column_value']
        if isinstance(val, list):
            val_str = "(" + ", ".join(f"'{v}'" for v in val) + ")"
        else:
            val_str = f"('{val}')"
    else:
        col = None
        val = None
        val_str = None

    country_selected_bool = selected_region != "Global"

    with state_province_dropdown:
        if not country_selected_bool:
            state_province_options = ['Select a Region/Country to Enable']
            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=True,
                key="state_province_selector_RO",
                index=0,
                on_change=mark_ro_recompute
            )
        else:
            state_province_options = ['-- Select State / Province --'] + sorted(
                row[0] for row in con.execute(
                    f"SELECT DISTINCT gadm_1_name FROM '{gadm_1_path}' WHERE {col} IN {val_str} and gadm_1_name is not null"
                ).fetchall()
            )

            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=False,
                key="state_province_selector_RO",
                index=0,
                on_change=reset_city
            )

    # --- COUNTY / DISTRICT ---
    with county_district_dropdown:
        if not country_selected_bool:
            county_district_options = ['Select a Region/Country to Enable']
            selected_county_district = st.selectbox(
                "County / District",
                county_district_options,
                disabled=True,
                key="county_district_selector_RO",
                index=0,
                on_change=mark_ro_recompute
            )
        else:
            if selected_state_province and not selected_state_province.startswith("--"):
                safe_state = selected_state_province.replace("'", "''")
                county_district_options = ['-- Select County / District --'] + sorted(
                    row[0] for row in con.execute(
                        f"SELECT DISTINCT gadm_2_name FROM '{gadm_2_path}' WHERE gadm_1_name = '{safe_state}'"
                    ).fetchall()
                )
            elif col and val is not None:
                # Convert val to string and escape any single quotes
                if isinstance(val, list):
                    sanitized_vals = [str(v).replace("'", "''") for v in val]
                    val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_vals) + ")"
                    filter_clause = f"{col} IN {val_str}"
                else:
                    val_str = str(val).replace("'", "''")
                    filter_clause = f"{col} = '{val_str}'"

                county_district_options = ['-- Select County / District --'] + sorted(
                    row[0] for row in con.execute(
                        f"SELECT DISTINCT gadm_2_name FROM '{gadm_2_path}' WHERE {filter_clause}"
                    ).fetchall()
                )
            else:
                county_district_options = ['-- Select County / District --']

            selected_county_district = st.selectbox(
                "County / District",
                county_district_options,
                disabled=False,
                key="county_district_selector_RO",
                index=0,
                on_change=reset_city
            )

    # --- CITY ---
    with city_dropdown:
        if not country_selected_bool:
            city_options = ['Select a Region/Country to Enable']
            selected_city = st.selectbox(
                "City",
                city_options,
                disabled=True,
                key="city_selector_RO",
                index=0,
                on_change=mark_ro_recompute
            )
        else:
            def duckdb_safe_val(v):
                if isinstance(v, bool):
                    return "TRUE" if v else "FALSE"
                safe_v = str(v).replace("'", "''")
                return f"'{safe_v}'"

            if isinstance(val, list):
                val_str = "(" + ", ".join([duckdb_safe_val(v) for v in val]) + ")"
            else:
                # Treat single value as a list of one
                val_str = f"({duckdb_safe_val(val)})"

            query = f"""
                SELECT DISTINCT city_name 
                FROM '{city_path}' 
                WHERE {col} IN {val_str} AND city_name IS NOT NULL
            """

            city_options = ['-- Select City --'] + sorted(
                row[0] for row in con.execute(query).fetchall()
            )

            selected_city = st.selectbox(
                "City",
                city_options,
                disabled=False,
                key="city_selector_RO",
                index=0,
                on_change=reset_state_and_county
            )
    
    with forestry_toggle:
        forestry_toggle = st.radio(
            "Forestry Data:",
            ["Exclude", "Include"],
            horizontal=True
        )

        if forestry_toggle == "Exclude":
            exclude_forestry = True
        else:
            exclude_forestry = False

    with download_col:
        st.markdown(
            """
            <style>
            .stDownloadButton button {
                white-space: nowrap;
                margin-left: -8px;
            }
            .custom-download-space {
                padding-top: 28px;
            }
            </style>
            <div class="custom-download-space"></div>
            """,
            unsafe_allow_html=True
        )

        download_placeholder = st.empty()



    if use_ct_ers is False:
        # ---------- DROPDOWN ROW 2 ---------
        benchmarking_group_dropdown, percentile_dropdown, proportion_scale_bar  = st.columns(3)

        with benchmarking_group_dropdown:    
            benchmarking_options = [
                'Global',
                'Country'
            ]

            benchmarking_help = (
                "This selection enables users to establish baseline emissions benchmarks by selecting either all assets "
                "within a sector globally or only those within a specific country. By comparing assets to these benchmarks, "
                "users can identify emissions reduction opportunities at both the national and global levels."
            )

            if is_country(selected_region):
                selected_benchmark = st.selectbox(
                    "Benchmarking Group",
                    benchmarking_options,
                    help=benchmarking_help,
                    disabled=use_ct_ers,
                    key="benchmarking_selector_RO",
                    on_change=mark_ro_recompute
                )

            else:
                selected_benchmark = st.selectbox(
                    "Benchmarking Group",
                    "Global",
                    disabled=True,
                    help=benchmarking_help,
                    key="benchmarking_selector_RO",
                    on_change=mark_ro_recompute
                )

            if selected_benchmark == "Global":
                benchmark_join = "AND pct.iso3_country = 'all' "
            else:
                benchmark_join = "AND ae.iso3_country = pct.iso3_country "

        with percentile_dropdown:
            percentile_options = [
                # '0th',
                '10th',
                '20th',
                '30th',
                '40th',
                '50th',
                '60th',
                '70th',
                '80th',
                '90th',
                '100th'
            ]

            percentile_help = (
                "Benchmarks are set using emissions factor percentiles within the selected Benchmarking Group. For example, "
                "the 10th percentile reflects the average emissions factor of the top-performing 10% of assets, while the "
                "50th percentile represents the midpoint range (40–50% asset bucket)."
            )

            selected_percentile = st.selectbox(
                "Emission Reduction Target (Percentile)",
                percentile_options,
                help=percentile_help,
                disabled=use_ct_ers,
                key="percentile_selector_RO",
                on_change=mark_ro_recompute
            )

            percentile_col = map_percentile_col(selected_percentile)

        with proportion_scale_bar:
            proportion_help = (
                "This input defines the fraction of each asset’s emissions reduction potential to include in the plan. "
                "Assets across all sectors are ranked by reduction potential, and the specified proportion is applied to "
                "each asset’s opportunity. For example, if an asset has a reduction potential of 100 tCO₂e and the proportion "
                "is set to 80%, only 80 tCO₂e is counted toward the plan."
            )

            selected_proportion = st.slider(
                label="Proportion",
                min_value=0,
                max_value=100,
                value=100,
                help=proportion_help,
                disabled=use_ct_ers,
                step=1,
                format="%d%%",
                key="proportion_scale_bar_RO",
                on_change=mark_ro_recompute
            )
    
    # --------- Calculate Display Text Based on Selections ----------
    if selected_city and not selected_city.startswith("--") and selected_region != "Global":
        display_region_text = f"{selected_city}, {selected_region}"
    elif selected_county_district and not selected_county_district.startswith("--") and selected_region != "Global":
        display_region_text = f"{selected_county_district}, {selected_region}"
    elif selected_state_province and not selected_state_province.startswith("--") and selected_region != "Global":
        display_region_text = f"{selected_state_province}, {selected_region}"
    else:
        display_region_text = selected_region

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown(
        f"""
        <div style="text-align:center; font-size:36px; font-weight:600; margin-top:10px;">
            Emissions Reduction Pathways: {display_region_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    use_the_countries = {"United Kingdom", "United States of America", "United Arab Emirates"}

    if selected_region == "Global":
        subject_phrase = "global emissions"
    else:
        country_prefix = "the " if display_region_text in use_the_countries else ""
        subject_phrase = f"{country_prefix}{display_region_text}'s facilities"

    summary_text = (
        f"Climate TRACE, a coalition of universities, NGOs, and tech companies founded by former "
        f"U.S. Vice President Al Gore, uses AI, satellites, and big data to estimate the emissions of "
        f"nearly every major emitting facility or plot of land worldwide. By analyzing its facilities, "
        f"the coalition has produced the following {display_region_text if display_region_text != 'Global' else 'global'} "
        f"emissions estimates for {selected_year}:"
    )

    # display text
    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    where_clauses = []
    where_clauses.append("subsector <> 'net-soil-organic-carbon'")
    if selected_region == "Global":
        table = gadm_0_path
        where_clauses.append("gas = 'co2e_100yr'")
        where_clauses.append("country_name is not null")
    elif selected_city and not selected_city.startswith("--"):
        table = city_path
        selected_city_cleaned = selected_city.replace("'","''")
        where_clauses.append(f"city_name = '{selected_city_cleaned}'")


    elif selected_county_district and not selected_county_district.startswith("--"):
        table = gadm_2_path
        selected_county_district_cleaned = selected_county_district.replace("'","''")
        where_clauses.append(f"gadm_2_name = '{selected_county_district_cleaned}'")


    elif selected_state_province and not selected_state_province.startswith("--"):
        table = gadm_1_path
        where_clauses.append(f"gadm_1_name = '{selected_state_province}'")
    else:
        table = gadm_0_path
        where_clauses.append("gas = 'co2e_100yr'")
        where_clauses.append("country_name is not null")

        if region_condition:
            col = region_condition['column_name']
            val = region_condition['column_value']
            if isinstance(val, list):
                val_str = "(" + ", ".join(f"'{v}'" for v in val) + ")"
                where_clauses.append(f" {col} IN {val_str}")
            else:
                val_str = f"'{val}'"
                where_clauses.append(f" {col} = {val_str}")


    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)
        where_sql += f" AND year = {selected_year}"
    else:
        where_sql = f"WHERE year = {selected_year}"
    

    query_country = build_country_sql(table, where_sql)
    # print(query_country)

    df_pie = con.execute(query_country).df()
    
    if exclude_forestry:
        df_pie = df_pie[df_pie['sector'] != 'forestry-and-land-use']
    
    df_pie["emissions_quantity"] = df_pie["country_emissions_quantity"]

    total_emissions = format_number_short(df_pie["country_emissions_quantity"].sum())
    total_emissions_text = (
        f"1) &nbsp; <span style='color:red;'> {total_emissions}</span> tons of CO2 equivalent emissions, "
        f"broken down into the following sectors:"
    )

    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            {total_emissions_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------- Visualize Sector Pie ---------------------------
    sector_color_map = {
        "agriculture": "#E8516C",                 # from --color-agriculture
        "buildings": "#03A0E3",                   # from --color-buildings = --color-seablue
        "fluorinated-gases": "#B6B4B4",           # from --color-fluorinated-gases
        "forestry-and-land-use": "#779608",       # from --color-forestry-and-land-use
        "fossil-fuel-operations": "#FF6F42",      # from --color-fossil-fuel-operations = --color-orange
        "manufacturing": "#9554FF",               # from --color-manufacturing = --color-violet
        "mineral-extraction": "#4380F5",          # from --color-mineral-extraction
        "power": "#56979F",                       # from --color-power
        "transportation": "#FBBA14",              # from --color-transportation = --color-yellow
        "waste": "#BBD421"                        # from --color-waste = --color-lightgreen
    }

    df_pie["sector"] = df_pie["sector"].str.lower()
    df_pie = df_pie[df_pie["sector"].isin(sector_color_map.keys())]

    fig = px.pie(
        df_pie,
        values="emissions_quantity",
        names="sector",
        hole=0.2,
        title="Emissions Allocation",
        color="sector",
        color_discrete_map=sector_color_map  
    )

    fig.update_traces(
        text = df_pie.apply(
            lambda row: f"{row['sector']}<br>{format_number_short(row['emissions_quantity'])} tCO₂e", axis=1
        ),
        textinfo="text+percent",
        textposition="outside",
        textfont_size=16,
        pull=[0.02] * len(df_pie)
    )

    fig.update_layout(
        title_font=dict(
            size=22,
            family="Arial",
        ),
        height=600,
        width=800,
        showlegend=True,
        margin=dict(t=80, b=50, l=40, r=40)
    )

    # center the chart 
    left_spacer, center_col, right_spacer = st.columns([1, 6, 1])
    with center_col:
        st.plotly_chart(fig, use_container_width=True)

    
    
    sector_reduction_text = (f"2) &nbsp; By comparing specific {display_region_text if display_region_text != 'Global' else 'global'} "
                             f"facilities to best widely available technology facilities elsewhere, the coalition " 
                             f"has also identified specific emissions reduction opportunities in every sector:"
                            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            {sector_reduction_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------- Stacked Bar ---------------------------
    reduction_where_clause = []
    if col and val:
        if isinstance(val, list):
            val_str = "(" + ", ".join(f"'{v}'" for v in val) + ")"
            reduction_where_clause.append(f"ae.{col} IN {val_str}")
        else:
            val_str = f"'{val}'"
            reduction_where_clause.append(f"ae.{col} = {val_str}")   
    
    reduction_where_sql = f"WHERE {' AND '.join(reduction_where_clause)}" if reduction_where_clause else ""

    dropdown_join = ""
    if selected_city and not selected_city.startswith("--") and country_selected_bool:
        dropdown_join = f""" 
            INNER JOIN (
                select distinct city_id
                    , city_name 
                from '{city_path}'
                where city_name = '{selected_city_cleaned}'
            ) c
                on c.city_id = regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g')
        """
    
    elif selected_county_district and not selected_county_district.startswith("--") and country_selected_bool:
        dropdown_join = f"""
            inner join (
                select distinct gadm_2_id

                from '{gadm_2_path}'

                where gadm_2_name = '{selected_county_district_cleaned}'
            ) g2
                on g2.gadm_2_id = ae.gadm_2
        """
        if reduction_where_sql:
            reduction_where_sql += " AND most_granular is true "
        else:
            reduction_where_sql = "WHERE most_granular is true "

    elif selected_state_province and not selected_state_province.startswith("--") and country_selected_bool:
        dropdown_join = f"""
            inner join (
                select distinct gadm_id

                from '{gadm_1_path}'

                where gadm_1_name = '{selected_state_province}'
            ) g1
                on g1.gadm_id = ae.gadm_1
        """

        if reduction_where_sql:
            reduction_where_sql += " AND most_granular is true "
        else:
            reduction_where_sql = "WHERE most_granular is true "

    else:
        if reduction_where_sql:
            reduction_where_sql += " AND most_granular is true "
        else:
            reduction_where_sql = "WHERE most_granular is true "

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Sector Reduction Opportunities ({ers_baseline_year})")

    if use_ct_ers is True:

        st.info(
            "The data in the chart below represents **Consequential (Net) Emissions Reduction Opportunity**. "
            "This measure reflects the *system-wide impact*, which combines a given sector's direct asset reductions with emissions "
            "induced (added or avoided) from other sectors when emissions reducing solutions are implemented. The formula for this is as follows:\n\n"
            "| `Sector Reduction Potential = Direct Asset Reductions – Net Inductions` |\n"
            "|----------------------------------------------------------------------|\n\n"
            "Inductions capture how actions in one sector ripple into others. For example, switching from a blast "
            "furnace to an electric arc furnace reduces on-site steel emissions (asset reduction) but increases "
            "electricity demand, which shows up as an **emissions induction** into the power sector. "
            "Conversely, some inductions can be numerically **negative** (as with power): adding renewables to the grid does not "
            "count as an asset-level reduction, but it reduces emissions in the power sector overall. In this case, the negative induction "
            "contributes positively to the system-wide outcome."
        )

        query_sector_reductions = build_sector_reduction_sql(use_ct_ers=use_ct_ers,
                                                             annual_asset_path=annual_asset_path,
                                                             dropdown_join=dropdown_join,
                                                             reduction_where_sql=reduction_where_sql,
                                                            )
        
        query_sector_inductions = build_sector_induction_sql(annual_asset_path=annual_asset_path,
                                                             dropdown_join=dropdown_join,
                                                             reduction_where_sql=reduction_where_sql
                                                            )
        
        df_induced = con.execute(query_sector_inductions).df()
    
    else:
        query_sector_reductions = build_sector_reduction_sql(use_ct_ers=use_ct_ers,
                                                             annual_asset_path=annual_asset_path,
                                                             dropdown_join=dropdown_join,
                                                             reduction_where_sql=reduction_where_sql,
                                                             percentile_path=percentile_path,
                                                             percentile_col=percentile_col,
                                                             selected_proportion=selected_proportion,
                                                             benchmark_join=benchmark_join                      
                                                            )
    

    df_stacked_bar = con.execute(query_sector_reductions).df()

    if exclude_forestry:
        df_stacked_bar = df_stacked_bar[df_stacked_bar['sector'] != 'forestry-and-land-use']

    # if use_ct_ers is True:
    #     # df_stacked_bar.drop(df_stacked_bar[df_stacked_bar["sector"] == "fossil-fuel-operations"].index, inplace=True)
    #     # print(df_stacked_bar)
    #     pass

    if use_ct_ers:
        
        df_stacked_bar = pd.merge(
            df_pie[["sector","country_emissions_quantity"]],
            df_stacked_bar[["sector",
                            "emissions_reduction_potential",
                            "emissions_reduced_at_asset", 
                            "induced_emissions"]],
            on="sector",
            how="outer"
        )


        # hover_texts = get_consequetial_hover_text(df_induced)

    else:
        df_stacked_bar = pd.merge(
            df_pie[["sector","country_emissions_quantity"]],
            df_stacked_bar[["sector",
                            "emissions_reduction_potential",
                            ]],
            on="sector",
            how="outer"
        )

    df_stacked_bar["emissions_reduction_potential"] = df_stacked_bar["emissions_reduction_potential"].fillna(0)

    # capping reduction potential at total emissions if it exceeds total emissions
    df_stacked_bar["emissions_reduction_potential"] = np.where(
        df_stacked_bar["emissions_reduction_potential"] > df_stacked_bar["country_emissions_quantity"],
        df_stacked_bar["country_emissions_quantity"],
        df_stacked_bar["emissions_reduction_potential"]
    )

    df_stacked_bar['static_emissions_q'] = df_stacked_bar["country_emissions_quantity"] - df_stacked_bar["emissions_reduction_potential"]

    df_stacked_bar["total"] = (
        df_stacked_bar["static_emissions_q"] + df_stacked_bar["emissions_reduction_potential"]
    )
    
    df_stacked_bar = df_stacked_bar.sort_values("total", ascending=False)

    df_stacked_bar["sector"] = pd.Categorical(
        df_stacked_bar["sector"],
        categories=df_stacked_bar["sector"],
        ordered=True
    )

    df_stacked_bar["formatted_static"] = df_stacked_bar["static_emissions_q"].apply(format_number_short)
    df_stacked_bar["formatted_avoided"] = df_stacked_bar["emissions_reduction_potential"].apply(format_number_short)

    if use_ct_ers:
        df_stacked_bar["induced_emissions"] = df_stacked_bar["induced_emissions"].fillna(0)
        df_stacked_bar["emissions_reduced_at_asset"] = df_stacked_bar["emissions_reduced_at_asset"].fillna(0)

        consequential_json = get_reduction_induction_json(df_stacked_bar, df_induced)
        # print(consequential_json)
    
    if use_ct_ers:
        df_stacked_bar["induced_emissions"] = (
            df_stacked_bar["induced_emissions"]
                .fillna(0)                           
                .apply(format_number_short)          
        )
        df_stacked_bar["emissions_reduced_at_asset"] = (
            df_stacked_bar["emissions_reduced_at_asset"]
                .fillna(0)                           
                .apply(format_number_short)          
        )

    # print(df_stacked_bar)

    fig = go.Figure()

    if use_ct_ers:
    
        sectors = [item["sector"] for item in consequential_json]
        static_vals = [item["static_emissions"] for item in consequential_json]
        reduction_vals = [item["reduction_potential"] for item in consequential_json]
        hover_texts = [item["hover_text"] for item in consequential_json]

        # gray static emissions bar
        fig.add_bar(
            name="Post-Reduction Emissions",
            x=sectors,
            y=static_vals,
            marker_color="#707070",
            customdata=[item["static_emissions_formatted"] for item in consequential_json],
            hovertemplate="<b>%{x}</b><br>Post-Reduction Emissions: %{customdata} tCO₂e<extra></extra>"
        )

        # green reduction potential bar
        fig.add_bar(
            name="Reduction Potential (tCO₂e)",
            x=sectors,
            y=reduction_vals,
            marker=dict(
                color="rgba(46,139,87,0.8)",  
                pattern=dict(
                    shape="/", 
                    fgcolor="rgba(46,139,87,0.7)",
                    size=8,
                    solidity=0.958
                )
            ),
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>"  # rich HTML box
        )

    else:
        fig.add_bar(
            name='Post-Reduction Emissions',
            x=df_stacked_bar["sector"],
            y=df_stacked_bar["static_emissions_q"],
            marker_color="#707070",
            customdata=df_stacked_bar["formatted_static"],
            hovertemplate='<b>%{x}</b><br>Post-Reduction Emissions: %{customdata} tCO₂e<extra></extra>'
        )

        fig.add_bar(
            name='Reduction Potential (tCO2e)',
            x=df_stacked_bar["sector"],
            y=df_stacked_bar["emissions_reduction_potential"],
            marker=dict(
                color="rgba(46,139,87,0.8)",  
                pattern=dict(
                    shape="/", 
                    fgcolor="rgba(46,139,87,0.7)",
                    size=8,
                    solidity=0.958
                )
            ),
            customdata=df_stacked_bar[[
                "formatted_avoided"
            ]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Reduction Potential: %{customdata[0]} tCO₂e"
            )
        )


    fig.update_layout(
        barmode='stack',
        height=600,
        margin=dict(t=30),

        xaxis=dict(
            type='category',
            title=dict(
                text="Sector",
                font=dict(
                    size=18,           # 👈 change x-axis title size here
                    family="Sans-Serif",
                )
            ),
            tickfont=dict(
                size=13,              # 👈 tick labels size
                family="Sans-Serif Italic"
            )
        ),

        yaxis=dict(
            title=dict(
                text="Emissions (tCO2e)",
                font=dict(
                    size=18,          # 👈 change y-axis title size here
                    family="Sans-Serif",
                )
            ),
            range=[0, df_stacked_bar["total"].max() * 1.15],
            tickfont=dict(
                size=13,              # 👈 y tick labels size
                family="Sans-Serif"
            )
        )
    )

    fig.add_trace(
        go.Bar(
            x=df_stacked_bar["sector"],
            y=[0] * len(df_stacked_bar),  # invisible base
            text=[format_number_short(v) for v in df_stacked_bar["total"]],
            textposition="outside",
            textfont=dict(
                size=14,             # increase size
                family="Sans-Serif Italic" # makes it bold
            ),
            marker=dict(color="rgba(0,0,0,0)"),  # transparent bar
            showlegend=False,
            hoverinfo="skip",
            cliponaxis=False,
            name="Total"
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="sector_reductions_chart")

    generic_text = "Intervention types vary. For example, in road transportation, Climate TRACE " \
                   "considered the emissions reduction potential from electrifying transport. For " \
                   "electricity, Climate TRACE considered the potential to replace fossil fuel power " \
                   "plants with clean renewable energy. In agriculture, Climate TRACE considered replacing " \
                   "existing agricultural practices with lower-emitting practices. But in every case, " \
                   "reduction potential is based on actual practices widely observed in other facilities, " \
                   "not speculative new technologies."
    
    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            {generic_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    top_5_reduction_df = df_stacked_bar.sort_values(by="emissions_reduction_potential",ascending=False).head(5)
    total_reduction_potential = top_5_reduction_df["emissions_reduction_potential"].sum()

    top_emitting_sectors_df = df_stacked_bar.sort_values(by="country_emissions_quantity",ascending=False).head(5)
    total_emissions_top = top_emitting_sectors_df["country_emissions_quantity"].sum()

    reduction_pct = int((total_reduction_potential / total_emissions_top) * 100)

    sentence_2_data = pd.merge(
            top_emitting_sectors_df[["sector","country_emissions_quantity"]],
            top_5_reduction_df[["sector","emissions_reduction_potential"]],
            on="sector",
            how="inner"
        ).sort_values(by="emissions_reduction_potential", ascending=False)

    # Store both raw and formatted values
    reductions_raw = sentence_2_data["emissions_reduction_potential"].tolist()
    reductions_formatted = sentence_2_data["emissions_reduction_potential"].apply(format_number_short).tolist()

    # get the sectors to display in the text
    top_emitting_sectors_list = list(sentence_2_data["sector"])

    # prep for sentence 3
    high_reduction_potential_low_emitter = []
    for sector in top_5_reduction_df['sector']:
        if sector not in top_emitting_sectors_list:
            high_reduction_potential_low_emitter.append(sector)
    
    high_reduction_low_emitter_df = top_5_reduction_df[
        top_5_reduction_df['sector'].isin(high_reduction_potential_low_emitter)
    ]


    # ------- Sentence 2: format sector list as natural language ------
    if len(top_emitting_sectors_list) > 1:
        sectors_text = ", ".join(top_emitting_sectors_list[:-1]) + " and " + top_emitting_sectors_list[-1]
    else:
        sectors_text = top_emitting_sectors_list[0]

    # Highlight formatted reductions in green (no commas highlighted)
    formatted_reductions = [
        f"<span style='color: green;'><strong>{r}</strong></span>" for r in reductions_formatted
    ]
    reductions_text = " , ".join(formatted_reductions)

    # Final sentence
    sentence_2 = f"The top emitting sectors, including {sectors_text}, have opportunities to reduce CO2e emissions by {reductions_text} metric tons, respectively."

    # ------------------ Building Sentence 3 ------------------
    hr_le_sectors = list(high_reduction_low_emitter_df["sector"])
    if not hr_le_sectors:
        s3_sector_text = ""
    elif len(hr_le_sectors) == 1:
        s3_sector_text = hr_le_sectors[0]
    else:
        s3_sector_text = ", ".join(hr_le_sectors[:-1]) + " and " + hr_le_sectors[-1]

    # Format reduction values as green-highlighted numbers (no decimals, commas OK)
    sentence_3_formatted_reductions = [
        f"<span style='color: green;'><strong>{format_number_short(val)}</strong></span>"
        for val in high_reduction_low_emitter_df["emissions_reduction_potential"]
    ]

    # Join reductions with proper punctuation
    sentence_3_formatted_reductions = [
        f"<span style='color: green;'><strong>{format_number_short(val)}</strong></span>"
        for val in high_reduction_low_emitter_df["emissions_reduction_potential"]
    ]

    if not sentence_3_formatted_reductions:
        sentence_3_text = ""
    elif len(sentence_3_formatted_reductions) == 1:
        sentence_3_text = sentence_3_formatted_reductions[0]
    else:
        sentence_3_text = ", ".join(sentence_3_formatted_reductions[:-1]) + " and " + sentence_3_formatted_reductions[-1]

    if not sentence_3_formatted_reductions:
        sentence_3 = ""
    else:
        sentence_3 = f"While the {s3_sector_text} sector{'s' if len(hr_le_sectors) > 1 else ''} {'are' if len(hr_le_sectors) > 1 else 'is'} not among the top emitting sectors, {'they' if len(hr_le_sectors) > 1 else 'it'} rank{'s' if len(hr_le_sectors) == 1 else ''} top 5 for emissions reduction opportunities with {sentence_3_text}  metric tons."


    # ----- Building Sentence 4 -------
    include_sectors = ", ".join(f"'{s.lower()}'" for s in top_emitting_sectors_list)

    sentence_4_sql = build_sentence_4_sql(table, where_sql, include_sectors)

    sentence_4_query = con.execute(sentence_4_sql).df()

    sector_to_subsectors = defaultdict(list)
    for _, row in sentence_4_query.iterrows():
        sector_to_subsectors[row['sector']].append(row['subsector'])

    # Construct sentence segments
    segments = []
    for sector, subsectors in sector_to_subsectors.items():
        if len(subsectors) > 1:
            subsector_text = ", ".join(subsectors[:-1]) + " and " + subsectors[-1]
        else:
            subsector_text = subsectors[0]
        segments.append(f"in the {sector} sector, high-emitting subsectors include {subsector_text}")

    # Final sentence
    sentence_4 = "Additionally, " + "; ".join(segments) + "."

    # quick formatting for the first sentence
    highlight_green_1 = f"<span style='color: green;'><strong>{format_number_short(total_reduction_potential)} (-{reduction_pct:.0f}%)</strong></span>"
    
    if sentence_3:
        sentence_3_and_4 = f"""{sentence_3} 

                               {sentence_4}"""
    else:
        sentence_3_and_4 = sentence_4

    reduction_text = f"""
        Using Climate TRACE emissions data, CO2e emissions in {display_region_text if display_region_text != 'Global' else 'the world'} could be reduced by {highlight_green_1} metric tons across 5 sectors of high emissions reduction opportunities.

        {sentence_2}

        {sentence_3_and_4}
    """

    st.markdown(f"""
        ### A Possible {display_region_text if display_region_text != 'Global' else 'Global'} Emissions Reduction Plan: {ers_baseline_year}

        <div style="border: 1px solid rgba(100,100,100,0.3); padding: 0px 18px 10px 18px; border-radius: 6px; font-size: 16px; line-height: 1.4;">
            <p style="margin-top: 0px;">{reduction_text.replace(chr(10), '<br>')}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            <em><strong>This analysis was based on emissions and technology type of the highest emitting facilities in {display_region_text if display_region_text != 'Global' else 'the world'}, including the following estimates:</strong></em>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------- Asset Table ---------------------------------
    st.markdown(f"### Top 100 Assets by Annual Reduction Potential ({ers_baseline_year})")

    asset_sorting_help = ("**Net Reduction Potential:** Top 100 assets by net emissions reduction potential, "
        "including emissions induced in other sectors from executing the respective reduction strategy.\n\n"
        "**Asset Reduction Potential:** Top 100 assets by direct emissions reduction at the asset alone.\n\n"
        "**Asset Annual Emissions:** Top 100 highest emitting assets by annual (2025) emissions.")

    if use_ct_ers is True:
        sorting_options = [
            "Net Reduction Potential",
            "Asset Reduction Potential",
            "Asset Annual Emissions"
        ]

        asset_sorting_preference = st.radio(
            "Sorting Preference",
            sorting_options, 
            horizontal=True,
            key="asset_sorting_preference_RO",
            help=asset_sorting_help,
            on_change=mark_ro_recompute
        )

        asset_table_sql = build_asset_reduction_sql(use_ct_ers=use_ct_ers,
                                                    annual_asset_path=annual_asset_path,
                                                    dropdown_join=dropdown_join,
                                                    reduction_where_sql=reduction_where_sql,
                                                    sorting_preference=asset_sorting_preference,
                                                    exclude_forestry=exclude_forestry)
    else:
        sorting_options = [
            "Asset Reduction Potential",
            "Asset Annual Emissions"
        ]

        asset_sorting_preference = st.radio(
            "Sorting Preference",
            sorting_options,
            horizontal=True,
            key="asset_sorting_preference_RO",
            help=asset_sorting_help,
            on_change=mark_ro_recompute
        )

        asset_table_sql = build_asset_reduction_sql(use_ct_ers=use_ct_ers,
                                                    percentile_col=percentile_col,
                                                    selected_proportion=selected_proportion,
                                                    annual_asset_path=annual_asset_path,
                                                    percentile_path=percentile_path,
                                                    benchmark_join=benchmark_join,
                                                    dropdown_join=dropdown_join,
                                                    reduction_where_sql=reduction_where_sql,
                                                    sorting_preference=asset_sorting_preference,
                                                    exclude_forestry=exclude_forestry
                                                    )

    asset_table_df = con.execute(asset_table_sql).df()

    # print(asset_table_df)

    # TEMPORARY FIX FOR CITY REMAINDERS
    # asset_table_df = asset_table_df[asset_table_df['asset_name'].notna()].reset_index()

    asset_table_df['asset_url'] = asset_table_df.apply(make_asset_url, axis=1)
    asset_table_df['country_url'] = asset_table_df.apply(make_country_url, axis=1)
    

    # Current Estimate
    baseline_year_col_name = f'''{ers_baseline_year} Emissions (tCO2e)'''
    asset_table_df[baseline_year_col_name] = asset_table_df["emissions_quantity"].apply(lambda x: f"{round(x):,}")
    asset_table_df["Asset Reduction Potential Per Year (tCO2e)"] = asset_table_df["emissions_reduction_potential"].fillna(0).apply(lambda x: f"{round(x):,}")

    if use_ct_ers is True:
        asset_table_df = asset_table_df[['asset_url',
                                         'country_url',
                                         'sector',
                                         'subsector',
                                         'asset_type',
                                         'strategy_name',
                                         baseline_year_col_name,
                                         'Asset Reduction Potential Per Year (tCO2e)',
                                         'total_emissions_reduced_per_year']]

        asset_table_df["Net Reduction Potential Per Year (tCO2e)"]  = asset_table_df["total_emissions_reduced_per_year"].apply(lambda x: f"{round(x):,}")
        asset_table_df = asset_table_df.drop(columns=["total_emissions_reduced_per_year"])
        styled_df = asset_table_df.style.applymap(
            lambda val: "color: red", subset=[baseline_year_col_name]
        ).applymap(
            lambda val: "color: green", subset=["Asset Reduction Potential Per Year (tCO2e)",
                                                "Net Reduction Potential Per Year (tCO2e)"]
        )
    else:
        asset_table_df = asset_table_df[['asset_url',
                                         'country_url',
                                         'sector',
                                         'subsector',
                                         'asset_type',
                                         baseline_year_col_name,
                                         'Asset Reduction Potential Per Year (tCO2e)']]
        styled_df = asset_table_df.style.applymap(
            lambda val: "color: red", subset=[baseline_year_col_name]
        ).applymap(
            lambda val: "color: green", subset=["Asset Reduction Potential Per Year (tCO2e)"]
        )

    row_height = 35  # pixels per row (adjust as needed)
    num_rows = 20
    table_height = row_height * num_rows + 35  # extra for header

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=table_height,
        column_config={
            "asset_url": st.column_config.LinkColumn("asset_name", display_text=r"admin=([^&]+)"),
            "country_url": st.column_config.LinkColumn("country_name", display_text=r"admin=([^:]+)"),
        },
    )


    if not use_ct_ers:
        subsector_download_sql = build_subsector_reduction_percentile_download(
                                    annual_asset_path=annual_asset_path,
                                    dropdown_join=dropdown_join,
                                    reduction_where_sql=reduction_where_sql,
                                    table=table,
                                    where_sql=where_sql,
                                    percentile_path=percentile_path,
                                    percentile_col=percentile_col,
                                    selected_proportion=selected_proportion,
                                    benchmark_join=benchmark_join,
                                    exclude_forestry=exclude_forestry
                                )
        
        # print(subsector_download_sql)

        subsector_reduction_download_df = con.execute(subsector_download_sql).df()

    else:
        country_subsector_ef_sql = build_country_subsector_ef_download(
                                    annual_asset_path,
                                    gadm_0_path,
                                    dropdown_join,
                                    reduction_where_sql,
                                    region_condition,
                                    selected_year,
                                    exclude_forestry
                                )
        
        # print(country_subsector_ef_sql)
        country_subsector_ef_download_df = con.execute(country_subsector_ef_sql).df()

    if not use_ct_ers:
        dfs_for_excel = {
            "Sector Emissions": df_pie,
            "Sector Reduction Data": df_stacked_bar,
            "Subsector Reduction Data": subsector_reduction_download_df,
            "Asset Reduction Data": asset_table_df
        }
    elif not df_pie.empty or not df_stacked_bar.empty or not asset_table_df.empty:
        # Create dictionary of DataFrames to export
        dfs_for_excel = {
            # "Sector Emissions": df_pie.drop(columns=["emissions_quantity"]),
            "Sector Reductions": df_stacked_bar,
            "Subsector Reductions": country_subsector_ef_download_df,
            "Asset Top 100 Reductions": asset_table_df
        }

    # Use the utility function to create the Excel file
    benchmarking_excel_file = create_excel_file(dfs_for_excel)

    # Fill in the placeholder with the actual download button
    download_placeholder.download_button(
        label="⬇ Download Data",
        data=benchmarking_excel_file,
        file_name="export_climate_trace_emissions_reduction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="The downloaded data will represent your dropdown selections."
    )

    con.close()

    # print(reduction_where_sql)