import os, psutil, time, sys, traceback
import streamlit as st
import duckdb
import re
import pandas as pd
import sys
from config import CONFIG
from utils.utils import *
from utils.queries import *


def show_abatement_curve():

    print("=== DEBUG: Starting tab02_abatement_curve_tab.py ===", flush=True)

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


    ##### SET UP -------
    # set up data pathways
    annual_asset_path = CONFIG['annual_asset_path']
    gadm_0_path = CONFIG['gadm_0_path']
    gadm_1_path = CONFIG['gadm_1_path']
    gadm_2_path = CONFIG['gadm_2_path']
    region_options = CONFIG['region_options']
    city_path = CONFIG['city_path']

    con = duckdb.connect()

    ##### INTRO -------
    summary_text = (
        "The Abatement Curve dashboard is an interative tool designed to help users visualize and assess emissions reduction solutions (ERS) across selected geographies and sectors on an asset-level. "
        "Identify abatement opportunities through ranked assets, compare strategies across sectors, and explore pathways to reduce emissions effectively."
    )
    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    ##### DROPDOWN FOR COUNTRY -------
    # add drop-down options for geography

    country_rows = con.execute(
            f"SELECT DISTINCT country_name, iso3_country FROM '{gadm_0_path}' WHERE country_name IS NOT NULL order by country_name"
    ).fetchall()

    country_map = {row[0]: row[1] for row in country_rows}
    unique_countries = list(country_map.keys())

    geography_col, sector_program_col = st.columns(2)
    with geography_col:
        with st.expander("Geography", expanded=True):

            region_country_options = region_options + unique_countries
            region_help = ("Select by region, continent, country, etc. **Global** is selected by default. When Global is selected, all regions are included.")

            selected_region = st.multiselect(
                "Region/Country", 
                region_country_options, 
                key="selected_region_AC",
                help=region_help,
                default="Global",
                on_change=mark_ac_recompute
            )

            region_conditions = []

            for i in selected_region:
                cond = map_region_condition(i, country_map)
                if cond is not None:
                    col = cond['column_name']
                    val = cond['column_value']

                    if isinstance(val, list):
                        sanitized_vals = [str(v).replace("'", "''") for v in val]
                        val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_vals) + ")"
                    else:
                        sanitized_val = str(val).replace("'", "''")
                        val_str = f"('{sanitized_val}')"

                    region_conditions.append(f"{col} IN {val_str}")

            if region_conditions:
                region_filter_clause = " OR ".join(region_conditions)
            else:
                region_filter_clause = None

            country_selected_bool = 'Global' not in selected_region

            selection_mode = st.radio(
                "Select by",
                ["State/Province + County", "City"],
                horizontal=True,
                key="selection_mode_AC",
                on_change=mark_ac_recompute
                )
        
            if selection_mode == "State/Province + County":

                if not country_selected_bool:
                    state_province_options = ['Select specific Regions/Countries to Enable']
                    selected_state_province = st.multiselect(
                        "State / Province",
                        state_province_options,
                        disabled=True,
                        key="state_province_selector_AC",
                        on_change=mark_ac_recompute
                    )
                else:
                    state_province_options = sorted(
                        row[0] for row in con.execute(
                            f"SELECT DISTINCT gadm_1_corrected_name FROM '{gadm_1_path}' "
                            f"WHERE ({region_filter_clause}) AND gadm_1_name IS NOT NULL"
                        ).fetchall()
                    )

                    selected_state_province = st.multiselect(
                        "State / Province",
                        state_province_options,
                        disabled=False,
                        key="state_province_selector_AC",
                        on_change=mark_ac_recompute
                    )

                if (not country_selected_bool) or (not selected_state_province):
                    county_district_options = ['Select a Region/Country to Enable']
                    selected_county_district = st.multiselect(
                        "County / District",
                        county_district_options,
                        disabled=True,
                        key="county_district_selector_AC",
                        on_change=mark_ac_recompute
                    )
                else:

                    sanitized_vals = [str(v).replace("'", "''") for v in selected_state_province]
                    val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_vals) + ")"

                    query = (
                        f"SELECT DISTINCT gadm_2_corrected_name FROM '{gadm_2_path}' "
                        f"WHERE gadm_1_corrected_name IN {val_str} AND gadm_2_name IS NOT NULL"
                    )

                    county_district_options = sorted(
                        row[0] for row in con.execute(query).fetchall()
                    )

                    selected_county_district = st.multiselect(
                        "County / District",
                        county_district_options,
                        disabled=False,
                        key="county_district_selector_AC",
                        on_change=mark_ac_recompute
                    )

            else:
                if not country_selected_bool:
                    city_options = ['Select a County/District to Enable']
                    selected_city = st.multiselect(
                        "City",
                        city_options,
                        disabled=True,
                        key="city_selector_AC",
                        on_change=mark_ac_recompute
                    )
                else:
                    def duckdb_safe_val(v):
                        if isinstance(v, bool):
                            return "TRUE" if v else "FALSE"
                        return f"'{str(v).replace("'", "''")}'"


                    if isinstance(val, list):
                        val_str = "(" + ", ".join([duckdb_safe_val(v) for v in val]) + ")"
                    else:
                        val_str = f"({duckdb_safe_val(val)})"


                    query = f"""
                        SELECT DISTINCT city_name 
                        FROM '{city_path}' 
                        WHERE {col} IN {val_str} AND city_name IS NOT NULL
                    """


                    city_options = sorted(
                        row[0] for row in con.execute(query).fetchall()
                    )


                    selected_city = st.multiselect(
                        "City",
                        city_options,
                        disabled=False,
                        key="city_selector_AC",
                        on_change=mark_ac_recompute
                    )

        # create SQL filters based on geography selection
        asset_geography_filters = []
        total_geography_filters = []
        total_path = gadm_0_path

        if region_conditions and country_selected_bool:
            asset_geography_filters.append(f"({region_filter_clause})")
            total_geography_filters = [(f"({region_filter_clause})")]
            total_path = gadm_0_path
            if (selection_mode == "State/Province + County"):

                if selected_state_province:
                    sanitized_states = [str(v).replace("'", "''") for v in selected_state_province]
                    val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_states) + ")"
                    asset_geography_filters.append("most_granular IS NOT False")
                    asset_geography_filters.append(f"gadm_1_name IN {val_str}")
                    total_geography_filters = [f"gadm_1_corrected_name IN {val_str}"]
                    total_path = gadm_1_path
                if selected_county_district:
                    sanitized_counties = [str(v).replace("'", "''") for v in selected_county_district]
                    val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_counties) + ")"
                    asset_geography_filters.append("most_granular IS NOT False")
                    asset_geography_filters.append(f"gadm_2_name IN {val_str}")
                    total_geography_filters = [f"gadm_2_corrected_name IN {val_str}"]
                    total_path = gadm_2_path
            else:

                if selected_city:
                    sanitized_city = [str(v).replace("'", "''") for v in selected_city]
                    val_str = "(" + ", ".join(f"'{v}'" for v in sanitized_city) + ")"
                    asset_geography_filters.append(f"city_name IN {val_str}")
                    total_geography_filters = [f"city_name IN {val_str}"]
                    total_path = city_path

        total_geography_filters_clause = " AND ".join(total_geography_filters) if total_geography_filters else "1=1"
        asset_geography_filters_clause = " AND ".join(asset_geography_filters) if asset_geography_filters else "1=1"
        asset_geography_filters_clause = asset_geography_filters_clause.replace("iso3_country", "ae.iso3_country")
        print(total_geography_filters_clause)
        print(asset_geography_filters_clause)

    ##### DROPDOWN MENU: SECTOR, SUBSECTOR, PROGRAM -------
    # add drop-down options for filtering data + program view for x/y axis

    all_sectors = list(abatement_subsector_options.keys())

    with sector_program_col:
        with st.expander("Sector & Subsector", expanded=True):
            selected_sector_user = st.multiselect(
                "Sector",
                options=all_sectors,
                default=['manufacturing'],
            )
            if not selected_sector_user:
                selected_sector = all_sectors
            else:
                selected_sector = selected_sector_user

            subsector_options = [
                subsector
                for sector in selected_sector
                for subsector in abatement_subsector_options[sector]
            ]

            selected_subsector_user = st.multiselect(
                "Subsector",
                options=subsector_options,
                default=subsector_options[0],
                on_change=mark_ac_recompute
            )
            if not selected_subsector_user:
                selected_subsector = subsector_options
            else:
                selected_subsector = selected_subsector_user

        selected_year = 2024

        if len(selected_subsector) > 1:
            multisector = True
        else:
            multisector = False

        with st.expander("Program View", expanded=True):
            program_options = ["Emissions Reduction Solutions"]
            program_help = ("**Emissions Reduction Solutions**: Emissions reduction potential ranked by difficulty score (ascending). Available for multisector selections.\n\n"
                            "**Emissions Factor Abatement Curve**: Activity ranked by emissions factor (descending). Only available for single sector selection.")

            if not multisector:
                program_options.append("Emissions Factor Abatement Curve")
            selected_program = st.radio(
                "Select your view",
                program_options,
                horizontal=True,
                help=program_help,
                key="selection_program_AC",
                on_change=mark_ac_recompute
                )

    if selected_program == "Emissions Reduction Solutions":
        selected_x = 'net_reduction_potential'
        selected_y = 'asset_difficulty_score'
    else:
        selected_x = 'activity'
        selected_y = 'emissions_factor'


    ##### QUERY DATA -------

    # query assets based on selection
    query_df_assets = find_sector_assets_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, asset_geography_filters_clause)

    print("Getting asset data")
    df_assets = con.execute(query_df_assets).df()

    if not df_assets.empty:
        df_assets =  relabel_regions(df_assets)
        print("✅ Retrieved asset data.", flush=True)

        ##### SUMMARIZE KEY METRICS -------
        # totals for ers, emissions, reduction, assets, countries

        query_totals = summarize_totals_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, asset_geography_filters_clause)
        df_totals = con.execute(query_totals).df()
        total_ers = df_totals['total_ers'][0]
        total_subsectors = df_totals['total_subsectors'][0]
        total_assets = df_totals['total_assets'][0]
        # # for COP30 slides -- electricity sector
        # if selected_subsector == ['electricity-generation']:
        #     NUMBER_OF_RENEWABLES = st.number_input('Number of renewable plants', value=1000)
        #     RENEWABLE_MWH = st.number_input('Renewable Energy (MWh)', value=9000000000)
        #     total_assets = df_totals['total_assets'][0] + int(NUMBER_OF_RENEWABLES)
        # else:
        #     total_assets = df_totals['total_assets'][0]
        total_countries = df_totals['total_countries'][0]

        query_reductions = summarize_reductions_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, asset_geography_filters_clause)
        df_reductions = con.execute(query_reductions).df()
        total_reductions = df_reductions['total_reductions'][0]

        query_emissions = summarize_emissions_sql(total_path, selected_subsector, selected_year, total_geography_filters_clause)
        df_emissions = con.execute(query_emissions).df()
        total_emissions = df_emissions['emissions_quantity'].sum()

        print("✅ Aggregated totals...", flush=True)

        ##### ADD HIGH-LEVEL EMISSIONS / REDUCTION INFO -------
        # emissions vs. reductions

        summary_text = (
            f"<b>Total Emissions:</b> {format_emissions(total_emissions)}<br>"
            f"<b>Total Emissions Reduction Potential:</b> {format_emissions(total_reductions)} ({round((total_reductions/total_emissions) * 100, 1)}%)")

        # display text
        st.markdown(
            f"""
            <div style="margin-top: 8px; font-size: 25px; line-height: 1.5;">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="text-align:center; font-size:36px; font-weight:600; margin-top:10px;">
                {selected_program}
            </div>
            <div style="text-align:center; font-size: 25px"><i>Overview of Assets</i></div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        ##### DROPDOWN MENU: AXES, GROUP, COLOR -------
        # set up selections

        #x_axis_col, y_axis_col, group_col, color_col, threshold_col = st.columns(5)
        group_col, color_col = st.columns(2)

        with group_col:
            if selected_program == "Emissions Reduction Solutions":
                group_options = ['asset', 'country', 'strategy_name']
            else:
                group_options = ['asset', 'country']
            selected_group = st.selectbox(
                "Group by",
                options=group_options,
                on_change=mark_ac_recompute)

        # with x_axis_col:
        #     if multisector:
        #         x_axis_options = ['emissions_quantity', 'net_reduction_potential', 'count']
        #     else:
        #         x_axis_options = ['activity', 'emissions_quantity', 'net_reduction_potential', 'count']
        #     selected_x = st.selectbox(
        #         "Set x-axis",
        #         options=x_axis_options)
            
        # with y_axis_col:
        #     if selected_x in ['count', 'activity']:
        #         if selected_group != 'strategy_name':
        #             y_axis_options = ['emissions_factor', 'asset_difficulty_score', 'emissions_quantity', 'net_reduction_potential']
        #         else:
        #             y_axis_options = ['asset_difficulty_score', 'emissions_quantity', 'net_reduction_potential']
        #     else:
        #         if selected_group != 'strategy_name':
        #             y_axis_options = ['emissions_factor', 'asset_difficulty_score']
        #         else:
        #             y_axis_options = ['asset_difficulty_score']
        #     selected_y = st.selectbox(
        #         "Set y-axis",
        #         options=y_axis_options
        #     )

        with color_col:
            if selected_program == "Emissions Reduction Solutions":
                if selected_group == 'strategy_name':
                    color_options = ['sector']
                else:
                    color_options = ['sector', 'continent', 'unfccc_annex']
            else:
                color_options = ['unfccc_annex', 'sector', 'continent']
            selected_color = st.selectbox(
                "Color by",
                options=color_options)
            
        # with threshold_col:
        #     selected_threshold = st.text_input("Add threshold")
        selected_threshold = 0

        if selected_group == 'asset':
            selected_list = 'selected_asset_list'
            highlight_text = "Assets"
            total_units = total_assets
            total_units_desc = 'total assets'
            chart_title = (f"<b>By Assets ({selected_year})</b> - <i>{total_units:,} {total_units_desc}</i>")
        elif selected_group == 'country':
            selected_list = 'selected_country_list'
            highlight_text = "Country-subsectors"
            total_units = total_countries
            total_units_desc = 'countries'
            chart_title = (f"<b>By Country ({selected_year})</b> - <i>{total_units:,} {total_units_desc}</i>")
        elif selected_group == 'strategy_name':
            selected_list = 'selected_strategy_list'
            highlight_text = "Strategies"
            total_units = total_ers
            total_units_desc = 'strategies'
            chart_title = (f"<b>By ERS ({selected_year})</b> - <i>{total_units:,} {total_units_desc}</i>")

        ##### ASSET QUERYING -------
        # highlight assets on curve

        if "selected_assets" not in st.session_state:
            st.session_state.selected_assets = []

        asset_options = df_assets[selected_list].unique()

        selected_assets = st.multiselect(
            highlight_text + " to highlight in curve",
            options=asset_options,
            default=st.session_state.selected_assets,
            key = "selected_assets"
        )

        st.markdown("<br>", unsafe_allow_html=True)


        ##### PLOT FIGURE -------
        # add abatement curve

        # define variables
        dict_color, dict_lines = define_color_lines(selected_y)
        dict_lines={'outlier': {}}
        # adding renewables for COP30 slides
        # if selected_subsector == ['electricity-generation']:
        #     columns = ['year', 'asset_id', 'asset_name', 'asset_type', 'iso3_country',
        #             'country_name', 'balancing_authority_region', 'continent', 'eu', 'oecd',
        #             'unfccc_annex', 'developed_un', 'em_finance', 'sector', 'subsector',
        #             'reduction_q_type', 'gid_0', 'gadm_1', 'gid_1', 'gadm_1_name', 'gadm_2',
        #             'gid_2', 'gadm_2_name', 'activity_units', 'strategy_name', 'activity',
        #             'capacity', 'emissions_quantity', 'emissions_factor',
        #             'asset_reduction_potential', 'net_reduction_potential',
        #             'asset_difficulty_score', 'selected_asset_list',
        #             'selected_country_list', 'selected_subsector_list',
        #             'selected_strategy_list']       

        #     renewables_df = pd.DataFrame(index=range(int(NUMBER_OF_RENEWABLES)))

        #     # Assign all columns at once
        #     renewables_df = renewables_df.assign(
        #         activity = float(RENEWABLE_MWH / NUMBER_OF_RENEWABLES),
        #         year = 2024,
        #         asset_id = renewables_df.index,
        #         asset_name = 'Renewables Dummy',
        #         asset_type = 'Renewables',
        #         iso3_country = 'USA',
        #         country_name = 'TEST',
        #         balancing_authority_region = 'Test',
        #         continent = 'North America',
        #         eu = True,
        #         oecd = True,
        #         unfccc_annex = True,
        #         developed_un = True,
        #         em_finance = True,
        #         sector = 'power',
        #         subsector = 'electricity-generation',
        #         reduction_q_type = False,
        #         gid_0 = None,
        #         gadm_1 = None,
        #         gid_1 = None,
        #         gadm_1_name = None,
        #         gadm_2 = None,
        #         gid_2 = None,
        #         gadm_2_name = None,
        #         activity_units = 'MWh',
        #         strategy_name = None,
        #         capacity = None,
        #         emissions_quantity = 0,
        #         emissions_factor = 0,
        #         asset_reduction_potential = 0,
        #         net_reduction_potential = 0,
        #         asset_difficulty_score = 0,
        #         selected_asset_list = None,
        #         selected_country_list = None,
        #         selected_subsector_list = None,
        #         selected_strategy_list = None
        #     )
        #     df_assets = pd.concat([df_assets, renewables_df])
        fig, df_csv = plot_abatement_curve(df_assets, selected_group, selected_color, dict_color, dict_lines, selected_list, selected_assets, selected_x, selected_y, selected_threshold, fill=True)
        print("✅ Plot generated", flush=True)

        title_col, download_col = st.columns([6, 1])
        with title_col:
            st.markdown(
                f"""
                <div style="text-align:left; font-size:24px; margin-top:10px;">
                    {chart_title}
                </div>
                """,
                unsafe_allow_html=True)
            
        with download_col:
            st.markdown("<div style='text-align: right; margin-top:10px;'>", unsafe_allow_html=True)
            st.download_button(
                label="Download plot data",
                data=df_csv,
                file_name="fig_plot_data.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)

        print("✅ Rendered abatement curve chart", flush=True)

        ##### EMISSIONS REDUCING SOLUTIONS -------
        # summarize all ers within selection

        st.markdown(f"### {total_ers} Emissions Reduction Solutions (ERS) Strategies")

        # create a table to summarize ers for sector
        query_ers = summarize_ers_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, asset_geography_filters_clause)
        ers_table = con.execute(query_ers).df()
        csv_ers = ers_table.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download CSV",
            data=csv_ers,
            file_name="ers_summary_data.csv",
            mime="text/csv",
        )

        st.dataframe(
            ers_table,
            use_container_width=True,
            row_height=80,
            column_config={
                "strategy_description": st.column_config.Column(width="large"),
                "assets_impacted": st.column_config.NumberColumn(format="localized"),
                "emissions_quantity": st.column_config.NumberColumn(format="localized"),
                "total_reduction_potential": st.column_config.NumberColumn(format="localized")})
        print(f"✅ ERS table loaded ({len(ers_table):,} rows)", flush=True)

        # create a table with all assets + ERS info
        query_table = create_table_assets_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, asset_geography_filters_clause)
        df_table = con.execute(query_table).df()

        if selected_program == 'Emissions Reduction Solutions':
            df_table = df_table.sort_values("asset_difficulty_score", ascending=True).reset_index()
            summary_text = (
            "Opportunities are ranked based on <b>difficulty score</b>. The difficulty score reflects the potential impact, effort, and capital cost required to implement "
            "an emissions reduction solution. Higher-ranked opportunities indicate easier, more practial solutions.")
        else:
            df_table = df_table.sort_values("emissions_factor", ascending=False).reset_index()
            summary_text = (
            "Opportunities are ranked based on <b>emissions factor</b>. Emissions factors are based on emissions per unit of activity. A higher emissions factor indicates higher-priority opportunities.")

        # create urls to link info to climate trace website
        df_table['asset_url'] = df_table.apply(make_asset_url, axis=1)
        df_table['country_url'] = df_table.apply(make_country_url, axis=1)
        df_table['gadm_1_url'] = df_table.apply(make_state_url, axis=1)
        df_table['gadm_1_url'] = df_table['gadm_1_url'].fillna('')
        df_table['gadm_2_url'] = df_table.apply(make_county_url, axis=1)
        df_table['gadm_2_url'] = df_table['gadm_2_url'].fillna('')
        print("✅ URL columns created", flush=True)

        # filter + format table
        csv_assets = df_table[['subsector', 'asset_name', 'asset_url', 'country_name', 'country_url', 'gadm_1_name', 'gadm_1_url', 'gadm_2_name', 'gadm_2_url', 'strategy_name', 'emissions_quantity (t CO2e)', 'emissions_factor', 'reduction_potential (t CO2e)']].to_csv(index=False).encode('utf-8')
        df_table = df_table[['subsector', 'asset_url', 'country_url', 'gadm_1_url', 'gadm_2_url', 'strategy_name', 'emissions_quantity (t CO2e)', 'emissions_factor', 'reduction_potential (t CO2e)']]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Top 200 Reduction Opportunities")
        
        st.markdown(
            f"""
            <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.download_button(
            label="Download CSV",
            data=csv_assets,
            file_name="asset_table_data.csv",
            mime="text/csv",
        )

        # display table
        st.dataframe(
            df_table,
            use_container_width=True,
            height=600,
            column_config={
                "asset_url": st.column_config.LinkColumn("asset_name", display_text=r"admin=([^&]+)"),
                "country_url": st.column_config.LinkColumn("country", display_text=r'admin=([^:]+)'),
                "gadm_1_url": st.column_config.LinkColumn("state / province", display_text=r'admin=(.+?)--'),
                "gadm_2_url": st.column_config.LinkColumn("county / municipality / district", display_text=r'admin=(.+?)--'),
                "emissions_quantity (t CO2e)": st.column_config.NumberColumn(format="localized"),
                "reduction_potential (t CO2e)": st.column_config.NumberColumn(format="localized")}
        )
        print("✅ Final table rendered", flush=True)

        con.close()
    else:
        summary_text = (
            "No assets found in your selected <b>geography</b> and <b>subsector(s)</b><br>"
            "Adjust your search criteria")

        # display text
        st.markdown(
            f"""
            <div style="margin-top: 8px; font-size: 25px; line-height: 1.5;">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True)