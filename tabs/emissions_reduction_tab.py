import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from calendar import month_name
import calendar
from config import CONFIG
from utils.utils import (format_dropdown_options, 
                         map_region_condition, 
                         format_number_short, 
                         create_excel_file, 
                         bordered_metric, 
                         map_percentile_col)


def show_emissions_reduction_plan():

    st.markdown("<br>", unsafe_allow_html=True)

    # configure data paths and region options for querying
    annual_asset_path = CONFIG['annual_asset_path']
    city_path = CONFIG['city_path']
    gadm_1_path = CONFIG['gadm_1_path']
    gadm_2_path = CONFIG['gadm_2_path']
    country_subsector_totals_path = CONFIG['country_subsector_totals_path']
    percentile_path = CONFIG['percentile_path']
    region_options = CONFIG['region_options']

    con = duckdb.connect()

    unique_countries = sorted(
        row[0] for row in con.execute(
            f"SELECT DISTINCT country_name FROM '{country_subsector_totals_path}' WHERE country_name IS NOT NULL"
        ).fetchall()
    )

    # --------- DROPDOWN ROW 1 ----------
    country_dropdown, state_province_dropdown, county_district_drodpdown, city_dropdown = st.columns(4)
    with country_dropdown:
        
        selected_region = st.selectbox(
            "Region/Country", 
            region_options + unique_countries, 
            key="selected_region"
        )

        region_condition = map_region_condition(selected_region)

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

    if not country_selected_bool:
        state_province_options = ['Select a Region/Country to Enable']
        county_district_options = ['Select a Region/Country to Enable']
        city_options = ['Select a Region/Country to Enable']
        disable_subregion_dropdowns = True
    else:
        state_province_options = ['-- Select State / Province --'] + sorted(
            row[0] for row in con.execute(
                f"SELECT DISTINCT gadm_1_name FROM '{gadm_1_path}' WHERE {col} in {val_str}"
            ).fetchall()
        )
        
        county_district_options = ['-- Select County / District --'] + sorted(
            row[0] for row in con.execute(
                f"SELECT DISTINCT gadm_2_name FROM '{gadm_2_path}' WHERE {col} in {val_str}"
            ).fetchall()
        )
        
        city_options = ['-- Select City --'] + sorted(
            row[0] for row in con.execute(
                f"SELECT DISTINCT city_name FROM '{city_path}' WHERE {col} in {val_str} and city_name is not null"
            ).fetchall()
        )
        disable_subregion_dropdowns = False

    with state_province_dropdown:
        selected_state_province = st.selectbox(
            "State / Province", 
            state_province_options, 
            disabled=not country_selected_bool, 
            key="state_province_selector",
            index=0 if disable_subregion_dropdowns else None
        )

    with county_district_drodpdown:
        if disable_subregion_dropdowns:
            county_district_options = ["Select a Region/Country to Enable"]
            selected_county_district = st.selectbox(
                "County / District",
                county_district_options,
                disabled=True,
                key="county_district_selector",
                index=0
            )
        else:
            if selected_state_province and not selected_state_province.startswith("--"):
                county_district_options = ['-- Select County / District --'] + sorted(
                    row[0] for row in con.execute(
                        f"SELECT DISTINCT gadm_2_name FROM '{gadm_2_path}' WHERE gadm_1_name = '{selected_state_province}'"
                    ).fetchall()
                )

            selected_county_district = st.selectbox(
                "County / District",
                county_district_options,
                disabled=False,
                key="county_district_selector"
            )

    with city_dropdown:
        selected_city = st.selectbox(
            "City", 
            city_options, 
            disabled=not country_selected_bool, 
            key="city_selector",
            index=0 if disable_subregion_dropdowns else None
        )

    # ---------- DROPDOWN ROW 2 ---------
    percentile_dropdown, year_dropdown, proportion_scale_bar = st.columns(3)

    with percentile_dropdown:
        percentile_options = [
            '0th',
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

        selected_percentile = st.selectbox(
            "Percentile",
            percentile_options,
            key="percentile_selector"
        )

        percentile_col = map_percentile_col(selected_percentile)

    with year_dropdown:
        # only displaying 2024 data for now, will update to query
        year_options = 2024

        selected_year = st.selectbox(
            "Year",
            year_options,
            disabled=True,
            key="year_selector"
        )

    with proportion_scale_bar:
        selected_proportion = st.slider(
            label="Proportion",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            format="%d%%"
        )


    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown(
        f"""
        <div style="text-align:center; font-size:36px; font-weight:600; margin-top:10px;">
            Emissions Reduction Pathways: {selected_region}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    use_the_countries = {"United Kingdom", "United States of America", "United Arab Emirates"}

    if selected_region == "Global":
        subject_phrase = "global emissions"
    else:
        country_prefix = "the " if selected_region in use_the_countries else ""
        subject_phrase = f"{country_prefix}{selected_region}'s facilities"

    # Format the message
    summary_text = (
        f"Climate TRACE, a coalition of universities, NGOs, and tech companies founded by former "
        f"U.S. Vice President Al Gore, uses AI, satellites, and big data to estimate the emissions of "
        f"nearly every major emitting facility or plot of land worldwide. By analyzing its facilities, "
        f"the coalition has produced the following {selected_region if selected_region != 'Global' else 'global'} "
        f"emissions estimates for {selected_year}:"
    )

    # Display in Streamlit
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
    if selected_region == "Global":
        table = country_subsector_totals_path
    elif selected_city and not selected_city.startswith("--"):
        table = city_path
        where_clauses.append(f"city_name = '{selected_city}'")

    elif selected_county_district and not selected_county_district.startswith("--"):
        table = gadm_2_path
        where_clauses.append(f"gadm_2_name = '{selected_county_district}'")

    elif selected_state_province and not selected_state_province.startswith("--"):
        table = gadm_1_path
        where_clauses.append(f"gadm_1_name = '{selected_state_province}'")
    else:
        table = country_subsector_totals_path
        where_clauses.append("gas = 'co2e_100yr'")

        if region_condition:
            col = region_condition['column_name']
            val = region_condition['column_value']
            if isinstance(val, list):
                val_str = "(" + ", ".join(f"'{v}'" for v in val) + ")"
                where_clauses.append(f"{col} IN {val_str}")
            else:
                val_str = f"'{val}'"
                where_clauses.append(f"{col} = {val_str}")


    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)
    else:
        where_sql = f"WHERE year = {selected_year}"  # If you always want to filter by year

    # If year filter is always needed:
    if where_clauses:
        where_sql += f" AND year = {selected_year}"
    # Otherwise, you could also include it in the GROUP BY only

    query_country = f"""
        SELECT 
            year,
            sector,
            SUM(emissions_quantity) AS country_emissions_quantity
        FROM '{table}'
        {where_sql}
        GROUP BY year, sector
        ORDER BY sector
    """

    query_sector_reductions = f'''
        SELECT 
            sector,
            SUM(emissions_quantity) AS emissions_quantity,
            SUM(emissions_reduction_potential) AS emissions_reduction_potential
        
        FROM (
            SELECT 
                ae.asset_id,
                ae.sector,
                ae.subsector,
                ae.iso3_country,
                ae.country_name,
                SUM(ae.activity) AS activity,
                AVG(ae.ef_12_moer) AS ef_12_moer,
                
                CASE 
                    WHEN AVG(ae.ef_12_moer) IS NULL 
                        THEN SUM(ae.emissions_quantity)
                    ELSE SUM(ae.activity) * AVG(ae.ef_12_moer)
                END AS emissions_quantity,
                
                GREATEST(
                    0,
                    CASE 
                        WHEN AVG(ae.ef_12_moer) IS NULL 
                            THEN SUM(ae.emissions_quantity) - SUM(ae.activity * pct.{percentile_col})
                        ELSE (SUM(ae.activity) * AVG(ae.ef_12_moer)) - SUM(ae.activity * pct.{percentile_col})
                    END
                ) AS emissions_reduction_potential
            
            FROM '{annual_asset_path}' ae
            LEFT JOIN '{percentile_path}' pct
                ON ae.iso3_country = pct.iso3_country
                AND ae.subsector = pct.original_inventory_sector
            
            GROUP BY 
                ae.asset_id,
                ae.sector,
                ae.subsector,
                ae.iso3_country,
                ae.country_name
        ) asset_level
        
        GROUP BY sector
    '''

    # --------------------------- Visualize Sector Pie ---------------------------
    df_pie = con.execute(query_country).df()

    sector_color_map = {
        "agriculture": "#0BCF42",
        "buildings": "#03A0E3",
        "fluorinated-gases": "#D3D3D3",
        "forestry-and-land-use": "#E8516C",
        "fossil-fuel-operations": "#FF6F42",
        "manufacturing": "#9554FF",
        "mineral-extraction": "#4380F5",
        "power": "#407076",
        "transportation": "#FFBBA1",
        "waste": "#BBD421"
    }

    df_pie["sector"] = df_pie["sector"].str.lower()
    df_pie = df_pie[df_pie["sector"].isin(sector_color_map.keys())]

    fig = px.pie(
        df_pie,
        values="country_emissions_quantity",
        names="sector",
        hole=0.2,
        title="Emissions Allocation",
        color="sector",
        color_discrete_map=sector_color_map  
    )


    # Update chart appearance
    fig.update_traces(
        textinfo='percent+label',
        textposition='outside',  # now outside, as intended
        textfont_size=16,
        pull=[0.02] * len(df_pie)  # Slight separation for clarity
    )

    fig.update_layout(
        title_font=dict(
            size=22,
            family="Arial",
        ),
        height=600,
        width=800,
        showlegend=True,
        margin=dict(t=80, b=40, l=40, r=40)
    )

    # center the chart 
    left_spacer, center_col, right_spacer = st.columns([1, 6, 1])
    with center_col:
        st.plotly_chart(fig, use_container_width=True)

    
    
    sector_reduction_text = (f"By comparing specific {selected_region if selected_region != 'Global' else 'global'} "
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

    # ---------------------------Visualize Stacked Bar ---------------------------
    df_stacked_bar = con.execute(query_sector_reductions).df()

    df_stacked_bar['static_emissions_q'] = df_stacked_bar["emissions_quantity"] - df_stacked_bar["emissions_reduction_potential"]

    df_stacked_bar["total"] = (
        df_stacked_bar["static_emissions_q"] + df_stacked_bar["emissions_reduction_potential"]
    )
    
    df_stacked_bar = df_stacked_bar.sort_values("total", ascending=False)
    #ordered_sectors = df_stacked_bar["sector"].to_list()

    df_stacked_bar["sector"] = pd.Categorical(
        df_stacked_bar["sector"],
        categories=df_stacked_bar["sector"],
        ordered=True
    )

    # print(df_stacked_bar)

    fig = go.Figure()

    fig.add_bar(
        name='Post-Reduction Emissions',
        x=df_stacked_bar["sector"],
        y=df_stacked_bar["static_emissions_q"],
        marker_color="#606060"
    )

    fig.add_bar(
        name='Avoided Emissions',
        x=df_stacked_bar["sector"],
        y=df_stacked_bar["emissions_reduction_potential"],
        marker_color="#C0C0C0"
    )

    fig.update_layout(
        barmode='stack',
        title="Sector Reduction Opportunities",
        title_font=dict(
            size=22,
            family="Arial",
        ),
        xaxis_title="Sector",
        yaxis_title="Emissions (tCO2e)",
        xaxis=dict(type='category'),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    con.close()
    
    # sup