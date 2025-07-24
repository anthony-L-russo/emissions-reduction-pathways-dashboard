import streamlit as st
import streamlit.components.v1 as components
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from calendar import month_name
import calendar
from collections import defaultdict
from config import CONFIG
from utils.utils import (format_dropdown_options, 
                         map_region_condition, 
                         format_number_short, 
                         create_excel_file, 
                         bordered_metric, 
                         map_percentile_col,
                         is_country,
                         reset_city,
                         reset_state_and_county)


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
    country_dropdown, state_province_dropdown, county_district_dropdown, city_dropdown = st.columns(4)
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

    with state_province_dropdown:
        if not country_selected_bool:
            state_province_options = ['Select a Region/Country to Enable']
            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=True,
                key="state_province_selector",
                index=0
            )
        else:
            state_province_options = ['-- Select State / Province --'] + sorted(
                row[0] for row in con.execute(
                    f"SELECT DISTINCT gadm_1_name FROM '{gadm_1_path}' WHERE {col} IN {val_str}"
                ).fetchall()
            )
            selected_state_province = st.selectbox(
                "State / Province",
                state_province_options,
                disabled=False,
                key="state_province_selector",
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
                key="county_district_selector",
                index=0
            )
        else:
            if selected_state_province and not selected_state_province.startswith("--"):
                county_district_options = ['-- Select County / District --'] + sorted(
                    row[0] for row in con.execute(
                        f"SELECT DISTINCT gadm_2_name FROM '{gadm_2_path}' WHERE gadm_1_name = '{selected_state_province.replace("'", "''")}'"
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
                key="county_district_selector",
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
                key="city_selector",
                index=0
            )
        else:
            def duckdb_safe_val(v):
                if isinstance(v, bool):
                    return "TRUE" if v else "FALSE"
                return f"'{str(v).replace("'", "''")}'"

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
                key="city_selector",
                index=0,
                on_change=reset_state_and_county
            )


    # ---------- DROPDOWN ROW 2 ---------
    benchmarking_group_dropdown, percentile_dropdown, proportion_scale_bar, year_dropdown  = st.columns(4)

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
                key="benchmarking_selector"
            )

        else:
            selected_benchmark = st.selectbox(
                "Benchmarking Group",
                "Global",
                disabled=True,
                help=benchmarking_help,
                key="benchmarking_selector"
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
            key="percentile_selector"
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
            step=1,
            format="%d%%"
        )


    with year_dropdown:
        year_col, download_col = st.columns([2, 1])  # Adjust ratio as needed

        with year_col:
            # Only displaying 2024 data for now, will update to query
            year_options = [2024]  # Needs to be a list for selectbox
            selected_year = st.selectbox(
                "Year",
                year_options,
                disabled=True,
                key="year_selector"
            )

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
    if selected_region == "Global":
        table = country_subsector_totals_path
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
        table = country_subsector_totals_path
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

    df_pie = con.execute(query_country).df()
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
        "agriculture": "#0BCF42",
        "buildings": "#03A0E3",
        "fluorinated-gases": "#D3D3D3",
        "forestry-and-land-use": "#E8516C",
        "fossil-fuel-operations": "#FF6F42",
        "manufacturing": "#9554FF",
        "mineral-extraction": "#4380F5",
        "power": "#407076",
        "transportation": "#FBBA1A",
        "waste": "#BBD421"
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

    # ---------------------------Visualize Stacked Bar ---------------------------
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
    elif selected_state_province and not selected_state_province.startswith("--") and country_selected_bool:
        dropdown_join = f"""
            inner join (
                select distinct gadm_id

                from '{gadm_1_path}'

                where gadm_1_name = '{selected_state_province}'
            ) g1
                on g1.gadm_id = ae.gadm_1
        """

    query_sector_reductions = f'''
        SELECT 
            sector sector,
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
                            THEN (SUM(ae.emissions_quantity) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                        ELSE ((SUM(ae.activity) * AVG(ae.ef_12_moer)) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                    END
                ) AS emissions_reduction_potential
            
            FROM '{annual_asset_path}' ae
            LEFT JOIN '{percentile_path}' pct
                ON ae.subsector = pct.original_inventory_sector
                AND ae.asset_type_2 = pct.asset_type
                {benchmark_join}
            {dropdown_join}

            {reduction_where_sql}
            
            GROUP BY 
                ae.asset_id,
                ae.sector,
                ae.subsector,
                ae.iso3_country,
                ae.country_name
        ) asset_level
                
        GROUP BY sector
    '''

    df_stacked_bar = con.execute(query_sector_reductions).df()


    df_stacked_bar = pd.merge(
        df_pie[["sector","country_emissions_quantity"]],
        df_stacked_bar[["sector","emissions_reduction_potential"]],
        on="sector",
        how="outer"
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

    fig = go.Figure()

    fig.add_bar(
        name='Post-Reduction Emissions',
        x=df_stacked_bar["sector"],
        y=df_stacked_bar["static_emissions_q"],
        marker_color="#606060",
        customdata=df_stacked_bar["formatted_static"],
        hovertemplate='<b>%{x}</b><br>Post-Reduction Emissions: %{customdata} tCO₂e<extra></extra>'
    )

    fig.add_bar(
        name='Avoided Emissions',
        x=df_stacked_bar["sector"],
        y=df_stacked_bar["emissions_reduction_potential"],
        marker_color="#C0C0C0",
        customdata=df_stacked_bar["formatted_avoided"],
        hovertemplate='<b>%{x}</b><br>Reduction Potential: %{customdata} tCO₂e<extra></extra>'
    )

    fig.update_layout(
        barmode='stack',
        title="Sector Reduction Opportunities (Annual)",
        title_font=dict(
            size=22,
            family="Arial",
        ),
        xaxis_title="Sector",
        yaxis_title="Emissions (tCO2e)",
        xaxis=dict(type='category'),
        height=600
    )

    fig.add_trace(
        go.Bar(
            x=df_stacked_bar["sector"],
            y=[0] * len(df_stacked_bar),  # invisible base
            text=[format_number_short(v) for v in df_stacked_bar["total"]],
            textposition="outside",
            marker=dict(color="rgba(0,0,0,0)"),  # transparent bar
            showlegend=False,
            hoverinfo="skip",
            cliponaxis=False,
            name="Total"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    generic_text = "Intervention types vary. For example, in road transportation, Climate TRACE " \
                   "considered the emissions reduction potential from electrifying transport. For " \
                   "elecricity, Climate TRACE considered the potential to replace fossil fuel power " \
                   "plants with clean renewable energy. In agriculture, Climate TRACE considered replacing " \
                   "existing agricultural practices with lower-emitting practices. But in every case, " \
                   "interventions are based on actual practices widely observed in other facilities, " \
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

    s4_query = f"""
        with sector as (
            select sector
                , sum(emissions_quantity) sector_emissions_quantity

            from '{table}'

            {where_sql}
                and lower(sector) <> 'power'
                and sector in ({include_sectors})

            group by sector
        ),

        subsector as (
            select sector
                , subsector
                , sum(emissions_quantity) subsector_emissions_quantity

            from '{table}'

            {where_sql}
                and lower(sector) <> 'power'
                and sector in ({include_sectors})

            group by sector
                , subsector
        ),

        agg as (
            select subsector.sector
                , subsector.subsector
                , sum(subsector.subsector_emissions_quantity) subsector_emissions_quantity

            from subsector
            inner join sector
                on sector.sector = subsector.sector

            group by subsector.sector
                , subsector.subsector

            having (sum(subsector.subsector_emissions_quantity) / sum(sector.sector_emissions_quantity)) >= 0.05
        ),
        
        subsector_rank as (
            select *
                , row_number() over (partition by sector order by subsector_emissions_quantity desc) as subsector_rank
            from agg
        )

        select *
        from subsector_rank
        where subsector_rank.subsector_rank <= 2

    """

    sentence_4_query = con.execute(s4_query).df()

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
        ### A possible {display_region_text if display_region_text != 'Global' else 'Global'} Emissions Reduction Plan

        <div style="border: 1px solid rgba(100,100,100,0.3); padding: 0px 18px 10px 18px; border-radius: 6px; font-size: 16px; line-height: 1.4;">
            <p style="margin-top: 0px;">{reduction_text.replace('\n', '<br>')}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="margin-top: 8px; font-size: 17px; line-height: 1.5;">
            <em><strong>This analysis was based on estimated emissions and technology type of the most emitting facilities in {display_region_text if display_region_text != 'Global' else 'the world'}, including the following estimates:</strong></em>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------- Asset Table ---------------------------------
    asset_table_query = f"""
        SELECT asset_name
            , country_name
            , sector
            , subsector
            , asset_type
            , emissions_quantity
            , emissions_reduction_potential
        
        FROM (
            SELECT 
                ae.asset_name,
                ae.asset_type,
                ae.country_name,
                ae.sector,
                ae.subsector,
                ae.asset_type,
                
                SUM(ae.emissions_quantity) AS emissions_quantity,
                
                --CASE 
                    --WHEN AVG(ae.ef_12_moer) IS NULL 
                       -- THEN SUM(ae.emissions_quantity)
                   -- ELSE SUM(ae.activity) * AVG(ae.ef_12_moer)
               -- END AS emissions_quantity,

                GREATEST(
                    0,
                    CASE 
                        WHEN pct.{percentile_col} is null then 0
                        WHEN AVG(ae.ef_12_moer) IS NULL 
                            THEN (SUM(ae.emissions_quantity) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                        ELSE ((SUM(ae.activity) * AVG(ae.ef_12_moer)) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                    END
                ) AS emissions_reduction_potential,

                ROW_NUMBER() OVER (
                    ORDER BY 
                        GREATEST(
                            0,
                            CASE 
                                WHEN pct.{percentile_col} is null then 0
                                WHEN AVG(ae.ef_12_moer) IS NULL 
                                    THEN (SUM(ae.emissions_quantity) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                                ELSE ((SUM(ae.activity) * AVG(ae.ef_12_moer)) - SUM(ae.activity * pct.{percentile_col})) * ({selected_proportion} / 100.0)
                            END
                        ) DESC
                ) AS rank
            
            FROM '{annual_asset_path}' ae
            LEFT JOIN '{percentile_path}' pct
                ON ae.subsector = pct.original_inventory_sector
                AND ae.asset_type_2 = pct.asset_type
                {benchmark_join}
            {dropdown_join}

            {reduction_where_sql}
            
            GROUP BY 
                ae.asset_name,                
                ae.country_name,
                ae.sector,
                ae.subsector,
                ae.asset_type,
                ae.asset_type,
                pct.{percentile_col}
        ) assets

        where rank <= 20

        order by rank asc
    """

    asset_table_df = con.execute(asset_table_query).df()

    # Current (2024) Estimated Emissions (tCO2e)
    # Estimated Emissions Reduction Potential Per Year (tCO2e)
    asset_table_df["2024 Emissions (tCO2e)"] = asset_table_df["emissions_quantity"].apply(lambda x: f"{round(x):,}")
    asset_table_df["Estimated Reduction Potential Per Year (tCO2e)"] = asset_table_df["emissions_reduction_potential"].apply(lambda x: f"{round(x):,}")  

    styled_df = asset_table_df.style.applymap(
        lambda val: "color: red", subset=["2024 Emissions (tCO2e)"]
            ).applymap(
                lambda val: "color: green", subset=["Estimated Reduction Potential Per Year (tCO2e)"]
            )

    st.markdown("### Top 20 Assets by Annual Reduction Potential")

    row_height = 35  # pixels per row (adjust as needed)
    num_rows = 20
    table_height = row_height * num_rows + 35  # extra for header

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=table_height
    )


    if not df_pie.empty or not df_stacked_bar.empty or not asset_table_df.empty:
        # Create dictionary of DataFrames to export
        dfs_for_excel = {
            "Sector Emissions": df_pie,
            "Sector Reduction Data": df_stacked_bar,
            "Asset Reduction Data": asset_table_df,
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
    