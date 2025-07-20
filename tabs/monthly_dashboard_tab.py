# UPDATE 4: MAKE PARQUET CONVERSION SCRIPT DELETE UNDERSCORE DATE AND MAKE STATIC FILE PATH

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from calendar import month_name
import calendar
from utils.utils import format_dropdown_options, map_region_condition, format_number_short, create_excel_file, bordered_metric
from config import CONFIG



def show_monthly_dashboard():

    # configure data paths (querying) and region options (dropdown selection)
    asset_path = CONFIG['asset_path']
    country_subsector_stats_path = CONFIG['country_subsector_stats_path']
    country_subsector_totals_path = CONFIG['country_subsector_totals_path']
    region_options = CONFIG['region_options']

    con = duckdb.connect()

    max_date = con.execute(f"""SELECT MAX(MAKE_DATE(year, month, 1)) AS max_date
            FROM '{country_subsector_totals_path}'
            WHERE country_name IS NOT NULL
    """).fetchone()[0]

    earliest_year = (max_date.year) - 3

    st.markdown("<br>", unsafe_allow_html=True)

    unique_countries = sorted(
        row[0] for row in con.execute(
            f"SELECT DISTINCT country_name FROM '{country_subsector_totals_path}' WHERE country_name IS NOT NULL"
        ).fetchall()
    )

    df_stats_all = pd.read_parquet(country_subsector_stats_path)
    
    df_stats_all = df_stats_all[df_stats_all['gas'].isin(['co2e_100yr', 'ch4'])]

    raw_sectors = sorted(df_stats_all['sector'].dropna().unique().tolist())

    def format_sector_label(sector):
        return ' '.join([w.capitalize() if w.lower() != 'and' else 'and' for w in sector.replace('-', ' ').split()])

    sector_labels = [format_sector_label(s) for s in raw_sectors]
    sector_map = dict(zip(sector_labels, raw_sectors))
    sector_labels.insert(0, "All")
    sector_map["All"] = None


    # --- ROW 1 ---
    region_dropdown, sector_dropdown, gas_drodpdown = st.columns(3)
    with region_dropdown:
        selected_scope = st.selectbox("Region/Country", region_options + unique_countries, key="selected_scope")
        region_condition = map_region_condition(selected_scope)

    with sector_dropdown:
        selected_sector_label = st.selectbox("Sector", sector_labels, key="sector_selector")
        selected_sector_raw = sector_map.get(selected_sector_label)

    with gas_drodpdown:
        selected_gas = st.selectbox("Gas", ["co2e_100yr", "ch4"], key="gas_selector")
        df_stats_all = df_stats_all[df_stats_all['gas'] == selected_gas]

    # Filter subsectors based on selected sector
    if selected_sector_raw:
        subsector_subset = df_stats_all[df_stats_all['sector'] == selected_sector_raw]
    else:
        subsector_subset = df_stats_all

    raw_subsectors = sorted(subsector_subset['subsector'].dropna().unique().tolist())
    subsector_labels, subsector_map = format_dropdown_options(raw_subsectors)

    # --- ROW 2 ---
    state_province_dropdown, subsector_dropdown, current_month_dropdown = st.columns(3)
    with state_province_dropdown:
        st.selectbox("State/Province", ["🚧 Coming Soon 🚧"], disabled=True, key="state_selector")

    with subsector_dropdown:
        # Reset subsector if sector changed
        if (
            "last_selected_sector" not in st.session_state
            or st.session_state["last_selected_sector"] != selected_sector_raw
        ):
            st.session_state["selected_subsector_label"] = []
            st.session_state["last_selected_sector"] = selected_sector_raw

        selected_subsector_label = st.multiselect(
            "Subsector",
            subsector_labels,
            key="selected_subsector_label"
        )

    with current_month_dropdown:
        # Split this column into two sub-columns
        #st.markdown("Latest Month") 
        month_col, download_col = st.columns([2, 1])  # adjust ratio if needed

        with month_col:
            if max_date:
                formatted_date = pd.to_datetime(max_date).strftime("%B %Y")
                st.selectbox(
                    label="Latest Month",
                    options=[formatted_date],
                    disabled=True,
                    #label_visibility="collapsed"
                )

        with download_col:
            # with download_col:
            st.markdown("<div style='margin-top: 28px; margin-left: 6px;'></div>", unsafe_allow_html=True)
            download_placeholder = st.empty()


    selected_subsector_raw = [subsector_map[label] for label in selected_subsector_label if label in subsector_map]

    spacer_col, download_col = st.columns([10, 1])

    st.markdown("<br>", unsafe_allow_html=True)

    # Emissions columns from parquet
    df_stats_all = df_stats_all[df_stats_all['country_name'].notna()].copy()
    emissions_columns = [col for col in df_stats_all.columns if col.startswith("emissions_quantity_")]
    emissions_columns_sorted = sorted(emissions_columns, reverse=True)
    emissions_column_latest = emissions_columns_sorted[0]
    emissions_column_prev = emissions_columns_sorted[1]

    country_name_map_asset = {
        "United States": "United States of America"
    }
    asset_scope = country_name_map_asset.get(selected_scope, selected_scope)

    # Build query for asset-level time series
    where_clauses = [f"gas = '{selected_gas}'"]

    # Sector/Subsector filters
    if selected_sector_raw:
        where_clauses.append(f"sector = '{selected_sector_raw}'")
    if selected_subsector_raw:
        if len(selected_subsector_raw) == 1:
            where_clauses.append(f"original_inventory_sector = '{selected_subsector_raw[0]}'")
        else:
            formatted_subsectors = ",".join([f"'{sub}'" for sub in selected_subsector_raw])
            where_clauses.append(f"original_inventory_sector IN ({formatted_subsectors})")



    if region_condition:
        col = region_condition['column_name']
        val = region_condition['column_value']

        # Special handling if it's actually a country filter
        if col == "country_name" and isinstance(val, list) and set(val) == {"United States", "United States of America"}:
            where_clauses.append("country_name IN ('United States', 'United States of America')")
        else:
            val_str = f"'{val}'" if isinstance(val, str) else str(val).upper()
            where_clauses.append(f"{col} = {val_str}")


    query = f"""
        SELECT 
            strftime(start_time, '%Y-%m') AS year_month,
            SUM(activity) AS activity,
            SUM(emissions_quantity) AS emissions_quantity
        FROM '{asset_path}'
        WHERE {' AND '.join(where_clauses)}
            and original_inventory_sector not in ('forest-land-clearing',
                                                    'forest-land-degradation',
                                                    'forest-land-fires',
                                                    'net-forest-land',
                                                    'net-shrubgrass',
                                                    'net-wetland',
                                                    'removals',
                                                    'shrubgrass-fires',
                                                    'water-reservoirs',
                                                    'wetland-fires')
        GROUP BY year_month
        ORDER BY year_month
    """
    # print(query)

    monthly_df = con.execute(query).df()
    monthly_df["year_month"] = pd.to_datetime(monthly_df["year_month"])
    if not monthly_df.empty:
        monthly_df["mean_emissions_factor"] = monthly_df["emissions_quantity"] / monthly_df["activity"]

    # Filter stats for table view
    if selected_sector_raw:
        df_stats_all = df_stats_all[df_stats_all['sector'] == selected_sector_raw]
    
    df_stats = df_stats_all[df_stats_all['country_name'].notna()].copy()
    
    df_stats['country_name'] = df_stats['country_name'].replace({
        'United States of America': 'United States',
        'Russian Federation': 'Russia'
    })

    if selected_scope != 'Global':
        if region_condition:
            col = region_condition['column_name']
            val = region_condition['column_value']
            if isinstance(val, (list, tuple, set)):
                df_stats = df_stats[df_stats[col].isin(val)]
            else:
                df_stats = df_stats[df_stats[col] == val]
        else:
            df_stats = df_stats[df_stats['country_name'] == selected_scope]

    # Determine if we're selecting everything ("All") or specific subsectors
    if not selected_subsector_raw or "All" in selected_subsector_label:
        # "All" selected or no subsector selected → use the full df_stats
        df_stats_filtered = df_stats.copy()
    else:
        # Specific subsectors selected → filter to those
        df_stats_filtered = df_stats[df_stats['subsector'].isin(selected_subsector_raw)]

    # Additional processing regardless of subsector filtering
    if 'emissions_slope_36_months_t_per_month' in df_stats_filtered.columns:
        df_stats_filtered['slope_times_emissions'] = (
            df_stats_filtered['emissions_slope_36_months_t_per_month'] * df_stats_filtered[emissions_column_latest]
        )
        slopes = df_stats_filtered.groupby('country_name').agg({
            'slope_times_emissions': 'sum',
            emissions_column_latest: 'sum'
        }).reset_index()
        slopes['emissions_slope_36_months_t_per_month'] = slopes['slope_times_emissions'] / slopes[emissions_column_latest]
        df_stats_agg = df_stats_filtered.groupby('country_name').agg({
            emissions_column_latest: 'sum',
            emissions_column_prev: 'sum',
            'mom_change': 'sum',
            'month_yoy_change': 'sum'
        }).reset_index()
        df_stats_agg['mom_percent_change'] = (df_stats_agg['mom_change'] / df_stats_agg[emissions_column_prev]) * 100
        df_stats_agg['month_yoy_percent_change'] = (
            df_stats_agg['month_yoy_change'] / (df_stats_agg[emissions_column_latest] - df_stats_agg['month_yoy_change'])
        ) * 100
        df_stats_filtered = df_stats_agg.merge(
            slopes[['country_name', 'emissions_slope_36_months_t_per_month']],
            on='country_name',
            how='left'
        )


    # Summary sentence using latest month from stats file
    if not df_stats_filtered.empty:
        summary_agg = df_stats_filtered.copy()
        summary_agg_row = summary_agg[[
            emissions_column_latest,
            emissions_column_prev,
            'month_yoy_change'
        ]].sum().to_dict()
    else:
        summary_agg_row = {
            emissions_column_latest: 0,
            emissions_column_prev: 0,
            'month_yoy_change': 0
        }


    latest_month = f"{month_name[int(emissions_column_latest[-2:])]} 20{emissions_column_latest[-4:-2]}"
    emissions_value = summary_agg_row[emissions_column_latest]
    prev_emissions_value = summary_agg_row[emissions_column_prev]
    mom_delta = ((emissions_value - prev_emissions_value) / prev_emissions_value) * 100 if prev_emissions_value != 0 else 0

    yoy_change = summary_agg_row['month_yoy_change']
    last_year_emissions = emissions_value - yoy_change
    yoy_delta = (yoy_change / last_year_emissions) * 100 if last_year_emissions != 0 else 0

    # Round to one decimal for display
    mom_delta_rounded = round(mom_delta, 1)
    yoy_delta_rounded = round(yoy_delta, 1)

    # Handle "no change" logic for month-over-month
    mom_color = None
    if mom_delta_rounded == 0.0:
        mom_text = "<span style='font-style: italic; font-weight: bold;'>no change</span>"
    else:
        mom_direction = "increase" if mom_delta > 0 else "decrease"
        mom_article = "an" if mom_direction[0] in "aeiou" else "a"
        mom_color = "red" if mom_delta > 0 else "green"
        mom_text = (
            f"{mom_article} <span style='color:{mom_color}; font-style: italic; font-weight: bold;'>"
            f"{mom_direction} of {abs(mom_delta_rounded):.1f}%</span>"
        )

    # Handle "no change" logic for year-over-year
    yoy_color = None
    if yoy_delta_rounded == 0.0:
        yoy_text = "<span style='font-style: italic; font-weight: bold;'>no change</span>"
    else:
        yoy_direction = "increase" if yoy_delta > 0 else "decrease"
        yoy_article = "an" if yoy_direction[0] in "aeiou" else "a"
        yoy_color = "red" if yoy_delta > 0 else "green"
        yoy_text = (
            f"{yoy_article} <span style='color:{yoy_color}; font-style: italic; font-weight: bold;'>"
            f"{yoy_direction} of {abs(yoy_delta_rounded):.1f}%</span>"
        )

    sector_text = f" {selected_sector_label}" if selected_sector_label and selected_sector_label != "All" else ""
    subsector_text = (
        f" ({selected_subsector_label})" if selected_sector_label and selected_sector_label != "All" and selected_subsector_label and selected_subsector_label != "All"
        else f" {selected_subsector_label}" if selected_subsector_label and selected_subsector_label != "All"
        else ""
    )

    gas_display = {'co2e_100yr': 'tCO₂e', 'ch4': 'tCH₄'}
    gas_unit = gas_display.get(selected_gas, selected_gas)

    st.markdown(
        f"<div style='font-size: 1.1em; line-height: 1.6em; margin: 16px 0;'>"
        f"In {latest_month}, {selected_scope}{sector_text}{subsector_text} emissions were "
        f"<span style='font-weight: bold; font-style: italic; text-decoration: underline;'>{emissions_value:,.0f}</span> {gas_unit}. "
        f"This represents {mom_text} compared to the previous month. This also represents {yoy_text} compared to the same month last year."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    
    # ----------------------- Summary Cards -----------------------
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        bordered_metric("Selected Region", selected_scope)

    with col2:
        if selected_subsector_label:
            display_value = selected_subsector_label
            tooltip_value = ", ".join(selected_subsector_label)
        else:
            display_value = f"All ({len(subsector_labels)})"
            tooltip_value = ", ".join(subsector_labels)

        bordered_metric("Selected Subsectors", display_value, tooltip_enabled=True, tooltip_value=tooltip_value)

    with col3:
        bordered_metric(f"Total Emissions {gas_unit}", format_number_short(emissions_value), value_color="red")

    with col4:
        if mom_delta == 0.0:
            arrow = ""
        elif mom_delta > 0.0:
            arrow = "🔼"
        else:
            arrow = "🔽"

        bordered_metric("MoM Change (%)", f'''{arrow} {abs(mom_delta_rounded):.1f}%''', value_color=mom_color)

    with col5:
        if yoy_delta_rounded == 0.0:
            arrow = ""
        elif yoy_delta > 0.0:
            arrow = "🔼"
        else:
            arrow = "🔽"
        bordered_metric("YoY Change (%)", f'''{arrow} {abs(yoy_delta_rounded)}%''', value_color=yoy_color)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------ Monthly Subsector Stacked Bar -----------------------------------------
    # st.subheader("Annual Emissions by Sector")

    # Optional name normalization for country dropdown
    country_name_map = {
        "United States of America": "United States",
        #"Russia": "Russian Federation"
    }
    raw_country = country_name_map.get(selected_scope, selected_scope)

    # Start building the query
    query = f"""
        SELECT 
            MAKE_DATE(year, month, 1) AS year_month,
            sector,
            SUM(emissions_quantity) AS emissions_quantity
        FROM '{country_subsector_totals_path}'
        WHERE gas = '{selected_gas}'
            AND year >= {earliest_year}
            and country_name is not null
    """

    # Apply region or country filter
    if selected_scope in region_options and region_condition:
        query += f" AND {region_condition['column_name']} = '{region_condition['column_value']}'"
    elif selected_scope != "Global":
        query += f" AND country_name = '{raw_country}'"

    # Add optional sector/subsector filters
    if selected_sector_raw:
        query += f" AND sector = '{selected_sector_raw}'"
    if selected_subsector_raw:
        if len(selected_subsector_raw) == 1:
            query += f" AND subsector = '{selected_subsector_raw[0]}'"
        else:
            formatted_subsectors = ",".join([f"'{sub}'" for sub in selected_subsector_raw])
            query += f" AND subsector IN ({formatted_subsectors})"

    # Finalize the query
    query += " GROUP BY year_month, sector ORDER BY year_month, sector"

    # Run the query
    df_monthly = con.execute(query).df()

    # Convert year_month to string for better display
    df_monthly["year_month"] = pd.to_datetime(df_monthly["year_month"])
    df_monthly["year_month_str"] = df_monthly["year_month"].dt.strftime("%b %Y")

    # Plot if data exists
    if not df_monthly.empty:
        df_totals = df_monthly.groupby("year_month_str", as_index=False).agg(
            total_emissions=("emissions_quantity", "sum")
        )

        fig_monthly = px.bar(
            df_monthly,
            x="year_month_str",
            y="emissions_quantity",
            color="sector",
            labels={
                "emissions_quantity": f"Emissions ({gas_unit})",
                "year_month_str": "Month",
                "sector": "Sector"
            }
        )

        fig_monthly.update_layout(
            barmode="stack",
            xaxis=dict(type="category"),
            legend_title="Sector",
            margin=dict(t=50, b=30)
        )

        fig_monthly.add_trace(
            go.Bar(
                x=df_totals["year_month_str"],
                y=[0] * len(df_totals),
                text=[format_number_short(v) for v in df_totals["total_emissions"]],
                textposition="outside",
                marker=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                cliponaxis=False,
                name="Total"
            )
        )

        st.subheader("Monthly Emissions by Sector")
        st.plotly_chart(fig_monthly, use_container_width=True)

    else:
        st.markdown(
            """
            <div style='border: 1px solid #ccc; height: 300px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>No data available for the selected filters</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    

    # Table display setup
    df_stats_filtered['abs_mom_change'] = df_stats_filtered['mom_change'].abs()
    df_stats_filtered = df_stats_filtered.sort_values(by='abs_mom_change', ascending=False).reset_index(drop=True)

    display_cols = [
        'country_name',
        emissions_column_prev,
        emissions_column_latest,
        'mom_change',
        'mom_percent_change',
        'month_yoy_percent_change',
        'emissions_slope_36_months_t_per_month'
    ]

    rename_map = {
        emissions_column_latest: 'Emissions ' + emissions_column_latest[-6:-2] + '-' + emissions_column_latest[-2:],
        emissions_column_prev: 'Emissions ' + emissions_column_prev[-6:-2] + '-' + emissions_column_prev[-2:],
        'emissions_slope_36_months_t_per_month': 'Average Monthly Change (3 Year)'
    }

    def color_change(val):
        return f'color: {"green" if val < 0 else "red"}'

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotting
    st.subheader(f"Emissions Over Time ({gas_unit}) - {selected_scope} | {selected_subsector_label}")

    subsector_condition = ""
    if selected_subsector_raw:
        if len(selected_subsector_raw) == 1:
            subsector_condition = f"AND subsector = '{selected_subsector_raw[0]}'"
        elif len(selected_subsector_raw) > 1:
            formatted_subsectors = ",".join([f"'{sub}'" for sub in selected_subsector_raw])
            subsector_condition = f"AND subsector IN ({formatted_subsectors})"

    region_clause = (
        f"AND {region_condition['column_name']} IN ('United States', 'United States of America')"
            if region_condition and isinstance(region_condition['column_value'], list)
        else f"AND {region_condition['column_name']} = '{region_condition['column_value']}'"
            if region_condition
        else ''
    )

    query_country = f"""
        WITH latest_month AS (
            SELECT MAX(MAKE_DATE(year, month, 1)) AS max_date
                , date_trunc('year', MAX(MAKE_DATE(year, month, 1))) - INTERVAL '3 years' AS cutoff_date
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
            AND country_name IS NOT NULL
        )
        SELECT 
            MAKE_DATE(year, month, 1) AS year_month,
            SUM(emissions_quantity) AS country_emissions_quantity
        FROM '{country_subsector_totals_path}', latest_month
        WHERE gas = '{selected_gas}'
            AND country_name IS NOT NULL
            AND MAKE_DATE(year, month, 1) >= cutoff_date
            {subsector_condition}
            {'AND sector = \'%s\'' % selected_sector_raw if selected_sector_raw else ''}
            {region_clause}
        GROUP BY year_month
        ORDER BY year_month
    """

    # --------------- Emissions Line Charts ---------------
    country_df = con.execute(query_country).df()

    if not country_df.empty:
        country_df['year_month'] = pd.to_datetime(country_df['year_month'])

    # Check which charts should be shown
    show_activity_and_ef = selected_subsector_raw and not monthly_df.empty

    # Dynamic row/height logic
    num_rows = 3 if show_activity_and_ef else 1
    subplot_titles = ["Emissions Over Time"]
    if show_activity_and_ef:
        subplot_titles += ["Activity Over Time", "Emission Factor Over Time"]

    # Create subplot figure
    fig_combined = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )

    # Row 1 — Emissions
    if not monthly_df.empty:
        fig_combined.add_trace(
            go.Scatter(
                x=monthly_df['year_month'],
                y=monthly_df['emissions_quantity'],
                mode='lines+markers',
                name='Assets'
            ),
            row=1, col=1
        )

    if not country_df.empty:
        fig_combined.add_trace(
            go.Scatter(
                x=country_df['year_month'],
                y=country_df['country_emissions_quantity'],
                mode='lines+markers',
                name='Total',
                line=dict(color='#E9967A')
            ),
            row=1, col=1
        )

        # Add vertical quarter lines across all rows
        min_date = country_df['year_month'].min()
        max_date = country_df['year_month'].max()
        quarter_starts = pd.date_range(
            start=min_date.to_period("Q").start_time,
            end=max_date.to_period("Q").end_time + pd.offsets.QuarterBegin(1),
            freq='QS'
        )

    # Row 2 — Activity
    if show_activity_and_ef:
        fig_combined.add_trace(
            go.Scatter(
                x=monthly_df['year_month'],
                y=monthly_df['activity'],
                mode='lines+markers',
                name='Activity'
            ),
            row=2, col=1
        )

    # Row 3 — Emissions Factor
    if show_activity_and_ef:
        fig_combined.add_trace(
            go.Scatter(
                x=monthly_df['year_month'],
                y=monthly_df['mean_emissions_factor'],
                mode='lines+markers',
                name='Emission Factor'
            ),
            row=3, col=1
        )

    for q_start in quarter_starts:
            # Just once — this applies to all subplots when x-axis is shared
            fig_combined.add_vline(
                x=q_start,
                line_width=1,
                line_dash='dash',
                line_color='gray'
            )

            # Add label only once (above row 1)
            fig_combined.add_annotation(
                x=q_start,
                y=1.01,
                xref="x",
                yref="paper",
                text=f"Q{((q_start.month - 1) // 3 + 1)} {q_start.year}",
                showarrow=False,
                font=dict(size=9),
                align="center"
            )

    # Layout adjustments
    fig_combined.update_layout(
        height=900 if show_activity_and_ef else 400,
        #title_text="Emissions Dashboard",
        showlegend=True,
        margin=dict(t=80, b=40)
    )

    st.plotly_chart(fig_combined, use_container_width=True)

    # else:
    #     st.markdown(
    #         """
    #         <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
    #             <h4>No asset-level or country-level data for this subsector</h4>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Top Movers by Absolute MoM Change")

    styled_df = (
        df_stats_filtered[display_cols]
        .rename(columns=rename_map)
        .style.format({
            rename_map[emissions_column_prev]: "{:,.0f}",
            rename_map[emissions_column_latest]: "{:,.0f}",
            'mom_change': "{:,.0f}",
            'mom_percent_change': "{:.1f}%",
            'month_yoy_percent_change': "{:.1f}%",
            rename_map['emissions_slope_36_months_t_per_month']: "{:,.0f}"
        })
        .applymap(color_change, subset=['mom_change', 'mom_percent_change', 'month_yoy_percent_change', rename_map['emissions_slope_36_months_t_per_month']])
    )

    st.dataframe(styled_df, use_container_width=True, height=450)


    # the code chunk to the end is for subsector stacked bar chart/
    st.markdown("<br>", unsafe_allow_html=True)


    # --------------- Cumulative Emissions Chart ----------------
    if not country_df.empty:
        import calendar

        country_df["year"] = country_df["year_month"].dt.year
        country_df["month"] = country_df["year_month"].dt.month
        country_df = country_df.sort_values(["year", "month"])
        country_df["cumulative_emissions"] = country_df.groupby("year")["country_emissions_quantity"].cumsum()
        country_df["month_name"] = country_df["month"].apply(lambda x: calendar.month_abbr[x])

        # Get unique years and month names
        years = country_df["year"].unique()
        month_abbrs = list(calendar.month_abbr)[1:]  # Jan, Feb, ..., Dec

        # Initialize figure
        fig_cumulative = go.Figure()

        # Add a bar for each year
        for year in years:
            df_year = country_df[country_df["year"] == year]

            # Ensure all months present
            month_values = []
            for m in range(1, 13):
                month_row = df_year[df_year["month"] == m]
                if not month_row.empty:
                    month_values.append(month_row["cumulative_emissions"].iloc[0])
                else:
                    month_values.append(0)

            fig_cumulative.add_bar(
                x=month_abbrs,
                y=month_values,
                name=str(year)
            )


        st.subheader("Cumulative Monthly Emissions by Year")

        # Update layout to group bars
        fig_cumulative.update_layout(
            barmode="group",
            # title="Cumulative Monthly Emissions by Year",
            xaxis_title="Month",
            yaxis_title="Cumulative Emissions (tCO₂e)",
            xaxis=dict(categoryorder="array", categoryarray=month_abbrs),
            legend_title="Year",
            margin=dict(t=50, b=30)
        )

        st.plotly_chart(fig_cumulative, use_container_width=True)

    else:
        st.markdown(
            """
            <div style='border: 1px solid #ccc; height: 300px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>No cumulative emissions data available for the selected filters</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    # loading da taframes into excel
    if not monthly_df.empty or not country_df.empty or not df_stats_filtered.empty or not df_monthly.empty:
        # Create dictionary of DataFrames to export
        dfs_for_excel = {
            "Monthly Sector Emissions": df_monthly,
            "Country Total Emissions": country_df,
            "Asset Total Emissions": monthly_df,
            "Stats Data": df_stats_filtered
        }

        # Use the utility function to create the Excel file
        excel_file = create_excel_file(dfs_for_excel)

        # Fill in the placeholder with the actual download button
        download_placeholder.download_button(
            label="   ⬇   Download Chart Data   ",
            data=excel_file,
            file_name="climate_trace_dashboard_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="The downloaded data will represent your dropdown selections."
        )

        # st.download_button( 
        #         label="⬇ Download Data ",  # keep it tight – can use "Download" if you want
        #         data="your,data,here\n1,2,3",  # Replace with real CSV
        #         file_name="emissions_data.csv",
        #         mime="text/csv"
        #     )

    con.close()
