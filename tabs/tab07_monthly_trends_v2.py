import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils import format_number_short, map_region_condition
from config import CONFIG
from calendar import month_name


def show_monthly_trends_v2():

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

    # Configure data paths
    country_subsector_stats_path = CONFIG['country_subsector_stats_path']
    country_subsector_totals_path = CONFIG['country_subsector_totals_path']
    asset_path = CONFIG['asset_emissions_country_subsector_path']
    gadm_0_path = CONFIG['gadm_0_path']
    region_options = [r for r in CONFIG['region_options'] if r != 'G20']

    con = duckdb.connect()

    st.markdown("<br>", unsafe_allow_html=True)

    # Create columns for Change View toggle and Region dropdown
    col_view, col_region, col_spacer = st.columns([3.5, 2, 9])

    with col_view:
        # Add title and help tooltip for the toggle
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 0.95em; font-weight: 600;">Change Time Period</span>
                <div style="position: relative; display: inline-block;">
                    <span style="cursor: help; color: #888; font-size: 0.9em;" title="Select how to compare emissions data">ⓘ</span>
                    <div style="position: absolute; bottom: 25px; left: -100px; width: 320px; background-color: #262730; border: 1px solid #444; border-radius: 8px; padding: 12px; font-size: 0.85em; line-height: 1.8; display: none; z-index: 1000;" class="tooltip-content">
                        <strong>Month YoY:</strong> Compare current month to same month last year<br><br>
                        <strong>Year-to-Date:</strong> Compare cumulative emissions from Jan to current month across years<br><br>
                        <strong>Month-over-Month:</strong> Compare current month to previous month
                    </div>
                </div>
            </div>
            <style>
                div:has(> span[title]):hover .tooltip-content {
                    display: block !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Three toggle options with new labels
        trend_view = st.segmented_control(
            "View",
            options=["Month YoY", "Year-to-Date", "Month-over-Month"],
            default="Month YoY",
            label_visibility="collapsed"
        )

    with col_region:
        # Region dropdown
        st.markdown(
            """
            <div style="margin-bottom: 10px;">
                <span style="font-size: 0.95em; font-weight: 600;">Region</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        selected_scope = st.selectbox(
            "Region",
            region_options,
            key="scope_selector",
            label_visibility="collapsed"
        )
        region_condition = map_region_condition(selected_scope, {})

    

    st.markdown("<br>", unsafe_allow_html=True)

    # Climate TRACE sector color map
    sector_color_map = {
        "agriculture": "#E8516C",
        "buildings": "#03A0E3",
        "fluorinated-gases": "#B6B4B4",
        "forestry-and-land-use": "#779608",
        "fossil-fuel-operations": "#FF6F42",
        "manufacturing": "#9554FF",
        "mineral-extraction": "#4380F5",
        "power": "#56979F",
        "transportation": "#FBBA14",
        "waste": "#BBD421"
    }

    # Helper function to create color shades for subsectors
    def create_color_shades(base_color, num_shades):
        """Create darker/lighter shades of a base color for subsectors, never reaching white"""
        import colorsys
        # Convert hex to RGB
        base_color = base_color.lstrip('#')
        r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))

        # Convert to HSL
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)

        # Generate shades - use base color as the first shade, then create variations
        shades = []
        max_lightness = 0.75  # Never go above 75% lightness to avoid white

        for i in range(num_shades):
            if i == 0:
                # First shade is the base color (darkest/most saturated)
                new_l = l
            else:
                # Gradually lighten but cap at max_lightness
                lightness_increment = (max_lightness - l) / max(1, num_shades - 1)
                new_l = min(max_lightness, l + (i * lightness_increment))

            new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(new_r * 255), int(new_g * 255), int(new_b * 255)
            )
            shades.append(hex_color)
        return shades

    # Get emissions columns from the stats data
    # Build WHERE clause based on region selection
    where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

    if region_condition:
        column_name = region_condition['column_name']
        column_value = region_condition['column_value']
        if isinstance(column_value, bool):
            where_clauses.append(f"{column_name} = {str(column_value).upper()}")
        else:
            where_clauses.append(f"{column_name} = '{column_value}'")

    where_clause = " AND ".join(where_clauses)

    # Get column names without loading all data (for performance)
    df_columns = con.execute(f"""
        SELECT *
        FROM '{country_subsector_stats_path}'
        WHERE {where_clause}
        LIMIT 0
    """).df()

    emissions_columns = [col for col in df_columns.columns if col.startswith("emissions_quantity_")]
    emissions_columns_sorted = sorted(emissions_columns, reverse=True)
    emissions_column_latest = emissions_columns_sorted[0]
    emissions_column_prev = emissions_columns_sorted[1]

    # Extract year and month from column names (format: emissions_quantity_YYYYMM)
    latest_year = int(emissions_column_latest[-6:-2])
    latest_month = int(emissions_column_latest[-2:])

    # Calculate global totals based on selected view
    if trend_view == "Month YoY":
        # Query DuckDB directly for aggregated data instead of loading all data
        global_totals_query = f"""
            SELECT
                SUM({emissions_column_latest}) as latest,
                SUM(month_yoy_change) as yoy_change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
        """
        df_global_totals = con.execute(global_totals_query).df()

        global_latest = df_global_totals['latest'].iloc[0]
        global_yoy_change = df_global_totals['yoy_change'].iloc[0]
        global_previous = global_latest - global_yoy_change

        absolute_change = global_yoy_change
        percent_change = (absolute_change / global_previous * 100) if global_previous != 0 else 0

    elif trend_view == "Year-to-Date":
        # Build WHERE clause with region filter for YTD
        ytd_global_where_clauses = [
            "gas = 'co2e_100yr'",
            "country_name IS NOT NULL",
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                ytd_global_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                ytd_global_where_clauses.append(f"{column_name} = '{column_value}'")

        ytd_global_where_clause = " AND ".join(ytd_global_where_clauses)

        # Query monthly data and calculate cumulative sums
        ytd_query = f"""
            SELECT
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {ytd_global_where_clause}
            GROUP BY year, month
            ORDER BY year, month
        """
        df_ytd = con.execute(ytd_query).df()

        # Calculate cumulative emissions per year
        df_ytd['cumulative'] = df_ytd.groupby('year')['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each year
        current_year_cumulative = df_ytd[
            (df_ytd['year'] == latest_year) & (df_ytd['month'] == latest_month)
        ]['cumulative'].sum()

        previous_year_cumulative = df_ytd[
            (df_ytd['year'] == latest_year - 1) & (df_ytd['month'] == latest_month)
        ]['cumulative'].sum()

        global_latest = current_year_cumulative
        global_previous = previous_year_cumulative
        absolute_change = global_latest - global_previous
        percent_change = (absolute_change / global_previous * 100) if global_previous != 0 else 0

    else:  # MoM
        # Query DuckDB directly for aggregated data instead of loading all data
        global_totals_query = f"""
            SELECT
                SUM({emissions_column_latest}) as latest,
                SUM({emissions_column_prev}) as previous,
                SUM(mom_change) as mom_change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
        """
        df_global_totals = con.execute(global_totals_query).df()

        global_latest = df_global_totals['latest'].iloc[0]
        global_previous = df_global_totals['previous'].iloc[0]
        absolute_change = df_global_totals['mom_change'].iloc[0]

        percent_change = (absolute_change / global_previous * 100) if global_previous != 0 else 0

    # Determine arrow and color using actual arrow symbols
    if absolute_change > 0:
        # Red upward arrow
        arrow = "↑"
        change_color = "red"
    elif absolute_change < 0:
        # Green downward arrow
        arrow = "↓"
        change_color = "green"
    else:
        # Horizontal arrow
        arrow = "→"
        change_color = "gray"

    # ==================== Five Summary Cards ====================

    # Card 1: Dynamic change based on selected view
    if trend_view == "Month YoY":
        card1_label = "Month YoY Change"
        # Get month name from latest_month
        import calendar
        current_month_name = calendar.month_name[latest_month]
        card1_current_label = f"{current_month_name} {latest_year}"
        card1_previous_label = f"{current_month_name} {latest_year - 1}"
    elif trend_view == "Year-to-Date":
        card1_label = "Year-to-Date Change"
        card1_current_label = f"YTD {latest_year}"
        card1_previous_label = f"YTD {latest_year - 1}"
    else:  # Month-over-Month
        card1_label = "Month-over-Month Change"
        import calendar
        current_month_name = calendar.month_name[latest_month]
        # Calculate previous month (handle year boundary)
        if latest_month == 1:
            prev_month = 12
            prev_year = latest_year - 1
        else:
            prev_month = latest_month - 1
            prev_year = latest_year
        prev_month_name = calendar.month_name[prev_month]
        card1_current_label = f"{current_month_name} {latest_year}"
        card1_previous_label = f"{prev_month_name} {prev_year}"

    card1_value = f"{format_number_short(abs(absolute_change))}"
    card1_subvalue = f"{abs(percent_change):.1f}%"

    # Add aggregate totals for display in card 1 (formatted with commas, no decimals)
    card1_current_total = f"{global_latest:,.0f}"
    card1_previous_total = f"{global_previous:,.0f}"

    # Calculate country-level changes for Cards 2 & 3
    if trend_view == "Year-to-Date":
        # Build WHERE clause with region filter
        country_ytd_where_clauses = [
            "gas = 'co2e_100yr'",
            "country_name IS NOT NULL",
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                country_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                country_ytd_where_clauses.append(f"{column_name} = '{column_value}'")

        country_ytd_where_clause = " AND ".join(country_ytd_where_clauses)

        # Query country-level YTD data from totals file
        country_ytd_query = f"""
            SELECT
                country_name,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {country_ytd_where_clause}
            GROUP BY country_name, year, month
            ORDER BY country_name, year, month
        """
        df_country_ytd_raw = con.execute(country_ytd_query).df()

        # Calculate cumulative emissions per country per year
        df_country_ytd_raw['cumulative'] = df_country_ytd_raw.groupby(['country_name', 'year'])['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each country for each year
        df_current = df_country_ytd_raw[
            (df_country_ytd_raw['year'] == latest_year) & (df_country_ytd_raw['month'] == latest_month)
        ][['country_name', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

        df_previous = df_country_ytd_raw[
            (df_country_ytd_raw['year'] == latest_year - 1) & (df_country_ytd_raw['month'] == latest_month)
        ][['country_name', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

        df_country_totals = df_current.merge(df_previous, on='country_name', how='outer').fillna(0)
        df_country_totals['change'] = df_country_totals['current_ytd'] - df_country_totals['previous_ytd']
    else:
        # Query DuckDB directly for aggregated data instead of pandas groupby
        change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'

        # Build WHERE clause for region filter
        country_totals_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                country_totals_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                country_totals_where_clauses.append(f"{column_name} = '{column_value}'")

        country_totals_where_clause = " AND ".join(country_totals_where_clauses)

        country_totals_query = f"""
            SELECT
                country_name,
                SUM({change_column}) as change
            FROM '{country_subsector_stats_path}'
            WHERE {country_totals_where_clause}
            GROUP BY country_name
        """
        df_country_totals = con.execute(country_totals_query).df()

    # Card 2: Largest Country Decrease
    largest_decrease = df_country_totals.loc[df_country_totals['change'].idxmin()]
    largest_decrease_country = largest_decrease['country_name']
    largest_decrease_value = largest_decrease['change']

    # Calculate percent change for largest decrease country
    if trend_view == "Year-to-Date":
        # For YTD, get cumulative values from the country totals we just calculated
        country_row = df_country_totals[df_country_totals['country_name'] == largest_decrease_country]
        if not country_row.empty:
            previous_total = country_row['previous_ytd'].iloc[0]
        else:
            previous_total = 0
    else:
        # Query DuckDB for this specific country's data
        country_latest_query = f"""
            SELECT
                SUM({emissions_column_latest}) as latest,
                SUM({emissions_column_prev}) as previous,
                SUM(month_yoy_change) as yoy_change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
                AND country_name = '{largest_decrease_country}'
        """
        df_country_latest = con.execute(country_latest_query).df()
        current_total = df_country_latest['latest'].iloc[0]
        if trend_view == "Month YoY":
            previous_total = current_total - df_country_latest['yoy_change'].iloc[0]
        else:  # MoM
            previous_total = df_country_latest['previous'].iloc[0]

    largest_decrease_percent = (largest_decrease_value / previous_total * 100) if previous_total != 0 else 0

    # Find the subsector driving the decrease for this country
    if trend_view == "Year-to-Date":
        # Query subsector-level YTD data for this specific country
        subsector_ytd_query = f"""
            SELECT
                subsector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name = '{largest_decrease_country}'
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
            GROUP BY subsector, year, month
            ORDER BY subsector, year, month
        """
        df_subsector_ytd = con.execute(subsector_ytd_query).df()

        # Calculate cumulative emissions per subsector per year
        df_subsector_ytd['cumulative'] = df_subsector_ytd.groupby(['subsector', 'year'])['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each subsector for each year
        df_current_sub = df_subsector_ytd[
            (df_subsector_ytd['year'] == latest_year) & (df_subsector_ytd['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

        df_previous_sub = df_subsector_ytd[
            (df_subsector_ytd['year'] == latest_year - 1) & (df_subsector_ytd['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

        df_subsector_change = df_current_sub.merge(df_previous_sub, on='subsector', how='outer').fillna(0)
        df_subsector_change['change'] = df_subsector_change['current_ytd'] - df_subsector_change['previous_ytd']

        driving_subsector_decrease = df_subsector_change.loc[df_subsector_change['change'].idxmin()]
        subsector_decrease_name = driving_subsector_decrease['subsector']
    else:
        # Query DuckDB for this specific country's subsector data
        change_col = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'
        country_subsector_query = f"""
            SELECT
                subsector,
                SUM({change_col}) as change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
                AND country_name = '{largest_decrease_country}'
            GROUP BY subsector
        """
        df_country_subsector = con.execute(country_subsector_query).df()

        driving_subsector_decrease = df_country_subsector.loc[df_country_subsector['change'].idxmin()]
        subsector_decrease_name = driving_subsector_decrease['subsector']

    # Card 3: Largest Country Increase
    largest_increase = df_country_totals.loc[df_country_totals['change'].idxmax()]
    largest_increase_country = largest_increase['country_name']
    largest_increase_value = largest_increase['change']

    # Calculate percent change for largest increase country
    if trend_view == "Year-to-Date":
        # For YTD, get cumulative values from the country totals we just calculated
        country_row_inc = df_country_totals[df_country_totals['country_name'] == largest_increase_country]
        if not country_row_inc.empty:
            previous_total_inc = country_row_inc['previous_ytd'].iloc[0]
        else:
            previous_total_inc = 0
    else:
        # Query DuckDB for this specific country's data
        country_latest_inc_query = f"""
            SELECT
                SUM({emissions_column_latest}) as latest,
                SUM({emissions_column_prev}) as previous,
                SUM(month_yoy_change) as yoy_change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
                AND country_name = '{largest_increase_country}'
        """
        df_country_latest_inc = con.execute(country_latest_inc_query).df()
        current_total_inc = df_country_latest_inc['latest'].iloc[0]
        if trend_view == "Month YoY":
            previous_total_inc = current_total_inc - df_country_latest_inc['yoy_change'].iloc[0]
        else:  # MoM
            previous_total_inc = df_country_latest_inc['previous'].iloc[0]

    largest_increase_percent = (largest_increase_value / previous_total_inc * 100) if previous_total_inc != 0 else 0

    # Find the subsector driving the increase for this country
    if trend_view == "Year-to-Date":
        # Query subsector-level YTD data for this specific country
        subsector_ytd_query_inc = f"""
            SELECT
                subsector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name = '{largest_increase_country}'
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
            GROUP BY subsector, year, month
            ORDER BY subsector, year, month
        """
        df_subsector_ytd_inc = con.execute(subsector_ytd_query_inc).df()

        # Calculate cumulative emissions per subsector per year
        df_subsector_ytd_inc['cumulative'] = df_subsector_ytd_inc.groupby(['subsector', 'year'])['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each subsector for each year
        df_current_sub_inc = df_subsector_ytd_inc[
            (df_subsector_ytd_inc['year'] == latest_year) & (df_subsector_ytd_inc['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

        df_previous_sub_inc = df_subsector_ytd_inc[
            (df_subsector_ytd_inc['year'] == latest_year - 1) & (df_subsector_ytd_inc['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

        df_subsector_change_inc = df_current_sub_inc.merge(df_previous_sub_inc, on='subsector', how='outer').fillna(0)
        df_subsector_change_inc['change'] = df_subsector_change_inc['current_ytd'] - df_subsector_change_inc['previous_ytd']

        driving_subsector_increase = df_subsector_change_inc.loc[df_subsector_change_inc['change'].idxmax()]
        subsector_increase_name = driving_subsector_increase['subsector']
    else:
        # Query DuckDB for this specific country's subsector data
        change_col_inc = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'
        country_subsector_inc_query = f"""
            SELECT
                subsector,
                SUM({change_col_inc}) as change
            FROM '{country_subsector_stats_path}'
            WHERE {where_clause}
                AND country_name = '{largest_increase_country}'
            GROUP BY subsector
        """
        df_country_subsector_inc = con.execute(country_subsector_inc_query).df()

        driving_subsector_increase = df_country_subsector_inc.loc[df_country_subsector_inc['change'].idxmax()]
        subsector_increase_name = driving_subsector_increase['subsector']

    # Card 4: Biggest Sector Move
    if trend_view == "Year-to-Date":
        # Build WHERE clause with region filter
        sector_card_ytd_where_clauses = [
            "gas = 'co2e_100yr'",
            "country_name IS NOT NULL",
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                sector_card_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                sector_card_ytd_where_clauses.append(f"{column_name} = '{column_value}'")

        sector_card_ytd_where_clause = " AND ".join(sector_card_ytd_where_clauses)

        # Query sector-level YTD data using cumulative approach
        sector_ytd_query = f"""
            SELECT
                sector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {sector_card_ytd_where_clause}
            GROUP BY sector, year, month
            ORDER BY sector, year, month
        """
        df_sector_ytd_raw = con.execute(sector_ytd_query).df()

        # Calculate cumulative emissions per sector per year
        df_sector_ytd_raw['cumulative'] = df_sector_ytd_raw.groupby(['sector', 'year'])['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each sector for each year
        df_current_sec = df_sector_ytd_raw[
            (df_sector_ytd_raw['year'] == latest_year) & (df_sector_ytd_raw['month'] == latest_month)
        ][['sector', 'cumulative']].rename(columns={'cumulative': 'current'})

        df_previous_sec = df_sector_ytd_raw[
            (df_sector_ytd_raw['year'] == latest_year - 1) & (df_sector_ytd_raw['month'] == latest_month)
        ][['sector', 'cumulative']].rename(columns={'cumulative': 'previous'})

        df_sector_totals = df_current_sec.merge(df_previous_sec, on='sector', how='outer').fillna(0)
        df_sector_totals['change'] = df_sector_totals['current'] - df_sector_totals['previous']
    else:
        # Query DuckDB directly for aggregated data
        change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'
        latest_column = emissions_column_latest
        prev_column = emissions_column_prev

        # Build WHERE clause for region filter
        sector_card_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                sector_card_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                sector_card_where_clauses.append(f"{column_name} = '{column_value}'")

        sector_card_where_clause = " AND ".join(sector_card_where_clauses)

        sector_totals_query = f"""
            SELECT
                sector,
                SUM({change_column}) as change,
                SUM({latest_column}) as current,
                SUM({prev_column}) as previous
            FROM '{country_subsector_stats_path}'
            WHERE {sector_card_where_clause}
            GROUP BY sector
        """
        df_sector_totals = con.execute(sector_totals_query).df()

        # Adjust previous for Month YoY
        if trend_view == "Month YoY":
            df_sector_totals['previous'] = df_sector_totals['current'] - df_sector_totals['change']

    df_sector_totals['abs_change'] = df_sector_totals['change'].abs()
    biggest_sector = df_sector_totals.loc[df_sector_totals['abs_change'].idxmax()]
    biggest_sector_name = biggest_sector['sector']
    biggest_sector_value = biggest_sector['change']
    biggest_sector_previous = biggest_sector['previous']
    biggest_sector_percent = (biggest_sector_value / biggest_sector_previous * 100) if biggest_sector_previous != 0 else 0

    # Card 5: Biggest Subsector Move
    if trend_view == "Year-to-Date":
        # Build WHERE clause with region filter
        subsector_card_ytd_where_clauses = [
            "gas = 'co2e_100yr'",
            "country_name IS NOT NULL",
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                subsector_card_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                subsector_card_ytd_where_clauses.append(f"{column_name} = '{column_value}'")

        subsector_card_ytd_where_clause = " AND ".join(subsector_card_ytd_where_clauses)

        # Query subsector-level YTD data using cumulative approach
        subsector_ytd_query = f"""
            SELECT
                subsector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {subsector_card_ytd_where_clause}
            GROUP BY subsector, year, month
            ORDER BY subsector, year, month
        """
        df_subsector_ytd_raw = con.execute(subsector_ytd_query).df()

        # Calculate cumulative emissions per subsector per year
        df_subsector_ytd_raw['cumulative'] = df_subsector_ytd_raw.groupby(['subsector', 'year'])['emissions_quantity'].cumsum()

        # Get the latest month's cumulative value for each subsector for each year
        df_current_subsec = df_subsector_ytd_raw[
            (df_subsector_ytd_raw['year'] == latest_year) & (df_subsector_ytd_raw['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'current'})

        df_previous_subsec = df_subsector_ytd_raw[
            (df_subsector_ytd_raw['year'] == latest_year - 1) & (df_subsector_ytd_raw['month'] == latest_month)
        ][['subsector', 'cumulative']].rename(columns={'cumulative': 'previous'})

        df_subsector_totals = df_current_subsec.merge(df_previous_subsec, on='subsector', how='outer').fillna(0)
        df_subsector_totals['change'] = df_subsector_totals['current'] - df_subsector_totals['previous']
    else:
        # Query DuckDB directly for aggregated data
        change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'
        latest_column = emissions_column_latest
        prev_column = emissions_column_prev

        # Build WHERE clause for region filter
        subsector_card_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

        if region_condition:
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                subsector_card_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                subsector_card_where_clauses.append(f"{column_name} = '{column_value}'")

        subsector_card_where_clause = " AND ".join(subsector_card_where_clauses)

        subsector_totals_query = f"""
            SELECT
                subsector,
                SUM({change_column}) as change,
                SUM({latest_column}) as current,
                SUM({prev_column}) as previous
            FROM '{country_subsector_stats_path}'
            WHERE {subsector_card_where_clause}
            GROUP BY subsector
        """
        df_subsector_totals = con.execute(subsector_totals_query).df()

        # Adjust previous for Month YoY
        if trend_view == "Month YoY":
            df_subsector_totals['previous'] = df_subsector_totals['current'] - df_subsector_totals['change']

    df_subsector_totals['abs_change'] = df_subsector_totals['change'].abs()
    biggest_subsector = df_subsector_totals.loc[df_subsector_totals['abs_change'].idxmax()]
    biggest_subsector_name = biggest_subsector['subsector']
    biggest_subsector_value = biggest_subsector['change']
    biggest_subsector_previous = biggest_subsector['previous']
    biggest_subsector_percent = (biggest_subsector_value / biggest_subsector_previous * 100) if biggest_subsector_previous != 0 else 0

    # Display the 5 cards with improved modern design
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.85em; font-weight: 600; margin-bottom: auto;">{card1_label}</div>
                <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: {change_color}; margin-bottom: 28px;">
                    {arrow} {card1_value} <span style="color: #888;">(</span><span style="color: {change_color};">{card1_subvalue}</span><span style="color: #888;">)</span>
                </div>
                <div style="font-size: 0.75em; text-align: left; color: #888; line-height: 1.3; padding-left: 8px;">
                    <div>{card1_current_label}: <span style="font-weight: 600; color: var(--text-color);">{card1_current_total}</span></div>
                    <div>{card1_previous_label}: <span style="font-weight: 600; color: var(--text-color);">{card1_previous_total}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        decrease_color = "green" if largest_decrease_value < 0 else "red"
        decrease_arrow = "↓" if largest_decrease_value < 0 else "↑"
        st.markdown(
            f"""
            <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column;">
                <div style="font-size: 0.85em; font-weight: 600; margin-bottom: auto;">Largest Country Decrease</div>
                <div style="font-size: 1.2em; font-weight: bold; text-align: center; margin-top: auto; margin-bottom: 8px;">{largest_decrease_country}</div>
                <div style="font-size: 1.0em; text-align: center; color: {decrease_color}; margin-bottom: 8px; font-weight: 600;">
                    {decrease_arrow} {format_number_short(largest_decrease_value)} <span style="color: #888;">(</span><span style="color: {decrease_color};">{abs(largest_decrease_percent):.1f}%</span><span style="color: #888;">)</span>
                </div>
                <div style="font-size: 0.75em; text-align: left; color: #888; padding-left: 8px;">Driven by: <span style="font-style: italic;">{subsector_decrease_name.replace('-', ' ').title()}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        increase_color = "red" if largest_increase_value > 0 else "green"
        increase_arrow = "↑" if largest_increase_value > 0 else "↓"
        st.markdown(
            f"""
            <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column;">
                <div style="font-size: 0.85em; font-weight: 600; margin-bottom: auto;">Largest Country Increase</div>
                <div style="font-size: 1.2em; font-weight: bold; text-align: center; margin-top: auto; margin-bottom: 8px;">{largest_increase_country}</div>
                <div style="font-size: 1.0em; text-align: center; color: {increase_color}; margin-bottom: 8px; font-weight: 600;">
                    {increase_arrow} {format_number_short(largest_increase_value)} <span style="color: #888;">(</span><span style="color: {increase_color};">{abs(largest_increase_percent):.1f}%</span><span style="color: #888;">)</span>
                </div>
                <div style="font-size: 0.75em; text-align: left; color: #888; padding-left: 8px;">Driven by: <span style="font-style: italic;">{subsector_increase_name.replace('-', ' ').title()}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        sector_color = "red" if biggest_sector_value > 0 else "green"
        sector_arrow = "↑" if biggest_sector_value > 0 else "↓"
        st.markdown(
            f"""
            <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column;">
                <div style="font-size: 0.85em; font-weight: 600; margin-bottom: auto;">Biggest Sector Move</div>
                <div style="font-size: 1.2em; font-weight: bold; text-align: center; margin-top: auto; margin-bottom: 8px; word-wrap: break-word; overflow-wrap: break-word;">{biggest_sector_name.replace('-', ' ').title()}</div>
                <div style="font-size: 1.0em; text-align: center; color: {sector_color}; font-weight: 600; margin-bottom: 8px;">
                    {sector_arrow} {format_number_short(abs(biggest_sector_value))} <span style="color: #888;">(</span><span style="color: {sector_color};">{abs(biggest_sector_percent):.1f}%</span><span style="color: #888;">)</span>
                </div>
                <div style="font-size: 0.75em; text-align: left; color: transparent; padding-left: 8px;">.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        subsector_color = "red" if biggest_subsector_value > 0 else "green"
        subsector_arrow = "↑" if biggest_subsector_value > 0 else "↓"
        st.markdown(
            f"""
            <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column;">
                <div style="font-size: 0.85em; font-weight: 600; margin-bottom: auto;">Biggest Subsector Move</div>
                <div style="font-size: 1.2em; font-weight: bold; text-align: center; margin-top: auto; margin-bottom: 8px; word-wrap: break-word; overflow-wrap: break-word;">{biggest_subsector_name.replace('-', ' ').title()}</div>
                <div style="font-size: 1.0em; text-align: center; color: {subsector_color}; font-weight: 600; margin-bottom: 8px;">
                    {subsector_arrow} {format_number_short(abs(biggest_subsector_value))} <span style="color: #888;">(</span><span style="color: {subsector_color};">{abs(biggest_subsector_percent):.1f}%</span><span style="color: #888;">)</span>
                </div>
                <div style="font-size: 0.75em; text-align: left; color: transparent; padding-left: 8px;">.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== Visualizations Row ====================
    # Use columns to align the visualizations - left for sector movers, right for country movers
    viz_col1, viz_col2 = st.columns(2)

    # ==================== Sector Movers Visualization ====================
    with viz_col1:
        st.markdown("#### Sector Movers")

        # Prepare sector movers data with top 3 subsectors
        if trend_view == "Year-to-Date":
            # Build WHERE clause for region filter
            sector_ytd_where_clauses = [
                "gas = 'co2e_100yr'",
                "country_name IS NOT NULL",
                f"year IN ({latest_year - 1}, {latest_year})",
                f"month <= {latest_month}"
            ]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    sector_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    sector_ytd_where_clauses.append(f"{column_name} = '{column_value}'")

            sector_ytd_where_clause = " AND ".join(sector_ytd_where_clauses)

            # Query sector-subsector YTD data using cumulative approach
            sector_subsector_ytd_query = f"""
                SELECT
                    sector,
                    subsector,
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE {sector_ytd_where_clause}
                GROUP BY sector, subsector, year, month
                ORDER BY sector, subsector, year, month
            """
            df_sector_sub_ytd = con.execute(sector_subsector_ytd_query).df()

            # Calculate cumulative emissions per sector-subsector per year
            df_sector_sub_ytd['cumulative'] = df_sector_sub_ytd.groupby(['sector', 'subsector', 'year'])['emissions_quantity'].cumsum()

            # Get the latest month's cumulative value for each sector-subsector for each year
            df_current_ss = df_sector_sub_ytd[
                (df_sector_sub_ytd['year'] == latest_year) & (df_sector_sub_ytd['month'] == latest_month)
            ][['sector', 'subsector', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

            df_previous_ss = df_sector_sub_ytd[
                (df_sector_sub_ytd['year'] == latest_year - 1) & (df_sector_sub_ytd['month'] == latest_month)
            ][['sector', 'subsector', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

            df_sector_movers = df_current_ss.merge(df_previous_ss, on=['sector', 'subsector'], how='outer').fillna(0)
            df_sector_movers['change'] = df_sector_movers['current_ytd'] - df_sector_movers['previous_ytd']
        else:
            # Query DuckDB directly for aggregated data instead of pandas groupby
            change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'

            # Build WHERE clause for region filter
            movers_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    movers_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    movers_where_clauses.append(f"{column_name} = '{column_value}'")

            movers_where_clause = " AND ".join(movers_where_clauses)

            sector_movers_query = f"""
                SELECT
                    sector,
                    subsector,
                    SUM({change_column}) as change
                FROM '{country_subsector_stats_path}'
                WHERE {movers_where_clause}
                GROUP BY sector, subsector
            """
            df_sector_movers = con.execute(sector_movers_query).df()

        # For each sector, get top 3 subsectors by absolute change
        sector_data = []
        for sector in df_sector_movers['sector'].unique():
            df_sec = df_sector_movers[df_sector_movers['sector'] == sector].copy()
            df_sec['abs_change'] = df_sec['change'].abs()
            df_sec = df_sec.sort_values('abs_change', ascending=False)

            # Get top 3 subsectors
            top_3 = df_sec.head(3)
            other = df_sec.iloc[3:]['change'].sum() if len(df_sec) > 3 else 0

            # Add top 3 to sector data
            for _, row in top_3.iterrows():
                sector_data.append({
                    'sector': sector,
                    'subsector': row['subsector'],
                    'change': row['change']
                })

            # Add "Other" if there are more than 3 subsectors
            if len(df_sec) > 3 and other != 0:
                sector_data.append({
                    'sector': sector,
                    'subsector': 'Other',
                    'change': other
                })

        df_sector_viz = pd.DataFrame(sector_data)

        # Create horizontal stacked bar chart for sectors
        fig_sector_movers = go.Figure()

        # Get unique sectors and sort by absolute change (highest to lowest)
        sector_totals = df_sector_viz.groupby('sector')['change'].sum()
        sector_abs_totals = sector_totals.abs().sort_values(ascending=False)
        sectors_sorted = sector_abs_totals.index.tolist()

        # Reverse for Plotly (which displays bottom to top for horizontal bars)
        sectors_sorted_reversed = sectors_sorted[::-1]

        # For each sector, add traces for its subsectors (largest first = at base)
        for sector in sectors_sorted_reversed:
            df_sec = df_sector_viz[df_sector_viz['sector'] == sector].copy()

            # Sort subsectors by absolute change (largest first) so largest is at base
            df_sec['abs_change'] = df_sec['change'].abs()
            df_sec = df_sec.sort_values('abs_change', ascending=False)

            # Create color shades for this sector's subsectors
            base_color = sector_color_map.get(sector, "#999999")
            subsectors_list = df_sec['subsector'].tolist()
            shades = create_color_shades(base_color, len(subsectors_list))

            # Add traces in order (largest first = base)
            for i, (_, row) in enumerate(df_sec.iterrows()):
                subsector = row['subsector']
                value = row['change']

                fig_sector_movers.add_trace(go.Bar(
                    name=f"{sector}: {subsector}",
                    y=[sector],
                    x=[value],
                    orientation='h',
                    marker_color=shades[i],
                    showlegend=False,
                    hovertemplate=f"<b>{sector.replace('-', ' ').title()}</b><br>" +
                                  f"{subsector.replace('-', ' ').title()}: %{{x:,.0f}}<extra></extra>"
                ))

        fig_sector_movers.update_layout(
            barmode='relative',  # Use 'relative' so positive/negative subsectors stack properly
            xaxis_title='Emissions Change (tCO₂e)',
            yaxis_title='Sector',
            height=640,
            showlegend=False,
            margin=dict(l=150, r=80, t=30, b=50),
            yaxis=dict(
                categoryorder='array',
                categoryarray=sectors_sorted_reversed  # Explicitly order Y-axis from bottom to top
            )
        )

        # Add thick vertical line at x=0 (gray works in both light and dark mode)
        fig_sector_movers.add_vline(x=0, line_width=3, line_color="#666666", opacity=0.9)

        # Add net change labels for each sector
        for sector in sectors_sorted:
            net_change = sector_totals[sector]

            # Get the subsector data for this sector to find bar extents
            df_sec_labels = df_sector_viz[df_sector_viz['sector'] == sector]
            positive_extent = df_sec_labels[df_sec_labels['change'] > 0]['change'].sum()
            negative_extent = df_sec_labels[df_sec_labels['change'] < 0]['change'].sum()

            # Handle zero case specially
            if abs(net_change) < 0.5:  # Essentially zero
                arrow = ""
                color = "white"
                text = "0"
                x_pos = 0
                x_anchor = "left"
                xshift = 10
            else:
                arrow = "↑" if net_change > 0 else "↓"
                color = "red" if net_change > 0 else "green"
                text = f"{arrow} {format_number_short(abs(net_change))}"

                # Position label at the furthest extent of bars, not at net change
                if net_change > 0:
                    x_pos = positive_extent  # Furthest positive extent
                    x_anchor = "left"
                    xshift = 10
                else:
                    x_pos = negative_extent  # Furthest negative extent
                    x_anchor = "right"
                    xshift = -10

            fig_sector_movers.add_annotation(
                x=x_pos,
                y=sector,
                text=text,
                showarrow=False,
                font=dict(size=11, color=color, family="Arial Black"),
                xanchor=x_anchor,
                xshift=xshift
            )

        st.plotly_chart(fig_sector_movers, use_container_width=True)

    # ==================== Country Sector Movers Visualizations ====================
    with viz_col2:
        st.markdown("#### Country Sector Movers")

        # Get country-level data broken down by sector with region filter
        if trend_view == "Year-to-Date":
            # Build WHERE clause for region filter
            ytd_where_clauses = [
                "gas = 'co2e_100yr'",
                "country_name IS NOT NULL",
                f"year IN ({latest_year - 1}, {latest_year})",
                f"month <= {latest_month}"
            ]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    ytd_where_clauses.append(f"{column_name} = '{column_value}'")

            ytd_where_clause = " AND ".join(ytd_where_clauses)

            country_sector_ytd_query = f"""
                SELECT
                    country_name,
                    sector,
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE {ytd_where_clause}
                GROUP BY country_name, sector, year, month
                ORDER BY country_name, sector, year, month
            """
            df_country_sec_ytd = con.execute(country_sector_ytd_query).df()

            # Calculate cumulative emissions per country-sector per year
            df_country_sec_ytd['cumulative'] = df_country_sec_ytd.groupby(['country_name', 'sector', 'year'])['emissions_quantity'].cumsum()

            # Get the latest month's cumulative value for each country-sector for each year
            df_current_cs = df_country_sec_ytd[
                (df_country_sec_ytd['year'] == latest_year) & (df_country_sec_ytd['month'] == latest_month)
            ][['country_name', 'sector', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

            df_previous_cs = df_country_sec_ytd[
                (df_country_sec_ytd['year'] == latest_year - 1) & (df_country_sec_ytd['month'] == latest_month)
            ][['country_name', 'sector', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

            df_country_sector = df_current_cs.merge(df_previous_cs, on=['country_name', 'sector'], how='outer').fillna(0)
            df_country_sector['change'] = df_country_sector['current_ytd'] - df_country_sector['previous_ytd']
        else:
            # Query DuckDB directly for aggregated data instead of pandas groupby
            change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'

            # Build WHERE clause for region filter
            country_movers_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    country_movers_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    country_movers_where_clauses.append(f"{column_name} = '{column_value}'")

            country_movers_where_clause = " AND ".join(country_movers_where_clauses)

            country_sector_query = f"""
                SELECT
                    country_name,
                    sector,
                    SUM({change_column}) as change
                FROM '{country_subsector_stats_path}'
                WHERE {country_movers_where_clause}
                GROUP BY country_name, sector
            """
            df_country_sector = con.execute(country_sector_query).df()

        # Get top countries by increase and decrease
        country_totals = df_country_sector.groupby('country_name')['change'].sum().sort_values(ascending=False)

        # Get all countries for rangeslider
        all_increases = [loc for loc in country_totals.index if country_totals[loc] > 0]
        all_decreases = [loc for loc in country_totals.index if country_totals[loc] < 0]

        # Check if there are any countries with net decreases
        has_decreases = len(all_decreases) > 0

        # --- Top Increases Chart ---
        increases_to_show = all_increases if len(all_increases) > 0 else []

        # Filter data for increases
        df_top_increases = df_country_sector[df_country_sector['country_name'].isin(increases_to_show)]

        # Create vertical stacked bar chart
        fig_increases = go.Figure()

        # Sort countries by total change (highest to lowest)
        country_totals_inc = df_top_increases.groupby('country_name')['change'].sum().sort_values(ascending=False)
        countries_sorted_inc = country_totals_inc.index.tolist()

        # Order sectors by their total absolute contribution (largest first = at base)
        sector_totals = df_top_increases.groupby('sector')['change'].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
        sectors_sorted_by_contribution = sector_totals.index.tolist()

        # For each sector (in order of contribution), create a bar trace
        for sector in sectors_sorted_by_contribution:
            df_sec = df_top_increases[df_top_increases['sector'] == sector]

            # Create data aligned with countries_sorted_inc
            values = []
            for loc in countries_sorted_inc:
                val = df_sec[df_sec['country_name'] == loc]['change'].sum()
                values.append(val)

            fig_increases.add_trace(go.Bar(
                name=sector.replace('-', ' ').title(),
                x=countries_sorted_inc,
                y=values,
                marker_color=sector_color_map.get(sector, "#999999"),
                hovertemplate=f"<b>%{{x}}</b><br>{sector.replace('-', ' ').title()}: %{{y:,.0f}}<extra></extra>"
            ))

        fig_increases.update_layout(
            barmode='relative',  # Use 'relative' to properly stack positive/negative values
            yaxis_title='Emissions Change (tCO₂e)',
            height=350,
            showlegend=False,
            margin=dict(l=70, r=50, t=30, b=70),
            xaxis=dict(
                tickangle=-45,
                rangeslider=dict(visible=True),
                type="category"
            )
        )

        # Set initial range to show top 10 if more than 10
        if len(countries_sorted_inc) > 10:
            top10_range_inc = [
                countries_sorted_inc.index(countries_sorted_inc[0]) - 0.5,
                countries_sorted_inc.index(countries_sorted_inc[9]) + 0.5,
            ]
            fig_increases.update_xaxes(range=top10_range_inc, type="category")

        # Add horizontal line at y=0 (gray works in both light and dark mode)
        fig_increases.add_hline(y=0, line_width=3, line_color="#666666", opacity=0.9)

        # Add net change labels for each country (positioned above bars in red)
        for loc in countries_sorted_inc:
            net_change = country_totals_inc[loc]

            # Get the sector data for this country to find bar extents
            df_country_labels = df_top_increases[df_top_increases['country_name'] == loc]
            positive_extent = df_country_labels[df_country_labels['change'] > 0]['change'].sum()

            # Handle zero case specially
            if abs(net_change) < 0.5:  # Essentially zero
                arrow = ""
                color = "white"
                text = "0"
                y_pos = 0
                y_anchor = "bottom"
                yshift = 10
            else:
                arrow = "↑"
                color = "red"
                text = f"{arrow} {format_number_short(abs(net_change))}"

                # Position label at the furthest extent of bars
                # For increases chart, bars extend upward, so use positive extent
                y_pos = positive_extent
                y_anchor = "bottom"
                yshift = 10

            fig_increases.add_annotation(
                x=loc,
                y=y_pos,
                text=text,
                showarrow=False,
                font=dict(size=10, color=color, family="Arial Black"),
                yanchor=y_anchor,
                yshift=yshift
            )

        st.plotly_chart(fig_increases, use_container_width=True)

        # --- Top Decreases Chart (only show if there are decreases) ---
        if has_decreases:
            # Show all decreases
            decreases_to_show = all_decreases

            # Filter data for decreases
            df_top_decreases = df_country_sector[df_country_sector['country_name'].isin(decreases_to_show)]

            # Create vertical stacked bar chart
            fig_decreases = go.Figure()

            # Sort countries by total change (lowest to highest, showing most negative first)
            country_totals_dec = df_top_decreases.groupby('country_name')['change'].sum().sort_values()
            countries_sorted_dec = country_totals_dec.index.tolist()

            # Order sectors by their total absolute contribution (largest first = at base)
            sector_totals_dec = df_top_decreases.groupby('sector')['change'].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
            sectors_sorted_dec = sector_totals_dec.index.tolist()

            # For each sector (in order of contribution), create a bar trace
            for sector in sectors_sorted_dec:
                df_sec = df_top_decreases[df_top_decreases['sector'] == sector]

                # Create data aligned with countries_sorted_dec
                values = []
                for loc in countries_sorted_dec:
                    val = df_sec[df_sec['country_name'] == loc]['change'].sum()
                    values.append(val)

                fig_decreases.add_trace(go.Bar(
                    name=sector.replace('-', ' ').title(),
                    x=countries_sorted_dec,
                    y=values,
                    marker_color=sector_color_map.get(sector, "#999999"),
                    hovertemplate=f"<b>%{{x}}</b><br>{sector.replace('-', ' ').title()}: %{{y:,.0f}}<extra></extra>"
                ))

            fig_decreases.update_layout(
                barmode='relative',  # Use 'relative' to properly stack positive/negative values
                yaxis_title='Emissions Change (tCO₂e)',
                height=350,
                showlegend=False,
                margin=dict(l=70, r=50, t=10, b=90),
                xaxis=dict(
                    tickangle=-45,
                    rangeslider=dict(visible=True),
                    type="category"
                )
            )

            # Set initial range to show top 10 if more than 10
            if len(countries_sorted_dec) > 10:
                # For decreases, show the 10 most negative (at the beginning of the sorted list)
                top10_range_dec = [
                    countries_sorted_dec.index(countries_sorted_dec[0]) - 0.5,
                    countries_sorted_dec.index(countries_sorted_dec[9]) + 0.5,
                ]
                fig_decreases.update_xaxes(range=top10_range_dec, type="category")

            # Add horizontal line at y=0 (gray works in both light and dark mode)
            fig_decreases.add_hline(y=0, line_width=3, line_color="#666666", opacity=0.9)

            # Add net change labels for each country (positioned below bars in green)
            for loc in countries_sorted_dec:
                net_change = country_totals_dec[loc]

                # Get the sector data for this country to find bar extents
                df_country_labels_dec = df_top_decreases[df_top_decreases['country_name'] == loc]
                negative_extent_dec = df_country_labels_dec[df_country_labels_dec['change'] < 0]['change'].sum()

                # Handle zero case specially
                if abs(net_change) < 0.5:  # Essentially zero
                    arrow = ""
                    color = "white"
                    text = "0"
                    y_pos = 0
                    y_anchor = "top"
                    yshift = -10
                else:
                    arrow = "↓"
                    color = "green"
                    text = f"{arrow} {format_number_short(abs(net_change))}"

                    # Position label at the furthest extent of bars
                    # For decreases chart, bars extend downward, so use negative extent
                    y_pos = negative_extent_dec
                    y_anchor = "top"
                    yshift = -10

                fig_decreases.add_annotation(
                    x=loc,
                    y=y_pos,
                    text=text,
                    showarrow=False,
                    font=dict(size=10, color=color, family="Arial Black"),
                    yanchor=y_anchor,
                    yshift=yshift
                )

            st.plotly_chart(fig_decreases, use_container_width=True)
        else:
            # Show placeholder when no decreases exist
            st.markdown(
                """
                <div style='border: 1px solid #444; border-radius: 8px; height: 180px; display: flex; align-items: center; justify-content: center; background-color: #1e1e1e;'>
                    <p style='color: #888; font-size: 1.1em; text-align: center; padding: 20px;'>
                        No countries experienced net emission decreases<br>during the selected time period
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==================== Country Subsector Movers Table ====================
    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Country Subsector Movers")
    with st.expander("# **View Top 10 country-subsector movers by absolute change for each sector**", expanded=False):

        # Query for country-subsector rankings by sector
        change_col = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'

        if trend_view == "Year-to-Date":
            # Build WHERE clause with region filter
            ranking_ytd_where_clauses = [
                "gas = 'co2e_100yr'",
                "country_name IS NOT NULL",
                f"year IN ({latest_year - 1}, {latest_year})",
                f"month <= {latest_month}"
            ]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    ranking_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    ranking_ytd_where_clauses.append(f"{column_name} = '{column_value}'")

            ranking_ytd_where_clause = " AND ".join(ranking_ytd_where_clauses)

            # Query YTD data for country-subsector rankings
            ranking_ytd_query = f"""
                SELECT
                    country_name,
                    sector,
                    subsector,
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE {ranking_ytd_where_clause}
                GROUP BY country_name, sector, subsector, year, month
                ORDER BY country_name, sector, subsector, year, month
            """
            df_ranking_ytd = con.execute(ranking_ytd_query).df()

            # Calculate cumulative emissions per country-subsector per year
            df_ranking_ytd['cumulative'] = df_ranking_ytd.groupby(['country_name', 'sector', 'subsector', 'year'])['emissions_quantity'].cumsum()

            # Get current and previous year cumulative values
            df_current_rank = df_ranking_ytd[
                (df_ranking_ytd['year'] == latest_year) & (df_ranking_ytd['month'] == latest_month)
            ][['country_name', 'sector', 'subsector', 'cumulative']].rename(columns={'cumulative': 'current'})

            df_previous_rank = df_ranking_ytd[
                (df_ranking_ytd['year'] == latest_year - 1) & (df_ranking_ytd['month'] == latest_month)
            ][['country_name', 'sector', 'subsector', 'cumulative']].rename(columns={'cumulative': 'previous'})

            df_rankings = df_current_rank.merge(df_previous_rank, on=['country_name', 'sector', 'subsector'], how='outer').fillna(0)
            df_rankings['change'] = df_rankings['current'] - df_rankings['previous']
        else:
            # Build WHERE clause for region filter
            ranking_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

            if region_condition:
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    ranking_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    ranking_where_clauses.append(f"{column_name} = '{column_value}'")

            ranking_where_clause = " AND ".join(ranking_where_clauses)

            # Query country-subsector data with current and previous values
            if trend_view == "Month YoY":
                ranking_query = f"""
                    SELECT
                        country_name,
                        sector,
                        subsector,
                        SUM({emissions_column_latest}) as current,
                        SUM(month_yoy_change) as change
                    FROM '{country_subsector_stats_path}'
                    WHERE {ranking_where_clause}
                    GROUP BY country_name, sector, subsector
                """
                df_rankings = con.execute(ranking_query).df()
                df_rankings['previous'] = df_rankings['current'] - df_rankings['change']
            else:  # MoM
                ranking_query = f"""
                    SELECT
                        country_name,
                        sector,
                        subsector,
                        SUM({emissions_column_latest}) as current,
                        SUM({emissions_column_prev}) as previous,
                        SUM(mom_change) as change
                    FROM '{country_subsector_stats_path}'
                    WHERE {ranking_where_clause}
                    GROUP BY country_name, sector, subsector
                """
                df_rankings = con.execute(ranking_query).df()

        # Calculate percentage change
        df_rankings['percent_change'] = df_rankings.apply(
            lambda row: (row['change'] / row['previous'] * 100) if row['previous'] != 0 else 0,
            axis=1
        )

        # Add absolute change for ranking
        df_rankings['abs_change'] = df_rankings['change'].abs()

        # Get top 10 per sector
        df_rankings = df_rankings.sort_values(['sector', 'abs_change'], ascending=[True, False])
        df_top10_per_sector = df_rankings.groupby('sector').head(10).reset_index(drop=True)

        # Create ranking within each sector
        df_top10_per_sector['rank'] = df_top10_per_sector.groupby('sector').cumcount() + 1

        # Sort sectors by absolute change (matching Sector Movers chart)
        sector_totals = df_rankings.groupby('sector')['change'].sum()
        sector_abs_totals = sector_totals.abs().sort_values(ascending=False)
        sectors = sector_abs_totals.index.tolist()

        # Create HTML table with multi-line formatted cells
        html_rows = []

        for sector in sectors:
            sector_data = df_top10_per_sector[df_top10_per_sector['sector'] == sector]
            row_html = f"<tr><td class='sector-cell'>{sector.replace('-', ' ').title()}</td>"

            for rank in range(1, 11):
                rank_data = sector_data[sector_data['rank'] == rank]
                if not rank_data.empty:
                    country = rank_data['country_name'].iloc[0]
                    subsector = rank_data['subsector'].iloc[0]
                    change_val = rank_data['change'].iloc[0]
                    percent_val = rank_data['percent_change'].iloc[0]

                    # Format the change value
                    arrow = "↑" if change_val > 0 else "↓"
                    formatted_change = format_number_short(abs(change_val))

                    # Determine color class for change line
                    color_class = 'increase' if change_val > 0 else 'decrease'

                    # Create multi-line cell HTML with CSS classes
                    cell_html = f"""
                    <div style='line-height: 1.5;'>
                        <div class='country-text-cell'>{country}</div>
                        <div class='subsector-text-cell'>{subsector.replace('-', ' ')}</div>
                        <div class='change-text-cell change-{color_class}'>{arrow} {formatted_change} ({abs(percent_val):.1f}%)</div>
                    </div>
                    """
                else:
                    cell_html = ''

                row_html += f"<td class='data-cell'>{cell_html}</td>"

            row_html += "</tr>"
            html_rows.append(row_html)

        # Build complete HTML table with theme-aware CSS
        table_html = f"""
        <style>
            .movers-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }}

            .movers-table {{
                border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
            }}

            .movers-table th {{
                border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
                padding: 10px;
                font-weight: 600;
                background-color: var(--background-color);
            }}

            .movers-table .sector-cell {{
                font-weight: 600;
                border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
                padding: 10px;
                font-size: 0.9em;
                background-color: var(--secondary-background-color);
            }}

            .movers-table .data-cell {{
                border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
                padding: 10px;
                vertical-align: top;
            }}

            .country-text-cell {{
                font-size: 0.9em;
                font-weight: 700;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                color: var(--text-color);
            }}

            .subsector-text-cell {{
                font-size: 0.72em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                opacity: 0.6;
            }}

            .change-text-cell {{
                font-weight: 600;
                font-size: 0.78em;
                white-space: nowrap;
                margin: 2px 0;
            }}

            .change-text-cell.change-increase {{
                color: #d9534f !important;
            }}

            .change-text-cell.change-decrease {{
                color: #5cb85c !important;
            }}

            .movers-table th:first-child {{
                text-align: left;
                position: sticky;
                left: 0;
                z-index: 1;
            }}

            .movers-table th:not(:first-child) {{
                text-align: center;
            }}
        </style>
        <div style='overflow-x: auto;'>
            <table class='movers-table'>
                <thead>
                    <tr>
                        <th>Sector</th>
                        {''.join([f"<th>Rank {i}</th>" for i in range(1, 11)])}
                    </tr>
                </thead>
                <tbody>
                    {''.join(html_rows)}
                </tbody>
            </table>
        </div>
        """

        st.markdown(table_html, unsafe_allow_html=True)

    # ==================== Country Subsector Drilldown ====================
    # st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Country Subsector Drilldown")

    with st.expander("**Explore detailed emissions trends by country, sector, and subsector**", expanded=False):

        # Get list of countries filtered by region
        country_rows = con.execute(
            f"SELECT DISTINCT country_name, iso3_country FROM '{gadm_0_path}' WHERE country_name IS NOT NULL ORDER BY country_name"
        ).fetchall()

        country_map = {row[0]: row[1] for row in country_rows}
        all_countries = list(country_map.keys())

        # Filter countries based on selected region
        if selected_scope == 'Global':
            available_countries = all_countries
        elif region_condition:
            # Query to get countries in the selected region
            region_col = region_condition['column_name']
            region_val = region_condition['column_value']

            if isinstance(region_val, bool):
                region_filter_query = f"""
                    SELECT DISTINCT country_name
                    FROM '{gadm_0_path}'
                    WHERE {region_col} = {str(region_val).upper()}
                    AND country_name IS NOT NULL
                    ORDER BY country_name
                """
            else:
                region_filter_query = f"""
                    SELECT DISTINCT country_name
                    FROM '{gadm_0_path}'
                    WHERE {region_col} = '{region_val}'
                    AND country_name IS NOT NULL
                    ORDER BY country_name
                """

            region_countries = con.execute(region_filter_query).fetchall()
            available_countries = [row[0] for row in region_countries]
        else:
            available_countries = all_countries

        # Get all sectors
        all_sectors_query = f"""
            SELECT DISTINCT sector
            FROM '{country_subsector_stats_path}'
            WHERE sector IS NOT NULL
            ORDER BY sector
        """
        all_sectors = [row[0] for row in con.execute(all_sectors_query).fetchall()]

        # Create mapping from country name to ISO3
        country_to_iso3 = {country: country_map[country] for country in available_countries if country in country_map}

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Create three columns for dropdowns
        col_country, col_sector, col_subsector = st.columns(3)

        with col_country:
            selected_country = st.selectbox(
                "Country",
                ["All Countries"] + available_countries,
                key="deep_dive_country"
            )

        # Get ISO3 code for selected country
        selected_country_iso3 = country_to_iso3.get(selected_country, None) if selected_country != "All Countries" else None

        with col_sector:
            selected_sector_dd = st.selectbox(
                "Sector",
                ["All Sectors"] + all_sectors,
                key="deep_dive_sector"
            )

        # Get subsectors based on selected sector
        if selected_sector_dd and selected_sector_dd != "All Sectors":
            subsector_query = f"""
                SELECT DISTINCT subsector
                FROM '{country_subsector_stats_path}'
                WHERE sector = '{selected_sector_dd}'
                AND subsector IS NOT NULL
                ORDER BY subsector
            """
        else:
            subsector_query = f"""
                SELECT DISTINCT subsector
                FROM '{country_subsector_stats_path}'
                WHERE subsector IS NOT NULL
                ORDER BY subsector
            """

        all_subsectors = [row[0] for row in con.execute(subsector_query).fetchall()]

        with col_subsector:
            selected_subsector_dd = st.selectbox(
                "Subsector",
                ["All Subsectors"] + all_subsectors,
                key="deep_dive_subsector"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==================== Calculate metrics for cards ====================

        # Build WHERE clauses for the query
        dd_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]

        # Add region filter
        if region_condition and selected_country == "All Countries":
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                dd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                dd_where_clauses.append(f"{column_name} = '{column_value}'")
        elif selected_country != "All Countries":
            dd_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")

        # Add sector filter
        if selected_sector_dd != "All Sectors":
            dd_where_clauses.append(f"sector = '{selected_sector_dd}'")

        # Add subsector filter
        if selected_subsector_dd != "All Subsectors":
            dd_where_clauses.append(f"subsector = '{selected_subsector_dd}'")

        dd_where_clause = " AND ".join(dd_where_clauses)

        # Calculate emissions change based on trend_view
        if trend_view == "Month YoY":
            dd_emissions_query = f"""
                SELECT
                    SUM({emissions_column_latest}) as latest,
                    SUM(month_yoy_change) as change
                FROM '{country_subsector_stats_path}'
                WHERE {dd_where_clause}
            """
            df_dd_emissions = con.execute(dd_emissions_query).df()
            dd_latest = df_dd_emissions['latest'].iloc[0]
            dd_change = df_dd_emissions['change'].iloc[0]
            dd_previous = dd_latest - dd_change

        elif trend_view == "Year-to-Date":
            # YTD calculation
            dd_ytd_where_clauses = [
                "gas = 'co2e_100yr'",
                "country_name IS NOT NULL",
                f"year IN ({latest_year - 1}, {latest_year})",
                f"month <= {latest_month}"
            ]

            # Add region/country filter
            if region_condition and selected_country == "All Countries":
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    dd_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    dd_ytd_where_clauses.append(f"{column_name} = '{column_value}'")
            elif selected_country != "All Countries":
                dd_ytd_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")

            # Add sector filter
            if selected_sector_dd != "All Sectors":
                dd_ytd_where_clauses.append(f"sector = '{selected_sector_dd}'")

            # Add subsector filter
            if selected_subsector_dd != "All Subsectors":
                dd_ytd_where_clauses.append(f"subsector = '{selected_subsector_dd}'")

            dd_ytd_where_clause = " AND ".join(dd_ytd_where_clauses)

            dd_ytd_query = f"""
                SELECT
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE {dd_ytd_where_clause}
                GROUP BY year, month
                ORDER BY year, month
            """
            df_dd_ytd = con.execute(dd_ytd_query).df()
            df_dd_ytd['cumulative'] = df_dd_ytd.groupby('year')['emissions_quantity'].cumsum()

            current_ytd = df_dd_ytd[
                (df_dd_ytd['year'] == latest_year) & (df_dd_ytd['month'] == latest_month)
            ]['cumulative'].sum()

            previous_ytd = df_dd_ytd[
                (df_dd_ytd['year'] == latest_year - 1) & (df_dd_ytd['month'] == latest_month)
            ]['cumulative'].sum()

            dd_latest = current_ytd
            dd_previous = previous_ytd
            dd_change = dd_latest - dd_previous

        else:  # Month-over-Month
            dd_emissions_query = f"""
                SELECT
                    SUM({emissions_column_latest}) as latest,
                    SUM({emissions_column_prev}) as previous,
                    SUM(mom_change) as change
                FROM '{country_subsector_stats_path}'
                WHERE {dd_where_clause}
            """
            df_dd_emissions = con.execute(dd_emissions_query).df()
            dd_latest = df_dd_emissions['latest'].iloc[0]
            dd_previous = df_dd_emissions['previous'].iloc[0]
            dd_change = df_dd_emissions['change'].iloc[0]

        dd_percent_change = (dd_change / dd_previous * 100) if dd_previous != 0 else 0

        # Calculate activity and emissions factor changes (only if subsector is selected)
        if selected_subsector_dd != "All Subsectors":
            # Build WHERE clauses for asset query
            asset_where_clauses = ["gas = 'co2e_100yr'"]

            # Add region/country filter
            if region_condition and selected_country == "All Countries":
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    asset_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    asset_where_clauses.append(f"{column_name} = '{column_value}'")
            elif selected_country != "All Countries":
                asset_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")

            # Add sector filter
            if selected_sector_dd != "All Sectors":
                asset_where_clauses.append(f"sector = '{selected_sector_dd}'")

            # Add subsector filter
            asset_where_clauses.append(f"original_inventory_sector = '{selected_subsector_dd}'")

            # Exclude certain subsectors
            asset_where_clauses.append("""original_inventory_sector NOT IN (
                'forest-land-clearing', 'forest-land-degradation', 'forest-land-fires',
                'net-forest-land', 'net-shrubgrass', 'net-wetland', 'removals',
                'shrubgrass-fires', 'water-reservoirs', 'wetland-fires'
            )""")

            asset_where_clause = " AND ".join(asset_where_clauses)

            # Query asset-level data for activity and emissions factor
            if trend_view == "Month YoY":
                # Get current month and same month last year
                current_month_str = f"{latest_year}-{latest_month:02d}"
                prev_year_month_str = f"{latest_year - 1}-{latest_month:02d}"

                asset_query = f"""
                    SELECT
                        strftime(start_time, '%Y-%m') AS year_month,
                        SUM(activity) AS activity,
                        SUM(emissions_quantity) AS emissions_quantity
                    FROM '{asset_path}'
                    WHERE {asset_where_clause}
                        AND strftime(start_time, '%Y-%m') IN ('{current_month_str}', '{prev_year_month_str}')
                    GROUP BY year_month
                    ORDER BY year_month
                """
                df_asset = con.execute(asset_query).df()

                if len(df_asset) >= 2:
                    dd_activity_current = df_asset.iloc[-1]['activity']
                    dd_activity_previous = df_asset.iloc[0]['activity']
                    dd_emissions_current = df_asset.iloc[-1]['emissions_quantity']
                    dd_emissions_previous = df_asset.iloc[0]['emissions_quantity']
                else:
                    dd_activity_current = dd_activity_previous = 0
                    dd_emissions_current = dd_emissions_previous = 0

            elif trend_view == "Year-to-Date":
                # Get YTD for current and previous year
                ytd_asset_query = f"""
                    SELECT
                        strftime(start_time, '%Y') AS year,
                        strftime(start_time, '%m') AS month,
                        SUM(activity) AS activity,
                        SUM(emissions_quantity) AS emissions_quantity
                    FROM '{asset_path}'
                    WHERE {asset_where_clause}
                        AND strftime(start_time, '%Y') IN ('{latest_year}', '{latest_year - 1}')
                        AND CAST(strftime(start_time, '%m') AS INTEGER) <= {latest_month}
                    GROUP BY year, month
                    ORDER BY year, month
                """
                df_asset_ytd = con.execute(ytd_asset_query).df()

                if not df_asset_ytd.empty:
                    # Calculate cumulative for current and previous year
                    df_current_year = df_asset_ytd[df_asset_ytd['year'] == str(latest_year)]
                    df_prev_year = df_asset_ytd[df_asset_ytd['year'] == str(latest_year - 1)]

                    dd_activity_current = df_current_year['activity'].sum()
                    dd_activity_previous = df_prev_year['activity'].sum()
                    dd_emissions_current = df_current_year['emissions_quantity'].sum()
                    dd_emissions_previous = df_prev_year['emissions_quantity'].sum()
                else:
                    dd_activity_current = dd_activity_previous = 0
                    dd_emissions_current = dd_emissions_previous = 0

            else:  # Month-over-Month
                # Get current month and previous month
                if latest_month == 1:
                    prev_month = 12
                    prev_year = latest_year - 1
                else:
                    prev_month = latest_month - 1
                    prev_year = latest_year

                current_month_str = f"{latest_year}-{latest_month:02d}"
                prev_month_str = f"{prev_year}-{prev_month:02d}"

                asset_query = f"""
                    SELECT
                        strftime(start_time, '%Y-%m') AS year_month,
                        SUM(activity) AS activity,
                        SUM(emissions_quantity) AS emissions_quantity
                    FROM '{asset_path}'
                    WHERE {asset_where_clause}
                        AND strftime(start_time, '%Y-%m') IN ('{current_month_str}', '{prev_month_str}')
                    GROUP BY year_month
                    ORDER BY year_month
                """
                df_asset = con.execute(asset_query).df()

                if len(df_asset) >= 2:
                    dd_activity_previous = df_asset.iloc[0]['activity']
                    dd_activity_current = df_asset.iloc[-1]['activity']
                    dd_emissions_previous = df_asset.iloc[0]['emissions_quantity']
                    dd_emissions_current = df_asset.iloc[-1]['emissions_quantity']
                else:
                    dd_activity_current = dd_activity_previous = 0
                    dd_emissions_current = dd_emissions_previous = 0

            # Calculate activity change
            dd_activity_change = dd_activity_current - dd_activity_previous
            dd_activity_percent_change = (dd_activity_change / dd_activity_previous * 100) if dd_activity_previous != 0 else 0

            # Calculate emissions factor change (percentage only)
            # For YTD, use average emissions factor
            if trend_view == "Year-to-Date":
                dd_ef_current = (dd_emissions_current / dd_activity_current) if dd_activity_current != 0 else 0
                dd_ef_previous = (dd_emissions_previous / dd_activity_previous) if dd_activity_previous != 0 else 0
            else:
                dd_ef_current = (dd_emissions_current / dd_activity_current) if dd_activity_current != 0 else 0
                dd_ef_previous = (dd_emissions_previous / dd_activity_previous) if dd_activity_previous != 0 else 0

            dd_ef_percent_change = ((dd_ef_current - dd_ef_previous) / dd_ef_previous * 100) if dd_ef_previous != 0 else 0

        # ==================== Display Cards ====================

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(
                f"""
                <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Selected Region</div>
                    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                            <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: var(--text-color); margin-bottom: 12px;">
                                {selected_scope}
                            </div>
                            <div style="font-size: 0.7em; text-align: center; color: #888; visibility: hidden;">
                                tCO₂e
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Change View</div>
                    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                            <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: var(--text-color); margin-bottom: 12px;">
                                {trend_view}
                            </div>
                            <div style="font-size: 0.7em; text-align: center; color: #888; visibility: hidden;">
                                tCO₂e
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            # Emissions change card
            dd_arrow = "↑" if dd_change > 0 else "↓"
            dd_color = "red" if dd_change > 0 else "green"

            st.markdown(
                f"""
                <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                    <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Emissions Change</div>
                    <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                            <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: {dd_color}; margin-bottom: 12px;">
                                {dd_arrow} {format_number_short(abs(dd_change))} <span style="color: #888;">(</span><span style="color: {dd_color};">{abs(dd_percent_change):.1f}%</span><span style="color: #888;">)</span>
                            </div>
                            <div style="font-size: 0.7em; text-align: center; color: #888;">
                                tCO₂e
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            # Activity change card (only if subsector selected)
            if selected_subsector_dd != "All Subsectors":
                dd_activity_arrow = "↑" if dd_activity_change > 0 else "↓"
                dd_activity_color = "red" if dd_activity_change > 0 else "green"

                st.markdown(
                    f"""
                    <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                        <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Activity Change</div>
                        <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: {dd_activity_color}; margin-bottom: 12px;">
                                    {dd_activity_arrow} {abs(dd_activity_percent_change):.1f}%
                                </div>
                                <div style="font-size: 0.7em; text-align: center; color: #888;">
                                    {format_number_short(abs(dd_activity_change))} units
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                        <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Activity Change</div>
                        <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <div style="font-size: 0.8em; text-align: center; color: #888; padding: 0 10px; margin-bottom: 12px;">
                                    Select a subsector to view activity data
                                </div>
                                <div style="font-size: 0.7em; text-align: center; color: #888; visibility: hidden;">
                                    units
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col5:
            # Emissions Factor change card (only if subsector selected)
            if selected_subsector_dd != "All Subsectors":
                dd_ef_arrow = "↑" if dd_ef_percent_change > 0 else "↓"
                dd_ef_color = "red" if dd_ef_percent_change > 0 else "green"

                st.markdown(
                    f"""
                    <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                        <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Emissions Factor Change</div>
                        <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <div style="font-size: 1.2em; font-weight: bold; text-align: center; color: {dd_ef_color}; margin-bottom: 12px;">
                                    {dd_ef_arrow} {abs(dd_ef_percent_change):.1f}%
                                </div>
                                <div style="font-size: 0.7em; text-align: center; color: #888;">
                                    tCO₂e per unit
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="border: 1px solid #999; border-radius: 10px; padding: 16px; height: 150px; display: flex; flex-direction: column;">
                        <div style="font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">Emissions Factor Change</div>
                        <div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;">
                            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <div style="font-size: 0.8em; text-align: center; color: #888; padding: 0 10px; margin-bottom: 12px;">
                                    Select a subsector to view emissions factor data
                                </div>
                                <div style="font-size: 0.7em; text-align: center; color: #888; visibility: hidden;">
                                    tCO₂e per unit
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )



        st.markdown("<br>", unsafe_allow_html=True)

        # ==================== Time Series Graphs ====================

        # Query for time series data (last 3 years)
        earliest_year_ts = latest_year - 3

        # Build WHERE clause for country/sector totals time series
        ts_where_clauses = [
            "gas = 'co2e_100yr'",
            "country_name IS NOT NULL",
            f"year >= {earliest_year_ts}"
        ]

        # Add region/country filter
        if region_condition and selected_country == "All Countries":
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                ts_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                ts_where_clauses.append(f"{column_name} = '{column_value}'")
        elif selected_country != "All Countries":
            ts_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")

        # Add sector filter
        if selected_sector_dd != "All Sectors":
            ts_where_clauses.append(f"sector = '{selected_sector_dd}'")

        # Add subsector filter
        if selected_subsector_dd != "All Subsectors":
            ts_where_clauses.append(f"subsector = '{selected_subsector_dd}'")

        ts_where_clause = " AND ".join(ts_where_clauses)

        # Query emissions time series
        ts_emissions_query = f"""
            SELECT
                MAKE_DATE(year, month, 1) AS year_month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {ts_where_clause}
            GROUP BY year_month
            ORDER BY year_month
        """
        df_ts_emissions = con.execute(ts_emissions_query).df()

        if not df_ts_emissions.empty:
            df_ts_emissions['year_month'] = pd.to_datetime(df_ts_emissions['year_month'])

        # Query asset-level data for activity and emissions factor (only if subsector selected)
        show_activity_and_ef = selected_subsector_dd != "All Subsectors"
        df_ts_asset = pd.DataFrame()

        if show_activity_and_ef:
            # Build WHERE clause for asset time series
            ts_asset_where_clauses = ["gas = 'co2e_100yr'"]

            # Add region/country filter
            if region_condition and selected_country == "All Countries":
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    ts_asset_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    ts_asset_where_clauses.append(f"{column_name} = '{column_value}'")
            elif selected_country != "All Countries":
                ts_asset_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")

            # Add sector filter
            if selected_sector_dd != "All Sectors":
                ts_asset_where_clauses.append(f"sector = '{selected_sector_dd}'")

            # Add subsector filter
            ts_asset_where_clauses.append(f"original_inventory_sector = '{selected_subsector_dd}'")

            # Exclude certain subsectors
            ts_asset_where_clauses.append("""original_inventory_sector NOT IN (
                'forest-land-clearing', 'forest-land-degradation', 'forest-land-fires',
                'net-forest-land', 'net-shrubgrass', 'net-wetland', 'removals',
                'shrubgrass-fires', 'water-reservoirs', 'wetland-fires'
            )""")

            ts_asset_where_clause = " AND ".join(ts_asset_where_clauses)

            # Query asset-level time series
            ts_asset_query = f"""
                SELECT
                    strftime(start_time, '%Y-%m') AS year_month,
                    SUM(activity) AS activity,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{asset_path}'
                WHERE {ts_asset_where_clause}
                GROUP BY year_month
                ORDER BY year_month
            """
            df_ts_asset = con.execute(ts_asset_query).df()

            if not df_ts_asset.empty:
                df_ts_asset['year_month'] = pd.to_datetime(df_ts_asset['year_month'])
                df_ts_asset['mean_emissions_factor'] = df_ts_asset['emissions_quantity'] / df_ts_asset['activity']

                # Filter to last 3 years
                cutoff_date = pd.Timestamp(year=earliest_year_ts, month=1, day=1)
                df_ts_asset = df_ts_asset[df_ts_asset['year_month'] >= cutoff_date]

        # Create time series subplot
        num_rows = 3 if show_activity_and_ef else 1
        subplot_titles = ["Emissions Over Time"]
        if show_activity_and_ef:
            subplot_titles += ["Activity Over Time", "Emission Factor Over Time"]

        fig_ts = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        # Row 1 — Emissions
        if not df_ts_emissions.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts_emissions['year_month'],
                    y=df_ts_emissions['emissions_quantity'],
                    mode='lines+markers',
                    name='Total Emissions',
                    line=dict(color='#E9967A')
                ),
                row=1, col=1
            )

        # Row 2 — Activity
        if show_activity_and_ef and not df_ts_asset.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts_asset['year_month'],
                    y=df_ts_asset['activity'],
                    mode='lines+markers',
                    name='Activity',
                    line=dict(color='#1f77b4')
                ),
                row=2, col=1
            )

        # Row 3 — Emissions Factor
        if show_activity_and_ef and not df_ts_asset.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts_asset['year_month'],
                    y=df_ts_asset['mean_emissions_factor'],
                    mode='lines+markers',
                    name='Emission Factor',
                    line=dict(color='#2ca02c')
                ),
                row=3, col=1
            )

        # Add quarterly vertical lines
        if not df_ts_emissions.empty:
            min_date = df_ts_emissions['year_month'].min()
            max_date = df_ts_emissions['year_month'].max()
            quarter_starts = pd.date_range(
                start=min_date.to_period("Q").start_time,
                end=max_date.to_period("Q").end_time + pd.offsets.QuarterBegin(1),
                freq='QS'
            )

            for q_start in quarter_starts:
                fig_ts.add_vline(
                    x=q_start,
                    line_width=1,
                    line_dash='dash',
                    line_color='gray'
                )

                fig_ts.add_annotation(
                    x=q_start,
                    y=1.01,
                    xref="x",
                    yref="paper",
                    text=f"Q{((q_start.month - 1) // 3 + 1)} {q_start.year}",
                    showarrow=False,
                    font=dict(size=9),
                    align="center"
                )

        # Update y-axis labels
        fig_ts.update_yaxes(title_text="Emissions (tCO₂e)", row=1, col=1)
        if show_activity_and_ef:
            fig_ts.update_yaxes(title_text="Activity", row=2, col=1)
            fig_ts.update_yaxes(title_text="Emission Factor (tCO₂e/unit)", row=3, col=1)

        # Layout adjustments
        fig_ts.update_layout(
            height=900 if show_activity_and_ef else 400,
            showlegend=True,
            margin=dict(t=80, b=40)
        )

        st.plotly_chart(fig_ts, use_container_width=True)

    con.close()
