import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.utils import format_number_short
from config import CONFIG


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

    con = duckdb.connect()

    # st.markdown("<br>", unsafe_allow_html=True)

    # ==================== Global Emissions Summary Section ====================
    #st.markdown("### Global Emissions Summary")
    #st.markdown("<br>", unsafe_allow_html=True)

    # Add title and help tooltip for the toggle
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <span style="font-size: 0.95em; font-weight: 600;">Change View</span>
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
    df_stats_global = con.execute(f"""
        SELECT *
        FROM '{country_subsector_stats_path}'
        WHERE gas = 'co2e_100yr'
        AND country_name IS NOT NULL
    """).df()

    emissions_columns = [col for col in df_stats_global.columns if col.startswith("emissions_quantity_")]
    emissions_columns_sorted = sorted(emissions_columns, reverse=True)
    emissions_column_latest = emissions_columns_sorted[0]
    emissions_column_prev = emissions_columns_sorted[1]

    # Extract year and month from column names (format: emissions_quantity_YYYYMM)
    latest_year = int(emissions_column_latest[-6:-2])
    latest_month = int(emissions_column_latest[-2:])

    # Calculate global totals based on selected view
    if trend_view == "Month YoY":
        # Compare current month to same month last year
        global_latest = df_stats_global[emissions_column_latest].sum()
        global_yoy_change = df_stats_global['month_yoy_change'].sum()
        global_previous = global_latest - global_yoy_change

        absolute_change = global_yoy_change
        percent_change = (absolute_change / global_previous * 100) if global_previous != 0 else 0

    elif trend_view == "Year-to-Date":
        # Year-to-date comparison - use cumulative approach like original tab
        # Define column ranges for later use
        ytd_columns_current = [col for col in emissions_columns_sorted
                               if int(col[-6:-2]) == latest_year and int(col[-2:]) <= latest_month]
        ytd_columns_previous = [col for col in emissions_columns_sorted
                                if int(col[-6:-2]) == latest_year - 1 and int(col[-2:]) <= latest_month]

        # Query monthly data and calculate cumulative sums
        ytd_query = f"""
            SELECT
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name IS NOT NULL
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
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
        # Month over month
        global_latest = df_stats_global[emissions_column_latest].sum()
        global_previous = df_stats_global[emissions_column_prev].sum()

        absolute_change = df_stats_global['mom_change'].sum()
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
        # Query country-level YTD data from totals file (same source as main YTD calc)
        country_ytd_query = f"""
            SELECT
                country_name,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name IS NOT NULL
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
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
        df_country_totals = df_stats_global.groupby('country_name').agg({
            emissions_column_latest: 'sum',
            emissions_column_prev: 'sum',
            'mom_change': 'sum',
            'month_yoy_change': 'sum'
        }).reset_index()

        if trend_view == "Month YoY":
            df_country_totals['change'] = df_country_totals['month_yoy_change']
        else:  # MoM
            df_country_totals['change'] = df_country_totals['mom_change']

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
        df_country_latest = df_stats_global[df_stats_global['country_name'] == largest_decrease_country]
        current_total = df_country_latest[emissions_column_latest].sum()
        if trend_view == "Month YoY":
            previous_total = current_total - df_country_latest['month_yoy_change'].sum()
        else:  # MoM
            previous_total = df_country_latest[emissions_column_prev].sum()

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
        df_country_subsector = df_stats_global[df_stats_global['country_name'] == largest_decrease_country].copy()
        if trend_view == "Month YoY":
            df_country_subsector['change'] = df_country_subsector['month_yoy_change']
        else:  # MoM
            df_country_subsector['change'] = df_country_subsector['mom_change']

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
        df_country_latest_inc = df_stats_global[df_stats_global['country_name'] == largest_increase_country]
        current_total_inc = df_country_latest_inc[emissions_column_latest].sum()
        if trend_view == "Month YoY":
            previous_total_inc = current_total_inc - df_country_latest_inc['month_yoy_change'].sum()
        else:  # MoM
            previous_total_inc = df_country_latest_inc[emissions_column_prev].sum()

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
        df_country_subsector_inc = df_stats_global[df_stats_global['country_name'] == largest_increase_country].copy()
        if trend_view == "Month YoY":
            df_country_subsector_inc['change'] = df_country_subsector_inc['month_yoy_change']
        else:  # MoM
            df_country_subsector_inc['change'] = df_country_subsector_inc['mom_change']

        driving_subsector_increase = df_country_subsector_inc.loc[df_country_subsector_inc['change'].idxmax()]
        subsector_increase_name = driving_subsector_increase['subsector']

    # Card 4: Biggest Sector Move
    if trend_view == "Year-to-Date":
        # Query sector-level YTD data using cumulative approach
        sector_ytd_query = f"""
            SELECT
                sector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name IS NOT NULL
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
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
        df_sector_totals = df_stats_global.groupby('sector').agg({
            emissions_column_latest: 'sum',
            emissions_column_prev: 'sum',
            'mom_change': 'sum',
            'month_yoy_change': 'sum'
        }).reset_index()

        if trend_view == "Month YoY":
            df_sector_totals['change'] = df_sector_totals['month_yoy_change']
            df_sector_totals['current'] = df_sector_totals[emissions_column_latest]
            df_sector_totals['previous'] = df_sector_totals[emissions_column_latest] - df_sector_totals['month_yoy_change']
        else:  # MoM
            df_sector_totals['change'] = df_sector_totals['mom_change']
            df_sector_totals['current'] = df_sector_totals[emissions_column_latest]
            df_sector_totals['previous'] = df_sector_totals[emissions_column_prev]

    df_sector_totals['abs_change'] = df_sector_totals['change'].abs()
    biggest_sector = df_sector_totals.loc[df_sector_totals['abs_change'].idxmax()]
    biggest_sector_name = biggest_sector['sector']
    biggest_sector_value = biggest_sector['change']
    biggest_sector_previous = biggest_sector['previous']
    biggest_sector_percent = (biggest_sector_value / biggest_sector_previous * 100) if biggest_sector_previous != 0 else 0

    # Card 5: Biggest Subsector Move
    if trend_view == "Year-to-Date":
        # Query subsector-level YTD data using cumulative approach
        subsector_ytd_query = f"""
            SELECT
                subsector,
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE gas = 'co2e_100yr'
                AND country_name IS NOT NULL
                AND year IN ({latest_year - 1}, {latest_year})
                AND month <= {latest_month}
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
        df_subsector_totals = df_stats_global.groupby('subsector').agg({
            emissions_column_latest: 'sum',
            emissions_column_prev: 'sum',
            'mom_change': 'sum',
            'month_yoy_change': 'sum'
        }).reset_index()

        if trend_view == "Month YoY":
            df_subsector_totals['change'] = df_subsector_totals['month_yoy_change']
            df_subsector_totals['current'] = df_subsector_totals[emissions_column_latest]
            df_subsector_totals['previous'] = df_subsector_totals[emissions_column_latest] - df_subsector_totals['month_yoy_change']
        else:  # MoM
            df_subsector_totals['change'] = df_subsector_totals['mom_change']
            df_subsector_totals['current'] = df_subsector_totals[emissions_column_latest]
            df_subsector_totals['previous'] = df_subsector_totals[emissions_column_prev]

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
                    <div>{card1_current_label}: <span style="font-weight: 600; color: #ddd;">{card1_current_total}</span></div>
                    <div>{card1_previous_label}: <span style="font-weight: 600; color: #ddd;">{card1_previous_total}</span></div>
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
        st.markdown("### Sector Movers")

        # Prepare sector movers data with top 3 subsectors
        if trend_view == "Year-to-Date":
            # Query sector-subsector YTD data using cumulative approach
            sector_subsector_ytd_query = f"""
                SELECT
                    sector,
                    subsector,
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE gas = 'co2e_100yr'
                    AND country_name IS NOT NULL
                    AND year IN ({latest_year - 1}, {latest_year})
                    AND month <= {latest_month}
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
            df_sector_movers = df_stats_global.groupby(['sector', 'subsector']).agg({
                'mom_change': 'sum',
                'month_yoy_change': 'sum'
            }).reset_index()

            if trend_view == "Month YoY":
                df_sector_movers['change'] = df_sector_movers['month_yoy_change']
            else:  # MoM
                df_sector_movers['change'] = df_sector_movers['mom_change']

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

        # Get unique sectors and sort by total change (highest to lowest)
        sector_totals = df_sector_viz.groupby('sector')['change'].sum().sort_values(ascending=False)
        sectors_sorted = sector_totals.index.tolist()

        # For each sector, add traces for its subsectors (largest first = at base)
        for sector in sectors_sorted:
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
            height=550,
            showlegend=False,
            margin=dict(l=150, r=80, t=30, b=50),
            yaxis=dict(
                categoryorder='array',
                categoryarray=sectors_sorted  # Explicitly order Y-axis from top to bottom
            )
        )

        # Add thick vertical line at x=0
        fig_sector_movers.add_vline(x=0, line_width=3, line_color="white", opacity=0.8)

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
        # Title and slider on same row
        st.markdown("### Country Sector Movers")
        num_countries = 10

        # Get country-level data broken down by sector
        if trend_view == "Year-to-Date":
            # Query country-sector YTD data using cumulative approach
            country_sector_ytd_query = f"""
                SELECT
                    country_name,
                    sector,
                    year,
                    month,
                    SUM(emissions_quantity) AS emissions_quantity
                FROM '{country_subsector_totals_path}'
                WHERE gas = 'co2e_100yr'
                    AND country_name IS NOT NULL
                    AND year IN ({latest_year - 1}, {latest_year})
                    AND month <= {latest_month}
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
            df_country_sector = df_stats_global.groupby(['country_name', 'sector']).agg({
                'mom_change': 'sum',
                'month_yoy_change': 'sum'
            }).reset_index()

            if trend_view == "Month YoY":
                df_country_sector['change'] = df_country_sector['month_yoy_change']
            else:  # MoM
                df_country_sector['change'] = df_country_sector['mom_change']

        # Get top countries by increase and decrease
        country_totals = df_country_sector.groupby('country_name')['change'].sum().sort_values(ascending=False)

        top_increases = country_totals.head(num_countries).index.tolist()
        top_decreases = country_totals.tail(num_countries).index.tolist()

        # Check if there are any countries with net decreases
        has_decreases = any(country_totals < 0)

        # --- Top Increases Chart ---
        # st.markdown(f"<p style='font-size: 0.9em; color: #888; margin-top: 0px;'>Top {num_countries} Country Increases</p>", unsafe_allow_html=True)

        # Add zoom slider for increases chart
        # if num_countries > 5:
        #     zoom_start_inc = st.slider(
        #         "Scroll through increases",
        #         min_value=0,
        #         max_value=max(0, num_countries - 5),
        #         value=0,
        #         step=1,
        #         key="increases_zoom_slider"
        #     )
        #     zoom_end_inc = zoom_start_inc + 5
        #     increases_to_show = top_increases[zoom_start_inc:zoom_end_inc]
        # else:
        increases_to_show = top_increases

        # Filter data for top increases
        df_top_increases = df_country_sector[df_country_sector['country_name'].isin(increases_to_show)]

        # Create vertical stacked bar chart
        fig_increases = go.Figure()

        # Sort countries by total change (highest to lowest)
        country_totals_inc = df_top_increases.groupby('country_name')['change'].sum().sort_values(ascending=False)
        countries_sorted_inc = country_totals_inc.index.tolist()

        # Order sectors by their total absolute contribution (largest first = at base)
        all_sectors = df_country_sector['sector'].unique()
        sector_totals = df_top_increases.groupby('sector')['change'].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
        sectors_sorted_by_contribution = sector_totals.index.tolist()

        # For each sector (in order of contribution), create a bar trace
        for sector in sectors_sorted_by_contribution:
            df_sec = df_top_increases[df_top_increases['sector'] == sector]

            # Create data aligned with countries_sorted_inc
            values = []
            for country in countries_sorted_inc:
                val = df_sec[df_sec['country_name'] == country]['change'].sum()
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
            height=280,
            showlegend=False,
            margin=dict(l=50, r=50, t=30, b=70),
            xaxis=dict(tickangle=-45)
        )

        # Add horizontal line at y=0
        fig_increases.add_hline(y=0, line_width=3, line_color="white", opacity=0.8)

        # Add net change labels for each country (positioned above bars in red)
        for country in countries_sorted_inc:
            net_change = country_totals_inc[country]

            # Get the sector data for this country to find bar extents
            df_country_labels = df_top_increases[df_top_increases['country_name'] == country]
            positive_extent = df_country_labels[df_country_labels['change'] > 0]['change'].sum()
            negative_extent = df_country_labels[df_country_labels['change'] < 0]['change'].sum()

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
                x=country,
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
            # st.markdown(f"<p style='font-size: 0.9em; color: #888; margin-top: -10px;'>Top {num_countries} Country Decreases</p>", unsafe_allow_html=True)

            # Add zoom slider for decreases chart
            # if num_countries > 5:
            #     zoom_start_dec = st.slider(
            #         "Scroll through decreases",
            #         min_value=0,
            #         max_value=max(0, num_countries - 5),
            #         value=0,
            #         step=1,
            #         key="decreases_zoom_slider"
            #     )
            #     zoom_end_dec = zoom_start_dec + 5
            #     decreases_to_show = top_decreases[zoom_start_dec:zoom_end_dec]
            # else:
            decreases_to_show = top_decreases

            # Filter data for top decreases
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
                for country in countries_sorted_dec:
                    val = df_sec[df_sec['country_name'] == country]['change'].sum()
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
                height=280,
                showlegend=False,
                margin=dict(l=50, r=50, t=10, b=90),
                xaxis=dict(tickangle=-45)
            )

            # Add horizontal line at y=0
            fig_decreases.add_hline(y=0, line_width=3, line_color="white", opacity=0.8)

            # Add net change labels for each country (positioned below bars in green)
            for country in countries_sorted_dec:
                net_change = country_totals_dec[country]

                # Get the sector data for this country to find bar extents
                df_country_labels_dec = df_top_decreases[df_top_decreases['country_name'] == country]
                positive_extent_dec = df_country_labels_dec[df_country_labels_dec['change'] > 0]['change'].sum()
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
                    x=country,
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
            st.markdown(f"<p style='font-size: 0.9em; color: #888; margin-top: 5px;'>Top {num_countries} Country Decreases</p>", unsafe_allow_html=True)
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

    con.close()
