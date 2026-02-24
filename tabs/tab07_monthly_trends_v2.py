import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from plotly.subplots import make_subplots
from utils.utils import format_number_short, map_region_condition, get_iso3_flag_table
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

    flag_df = get_iso3_flag_table()

    con.register("country_flags", flag_df)

    max_date = con.execute(f"""SELECT MAX(MAKE_DATE(year, month, 1)) AS max_date
            FROM '{country_subsector_totals_path}'
            WHERE country_name IS NOT NULL
    """).fetchone()[0]

    # st.markdown("<br>", unsafe_allow_html=True)

    # st.markdown("""
    #     <style>
    #     /* Find the anchor, then style the first selectbox that follows it */
    #     #region_anchor + div [data-testid="stSelectbox"] label {
    #         margin-bottom: -10px !important;   /* moves label closer to box */
    #         padding-bottom: 0 !important;
    #         line-height: 1.0 !important;
    #     }

    #     #region_anchor + div [data-testid="stSelectbox"] div[data-baseweb="select"] {
    #         margin-top: -30px !important;      /* pulls the dropdown up */
    #     }

    #     #region_anchor + div [data-testid="stSelectbox"] {
    #         max-width: 280px;
    #         margin-left: auto;
    #         margin-right: auto;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)

    # Create columns for Change View toggle and Region dropdown
    col_view, col_region, col_month = st.columns([3, 1.6, 3])

    with col_view:
        trend_view = st.segmented_control(
            label="Change Time Period",
            options=["Month YoY", "Year-to-Date YoY", "Month-over-Month"],
            default="Month YoY",
        )

    with col_region:
        # Pull the entire region block up
        # st.markdown("<div style='margin-top:-300px'></div>", unsafe_allow_html=True)

        # Streamlit-like label (left aligned, subtle)
        # st.markdown(
        #     "<div style='font-size:0.86rem; color:rgba(49,51,63,0.6); margin-bottom:0.25rem;'>Region</div>",
        #     unsafe_allow_html=True,
        # )
        # Custom label styled like Streamlit
        # st.markdown(
        #     "<div style='font-size:0.86rem; color:rgba(49,51,63,0.6); margin-bottom:0.25rem;'>Region</div>",
        #     unsafe_allow_html=True,
        # )

        selected_scope = st.selectbox(
            label="Region",
            options=region_options,
            key="scope_selector",
            # label_visibility="collapsed",
        )


        region_condition = map_region_condition(selected_scope, {})

    with col_month:
        if max_date:
            formatted_date = pd.to_datetime(max_date).strftime("%B %Y")

            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; margin-top: 4px;">
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem; color:#6b7280;">Latest Month</div>
                        <div style="font-size:1.0rem; font-weight:600;">{formatted_date}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
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

    elif trend_view == "Year-to-Date YoY":
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
    elif trend_view == "Year-to-Date YoY":
        card1_label = "Year-to-Date YoY Change"
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
    if trend_view == "Year-to-Date YoY":
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
                and country_name <> 'Unknown'
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
                and country_name <> 'Unknown'
            GROUP BY country_name
        """
        df_country_totals = con.execute(country_totals_query).df()

    # Card 2: Largest Country Decrease
    largest_decrease = df_country_totals.loc[df_country_totals['change'].idxmin()]
    largest_decrease_country = largest_decrease['country_name']
    largest_decrease_value = largest_decrease['change']

    # Calculate percent change for largest decrease country
    if trend_view == "Year-to-Date YoY":
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
    if trend_view == "Year-to-Date YoY":
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
                and subsector <> 'non-broadcasting-vessels'
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
                and subsector <> 'non-broadcasting-vessels'
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
    if trend_view == "Year-to-Date YoY":
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
    if trend_view == "Year-to-Date YoY":
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
    if trend_view == "Year-to-Date YoY":
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
    if trend_view == "Year-to-Date YoY":
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
                and subsector <> 'non-broadcasting-vessels'
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
                and subsector <> 'non-broadcasting-vessels'
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

        # Add Sector Movers breakout toggle with label to the left
        label_col, toggle_col = st.columns([0.8, 3.2], gap="small")
        with label_col:
            st.markdown("<div style='padding-top: 8px; font-size: 0.9em; font-weight: 500;'>Breakdown Sector by:</div>", unsafe_allow_html=True)
        with toggle_col:
            sector_breakout = st.segmented_control(
                "Sector Movers Breakout",
                options=["Total Net Change", "Subsector", "Global North/South"],
                default="Total Net Change",
                key="sector_movers_breakout",
                label_visibility="collapsed"
            )

        # Prepare sector movers data with top 3 subsectors
        if trend_view == "Year-to-Date YoY":
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

        # Apply breakout transformations based on selected view
        if sector_breakout == "Total Net Change":
            # Aggregate to sector level only (no subsector breakout)
            df_sector_viz = df_sector_movers.groupby('sector')['change'].sum().reset_index()
            df_sector_viz['subsector'] = 'Net Change'  # Placeholder for compatibility
        elif sector_breakout == "Global North/South":
            # Query with developed_un to classify by Global North/South
            # Re-query with global classification
            if trend_view == "Year-to-Date YoY":
                global_ytd_query = f"""
                    SELECT
                        sector,
                        CASE WHEN developed_un = TRUE THEN 'Global North' ELSE 'Global South' END AS global_classification,
                        year,
                        month,
                        SUM(emissions_quantity) AS emissions_quantity
                    FROM '{country_subsector_totals_path}'
                    WHERE {sector_ytd_where_clause}
                    GROUP BY sector, global_classification, year, month
                    ORDER BY sector, global_classification, year, month
                """
                df_global_ytd = con.execute(global_ytd_query).df()
                df_global_ytd['cumulative'] = df_global_ytd.groupby(['sector', 'global_classification', 'year'])['emissions_quantity'].cumsum()

                df_current_global = df_global_ytd[
                    (df_global_ytd['year'] == latest_year) & (df_global_ytd['month'] == latest_month)
                ][['sector', 'global_classification', 'cumulative']].rename(columns={'cumulative': 'current_ytd'})

                df_previous_global = df_global_ytd[
                    (df_global_ytd['year'] == latest_year - 1) & (df_global_ytd['month'] == latest_month)
                ][['sector', 'global_classification', 'cumulative']].rename(columns={'cumulative': 'previous_ytd'})

                df_sector_viz = df_current_global.merge(df_previous_global, on=['sector', 'global_classification'], how='outer').fillna(0)
                df_sector_viz['change'] = df_sector_viz['current_ytd'] - df_sector_viz['previous_ytd']
                df_sector_viz = df_sector_viz.rename(columns={'global_classification': 'subsector'})  # Rename for compatibility
            else:
                change_column = 'month_yoy_change' if trend_view == "Month YoY" else 'mom_change'
                global_query = f"""
                    SELECT
                        sector,
                        CASE WHEN developed_un = TRUE THEN 'Global North' ELSE 'Global South' END AS global_classification,
                        SUM({change_column}) as change
                    FROM '{country_subsector_stats_path}'
                    WHERE {movers_where_clause}
                    GROUP BY sector, global_classification
                """
                df_sector_viz = con.execute(global_query).df()
                df_sector_viz = df_sector_viz.rename(columns={'global_classification': 'subsector'})  # Rename for compatibility

        # Create horizontal stacked bar chart for sectors
        fig_sector_movers = go.Figure()

        # Get unique sectors and sort by absolute change (highest to lowest)
        sector_totals = df_sector_viz.groupby('sector')['change'].sum()
        sector_abs_totals = sector_totals.abs().sort_values(ascending=False)
        sectors_sorted = sector_abs_totals.index.tolist()

        # Reverse for Plotly (which displays bottom to top for horizontal bars)
        sectors_sorted_reversed = sectors_sorted[::-1]

        # For each sector, add traces based on breakout view
        if sector_breakout == "Total Net Change":
            # Simple bars with sector colors, no breakout
            for sector in sectors_sorted_reversed:
                df_sec = df_sector_viz[df_sector_viz['sector'] == sector]
                if not df_sec.empty:
                    value = df_sec['change'].iloc[0]
                    base_color = sector_color_map.get(sector, "#999999")

                    fig_sector_movers.add_trace(go.Bar(
                        name=sector,
                        y=[sector],
                        x=[value],
                        orientation='h',
                        marker_color=base_color,
                        showlegend=False,
                        hovertemplate=f"<b>{sector.replace('-', ' ').title()}</b><br>" +
                                      f"Net Change: %{{x:,.0f}}<extra></extra>"
                    ))
        elif sector_breakout == "Global North/South":
            # Global North/South view - use shades of sector color
            for sector in sectors_sorted_reversed:
                df_sec = df_sector_viz[df_sector_viz['sector'] == sector].copy()

                # Sort by Global North first, then Global South
                df_sec['sort_order'] = df_sec['subsector'].map({'Global North': 0, 'Global South': 1})
                df_sec = df_sec.sort_values('sort_order')

                # Create color shades for this sector (2 shades: Global North and Global South)
                base_color = sector_color_map.get(sector, "#999999")
                classifications = df_sec['subsector'].tolist()
                shades = create_color_shades(base_color, len(classifications))

                # Add traces
                for i, (_, row) in enumerate(df_sec.iterrows()):
                    classification = row['subsector']
                    value = row['change']

                    fig_sector_movers.add_trace(go.Bar(
                        name=f"{sector}: {classification}",
                        y=[sector],
                        x=[value],
                        orientation='h',
                        marker_color=shades[i],
                        showlegend=False,
                        hovertemplate=f"<b>{sector.replace('-', ' ').title()}</b><br>" +
                                      f"{classification}: %{{x:,.0f}}<extra></extra>"
                    ))
        else:
            # Subsector view (default) - stacked bars with shades
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
                font=dict(size=13, color=color, family="Arial"),
                xanchor=x_anchor,
                xshift=xshift
            )

        st.plotly_chart(fig_sector_movers, use_container_width=True)

    # ==================== Country Sector Movers Visualizations ====================
    with viz_col2:
        st.markdown("#### Country Movers")

        # Add Country Sector Movers breakout toggle with label to the left
        label_col2, toggle_col2 = st.columns([0.8, 3.2], gap="small")
        with label_col2:
            st.markdown("<div style='padding-top: 8px; font-size: 0.9em; font-weight: 500;'>Breakdown Country by:</div>", unsafe_allow_html=True)
        with toggle_col2:
            country_breakout = st.segmented_control(
                "Country Sector Movers Breakout",
                options=["Total Net Change", "Sector"],
                default="Total Net Change",
                key="country_movers_breakout",
                label_visibility="collapsed"
            )

        # Get country-level data broken down by sector with region filter
        if trend_view == "Year-to-Date YoY":
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
                    and country_name <> 'Unknown'
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
                    and country_name <> 'Unknown'
                GROUP BY country_name, sector
            """
            df_country_sector = con.execute(country_sector_query).df()

        # Apply breakout transformations based on selected view
        if country_breakout == "Total Net Change":
            # Aggregate to country level only (no sector breakout)
            df_country_sector = df_country_sector.groupby('country_name')['change'].sum().reset_index()
            df_country_sector['sector'] = 'Net Change'  # Placeholder for compatibility

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

        # Create bar traces based on breakout view
        if country_breakout == "Total Net Change":
            # Simple bars with neutral color (works in light/dark mode)
            values = [country_totals_inc[loc] for loc in countries_sorted_inc]

            fig_increases.add_trace(go.Bar(
                name='Net Change',
                x=countries_sorted_inc,
                y=values,
                marker_color='#4A90E2',  # Neutral blue
                hovertemplate="<b>%{x}</b><br>Net Change: %{y:,.0f}<extra></extra>"
            ))
        else:
            # Sector view (default) - stacked bars with sector colors
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
                font=dict(size=12, color=color, family="Arial"),
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

            # Create bar traces based on breakout view
            if country_breakout == "Total Net Change":
                # Simple bars with neutral color
                values = [country_totals_dec[loc] for loc in countries_sorted_dec]

                fig_decreases.add_trace(go.Bar(
                    name='Net Change',
                    x=countries_sorted_dec,
                    y=values,
                    marker_color='#4A90E2',  # Neutral blue
                    hovertemplate="<b>%{x}</b><br>Net Change: %{y:,.0f}<extra></extra>"
                ))
            else:
                # Sector view (default) - stacked bars with sector colors
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
                    font=dict(size=12, color=color, family="Arial"),
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

    # ==================== Unified Country Rankings Table ====================
    # st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Country Rankings")

    # Add dropdowns for ranking configuration and download button
    rank_col1, rank_col2, rank_spacer = st.columns([2.5, 2.5, 8])

    with rank_col1:
        rank_metric = st.segmented_control(
            "Ranking Metric",
            options=["Emissions Changes", "Total Inventory"],
            default="Emissions Changes",
            key="rank_metric_selector",
        )

    with rank_col2:
        rank_breakdown = st.segmented_control(
            "Grouping Level",
            options=["Country-Sector", "Country-Subsector"],
            default="Country-Subsector",
            key="rank_breakdown_selector",
        )
    # Determine grouping columns based on breakdown selection
    if rank_breakdown == "Country-Subsector":
        group_by_cols = ["country_name", "iso3_country", "sector", "subsector"]
        detail_col = "subsector"
    else:  # "Country-Sector"
        group_by_cols = ["country_name", "iso3_country", "sector"]
        detail_col = "sector"

    group_by_sql = ", ".join(group_by_cols)

    # Build WHERE clause with region filter
    rank_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL",]
    if region_condition:
        rc_col = region_condition['column_name']
        rc_val = region_condition['column_value']
        if isinstance(rc_val, bool):
            rank_where_clauses.append(f"{rc_col} = {str(rc_val).upper()}")
        else:
            rank_where_clauses.append(f"{rc_col} = '{rc_val}'")

    # Query data based on trend_view - unified for both rank metrics
    if trend_view == "Year-to-Date YoY":
        rank_ytd_where = rank_where_clauses + [
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]
        rank_ytd_where_clause = " AND ".join(rank_ytd_where)

        rank_ytd_query = f"""
            SELECT
                {group_by_sql},
                year,
                month,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {rank_ytd_where_clause}
                and country_name <> 'Unknown'
            GROUP BY {group_by_sql}, year, month
            ORDER BY country_name, sector, year, month
        """
        df_rank_ytd = con.execute(rank_ytd_query).df()

        cumsum_keys = group_by_cols + ['year']
        df_rank_ytd['cumulative'] = df_rank_ytd.groupby(cumsum_keys)['emissions_quantity'].cumsum()

        df_current_rank = df_rank_ytd[
            (df_rank_ytd['year'] == latest_year) & (df_rank_ytd['month'] == latest_month)
        ][group_by_cols + ['cumulative']].rename(columns={'cumulative': 'current'})

        df_previous_rank = df_rank_ytd[
            (df_rank_ytd['year'] == latest_year - 1) & (df_rank_ytd['month'] == latest_month)
        ][group_by_cols + ['cumulative']].rename(columns={'cumulative': 'previous'})

        df_rankings = df_current_rank.merge(df_previous_rank, on=group_by_cols, how='outer').fillna(0)
        df_rankings['change'] = df_rankings['current'] - df_rankings['previous']

    else:
        rank_where_clause = " AND ".join(rank_where_clauses)

        if trend_view == "Month YoY":
            rank_query = f"""
                SELECT
                    {group_by_sql},
                    SUM({emissions_column_latest}) as current,
                    SUM(month_yoy_change) as change
                FROM '{country_subsector_stats_path}'
                WHERE {rank_where_clause}
                    and country_name <> 'Unknown'
                GROUP BY {group_by_sql}
            """
            df_rankings = con.execute(rank_query).df()
            df_rankings['previous'] = df_rankings['current'] - df_rankings['change']
        else:  # Month-over-Month
            rank_query = f"""
                SELECT
                    {group_by_sql},
                    SUM({emissions_column_latest}) as current,
                    SUM({emissions_column_prev}) as previous,
                    SUM(mom_change) as change
                FROM '{country_subsector_stats_path}'
                WHERE {rank_where_clause}
                    and country_name <> 'Unknown'
                GROUP BY {group_by_sql}
            """
            df_rankings = con.execute(rank_query).df()

    # Derived columns
    df_rankings['pct_change'] = df_rankings.apply(
        lambda row: (row['change'] / row['previous'] * 100) if row['previous'] != 0 else None,
        axis=1
    )
    df_rankings['abs_change'] = df_rankings['change'].abs()

    # Join flags
    df_flags = con.execute("SELECT iso3, flag FROM country_flags").df()
    df_flags = df_flags.rename(columns={'iso3': 'iso3_country'})
    df_rankings = df_rankings.merge(df_flags, on='iso3_country', how='left')
    df_rankings['flag'] = df_rankings['flag'].fillna('')

    # Determine ranking column and sector sort based on rank_metric
    if rank_metric == "Emissions Changes":
        rank_col = 'abs_change'
        sector_sort_agg = df_rankings.groupby('sector')['change'].sum()
        sectors = sector_sort_agg.abs().sort_values(ascending=False).index.tolist()
    else:  # "Total Inventory"
        rank_col = 'current'
        sector_sort_agg = df_rankings.groupby('sector')['current'].sum()
        sectors = sector_sort_agg.sort_values(ascending=False).index.tolist()

    # Build global top-10
    df_top10_global = df_rankings.nlargest(10, rank_col).copy()
    df_top10_global['rank'] = range(1, len(df_top10_global) + 1)

    # Build per-sector top-10
    df_rankings_sorted = df_rankings.sort_values(['sector', rank_col], ascending=[True, False])
    df_top10_per_sector = df_rankings_sorted.groupby('sector').head(10).reset_index(drop=True)
    df_top10_per_sector['rank'] = df_top10_per_sector.groupby('sector').cumcount() + 1

    # Cell rendering helper - returns compact single-line HTML
    def render_cell(row_data, is_sector_row=False):
        if row_data.empty:
            return ''

        country = row_data['country_name'].iloc[0]
        flag = row_data['flag'].iloc[0]
        detail_val = row_data[detail_col].iloc[0]
        detail_text = detail_val.replace('-', ' ')
        if detail_col == "sector":
            detail_text = detail_text.title()

        # Hide detail line when it's redundant (Country-Sector mode in per-sector rows)
        show_detail = not (detail_col == "sector" and is_sector_row)
        detail_html = "<div class='detail-text-cell'>" + detail_text + "</div>" if show_detail else ""

        country_line = "<div class='country-text-cell'>" + flag + " " + country + "</div>"

        if rank_metric == "Emissions Changes":
            change_val = row_data['change'].iloc[0]
            pct_val = row_data['pct_change'].iloc[0]
            arrow = "↑" if change_val > 0 else "↓"
            formatted_change = format_number_short(abs(change_val))
            color_class = 'increase' if change_val > 0 else 'decrease'
            if pct_val is not None and not pd.isna(pct_val):
                pct_display = str(round(abs(pct_val), 1))
            else:
                pct_display = "N/A"
            value_line = "<div class='change-text-cell change-" + color_class + "'>" + arrow + " " + formatted_change + " (" + pct_display + "%)</div>"
        else:
            inventory_val = row_data['current'].iloc[0]
            pct_val = row_data['pct_change'].iloc[0]
            formatted_inventory = format_number_short(inventory_val)

            if pd.isna(pct_val) or pct_val is None:
                pct_text = 'N/A'
                color_class = 'neutral'
            elif pct_val > 0:
                pct_text = '↑ ' + str(round(abs(pct_val), 1)) + '%'
                color_class = 'increase'
            elif pct_val < 0:
                pct_text = '↓ ' + str(round(abs(pct_val), 1)) + '%'
                color_class = 'decrease'
            else:
                pct_text = '0.0%'
                color_class = 'neutral'
            value_line = "<div class='metric-text-cell'>" + formatted_inventory + " <span class='pct-text-cell pct-" + color_class + "'>(" + pct_text + ")</span></div>"

        return "<div style='line-height:1.5'>" + country_line + detail_html + value_line + "</div>"

    # Build HTML rows
    html_rows = []

    # Global "ALL" row
    global_cells = []
    for r in range(1, 11):
        rd = df_top10_global[df_top10_global['rank'] == r]
        global_cells.append("<td class='data-cell'>" + render_cell(rd, is_sector_row=False) + "</td>")
    html_rows.append("<tr class='global-row'><td class='sector-cell global-sector-cell'>ALL</td>" + "".join(global_cells) + "</tr>")

    # Per-sector rows
    for sector in sectors:
        sector_data = df_top10_per_sector[df_top10_per_sector['sector'] == sector]
        sector_label = sector.replace('-', ' ').title()
        sector_cells = []
        for r in range(1, 11):
            rd = sector_data[sector_data['rank'] == r]
            sector_cells.append("<td class='data-cell'>" + render_cell(rd, is_sector_row=True) + "</td>")
        html_rows.append("<tr><td class='sector-cell'>" + sector_label + "</td>" + "".join(sector_cells) + "</tr>")

    # Build header row
    rank_headers = "".join(["<th>Rank " + str(i) + "</th>" for i in range(1, 11)])
    thead = "<thead><tr><th>Sector</th>" + rank_headers + "</tr></thead>"
    tbody = "<tbody>" + "".join(html_rows) + "</tbody>"

    # CSS (using regular string, not f-string, to avoid brace escaping issues)
    table_css = """
    <style>
        .rankings-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
        }
        .rankings-table th {
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
            padding: 10px;
            font-weight: 600;
            background-color: var(--background-color);
        }
        .rankings-table .sector-cell {
            font-weight: 600;
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
            padding: 10px;
            font-size: 0.9em;
            background-color: var(--secondary-background-color);
        }
        .rankings-table .global-row {
            background-color: rgba(31, 119, 255, 0.1);
        }
        .rankings-table .global-sector-cell {
            background-color: rgba(31, 119, 255, 0.15);
        }
        .rankings-table .data-cell {
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.3));
            padding: 10px;
            vertical-align: top;
        }
        .country-text-cell {
            font-size: 0.9em;
            font-weight: 700;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-color);
        }
        .detail-text-cell {
            font-size: 0.72em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            opacity: 0.6;
        }
        .change-text-cell {
            font-weight: 600;
            font-size: 0.78em;
            white-space: nowrap;
            margin: 2px 0;
        }
        .change-text-cell.change-increase {
            color: #d9534f !important;
        }
        .change-text-cell.change-decrease {
            color: #5cb85c !important;
        }
        .metric-text-cell {
            font-size: 0.78em;
            font-weight: 600;
            color: var(--text-color);
            margin: 2px 0;
            white-space: nowrap;
        }
        .pct-text-cell {
            font-weight: 600;
            white-space: nowrap;
        }
        .pct-text-cell.pct-increase {
            color: #d9534f !important;
        }
        .pct-text-cell.pct-decrease {
            color: #5cb85c !important;
        }
        .pct-text-cell.pct-neutral {
            color: var(--text-color);
            opacity: 0.6;
        }
        .rankings-table th:first-child {
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 1;
        }
        .rankings-table th:not(:first-child) {
            text-align: center;
        }
    </style>
    """

    table_html = table_css + "<div style='overflow-x:auto'><table class='rankings-table'>" + thead + tbody + "</table></div>"

    st.markdown(table_html, unsafe_allow_html=True)

    # ==================== Monthly Time Series ====================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Monthly Time Series: " + selected_scope)

    with st.expander("**Explore monthly emissions, activity, and emissions factor trends. Drill down by country, sector, and subsector**", expanded=False):

        # Get list of countries from the actual data table (avoids name mismatches with gadm_0)
        ts_country_base_filter = "country_name IS NOT NULL AND gas = 'co2e_100yr'"
        country_rows = con.execute(
            "SELECT DISTINCT country_name, iso3_country FROM '" + country_subsector_totals_path + "' WHERE " + ts_country_base_filter + " ORDER BY country_name"
        ).fetchall()

        country_map = {row[0]: row[1] for row in country_rows}
        all_countries = list(country_map.keys())

        # Filter countries based on selected region
        if selected_scope == 'Global':
            available_countries = all_countries
        elif region_condition:
            region_col = region_condition['column_name']
            region_val = region_condition['column_value']

            if isinstance(region_val, bool):
                region_filter_sql = region_col + " = " + str(region_val).upper()
            else:
                region_filter_sql = region_col + " = '" + region_val + "'"

            region_countries = con.execute(
                "SELECT DISTINCT country_name FROM '" + country_subsector_totals_path + "' WHERE " + ts_country_base_filter + " AND " + region_filter_sql + " AND country_name <> 'Unknown' " + " ORDER BY country_name"
            ).fetchall()
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

        # ==================== Calculate metrics for ALL 3 time periods ====================

        # Scope label for card titles
        if selected_country != "All Countries":
            scope_label = selected_country
        else:
            scope_label = selected_scope

        # --- Build shared WHERE clauses ---
        dd_where_clauses = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]
        if region_condition and selected_country == "All Countries":
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                dd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                dd_where_clauses.append(f"{column_name} = '{column_value}'")
        elif selected_country != "All Countries":
            dd_where_clauses.append(f"country_name = '{selected_country}'")
        if selected_sector_dd != "All Sectors":
            dd_where_clauses.append(f"sector = '{selected_sector_dd}'")
        if selected_subsector_dd != "All Subsectors":
            dd_where_clauses.append(f"subsector = '{selected_subsector_dd}'")
        dd_where_clause = " AND ".join(dd_where_clauses)

        # --- Emissions: MoM + YoY from stats table (single query) ---
        dd_stats_query = f"""
            SELECT
                SUM({emissions_column_latest}) as latest,
                SUM({emissions_column_prev}) as prev_month,
                SUM(mom_change) as mom_change,
                SUM(month_yoy_change) as yoy_change
            FROM '{country_subsector_stats_path}'
            WHERE {dd_where_clause}
        """
        df_dd_stats = con.execute(dd_stats_query).df()
        em_latest = df_dd_stats['latest'].iloc[0] or 0
        em_prev_month = df_dd_stats['prev_month'].iloc[0] or 0
        em_mom_change = df_dd_stats['mom_change'].iloc[0] or 0
        em_yoy_change = df_dd_stats['yoy_change'].iloc[0] or 0
        em_yoy_previous = em_latest - em_yoy_change

        em_mom_pct = (em_mom_change / em_prev_month * 100) if em_prev_month != 0 else 0
        em_yoy_pct = (em_yoy_change / em_yoy_previous * 100) if em_yoy_previous != 0 else 0

        # --- Emissions: YTD from totals table ---
        dd_ytd_where_clauses = [
            "gas = 'co2e_100yr'", "country_name IS NOT NULL",
            f"year IN ({latest_year - 1}, {latest_year})",
            f"month <= {latest_month}"
        ]
        if region_condition and selected_country == "All Countries":
            column_name = region_condition['column_name']
            column_value = region_condition['column_value']
            if isinstance(column_value, bool):
                dd_ytd_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
            else:
                dd_ytd_where_clauses.append(f"{column_name} = '{column_value}'")
        elif selected_country != "All Countries":
            dd_ytd_where_clauses.append(f"country_name = '{selected_country}'")
        if selected_sector_dd != "All Sectors":
            dd_ytd_where_clauses.append(f"sector = '{selected_sector_dd}'")
        if selected_subsector_dd != "All Subsectors":
            dd_ytd_where_clauses.append(f"subsector = '{selected_subsector_dd}'")
        dd_ytd_where_clause = " AND ".join(dd_ytd_where_clauses)

        df_dd_ytd = con.execute(f"""
            SELECT year, month, SUM(emissions_quantity) AS emissions_quantity
            FROM '{country_subsector_totals_path}'
            WHERE {dd_ytd_where_clause}
            GROUP BY year, month ORDER BY year, month
        """).df()
        df_dd_ytd['cumulative'] = df_dd_ytd.groupby('year')['emissions_quantity'].cumsum()
        em_ytd_current = df_dd_ytd[
            (df_dd_ytd['year'] == latest_year) & (df_dd_ytd['month'] == latest_month)
        ]['cumulative'].sum()
        em_ytd_previous = df_dd_ytd[
            (df_dd_ytd['year'] == latest_year - 1) & (df_dd_ytd['month'] == latest_month)
        ]['cumulative'].sum()
        em_ytd_change = em_ytd_current - em_ytd_previous
        em_ytd_pct = (em_ytd_change / em_ytd_previous * 100) if em_ytd_previous != 0 else 0

        # --- Activity / EF metrics (only when subsector selected) ---
        show_activity_and_ef_cards = selected_subsector_dd != "All Subsectors"

        if show_activity_and_ef_cards:
            asset_where_clauses = ["gas = 'co2e_100yr'"]
            if region_condition and selected_country == "All Countries":
                column_name = region_condition['column_name']
                column_value = region_condition['column_value']
                if isinstance(column_value, bool):
                    asset_where_clauses.append(f"{column_name} = {str(column_value).upper()}")
                else:
                    asset_where_clauses.append(f"{column_name} = '{column_value}'")
            elif selected_country != "All Countries":
                asset_where_clauses.append(f"iso3_country = '{selected_country_iso3}'")
            if selected_sector_dd != "All Sectors":
                asset_where_clauses.append(f"sector = '{selected_sector_dd}'")
            asset_where_clauses.append(f"original_inventory_sector = '{selected_subsector_dd}'")
            asset_where_clauses.append("""original_inventory_sector NOT IN (
                'forest-land-clearing', 'forest-land-degradation', 'forest-land-fires',
                'net-forest-land', 'net-shrubgrass', 'net-wetland', 'removals',
                'shrubgrass-fires', 'water-reservoirs', 'wetland-fires')""")
            asset_where_clause = " AND ".join(asset_where_clauses)

            # Month strings for MoM + YoY
            if latest_month == 1:
                prev_m, prev_y = 12, latest_year - 1
            else:
                prev_m, prev_y = latest_month - 1, latest_year
            current_month_str = f"{latest_year}-{latest_month:02d}"
            prev_month_str = f"{prev_y}-{prev_m:02d}"
            yoy_prev_str = f"{latest_year - 1}-{latest_month:02d}"

            # Single query for MoM + YoY months
            months_in = ", ".join([f"'{m}'" for m in set([current_month_str, prev_month_str, yoy_prev_str])])
            df_asset_card = con.execute(f"""
                SELECT strftime(start_time, '%Y-%m') AS year_month,
                       SUM(activity) AS activity, SUM(emissions_quantity) AS emissions_quantity
                FROM '{asset_path}'
                WHERE {asset_where_clause} AND strftime(start_time, '%Y-%m') IN ({months_in})
                GROUP BY year_month ORDER BY year_month
            """).df()

            def _get_asset(df, ym):
                row = df[df['year_month'] == ym]
                return (row.iloc[0]['activity'], row.iloc[0]['emissions_quantity']) if not row.empty else (0, 0)

            act_cur, em_asset_cur = _get_asset(df_asset_card, current_month_str)
            act_prev_m, em_asset_prev_m = _get_asset(df_asset_card, prev_month_str)
            act_yoy_prev, em_asset_yoy_prev = _get_asset(df_asset_card, yoy_prev_str)

            act_mom_change = act_cur - act_prev_m
            act_mom_pct = (act_mom_change / act_prev_m * 100) if act_prev_m != 0 else 0
            act_yoy_change = act_cur - act_yoy_prev
            act_yoy_pct = (act_yoy_change / act_yoy_prev * 100) if act_yoy_prev != 0 else 0

            ef_cur = (em_asset_cur / act_cur) if act_cur != 0 else 0
            ef_prev_m = (em_asset_prev_m / act_prev_m) if act_prev_m != 0 else 0
            ef_yoy_prev = (em_asset_yoy_prev / act_yoy_prev) if act_yoy_prev != 0 else 0
            ef_mom_change = ef_cur - ef_prev_m
            ef_mom_pct = (ef_mom_change / ef_prev_m * 100) if ef_prev_m != 0 else 0
            ef_yoy_change = ef_cur - ef_yoy_prev
            ef_yoy_pct = (ef_yoy_change / ef_yoy_prev * 100) if ef_yoy_prev != 0 else 0

            # YTD from asset table
            df_asset_ytd = con.execute(f"""
                SELECT strftime(start_time, '%Y') AS year,
                       SUM(activity) AS activity, SUM(emissions_quantity) AS emissions_quantity
                FROM '{asset_path}'
                WHERE {asset_where_clause}
                    AND strftime(start_time, '%Y') IN ('{latest_year}', '{latest_year - 1}')
                    AND CAST(strftime(start_time, '%m') AS INTEGER) <= {latest_month}
                GROUP BY year ORDER BY year
            """).df()

            def _get_ytd_asset(df, yr):
                row = df[df['year'] == str(yr)]
                return (row.iloc[0]['activity'], row.iloc[0]['emissions_quantity']) if not row.empty else (0, 0)

            act_ytd_cur, em_ytd_cur_asset = _get_ytd_asset(df_asset_ytd, latest_year)
            act_ytd_prev, em_ytd_prev_asset = _get_ytd_asset(df_asset_ytd, latest_year - 1)
            act_ytd_change = act_ytd_cur - act_ytd_prev
            act_ytd_pct = (act_ytd_change / act_ytd_prev * 100) if act_ytd_prev != 0 else 0

            ef_ytd_cur = (em_ytd_cur_asset / act_ytd_cur) if act_ytd_cur != 0 else 0
            ef_ytd_prev = (em_ytd_prev_asset / act_ytd_prev) if act_ytd_prev != 0 else 0
            ef_ytd_change = ef_ytd_cur - ef_ytd_prev
            ef_ytd_pct = (ef_ytd_change / ef_ytd_prev * 100) if ef_ytd_prev != 0 else 0

        # ==================== Display 3 Cards ====================

        def _fmt_ef(val):
            if val == 0:
                return "0"
            if abs(val) >= 1000:
                return format_number_short(val)
            if abs(val) >= 1:
                return f"{val:,.2f}"
            return f"{val:.4f}"

        # Pick the right change/inventory values based on trend_view
        if trend_view == "Month YoY":
            tv_short = "Month YoY"
            em_change, em_pct = em_yoy_change, em_yoy_pct
            em_cur_val, em_prev_val = em_latest, em_yoy_previous
            cur_label = month_name[latest_month][:3] + " " + str(latest_year)
            prev_label = month_name[latest_month][:3] + " " + str(latest_year - 1)
            if show_activity_and_ef_cards:
                act_change_val, act_pct_val = act_yoy_change, act_yoy_pct
                act_cur_val, act_prev_val = act_cur, act_yoy_prev
                ef_change_val, ef_pct_val = ef_yoy_change, ef_yoy_pct
                ef_cur_val, ef_prev_val = ef_cur, ef_yoy_prev
        elif trend_view == "Year-to-Date YoY":
            tv_short = "YTD YoY"
            em_change, em_pct = em_ytd_change, em_ytd_pct
            em_cur_val, em_prev_val = em_ytd_current, em_ytd_previous
            cur_label = "Jan-" + month_name[latest_month][:3] + " " + str(latest_year)
            prev_label = "Jan-" + month_name[latest_month][:3] + " " + str(latest_year - 1)
            if show_activity_and_ef_cards:
                act_change_val, act_pct_val = act_ytd_change, act_ytd_pct
                act_cur_val, act_prev_val = act_ytd_cur, act_ytd_prev
                ef_change_val, ef_pct_val = ef_ytd_change, ef_ytd_pct
                ef_cur_val, ef_prev_val = ef_ytd_cur, ef_ytd_prev
        else:  # Month-over-Month
            tv_short = "MoM"
            em_change, em_pct = em_mom_change, em_mom_pct
            em_cur_val, em_prev_val = em_latest, em_prev_month
            cur_label = month_name[latest_month][:3] + " " + str(latest_year)
            if latest_month == 1:
                prev_label = month_name[12][:3] + " " + str(latest_year - 1)
            else:
                prev_label = month_name[latest_month - 1][:3] + " " + str(latest_year)
            if show_activity_and_ef_cards:
                act_change_val, act_pct_val = act_mom_change, act_mom_pct
                act_cur_val, act_prev_val = act_cur, act_prev_m
                ef_change_val, ef_pct_val = ef_mom_change, ef_mom_pct
                ef_cur_val, ef_prev_val = ef_cur, ef_prev_m

        card_css = "border: 1px solid #999; border-radius: 10px; padding: 16px; height: 180px; display: flex; flex-direction: column;"

        col1, col2, col3 = st.columns(3)

        em_arrow = "↑" if em_change > 0 else "↓"
        em_color = "red" if em_change > 0 else "green"

        with col1:
            st.markdown(f"""
                <div style="{card_css}">
                    <div style="font-size: 0.95em; font-weight: 600; margin-bottom: auto;">{scope_label} Emissions ({tv_short})</div>
                    <div style="font-size: 1.4em; font-weight: bold; text-align: center; color: {em_color}; margin-bottom: 28px;">
                        {em_arrow} {format_number_short(abs(em_change))} <span style="color: #888;">(</span><span style="color: {em_color};">{abs(em_pct):.1f}%</span><span style="color: #888;">)</span>
                    </div>
                    <div style="font-size: 0.82em; text-align: left; color: #888; line-height: 1.4; padding-left: 8px;">
                        <div>{cur_label}: <span style="font-weight: 600; color: var(--text-color);">{em_cur_val:,.0f}</span> tCO₂e</div>
                        <div>{prev_label}: <span style="font-weight: 600; color: var(--text-color);">{em_prev_val:,.0f}</span> tCO₂e</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            if show_activity_and_ef_cards:
                act_arrow = "↑" if act_change_val > 0 else "↓"
                act_color = "red" if act_change_val > 0 else "green"
                st.markdown(f"""
                    <div style="{card_css}">
                        <div style="font-size: 0.95em; font-weight: 600; margin-bottom: auto;">{scope_label} Activity ({tv_short})</div>
                        <div style="font-size: 1.4em; font-weight: bold; text-align: center; color: {act_color}; margin-bottom: 28px;">
                            {act_arrow} {format_number_short(abs(act_change_val))} <span style="color: #888;">(</span><span style="color: {act_color};">{abs(act_pct_val):.1f}%</span><span style="color: #888;">)</span>
                        </div>
                        <div style="font-size: 0.82em; text-align: left; color: #888; line-height: 1.4; padding-left: 8px;">
                            <div>{cur_label}: <span style="font-weight: 600; color: var(--text-color);">{act_cur_val:,.0f}</span></div>
                            <div>{prev_label}: <span style="font-weight: 600; color: var(--text-color);">{act_prev_val:,.0f}</span></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="{card_css} justify-content: center; align-items: center;">
                        <div style="font-size: 0.95em; color: #888; text-align: center;">Select a subsector to view activity data</div>
                    </div>""", unsafe_allow_html=True)

        with col3:
            if show_activity_and_ef_cards:
                ef_arrow = "↑" if ef_change_val > 0 else "↓"
                ef_color = "red" if ef_change_val > 0 else "green"
                st.markdown(f"""
                    <div style="{card_css}">
                        <div style="font-size: 0.95em; font-weight: 600; margin-bottom: auto;">{scope_label} Emissions Factor ({tv_short})</div>
                        <div style="font-size: 1.4em; font-weight: bold; text-align: center; color: {ef_color}; margin-bottom: 28px;">
                            {ef_arrow} {abs(ef_pct_val):.1f}%
                        </div>
                        <div style="font-size: 0.82em; text-align: left; color: #888; line-height: 1.4; padding-left: 8px;">
                            <div>{cur_label}: <span style="font-weight: 600; color: var(--text-color);">{_fmt_ef(ef_cur_val)}</span> tCO₂e/unit</div>
                            <div>{prev_label}: <span style="font-weight: 600; color: var(--text-color);">{_fmt_ef(ef_prev_val)}</span> tCO₂e/unit</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="{card_css} justify-content: center; align-items: center;">
                        <div style="font-size: 0.95em; color: #888; text-align: center;">Select a subsector to view emissions factor data</div>
                    </div>""", unsafe_allow_html=True)

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
            ts_where_clauses.append(f"country_name = '{selected_country}'")

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

        # Query asset-level time series (always runs for emissions line, like tab03)
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

        # Add subsector filter (only when selected)
        if selected_subsector_dd != "All Subsectors":
            ts_asset_where_clauses.append(f"original_inventory_sector = '{selected_subsector_dd}'")

        # Exclude forestry subsectors
        ts_asset_where_clauses.append("""original_inventory_sector NOT IN (
            'forest-land-clearing', 'forest-land-degradation', 'forest-land-fires',
            'net-forest-land', 'net-shrubgrass', 'net-wetland', 'removals',
            'shrubgrass-fires', 'water-reservoirs', 'wetland-fires')""")

        ts_asset_where_clause = " AND ".join(ts_asset_where_clauses)

        df_ts_asset = con.execute(f"""
            SELECT
                strftime(start_time, '%Y-%m') AS year_month,
                SUM(activity) AS activity,
                SUM(emissions_quantity) AS emissions_quantity
            FROM '{asset_path}'
            WHERE {ts_asset_where_clause}
            GROUP BY year_month
            ORDER BY year_month
        """).df()

        if not df_ts_asset.empty:
            df_ts_asset['year_month'] = pd.to_datetime(df_ts_asset['year_month'])
            df_ts_asset['mean_emissions_factor'] = df_ts_asset['emissions_quantity'] / df_ts_asset['activity']
            cutoff_date = pd.Timestamp(year=earliest_year_ts, month=1, day=1)
            df_ts_asset = df_ts_asset[df_ts_asset['year_month'] >= cutoff_date]

        # Create time series subplot
        show_activity_and_ef = selected_subsector_dd != "All Subsectors"
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

        # Row 1 — Asset emissions line (always shown)
        if not df_ts_asset.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts_asset['year_month'],
                    y=df_ts_asset['emissions_quantity'],
                    mode='lines+markers',
                    name='Assets'
                ),
                row=1, col=1
            )

        # Row 1 — Total emissions line
        if not df_ts_emissions.empty:
            fig_ts.add_trace(
                go.Scatter(
                    x=df_ts_emissions['year_month'],
                    y=df_ts_emissions['emissions_quantity'],
                    mode='lines+markers',
                    name='Total',
                    line=dict(color='#E9967A')
                ),
                row=1, col=1
            )

        # Row 2 — Activity (subsector only)
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

        # Row 3 — Emissions Factor (subsector only)
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

    # ==================== Download Section ====================
    st.markdown("<br>", unsafe_allow_html=True)

    # Build time period description for metadata
    if trend_view == "Month YoY":
        time_period_desc = month_name[latest_month] + " " + str(latest_year) + " vs " + month_name[latest_month] + " " + str(latest_year - 1)
    elif trend_view == "Year-to-Date YoY":
        time_period_desc = "January - " + month_name[latest_month] + " " + str(latest_year) + " vs January - " + month_name[latest_month] + " " + str(latest_year - 1)
    else:  # MoM
        if latest_month == 1:
            prev_m, prev_y = 12, latest_year - 1
        else:
            prev_m, prev_y = latest_month - 1, latest_year
        time_period_desc = month_name[latest_month] + " " + str(latest_year) + " vs " + month_name[prev_m] + " " + str(prev_y)

    # Snapshot all data needed by the download fragment so checkbox clicks
    # only rerun this small section, not the whole page.
    _dl_data = {
        "trend_view": trend_view,
        "time_period_desc": time_period_desc,
        "selected_scope": selected_scope,
        "region_condition": region_condition,
        "sector_breakout": sector_breakout,
        "country_breakout": country_breakout,
        "rank_metric": rank_metric,
        "rank_breakdown": rank_breakdown,
        "selected_country": selected_country,
        "selected_sector_dd": selected_sector_dd,
        "selected_subsector_dd": selected_subsector_dd,
        "df_sector_viz": df_sector_viz,
        "df_country_sector": df_country_sector,
        "all_increases": all_increases,
        "all_decreases": all_decreases,
        "df_rankings": df_rankings,
        "df_ts_emissions": df_ts_emissions,
        "df_ts_asset": df_ts_asset,
        "stats_path": country_subsector_stats_path,
    }

    @st.fragment
    def render_download_section(data):

        # Scoped CSS (only affects elements inside .dl-wrap)
        st.markdown(
            """
            <style>
            .dl-wrap .dl-highlight {
                background: #fef9e7;
                border: 1px solid #f0e0a0;
                border-radius: 10px;
                padding: 0.5rem 0.75rem;
                margin-top: 0.25rem;
            }
            @media (prefers-color-scheme: dark) {
                .dl-wrap .dl-highlight {
                background: rgba(120, 100, 30, 0.18);
                border-color: rgba(180, 160, 60, 0.35);
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Wrap the whole section so CSS is scoped
        st.markdown('<div class="dl-wrap">', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:1.1em; font-weight:600; margin-bottom:6px;'>Download Data</div>",
                unsafe_allow_html=True
            )

            # Only this block is highlighted
            st.markdown('<div class="dl-highlight">', unsafe_allow_html=True)

            dl_cols = st.columns([0.5, 1.1, 1.2, 1.2, 1.3, 1.1, 1.5])

            with dl_cols[1]:
                dl_sector = st.checkbox("Sector Movers", key="dl_sector", value=True)
            with dl_cols[2]:
                dl_country = st.checkbox("Country Movers", key="dl_country", value=True)
            with dl_cols[3]:
                dl_rankings_cb = st.checkbox("Country Rankings", key="dl_rankings", value=True)
            with dl_cols[4]:
                dl_timeseries = st.checkbox("Monthly Time Series", key="dl_timeseries", value=True)
            with dl_cols[5]:
                dl_raw = st.checkbox("Raw Statistics", key="dl_raw", value=True)

            st.markdown("</div>", unsafe_allow_html=True)  # end highlight

            any_selected = dl_sector or dl_country or dl_rankings_cb or dl_timeseries or dl_raw

            with dl_cols[6]:
                if not any_selected:
                    st.download_button(
                        label="Download Data",
                        data=b"",
                        file_name="emissions_data.xlsx",
                        disabled=True,
                        icon=":material/download:",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)  # end dl-wrap
                    return

                # Build Excel — keyed on all user inputs so cache invalidates properly
                def _build_excel(data_dict, selections):
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:

                        def write_metadata(ws, metadata_pairs):
                            bold = writer.book.add_format({"bold": True})
                            for i, (k, v) in enumerate(metadata_pairs):
                                ws.write(i, 0, k, bold)
                                ws.write(i, 1, str(v))

                        if selections["sector"]:
                            metadata = [
                                ("Time Period", data_dict["trend_view"]),
                                ("Comparison", data_dict["time_period_desc"]),
                                ("Region", data_dict["selected_scope"]),
                                ("Breakout", data_dict["sector_breakout"]),
                            ]
                            start = len(metadata) + 1
                            df_sec_out = data_dict["df_sector_viz"].copy()
                            # Sort: sector by highest absolute change, then subsector by highest absolute change within sector
                            sector_abs_totals = df_sec_out.groupby("sector")["change"].sum().abs().rename("_sector_abs_total")
                            df_sec_out = df_sec_out.merge(sector_abs_totals, on="sector")
                            df_sec_out["_abs_change"] = df_sec_out["change"].abs()
                            df_sec_out = df_sec_out.sort_values(["_sector_abs_total", "_abs_change"], ascending=[False, False]).drop(columns=["_sector_abs_total", "_abs_change"])
                            if data_dict["sector_breakout"] == "Total Net Change":
                                df_sec_out = df_sec_out.drop(columns=["subsector"], errors="ignore")
                            df_sec_out.to_excel(writer, sheet_name="Sector Movers", startrow=start, index=False)
                            write_metadata(writer.sheets["Sector Movers"], metadata)

                        if selections["country"]:
                            base_meta = [
                                ("Time Period", data_dict["trend_view"]),
                                ("Comparison", data_dict["time_period_desc"]),
                                ("Region", data_dict["selected_scope"]),
                                ("Breakout", data_dict["country_breakout"]),
                            ]

                            inc_meta = base_meta + [("Direction", "Increases")]
                            start = len(inc_meta) + 1
                            df_inc = data_dict["df_country_sector"][data_dict["df_country_sector"]["country_name"].isin(data_dict["all_increases"])].copy()
                            # Sort: country total change desc, then subsector change desc within each country
                            country_totals_inc = df_inc.groupby("country_name")["change"].sum().rename("_country_total")
                            df_inc = df_inc.merge(country_totals_inc, on="country_name")
                            df_inc = df_inc.sort_values(["_country_total", "change"], ascending=[False, False]).drop(columns=["_country_total"])
                            if data_dict["country_breakout"] == "Total Net Change":
                                df_inc = df_inc.drop(columns=["sector"], errors="ignore")
                            df_inc.to_excel(writer, sheet_name="Country Movers - Increase", startrow=start, index=False)
                            write_metadata(writer.sheets["Country Movers - Increase"], inc_meta)

                            dec_meta = base_meta + [("Direction", "Decreases")]
                            start = len(dec_meta) + 1
                            df_dec = data_dict["df_country_sector"][data_dict["df_country_sector"]["country_name"].isin(data_dict["all_decreases"])].copy()
                            # Sort: country total change asc (largest decrease first), then subsector change asc
                            country_totals_dec = df_dec.groupby("country_name")["change"].sum().rename("_country_total")
                            df_dec = df_dec.merge(country_totals_dec, on="country_name")
                            df_dec = df_dec.sort_values(["_country_total", "change"], ascending=[True, True]).drop(columns=["_country_total"])
                            if data_dict["country_breakout"] == "Total Net Change":
                                df_dec = df_dec.drop(columns=["sector"], errors="ignore")
                            df_dec.to_excel(writer, sheet_name="Country Movers - Decrease", startrow=start, index=False)
                            write_metadata(writer.sheets["Country Movers - Decrease"], dec_meta)

                        if selections["rankings"]:
                            metadata = [
                                ("Time Period", data_dict["trend_view"]),
                                ("Comparison", data_dict["time_period_desc"]),
                                ("Region", data_dict["selected_scope"]),
                                ("Ranking Metric", data_dict["rank_metric"]),
                                ("Grouping Level", data_dict["rank_breakdown"]),
                            ]
                            start = len(metadata) + 1
                            data_dict["df_rankings"].to_excel(writer, sheet_name="Country Rankings", startrow=start, index=False)
                            write_metadata(writer.sheets["Country Rankings"], metadata)

                        if selections["timeseries"]:
                            ts_meta = [
                                ("Time Period", data_dict["trend_view"]),
                                ("Region", data_dict["selected_scope"]),
                                ("Country", data_dict["selected_country"]),
                                ("Sector", data_dict["selected_sector_dd"]),
                                ("Subsector", data_dict["selected_subsector_dd"]),
                            ]
                            start = len(ts_meta) + 1

                            df_em = data_dict["df_ts_emissions"]
                            df_asset = data_dict["df_ts_asset"]

                            if not df_em.empty:
                                df_ts = df_em[["year_month", "emissions_quantity"]].copy()

                                # Add asset emissions as a separate column
                                if not df_asset.empty:
                                    df_asset_merge = df_asset[["year_month", "emissions_quantity"]].rename(
                                        columns={"emissions_quantity": "asset_emissions_quantity"}
                                    )
                                    df_ts = df_ts.merge(df_asset_merge, on="year_month", how="left")

                                    # Only include activity/EF when a subsector is selected
                                    if data_dict["selected_subsector_dd"] != "All Subsectors":
                                        df_act_ef = df_asset[["year_month", "activity", "mean_emissions_factor"]]
                                        df_ts = df_ts.merge(df_act_ef, on="year_month", how="left")

                                df_ts.to_excel(writer, sheet_name="Monthly Time Series", startrow=start, index=False)
                                write_metadata(writer.sheets["Monthly Time Series"], ts_meta)

                        if selections["raw"]:
                            raw_where = ["gas = 'co2e_100yr'", "country_name IS NOT NULL"]
                            rc = data_dict["region_condition"]
                            if rc:
                                if isinstance(rc["column_value"], bool):
                                    raw_where.append(f"{rc['column_name']} = {str(rc['column_value']).upper()}")
                                else:
                                    raw_where.append(f"{rc['column_name']} = '{rc['column_value']}'")
                            raw_where_clause = " AND ".join(raw_where)

                            raw_con = duckdb.connect()
                            df_raw = raw_con.execute(
                                f"SELECT * FROM '{data_dict['stats_path']}' WHERE {raw_where_clause}"
                            ).df()
                            raw_con.close()

                            metadata = [
                                ("Region", data_dict["selected_scope"]),
                                ("Filters", "gas = co2e_100yr, country_name IS NOT NULL"),
                            ]
                            start = len(metadata) + 1
                            df_raw.to_excel(writer, sheet_name="Raw Statistics", startrow=start, index=False)
                            write_metadata(writer.sheets["Raw Statistics"], metadata)

                    buf.seek(0)
                    return buf.getvalue()

                sels = {
                    "sector": dl_sector,
                    "country": dl_country,
                    "rankings": dl_rankings_cb,
                    "timeseries": dl_timeseries,
                    "raw": dl_raw,
                }

                excel_bytes = _build_excel(data, sels)

                st.download_button(
                    label="Download Data",
                    data=excel_bytes,
                    file_name="emissions_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    icon=":material/download:",
                    use_container_width=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)  # end dl-wrap

    render_download_section(_dl_data)

    con.close()
