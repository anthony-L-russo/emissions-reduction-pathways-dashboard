# UPDATE 4: MAKE PARQUET CONVERSION SCRIPT DELETE UNDERSCORE DATE AND MAKE STATIC FILE PATH
# UPDATE 5: ADD GRAPHS (NOTION TASK)

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from calendar import month_name
from utils.utils import format_dropdown_options, map_region_condition, format_number_short, create_excel_file

st.set_page_config(layout="wide")

# Load CT logo
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")

# data paths
asset_path = "data/asset_emissions_country_subsector.parquet"
country_subsector_stats_path = "data/country_subsector_emissions_statistics_202505.parquet"
country_subsector_totals_path = 'data/country_subsector_emissions_totals_202505.parquet'

con = duckdb.connect()

release_version = con.execute(f"SELECT DISTINCT release FROM '{asset_path}'").fetchone()[0]

col1, col2 = st.columns([10, 1])

with col1:
    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
            <h1 style="margin: 0; font-size: 2.8em;">Climate TRACE Monthly Dashboard</h1>
        </div>
        <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
            The data in this dashboard is from Climate TRACE release <span style='color: red;'><strong>{release_version}</strong></span>. It excludes all Forestry data.
        </p>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    download_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# Scope dropdown options
region_options = [
    'Global', 'EU', 'OECD', 'Non-OECD',
    'UNFCCC Annex', 'UNFCCC Non-Annex',
    'Global North', 'Global South',
    'Developed Markets', 'Emerging Markets',
    'Africa', 'Antarctica', 'Asia', 'Europe',
    'North America', 'Oceania', 'South America'
]

unique_countries = sorted(
    row[0] for row in con.execute(
        f"SELECT DISTINCT country_name FROM '{country_subsector_totals_path}' WHERE country_name IS NOT NULL"
    ).fetchall()
)

# Load and prepare statistics CSV

df_stats_all = pd.read_parquet(country_subsector_stats_path)
df_stats_all = df_stats_all[df_stats_all['gas'] == 'co2e_100yr']

raw_sectors = sorted(df_stats_all['sector'].dropna().unique().tolist())

def format_sector_label(sector):
    return ' '.join([w.capitalize() if w.lower() != 'and' else 'and' for w in sector.replace('-', ' ').split()])

sector_labels = [format_sector_label(s) for s in raw_sectors]
sector_map = dict(zip(sector_labels, raw_sectors))
sector_labels.insert(0, "All")
sector_map["All"] = None

# Subsector dropdown setup
col1, col2, col3 = st.columns(3)
with col1:
    selected_scope = st.selectbox("Select a Region/Country", region_options + unique_countries, key="selected_scope")
    region_condition = map_region_condition(selected_scope)
with col2:
    selected_sector_label = st.selectbox("Select Sector", sector_labels, key="sector_selector")
    selected_sector_raw = sector_map.get(selected_sector_label)
if selected_sector_raw:
    subsector_subset = df_stats_all[df_stats_all['sector'] == selected_sector_raw]
else:
    subsector_subset = df_stats_all

raw_subsectors = sorted(subsector_subset['subsector'].dropna().unique().tolist())
subsector_labels, subsector_map = format_dropdown_options(raw_subsectors)
subsector_labels.insert(0, "All")
subsector_map["All"] = None

with col3:
    default_index = 0
    if "selected_subsector_label" in st.session_state:
        try:
            default_index = subsector_labels.index(st.session_state["selected_subsector_label"])
        except ValueError:
            pass
    selected_subsector_label = st.selectbox(
        "Select Subsector", subsector_labels, index=default_index, key="selected_subsector_label"
    )

selected_subsector_raw = subsector_map.get(selected_subsector_label)

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
where_clauses = ["gas = 'co2e_100yr'"]
# Sector/Subsector filters
if selected_sector_raw:
    where_clauses.append(f"sector = '{selected_sector_raw}'")
if selected_subsector_raw:
    where_clauses.append(f"original_inventory_sector = '{selected_subsector_raw}'")

if region_condition:
    col = region_condition['column_name']
    val = region_condition['column_value']

    # Special handling if it's actually a country filter
    if col == "country_name" and val in ("United States", "United States of America"):
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
        df_stats = df_stats[df_stats[region_condition['column_name']] == region_condition['column_value']]
    else:
        df_stats = df_stats[df_stats['country_name'] == selected_scope]

if selected_subsector_raw:
    df_stats_filtered = df_stats[df_stats['subsector'] == selected_subsector_raw]
else:
    df_stats_filtered = df_stats.copy()
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
        df_stats_filtered = df_stats_agg.merge(slopes[['country_name', 'emissions_slope_36_months_t_per_month']], on='country_name', how='left')

# Summary sentence using latest month from stats file
if not df_stats_filtered.empty:
    summary_agg = df_stats_filtered.copy()
    summary_agg_row = summary_agg[[
        emissions_column_latest,
        emissions_column_prev,
        'month_yoy_change'
    ]].sum().to_dict()


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

st.markdown(
    f"<div style='font-size: 1.1em; line-height: 1.6em; margin: 16px 0;'>"
    f"In {latest_month}, {selected_scope}{sector_text}{subsector_text} emissions were "
    f"<span style='font-weight: bold; font-style: italic; text-decoration: underline;'>{emissions_value:,.0f}</span> tCO₂e. "
    f"This represents {mom_text} compared to the previous month. This also represents {yoy_text} compared to the same month last year."
    f"</div>",
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
st.subheader(f"Emissions Over Time (tCO2e) - {selected_scope} | {selected_subsector_label}")

query_country = f"""
    WITH latest_month AS (
        SELECT MAX(MAKE_DATE(year, month, 1)) AS max_date
        FROM '{country_subsector_totals_path}'
        WHERE gas = 'co2e_100yr'
          AND country_name IS NOT NULL
    )
    SELECT 
        MAKE_DATE(year, month, 1) AS year_month,
        SUM(emissions_quantity) AS country_emissions_quantity
    FROM '{country_subsector_totals_path}', latest_month
    WHERE gas = 'co2e_100yr'
      AND country_name IS NOT NULL
      AND MAKE_DATE(year, month, 1) >= (max_date - INTERVAL '36' MONTH)
      {'AND subsector = \'%s\'' % selected_subsector_raw if selected_subsector_raw else ''}
      {'AND sector = \'%s\'' % selected_sector_raw if selected_sector_raw else ''}
      {f"AND {region_condition['column_name']} = '{region_condition['column_value']}'" if region_condition else ''}
    GROUP BY year_month
    ORDER BY year_month
"""

country_df = con.execute(query_country).df()
if not country_df.empty:
    country_df['year_month'] = pd.to_datetime(country_df['year_month'])

if not monthly_df.empty or not country_df.empty:
    fig_emissions = px.line()
    fig_emissions.update_layout(showlegend=True)
    if not monthly_df.empty:
        fig_emissions.add_scatter(x=monthly_df['year_month'], y=monthly_df['emissions_quantity'], mode='lines+markers', name='Assets')
    else:
        # Add empty trace to preserve Assets legend
        fig_emissions.add_scatter(x=country_df['year_month'], y=[None] * len(country_df), mode='lines', name='Assets')

    if not country_df.empty:
        fig_emissions.add_scatter(x=country_df['year_month'], y=country_df['country_emissions_quantity'], mode='lines+markers', name='Total', line=dict(color='#E9967A'))
    st.plotly_chart(fig_emissions, use_container_width=True)
else:
    st.markdown(
        """
        <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
            <h4>No asset-level data for this subsector</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2 = st.columns(2)
with col1:
    st.subheader("Activity Over Time")
    fig_placeholder1 = st.empty()
    if selected_subsector_raw and not monthly_df.empty:
        fig_activity = px.line(
            monthly_df, x='year_month', y='activity', markers=True,
            title='Activity'
        )
        fig_placeholder1.plotly_chart(fig_activity, use_container_width=True)
    elif not selected_subsector_raw:
        fig_placeholder1.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>Select a Subsector to see Activity Over Time</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        fig_placeholder1.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>No asset-level data for this subsector</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.subheader("Emissions Factor Over Time")
    fig_placeholder2 = st.empty()
    if selected_subsector_raw and not monthly_df.empty:
        fig_ef = px.line(
            monthly_df, x='year_month', y='mean_emissions_factor', markers=True,
            title='Emission Factor (tCO2e / unit activity)'
        )
        fig_placeholder2.plotly_chart(fig_ef, use_container_width=True)
    elif not selected_subsector_raw:
        fig_placeholder2.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>Select a Subsector to see Emission Factor Over Time</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        fig_placeholder2.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>No asset-level data for this subsector</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

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
    .applymap(color_change, subset=['mom_change', rename_map['emissions_slope_36_months_t_per_month']])
)

st.dataframe(styled_df, use_container_width=True, height=450)


# the code chunk to the end is for subsector stacked bar chart/
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Annual Emissions by Sector")

# Optional name normalization for country dropdown
country_name_map = {
    "United States of America": "United States",
    #"Russia": "Russian Federation"
}
raw_country = country_name_map.get(selected_scope, selected_scope)

# Start building the query
query = f"""
    SELECT 
        year,
        sector,
        SUM(emissions_quantity) AS emissions_quantity
    FROM '{country_subsector_totals_path}'
    WHERE gas = 'co2e_100yr'
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
    query += f" AND subsector = '{selected_subsector_raw}'"

# Finalize the query
query += " GROUP BY year, sector ORDER BY year, sector"

# Run the query
df_annual = con.execute(query).df()
df_annual["year"] = df_annual["year"].astype(str)

# Plot the results
if not df_annual.empty:

    df_totals = df_annual.groupby("year", as_index=False).agg(
        total_emissions=("emissions_quantity", "sum")
    )

    # Ensure year is treated as categorical
    df_annual["year"] = df_annual["year"].astype(str)

    # Base stacked bar chart with dynamic Y scaling
    fig_annual = px.bar(
        df_annual,
        x="year",
        y="emissions_quantity",
        color="sector",
        labels={
            "emissions_quantity": "Emissions (tCO₂e)",
            "year": "Year",
            "sector": "Sector"
        }
    )

    fig_annual.update_layout(
        barmode="stack",
        xaxis=dict(type="category"),
        legend_title="Sector",
        margin=dict(t=50, b=30)
    )

    fig_annual.add_trace(
    go.Bar(
        x=df_totals["year"],
        y=[0] * len(df_totals),  # Invisible bars
        text=[format_number_short(v) for v in df_totals["total_emissions"]],
        textposition="outside",
        marker=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
        name="Total"
    )
)

    st.plotly_chart(fig_annual, use_container_width=True)

else:
    st.markdown(
        """
        <div style='border: 1px solid #ccc; height: 300px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
            <h4>No data available for the selected filters</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# loading dataframes into excel
if not monthly_df.empty or not country_df.empty or not df_stats_filtered.empty or not df_annual.empty:
    # Create dictionary of DataFrames to export
    dfs_for_excel = {
        "Country Total Emissions": country_df,
        "Asset Total Emissions": monthly_df,
        "Stats Data": df_stats_filtered,
        "Annual Sector Emissions": df_annual
    }

    # Use the utility function to create the Excel file
    excel_file = create_excel_file(dfs_for_excel)

    # Fill in the placeholder with the actual download button
    download_placeholder.download_button(
        label="Download Data",
        data=excel_file,
        file_name="climate_trace_dashboard_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="The downloaded data will represent your dropdown selections."
    )

con.close()
