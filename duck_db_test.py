import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import base64
from utils.utils import format_dropdown_options, map_region_condition

st.set_page_config(layout="wide")

# Load logo
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")

# Connect to DuckDB and load subsectors
parquet_path = "data/asset_parquet/asset_emissions_most_granular.parquet"
con = duckdb.connect()

release_version = con.execute(f"SELECT DISTINCT release FROM '{parquet_path}'").fetchone()[0]

# UI Header
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
        <h1 style="margin: 0;">Climate TRACE Emissions Dashboard</h1>
    </div>
    <p style="margin-top: 5px; font-size: 1em; font-style: italic; color: white;">
        The data in this dashboard is from Climate TRACE <span style='color: red;'><strong>Release {release_version}</strong></span>
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("<br><br>", unsafe_allow_html=True)

# Build scope dropdown
region_options = [
    'Global',
    'EU', 'OECD', 'Non-OECD',
    'UNFCCC Annex', 'UNFCCC Non-Annex',
    'Global North', 'Global South',
    'Developed Markets', 'Emerging Markets',
    'Africa', 'Antarctica', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'
]

unique_countries = sorted([
    row[0] for row in con.execute(f"SELECT DISTINCT country_name FROM '{parquet_path}' WHERE country_name IS NOT NULL").fetchall()
])
scope_options = region_options + unique_countries

# Subsector options
raw_subsectors = con.execute(f"""
    SELECT DISTINCT original_inventory_sector
    FROM '{parquet_path}'
    WHERE gas = 'co2e_100yr'
""").fetchall()
raw_subsectors = sorted([row[0] for row in raw_subsectors if row[0]])

subsector_labels, subsector_map = format_dropdown_options(raw_subsectors)
subsector_labels.insert(0, "All")
subsector_map["All"] = None

# Sidebar filters
col1, col2 = st.columns(2)
with col1:
    selected_scope = st.selectbox("Scope", scope_options)
    region_condition = map_region_condition(selected_scope)
with col2:
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

st.markdown("<br><br>", unsafe_allow_html=True)

# Load and prepare statistics CSV
df_stats = pd.read_csv("data/statistics/country_subsector_emissions_statistics_202504.csv")
emissions_columns = [col for col in df_stats.columns if col.startswith("emissions_quantity_")]
emissions_columns_sorted = sorted(emissions_columns, reverse=True)
emissions_column_latest = emissions_columns_sorted[0]
emissions_column_prev = emissions_columns_sorted[1]

# Build query to DuckDB
where_clauses = ["gas = 'co2e_100yr'"]
if selected_subsector_raw:
    where_clauses.append(f"original_inventory_sector = '{selected_subsector_raw}'")
if region_condition:
    value = region_condition['column_value']
    value_str = f"'{value}'" if isinstance(value, str) else str(value).upper()
    where_clauses.append(f"{region_condition['column_name']} = {value_str}")

query = f"""
    SELECT 
        strftime(start_time, '%Y-%m') AS year_month,
        SUM(activity) AS activity,
        SUM(emissions_quantity) AS emissions_quantity
    FROM '{parquet_path}'
    WHERE {' AND '.join(where_clauses)}
    GROUP BY year_month
    ORDER BY year_month
"""

monthly_df = con.execute(query).df()
monthly_df["year_month"] = pd.to_datetime(monthly_df["year_month"])
monthly_df["mean_emissions_factor"] = monthly_df["emissions_quantity"] / monthly_df["activity"]

# Stats table logic
df_stats_filtered = df_stats[df_stats['gas'] == 'co2e_100yr']
df_stats_filtered = df_stats_filtered[df_stats_filtered['country_name'].notna()]

country_name_map = {
    'United States of America': 'United States',
    'Russian Federation': 'Russia',
}
df_stats_filtered['country_name'] = df_stats_filtered['country_name'].replace(country_name_map)

if selected_scope != 'Global':
    region_cond = map_region_condition(selected_scope)
    if region_cond:
        df_stats_filtered = df_stats_filtered[df_stats_filtered[region_cond['column_name']] == region_cond['column_value']]
    else:
        df_stats_filtered = df_stats_filtered[df_stats_filtered['country_name'] == selected_scope]

if selected_subsector_raw:
    df_stats_filtered = df_stats_filtered[df_stats_filtered['subsector'] == selected_subsector_raw]
else:
    if 'emissions_slope_36_months_t_per_month' in df_stats_filtered.columns:
        df_stats_filtered['slope_times_emissions'] = (
            df_stats_filtered['emissions_slope_36_months_t_per_month'] * df_stats_filtered[emissions_column_latest]
        )
    agg_dict = {
        emissions_column_latest: 'sum',
        emissions_column_prev: 'sum',
        'mom_change': 'sum',
        'month_yoy_change': 'sum'
    }
    df_stats_agg = df_stats_filtered.groupby('country_name').agg(agg_dict).reset_index()
    df_stats_agg['mom_percent_change'] = (df_stats_agg['mom_change'] / df_stats_agg[emissions_column_prev]) * 100
    df_stats_agg['month_yoy_percent_change'] = (
        df_stats_agg['month_yoy_change'] / (df_stats_agg[emissions_column_latest] - df_stats_agg['month_yoy_change'])
    ) * 100
    if 'slope_times_emissions' in df_stats_filtered.columns:
        slopes = df_stats_filtered.groupby('country_name').agg({
            'slope_times_emissions': 'sum',
            emissions_column_latest: 'sum'
        }).reset_index()
        slopes['emissions_slope_36_months_t_per_month'] = (
            slopes['slope_times_emissions'] / slopes[emissions_column_latest]
        )
        df_stats_agg = df_stats_agg.merge(
            slopes[['country_name', 'emissions_slope_36_months_t_per_month']],
            on='country_name',
            how='left'
        )
    df_stats_filtered = df_stats_agg

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

# Plotting
st.subheader(f"Emissions Over Time - {selected_scope} | {selected_subsector_label}")
fig_emissions = px.line(
    monthly_df,
    x='year_month',
    y='emissions_quantity',
    markers=True,
    title='Emissions Quantity (tCO2e)'
)
st.plotly_chart(fig_emissions, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Activity Over Time")
    fig_placeholder1 = st.empty()
    if selected_subsector_raw:
        fig_activity = px.line(
            monthly_df,
            x='year_month',
            y='activity',
            markers=True,
            title='Shipping Activity'
        )
        fig_placeholder1.plotly_chart(fig_activity, use_container_width=True)
    else:
        fig_placeholder1.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>Select a Subsector to see Activity Over Time</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.subheader("Emissions Factor Over Time")
    fig_placeholder2 = st.empty()
    if selected_subsector_raw:
        fig_ef = px.line(
            monthly_df,
            x='year_month',
            y='mean_emissions_factor',
            markers=True,
            title='Emission Factor (tCO2e / unit activity)'
        )
        fig_placeholder2.plotly_chart(fig_ef, use_container_width=True)
    else:
        fig_placeholder2.markdown(
            """
            <div style='border: 1px solid #ccc; height: 400px; opacity: 0.5; display: flex; align-items: center; justify-content: center;'>
                <h4>Select a Subsector to see Emission Factor Over Time</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('**Top Movers by Absolute MoM Change (Scrollable Table)**')

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

con.close()
