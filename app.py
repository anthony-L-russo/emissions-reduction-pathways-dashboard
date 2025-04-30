import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px
import base64


def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.set_page_config(layout="wide")

# Load the monthly shipping emissions dataset
# This contains emissions data for each country-subsector-month
df = pd.read_csv('data/country_monthly/month_country_subsector.csv')
df['year_month'] = pd.to_datetime(df['year_month'])  # Convert to datetime for plotting

df['year'] = df['year_month'].dt.year  # Extract year for potential grouping
countries = df['country_name'].dropna().unique()  # Unique list of countries for dropdown

# Load the statistics file (includes month over month change, slope, etc.)
df_stats = pd.read_csv('data/statistics/country_subsector_emissions_statistics_202504.csv')

# Dynamically find the most recent emissions quantity column from statistics
emissions_columns = [col for col in df_stats.columns if col.startswith('emissions_quantity_')]
emissions_columns_sorted = sorted(emissions_columns, reverse=True)
emissions_column_latest = emissions_columns_sorted[0]  # Most recent month
emissions_column_prev = emissions_columns_sorted[1]    # Previous month

# Streamlit page config and title


# Logo and Title side by side
logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
        <h1 style="margin: 0;">Climate TRACE Emissions Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# adding some space between title and toggle/dropdowns
st.markdown("<br><br>", unsafe_allow_html=True)


# Sidebar selectors for global/country and subsector filter
col1, col2 = st.columns(2)
with col1:
    scope = st.radio("View Scope", options=['Global', 'Country'], horizontal=True)

with col2:
    subsector = st.selectbox("Select Subsector", options=['All', 'Domestic Shipping', 'International Shipping'])

# Mapping UI label to internal field value in dataset
subsector_mapping = {
    'All': None,
    'Domestic Shipping': 'domestic-shipping',
    'International Shipping': 'international-shipping',
    'Electricity Generation': 'electricity-generation'
}
internal_subsector = subsector_mapping[subsector]

# Country selector if "Country" scope is selected
if scope == 'Country':
    # st.subheader('Select a Country')
    country_options = ["-- All Countries --"] + sorted(countries)
    selected_country = st.selectbox('Select a Country', options=country_options)
    if selected_country == "-- All Countries --":
        selected_country = None
else:
    selected_country = None

# adding some space for visual appeal
st.markdown("<br><br>", unsafe_allow_html=True)

# Get the latest month in the monthly emissions dataset (not statistics)
df_for_latest_month = df[df['gas'] == 'co2e_100yr']
df_for_latest_month = df_for_latest_month[df_for_latest_month['country_name'].notna()]
if internal_subsector:
    df_for_latest_month = df_for_latest_month[df_for_latest_month['original_inventory_sector'] == internal_subsector]
latest_month = df_for_latest_month['year_month'].max()
latest_month_str = latest_month.strftime("%B %Y")

# Filter the statistics dataset for shipping subsectors and non-null countries
df_stats_filtered = df_stats[df_stats['gas'] == 'co2e_100yr']
df_stats_filtered = df_stats_filtered[df_stats_filtered['country_name'].notna()]
df_stats_filtered = df_stats_filtered[df_stats_filtered['subsector'].isin(['domestic-shipping', 'international-shipping'])]
if internal_subsector:
    df_stats_filtered = df_stats_filtered[df_stats_filtered['subsector'] == internal_subsector]

# If viewing "All" subsectors, aggregate metrics by country and recalculate percent change
if subsector == 'All':
    if 'emissions_slope_36_months_t_per_month' in df_stats_filtered.columns:
        df_stats_filtered['slope_times_emissions'] = df_stats_filtered['emissions_slope_36_months_t_per_month'] * df_stats_filtered[emissions_column_latest]

    # Sum emissions, previous emissions, and mom_change; we'll recalculate percentages manually
    agg_dict = {
        emissions_column_latest: 'sum',
        emissions_column_prev: 'sum',
        'mom_change': 'sum',
        'month_yoy_change': 'sum'
    }

    df_stats_agg = df_stats_filtered.groupby('country_name').agg(agg_dict).reset_index()

    # Recalculate percent changes properly and convert to percentage units (multiply by 100)
    df_stats_agg['mom_percent_change'] = (df_stats_agg['mom_change'] / df_stats_agg[emissions_column_prev]) * 100
    df_stats_agg['month_yoy_percent_change'] = (df_stats_agg['month_yoy_change'] / (df_stats_agg[emissions_column_latest] - df_stats_agg['month_yoy_change'])) * 100

    # Normalize weighted slope back to avg slope
    if 'slope_times_emissions' in df_stats_filtered.columns:
        slopes = df_stats_filtered.groupby('country_name').agg({
            'slope_times_emissions': 'sum',
            emissions_column_latest: 'sum'
        }).reset_index()
        slopes['emissions_slope_36_months_t_per_month'] = slopes['slope_times_emissions'] / slopes[emissions_column_latest]
        df_stats_agg = df_stats_agg.merge(slopes[['country_name', 'emissions_slope_36_months_t_per_month']], on='country_name', how='left')

    df_stats_filtered = df_stats_agg
else:
    # for specific subsector, use the raw columns, no aggregation or recalculation
    df_stats_filtered = df_stats_filtered[[
        'country_name',
        emissions_column_latest,
        'mom_change',
        'mom_percent_change',
        'month_yoy_percent_change',
        'emissions_slope_36_months_t_per_month'
    ]]

# Display Top Movers Table (Country Scope only)
if scope == 'Country':
    st.markdown('**Top 10 Movers (by Absolute MoM Change)**')

    df_stats_filtered['abs_mom_change'] = df_stats_filtered['mom_change'].abs()
    top_emitters_df = (
        df_stats_filtered
        .sort_values(by='abs_mom_change', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Keep consistent display order for table
    display_cols = ['country_name', emissions_column_latest, 'mom_change', 'mom_percent_change', 'month_yoy_percent_change', 'emissions_slope_36_months_t_per_month']
    rename_map = {emissions_column_latest: 'emissions_quantity'}

    def color_change(val):
        color = 'green' if val < 0 else 'red'
        return f'color: {color}'

    st.dataframe(
        top_emitters_df[display_cols].rename(columns=rename_map).style.format({
            'emissions_quantity': "{:,.0f}",
            'mom_change': "{:,.0f}",
            'mom_percent_change': "{:.1f}%",
            'month_yoy_percent_change': "{:.1f}%",
            'emissions_slope_36_months_t_per_month': "{:,.0f}"
        }).applymap(color_change, subset=['mom_change','emissions_slope_36_months_t_per_month'])
    )

# Build data for time series plots (monthly aggregation)
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered['gas'] == 'co2e_100yr']
df_filtered = df_filtered[df_filtered['original_inventory_sector'].isin(['domestic-shipping', 'international-shipping'])]
if internal_subsector:
    df_filtered = df_filtered[df_filtered['original_inventory_sector'] == internal_subsector]
if scope == 'Country' and selected_country:
    df_filtered = df_filtered[df_filtered['country_name'] == selected_country]

# Monthly time series aggregation
total_monthly_df = df_filtered.groupby('year_month').agg({
    'activity': 'sum',
    'mean_emissions_factor': 'mean',
    'emissions_quantity': 'sum'
}).reset_index()

# Emissions trend plot
st.subheader(f"Emissions Over Time - {selected_country or 'Global'} | {subsector}")
fig_emissions = px.line(
    total_monthly_df,
    x='year_month',
    y='emissions_quantity',
    markers=True,
    title='Emissions Quantity (tCO2e)'
)
st.plotly_chart(fig_emissions, use_container_width=True)

# Activity and Emission Factor side-by-side charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Activity Over Time")
    fig_activity = px.line(
        total_monthly_df,
        x='year_month',
        y='activity',
        markers=True,
        title='Shipping Activity'
    )
    st.plotly_chart(fig_activity, use_container_width=True)

with col2:
    st.subheader("Emissions Factor Over Time")
    fig_ef = px.line(
        total_monthly_df,
        x='year_month',
        y='mean_emissions_factor',
        markers=True,
        title='Emission Factor (tCO2e / unit activity)'
    )
    st.plotly_chart(fig_ef, use_container_width=True)
