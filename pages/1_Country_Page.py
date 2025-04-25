import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv('/Users/anthonyrusso/Documents/statistics-visualization-tool/data/country_subsector.csv' \
'')

# filter only to co2e_100yr for consistency
df = df[
    (df["gas"] == "co2e_100yr") &
    (df["country_name"].notnull())
].copy()

# filter for electricity-gen, international & domestic shipping
target_subsectors = ["electricity-generation", "international-shipping", "domestic-shipping"]

st.title("📊 Monthly Emissions Statistics Dashboard")
st.caption("Tracking changes in emissions for electricity generation and shipping subsectors (Jan 2025)")

# build insights per subsector
for subsector in target_subsectors:
    st.subheader(f"🔍 {subsector.title().replace('-', ' ')}")

    # Filter for this subsector
    sub_df = df[df["subsector"] == subsector].copy()

    # estimate prior month’s emissions? is the month yoy comparing january TY to january LY?
    sub_df["monthly_emission_change"] = sub_df["mom_change"]
    sub_df["abs_change"] = sub_df["monthly_emission_change"].abs()

    # calculate overall total change MoM for the respective sector
    total_change = sub_df["monthly_emission_change"].sum()
    total_slope = sub_df["emissions_slope_36_months_t_per_month"].sum()

    # summarize the total change and the 36 month slope (average monthly emission change past 3 years)
    st.markdown(f"**Total Emission Change (Jan 2025 vs. prior month):** `{total_change:,.0f}` tCO₂e")
    st.markdown(f"**Total Trend Slope (3-year):** `{total_slope:,.2f}` tCO₂e/month")

    # top 5 contributors to monthly change
    st.markdown("**Top 5 Contributors to Monthly Change:**")

    top_movers = sub_df.sort_values("abs_change", ascending=False).head(5)

    st.table(top_movers[[
        "country_name",
        "emissions_quantity_202501",
        "monthly_emission_change",
        "emissions_slope_36_months_t_per_month"
    ]].rename(columns={
        "country_name": "Country",
        "emissions_quantity_202501": "Jan 2025 Emissions (tCO₂e)",
        "monthly_emission_change": "Change from Dec 2024 (tCO₂e)",
        "emissions_slope_36_months_t_per_month": "3-Yr Trend (tCO₂e/month)"
    }).style.format({
        "Jan 2025 Emissions (tCO₂e)": "{:,.0f}",
        "Change from Dec 2024 (tCO₂e)": "{:,.0f}",
        "3-Yr Trend (tCO₂e/month)": "{:,.0f}"
    }))

    st.markdown("---")
