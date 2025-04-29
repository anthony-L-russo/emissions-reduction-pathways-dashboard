import streamlit as st
import pandas as pd

 
df = pd.read_csv('data/country_monthly/month_country_subsector.csv')

# agg to the global-month level
global_df = df.groupby('year_month').agg({
    'activity': 'sum',
    'mean_emissions_factor': 'mean',
    'emissions_quantity': 'sum'
}).reset_index()



