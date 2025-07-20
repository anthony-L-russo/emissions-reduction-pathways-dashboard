import streamlit as st
import base64
import duckdb

from config import CONFIG
from utils.utils import get_release_version
from tabs.monthly_dashboard_tab import show_monthly_dashboard
from tabs.emissions_reduction_tab import show_emissions_reduction_plan


st.set_page_config(layout="wide")

# load CT logo
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")

asset_path = CONFIG['asset_path']

con = duckdb.connect()

st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
            <h1 style="margin: 0; font-size: 2.8em;">Climate TRACE Benchmarking</h1>
        </div>
        <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
            The data in this dashboard is from Climate TRACE release <span style='color: red;'><strong>{get_release_version(con, asset_path)}</strong></span> (excluding forestry), covering 660 million assets globally.
        </p>
        <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
            This web application is for the internal use of Climate TRACE and its partners only. The data displayed may be revised, updated, rearranged, or deleted without prior communication to users, and is not warranted to be error free.
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Benchmarking", "Monthly Dashboard"])

with tab1:
    show_emissions_reduction_plan()
    # pass
with tab2:
    show_monthly_dashboard()

