import streamlit as st
import base64
import duckdb

from config import CONFIG
from utils.utils import get_release_version
#from tabs.tab01_emissions_reduction_tab import show_emissions_reduction_plan
#from tabs.tab02_abatement_curve_tab import show_abatement_curve
from tabs.tab06_reduction_heatmap import show_reduction_heatmap

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    /* Remove default top padding */
    .block-container {
        padding-top: 1rem; /* adjust this number (default ~6rem) */
    }
    /* Hide the sidebar completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    /* Hide the sidebar collapse/expand arrow */
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stPageLink-NavLink"],
    [data-testid="stPageLink-NavLink"] span,
    [data-testid="stPageLink-NavLink"] p {
        color: #ff4b4b !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# load CT logo
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.markdown("<br>", unsafe_allow_html=True)

st.page_link("pages/0_Home.py", label="⬅️ Back to Home")


logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")

asset_path = CONFIG['asset_emissions_country_subsector_path']

con = duckdb.connect()

st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
            <h1 style="margin: 0; font-size: 2.8em;">Climate TRACE Heat Map (Beta)</h1>
        </div>
        <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
            The data in this dashboard is from Climate TRACE release <span style='color: red;'><strong>{get_release_version(con, asset_path)}</strong></span> (excluding forestry), covering 740 million assets globally.
        </p>
        <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
            This web application is for the internal use of Climate TRACE and its partners only. The data displayed may be revised, updated, rearranged, or deleted without prior communication to users, and is not warranted to be error free.
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


tab1, = st.tabs(["Heat Map"])
with tab1:
    show_reduction_heatmap()

# with tab2:
#     show_abatement_curve()

# with tab3:
#     show_reduction_heatmap()

