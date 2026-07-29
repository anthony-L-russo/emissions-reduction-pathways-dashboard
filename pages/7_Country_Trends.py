import streamlit as st
import base64
import duckdb

from config import CONFIG
from utils.utils import get_release_version
from tabs.tab08_country_trends import show_country_trends

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stPageLink-NavLink"],
    [data-testid="stPageLink-NavLink"] span,
    [data-testid="stPageLink-NavLink"] p { color: #ff4b4b !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


st.markdown("<br>", unsafe_allow_html=True)

st.page_link("pages/0_Home.py", label="⬅️ Back to Home")

logo_base64 = get_base64_of_bin_file("Climate TRACE Logo.png")
asset_path = CONFIG['asset_emissions_country_subsector_path']
con = duckdb.connect()

st.markdown(
    f"""
    <div style='display: flex; align-items: center;'>
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;" />
        <h1 style="margin: 0; font-size: 2.8em;">Climate TRACE Country Trends (Beta)</h1>
    </div>
    <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
        The data in this dashboard is from Climate TRACE release
        <span style='color: red;'><strong>{get_release_version(con, asset_path)}</strong></span>,
        covering 740 million assets globally (forestry configurable below).
    </p>
    <p style="margin-top: 2px; font-size: 1em; font-style: italic;">
        This web application is for the internal use of Climate TRACE and its partners only.
        The data displayed may be revised, updated, rearranged, or deleted without prior communication
        to users, and is not warranted to be error free.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

tab1, = st.tabs(["Country Trends"])
with tab1:
    show_country_trends()
