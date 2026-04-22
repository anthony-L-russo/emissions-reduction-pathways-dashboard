import streamlit as st
import base64

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# Tag added to avoid search engines
st.markdown("""
<meta name="robots" content="noindex,nofollow">
""", unsafe_allow_html=True)

# --- Style cleanup ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        height: 100vh !important;
    }

    .tool-card {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: inherit;
        height: 340px;
        width: 100%;
        box-sizing: border-box;
    }
    .tool-card:hover {
        border-color: #ff4b4b;
        transform: translateY(-4px);
    }
    .tool-icon {
        font-size: 5rem;
    }
    .tool-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 6px;
        color: #1f77ff;
        line-height: 1.2;
    }
    .tool-desc {
        font-size: 0.78rem;
        color: #bbb;
        margin-top: 5px;
        line-height: 1.3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Load logo ----
def get_base64_logo():
    with open("Climate TRACE Logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_logo()

# ---- Header ----
st.markdown(
    f"""
    <div style='display: flex; align-items: center; justify-content: center;'>
        <img src="data:image/png;base64,{logo_base64}" width="70" style="margin-right: 15px;" />
        <h1 style="margin: 0; font-size: 2.8em;">Climate TRACE Sandbox</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("<br>", unsafe_allow_html=True)

# ---- 3-column grid: top row ----
top1, top2, top3 = st.columns(3, gap="medium")

with top1:
    st.markdown(
        """
        <a href="/Sector_Reduction_Pathways" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">🧭</div>
                <div class="tool-title">Sector Reduction Pathways</div>
                <div class="tool-desc">Top-down reduction opportunities by sector and region.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with top2:
    st.markdown(
        """
        <a href="/Abatement_Curve" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">📉</div>
                <div class="tool-title">Abatement Curve</div>
                <div class="tool-desc">Bottom-up reduction potential comparing ERS by difficulty and scale.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with top3:
    st.markdown(
        """
        <a href="/Heat_Map" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">🌡️</div>
                <div class="tool-title">Heat Map</div>
                <div class="tool-desc">Geographic hotspots of reduction potential across sectors and regions.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---- 3-column grid: bottom row ----
bot1, bot2, bot3 = st.columns(3, gap="medium")

with bot1:
    st.markdown(
        """
        <a href="/Global_Trends" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">🌍</div>
                <div class="tool-title">Global Trends</div>
                <div class="tool-desc">Explore emissions change drivers by geography and sector.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with bot2:
    st.markdown(
        """
        <a href="/Ownership" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">🗂️</div>
                <div class="tool-title">Ownership</div>
                <div class="tool-desc">Identify owners and evaluate their asset portfolios.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with bot3:
    st.markdown(
        """
        <a href="/Country_Trends" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card">
                <div class="tool-icon">🌐</div>
                <div class="tool-title">Country Trends</div>
                <div class="tool-desc">Analyze a country's emissions profile and see how it compares globally.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# ---- Footer ----
# st.markdown(
#     """
#     <div style='text-align: center; margin-top: 3rem;'>
#         <a href='https://climatetrace.org' target='_blank'
#            style='text-decoration: none; font-size: 1.5rem; color: #1f77ff;'>
#            🌍 Visit the <span style='font-weight:600;'>Climate TRACE</span> website!
#         </a>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
