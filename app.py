import streamlit as st
import base64

st.set_page_config(layout="wide")

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
        padding: 25px;
        text-align: center;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
        display: block;
        color: inherit;
        width: 90%;
        margin: 0 auto;
    }
    .tool-card:hover {
        border-color: #ff4b4b;
        transform: translateY(-4px);
    }
    .tool-icon {
        font-size: 2.5rem;
    }
    .tool-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 10px;
        color: #1f77ff;
    }
    .tool-desc {
        font-size: 1rem;
        color: #bbb;
        margin-top: 8px;
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
st.markdown("<br>", unsafe_allow_html=True)
#st.markdown("<br>", unsafe_allow_html=True)




# ---- 2-column layout ----
col1, col2 = st.columns(2)


with col1:
    st.markdown(
    """
        <a href="/Sector_Reduction_Pathways" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card"
                style="min-height: 200px; padding: 35px 25px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div class="tool-icon" style="font-size: 6rem;">🧭</div>
                <div class="tool-title">Sector Reduction Pathways</div>
                <div class="tool-desc">View top-down reduction opportunities by sector and region to identify the largest impacts.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
        <a href="/Heat_Map" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card"
                style="min-height: 200px; padding: 35px 25px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div class="tool-icon" style="font-size: 6rem;">🌡️</div>
                <div class="tool-title">Heat Map</div>
                <div class="tool-desc">Pinpoint geographic hotspots of reduction potential across sectors and regions.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
        <a href="/Ownership" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card"
                style="min-height: 200px; padding: 35px 25px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div class="tool-icon" style="font-size: 6rem;">🗂️</div>
                <div class="tool-title">Ownership</div>
                <div class="tool-desc">Identify owners and evaluate their asset portfolios.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
    

with col2:

    st.markdown(
    """
        <a href="/Abatement_Curve" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card"
                style="min-height: 200px; padding: 35px 25px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div class="tool-icon" style="font-size: 6rem;">📉</div>
                <div class="tool-title">Abatement Curve</div>
                <div class="tool-desc">View bottom-up reduction potential to compare ERS by difficulty and scale.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
        <a href="/Monthly_Trends" target="_self" style="text-decoration: none; display: block;">
            <div class="tool-card"
                style="min-height: 200px; padding: 35px 25px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div class="tool-icon" style="font-size: 6rem;">📊</div>
                <div class="tool-title">Monthly Trends</div>
                <div class="tool-desc">Track month-over-month emissions patterns and sector activity worldwide.</div>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

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
