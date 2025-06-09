import pandas as pd
import io
import streamlit as st
import html
# import duckdb
# from config import CONFIG

def format_dropdown_options(raw_values, lowercase_words=None):
    if lowercase_words is None:
        lowercase_words = {"and"}

    def format_label(value):
        words = value.replace("-", " ").split()
        return " ".join([
            word.capitalize() if word.lower() not in lowercase_words else word.lower()
            for word in words
        ])

    # generate labels and check for duplicates
    seen = {}
    options = []
    mapping = {}
    for raw in raw_values:
        label = format_label(raw)
        if label in seen:
            label += f" ({raw})"  # Ensure uniqueness
        seen[label] = True
        options.append(label)
        mapping[label] = raw

    return options, mapping


# function finds us the correct column and value to use as a condition based on the dropdown selection
def map_region_condition(region_selection):
    
    list_of_continents = ['Africa',
                          'Antarctica',
                          'Asia',
                          'Europe',
                          'North America',
                          'Oceania',
                          'South America']
    
    region_mapping = {
        'EU': {
            'column_name': 'eu',
            'column_value': True
        },
        'OECD': {
            'column_name': 'oecd',
            'column_value': True
        },
        'Non-OECD': {
            'column_name': 'oecd',
            'column_value': False
        },
        'UNFCCC Annex': {
            'column_name': 'unfccc_annex',
            'column_value': True
        },
        'UNFCCC Non-Annex': {
            'column_name': 'unfccc_annex',
            'column_value': False
        },
        'Global North': {
            'column_name': 'developed_un',
            'column_value': True
        },
        'Global South': {
            'column_name': 'developed_un',
            'column_value': False
        },
        'Developed Markets': {
            'column_name': 'em_finance',
            'column_value': False
        },
        'Emerging Markets': {
            'column_name': 'em_finance',
            'column_value': True
        }
    }

    if region_selection == 'Global':
        return None
    
    elif region_selection in list_of_continents:
        return {
            'column_name': 'continent',
            'column_value': region_selection
        }
    
    elif region_selection in region_mapping:
        return region_mapping[region_selection]

    else:
        return {
            'column_name': 'country_name',
            'column_value': region_selection
        }
    

def format_number_short(n):
    if abs(n) >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1e3:.0f}K"
    else:
        return f"{n:.0f}"
    

def create_excel_file(dataframes_dict):
    
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output

def get_release_version(con, path):
    
    release_version = con.execute(f"SELECT DISTINCT release FROM '{path}'").fetchone()[0]
    con.close()

    return release_version



def bordered_metric(
    label, 
    value, 
    tooltip_enabled=False, 
    total_options_in_scope=None, 
    tooltip_value=None, 
    value_color=None,
    font_size="2em"
):
    # Format the display value
    if isinstance(value, list):
        if total_options_in_scope and len(value) == total_options_in_scope:
            display_val = f"All ({len(value)})"
            tooltip = ", ".join(value)
        else:
            tooltip = ", ".join(value)
            total_char_len = sum(len(v) for v in value)
            if total_char_len > 19:
                display_val = value[0] + f" +{len(value) - 1} more"
            else:
                display_val = ", ".join(value[:2])
                if len(value) > 2:
                    display_val += f" +{len(value) - 2} more"
    else:
        display_val = str(value)
        tooltip = tooltip_value if tooltip_value else display_val

    # Escape all text to avoid breaking markup
    # display_val = html.escape(display_val)
    tooltip = html.escape(tooltip)
    label = html.escape(label)

    # Build style dynamically
    base_style = (
        "flex-grow: 1; display: flex; align-items: center; justify-content: center;"
        f"font-size: {font_size}; font-weight: bold; text-align: center; padding: 0 4px;"
    )
    if value_color:
        base_style += f" color: {value_color};"

    card_html = f"""
        <div style="
            border: 1px solid #999;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            min-height: 160px;
            display: flex;
            flex-direction: column;
        ">
            <div style="
                font-weight: 600;
                text-align: left;
                margin-bottom: -18px;
                margin-top: -4px;
                padding: 0;
            ">
                {label}
            </div>
            <div style="{base_style}">
                {display_val}
            </div>
        </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)