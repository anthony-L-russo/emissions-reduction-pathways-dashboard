import pandas as pd
import io
import streamlit as st
import html
import numpy as np
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
    
    elif region_selection in ["United States", "United States of America"]:
        return {
            'column_name': 'country_name',
            'column_value': ["United States", "United States of America"]
    }

    else:
        return {
            'column_name': 'country_name',
            'column_value': region_selection
        }
    

def format_number_short(n):
    if abs(n) >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1e6:.1f}M"
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
    value_color=None
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
    display_val = html.escape(display_val)
    tooltip = html.escape(tooltip)
    label = html.escape(label)

    # Build style dynamically
    base_style = (
        "flex-grow: 1; display: flex; align-items: center; justify-content: center; "
        "font-size: 2em; font-weight: bold; text-align: center; padding: 0 4px;"
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


def map_percentile_col(selected_percentile):

    percentile_dict = {
        # "0th": "percentile_0",
        "10th": "percentile_avg_0_to_10",
        "20th": "percentile_avg_10_to_20",
        "30th": "percentile_avg_20_to_30",
        "40th": "percentile_avg_30_to_40",
        "50th": "percentile_avg_40_to_50",
        "60th": "percentile_avg_50_to_60",
        "70th": "percentile_avg_60_to_70",
        "80th": "percentile_avg_70_to_80",
        "90th": "percentile_avg_80_to_90",
        "100th": "percentile_avg_90_to_100"
    }

    return percentile_dict[selected_percentile]


def data_add_moer(df, cond={}):
    """
    Adding MOER data to the analysis (temporary: monthly 2023 only)
    df:  straightly from ct.get_data_asset()
    """

    cond0 = {'moer': False}
    for k,v in cond0.items():
        if k not in cond:
            cond[k] = v

    #Get MOER data (ask Zoheyr)  --  *** ADD codes to check every asset is covered ***
    fpath = 'data/static/asset_moer_2023.parquet'
    df_moer = pd.read_parquet(fpath)
    df_moer['ef_moer'] = df_moer['moer_avg']*0.4536/1000 #Convert lbs to tons

    #Map MOER to assets using asset_id
    df = pd.merge(
        df,
        df_moer[['asset_id', 'original_inventory_sector', 'ef_moer']],
        left_on=['asset_id', 'subsector'],
        right_on=['asset_id', 'original_inventory_sector'],
        how='left'
    )  

    df = df.drop(columns=['original_inventory_sector'])

    #Customize MOER data based on subsectors:
    df[['other1', 'other2', 'other3', 'other4', 'other5', 'other7', 'other9']] = df[['other1', 'other2', 'other3', 'other4', 'other5', 'other7', 'other9']].apply(pd.to_numeric, errors='coerce')

    df['eq_12'] = df['emissions_quantity']
    df['ef_12'] = df['average_emissions_factor']
    df['eq_12_moer'] = np.nan
    df['ef_12_moer'] = np.nan

    if cond['moer']:
        for sec in df.subsector.unique():
            mask_sec = df['subsector']==sec
            df_sec = df.loc[mask_sec, :].copy()

            if sec == 'electricity-generation':
                mask = df_sec['asset_type']=='biomass'
                df_sec.loc[mask,'eq_12_moer'] = df_sec.loc[mask,'other4']
                df_sec.loc[~mask,'eq_12_moer'] = df_sec.loc[~mask,'activity']*(df_sec.loc[~mask,'other7'].fillna(df_sec.loc[~mask,'average_emissions_factor']))            

                df_sec['ef_12_moer'] = df_sec['eq_12_moer']/df_sec['activity']
                
            elif sec == 'iron-and-steel':
                df_sec['eq_12'] = df_sec['other2']
                df_sec['ef_12'] = df_sec['other1']

                # df_sec['eq_12_moer'] = df_sec['other2'] + df_sec['activity']*df_sec['other3']*(df_sec['other7']-df_sec['other5'])
                df_sec['eq_12_moer'] = df_sec['other2']
                df_sec['ef_12_moer'] = df_sec['eq_12_moer']/df_sec['activity']

            # elif sec == 'aluminum':
            #     df_sec['eq_12'] = df_sec['other2']
            #     df_sec['ef_12'] = df_sec['other1']

            #     df_sec['eq_12_moer'] = df_sec['other2'] + df_sec['activity'] * df_sec['other3'] * ((df_sec['ef_moer'].fillna(df_sec['other5'])) - df_sec['other5'])
            #     df_sec['ef_12_moer'] = df_sec['eq_12_moer']/df_sec['activity']

            elif sec == 'cement':
                df_sec['eq_12'] = df_sec['other2']
                df_sec['ef_12'] = df_sec['other1']

                df_sec['eq_12_moer'] = df_sec['other2'] + df_sec['activity'] * df_sec['other7'] * ((df_sec['ef_moer'].fillna(df_sec['other9'])) - df_sec['other9'])
                df_sec['ef_12_moer'] = df_sec['eq_12_moer']/df_sec['activity']

            # elif sec == 'road-transportation':
            #     df_sec['ef_12_moer'] = df_sec['ef_moer']*35/1.6093/100/1000  #35 refers to distrance travelled to kWh based on EPA study among average car (high is 48, low is 24)
            #     df_sec['eq_12_moer'] = df_sec['activity']*df_sec['ef_12_moer'] #

            df.loc[mask_sec, ['eq_12','ef_12','eq_12_moer','ef_12_moer']] = df_sec[['eq_12','ef_12','eq_12_moer','ef_12_moer']]
    return df

def is_country(region_selection):
    if region_selection in [
        'Global',
        'Africa',
        'Antarctica',
        'Asia',
        'Europe',
        'North America',
        'Oceania',
        'South America',
        'EU',
        'OECD',
        'Non-OECD',
        'UNFCCC Annex',
        'UNFCCC Non-Annex',
        'Global North',
        'Global South',
        'Developed Markets',
        'Emerging Markets'
    ]:
        return False
    
    else:
        return True
    
def reset_city():
    st.session_state["city_selector"] = "-- Select City --"

def reset_state_and_county():
    st.session_state["state_province_selector"] = "-- Select State / Province --"
    st.session_state["county_district_selector"] = "-- Select County / District --"