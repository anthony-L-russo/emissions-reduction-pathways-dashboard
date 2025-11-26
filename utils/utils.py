import pandas as pd
import io
import streamlit as st
import urllib.parse
import html
import numpy as np
import math
import plotly.graph_objects as go
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
def map_region_condition(region_selection, country_map=None):
    
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
        },
        'G20': {
            'column_name': 'g20',
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
                'column_name': 'iso3_country', 
                'column_value': country_map.get(region_selection)
        }
    
def relabel_regions(df):
    dict_relabel = {
        'unfccc_annex': {True: 'Annex1', False: 'Non-Annex1'},
        'em_finance': {True: 'Emerging Markets', False: 'Developed Markets'},
        'developed_un': {True: 'Global North', False: 'Global South'},
        'continent': {'Unlisted': 'Unknown/Unlisted'}
    }
    for col, mapping in dict_relabel.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df 
    

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

def bordered_metric_abatement(
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

    # Build style dynamically
    base_style = (
        "flex-grow: 1; "
        "font-size: 2em; font-weight: bold; padding: 0 4px; "
        "display: flex; flex-direction: column; align-items: center; justify-content: center; "
        "line-height: 0.7;"
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
                <div style="text-align: center; white-space: normal; overflow-wrap: anywhere;">
                    {display_val}
                </div>
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
    fpath = 'percentile_moer/asset_moer_2023.parquet'
    df_moer = pd.read_parquet(fpath)
    df_moer['ef_moer'] = df_moer['moer_avg']*0.4536/1000 #Convert lbs to tons

    df["asset_id"] = df["asset_id"].astype(str)
    df_moer["asset_id"] = df_moer["asset_id"].astype(str)


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
    st.session_state["city_selector_RO"] = "-- Select City --"
    st.session_state.needs_recompute_RO = True

def reset_state_and_county():
    st.session_state["state_province_selector_RO"] = "-- Select State / Province --"
    st.session_state["county_district_selector_RO"] = "-- Select County / District --"
    st.session_state.needs_recompute_RO = True


abatement_subsector_options = {
    'agriculture': [
        'crop-residues',
        'cropland-fires',
        'enteric-fermentation-cattle-operation',
        'enteric-fermentation-cattle-pasture',
        'enteric-fermentation-other',
        'manure-applied-to-soils',
        'manure-left-on-pasture-cattle',
        'manure-management-cattle-operation',
        'manure-management-other',
        'other-agricultural-soil-emissions',
        'rice-cultivation',
        'synthetic-fertilizer-application',
    ],
    'buildings': [
        'non-residential-onsite-fuel-usage',
        'residential-onsite-fuel-usage',
        'other-onsite-fuel-usage'
    ],
    'forestry-and-land-use': [
        'forest-land-clearing',
        'forest-land-degradation',
        'forest-land-fires',
        'net-forest-land',
        'net-shrubgrass',
        'net-wetland',
        'removals',
        'shrubgrass-fires',
        'water-reservoirs',
        'wetland-fires',
    ],
    'fluorinated-gases': [
        'fluorinated-gases'
    ],
    'fossil-fuel-operations': [
        'coal-mining',
        'oil-and-gas-production',
        'oil-and-gas-refining',
        'oil-and-gas-transport',
        'other-fossil-fuel-operations',
        'other-solid-fuels'
    ],
    'manufacturing': [
        'iron-and-steel',
        'aluminum',
        'cement',
        'chemicals',
        'food-beverage-tobacco',
        'glass',
        'lime',
        'other-chemicals',
        'other-manufacturing',
        'other-metals',
        'petrochemical-steam-cracking',
        'pulp-and-paper',
        'textiles-leather-apparel',
        'wood-and-wood-products'
    ],
    'mineral-extraction': [
        'bauxite-mining',
        'copper-mining',
        'iron-mining',
        'sand-quarrying',
        'rock-quarrying',
        'other-mining-quarrying'
    ],
    'power': [
        'electricity-generation',
        'heat-plants',
        'other-energy-use'
    ],
    'transportation': [
        'domestic-aviation',
        'domestic-shipping',
        'international-aviation',
        'international-shipping',
        'non-broadcasting-vessels',
        'railways',
        'road-transportation',
        'other-transport'
    ],
    'waste': [
        'biological-treatment-of-solid-waste-and-biogenic',
        'domestic-wastewater-treatment-and-discharge',
        'incineration-and-open-burning-of-waste',
        'industrial-wastewater-treatment-and-discharge',
        'solid-waste-disposal',
    ],
}

def define_color_lines(metric):

    # dictionary for asset colors
    dict_color = {}
    dict_color['unfccc_annex'] = {
        'Annex1': '#407076',
        'Non-Annex1': '#FBBA1A'
    }
    dict_color['em_finance'] = {
        'Developed Markets': '#407076',
        'Emerging Markets': '#FBBA1A'
    }
    dict_color['developed_un'] = {
        'Global North': '#407076',
        'Global South': '#FBBA1A'
    }
    dict_color['asset_type'] = {
        'Smelting': '#407076',
        'Refinery': '#FBBA1A'
    }
    dict_color['continent'] = {
        'Europe': '#4878A8',
        'North America': '#6D4DA8',
        'Asia': '#FBBA1A',
        'Africa': '#C75B39',
        'South America': '#4C956C',
        'Oceania': '#91643A',
        'Antarctica': '#D3D3D3',
        'Unknown/Unlisted': '#9E9C9C'
    }
    dict_color['sector'] = {
        'forestry-and-land-use': '#779608',
        'manufacturing': '#9554FF',
        'fossil-fuel-operations': '#FF6F42',
        'waste': '#BBD421',
        'transportation': '#FBBA14',
        'agriculture':  '#E8516C',
        'buildings':  '#03A0E3',
        'fluorinated-gases': '#B6B4B4',
        'mineral-extraction': '#4380F5',
        'power': '#56979F'
    }
    dict_color['background'] = {
        'background0': '#EBE6E6',
        'background1': '#D9D4D4',
        'background2': '#556063',
        'background3': '#444546',
    }

    if metric == 'emissions_factor':
        outlier_values = {
        'solid-waste-disposal': {'more landfills above': 1},
        'food-beverage-tobacco': {'more facilities above': 0.0000814059958583646}}
        dict_lines = {
            subsector: outlier_values.get(subsector, {})
            for subsector, sublist in abatement_subsector_options.items()
            for subsector in sublist}      
    else:
        outlier_values = {}
        dict_lines = {
            subsector: outlier_values.get(subsector, {})
            for subsector, sublist in abatement_subsector_options.items()
            for subsector in sublist}
        
    return dict_color, dict_lines


def plot_abatement_curve(gdf_asset, selected_group, selected_color, dict_color, dict_lines, selected_list, selected_assets, selected_x, selected_y, threshold, fill=False, cond={}):

    def weighted_avg(group, x_col, y_col, weight_x=0.5, weight_y=0.5):
        def min_max_normalize(series):
            return (series - series.min()) / (series.max() - series.min())
        # Normalize x and y within each group
        x_norm = min_max_normalize(group[x_col])
        y_norm = min_max_normalize(group[y_col])

        # Invert y so lower original y means higher transformed score
        y_transformed = 1 - y_norm

        # Compute weighted average composite score for the group (return scalar)
        composite_scores = weight_x * x_norm + weight_y * y_transformed
        return composite_scores.mean()

    def hex_to_rgba(hex_color, opacity=0.3):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {opacity})'

    def bucket_and_aggregate(df, group_col, group_2_col, total_col, bucket_col, asset_id_col, asset_name_col, sum_cols=None, first_cols=None, max_buckets=1000):
        sum_cols = sum_cols or []
        first_cols = first_cols or []
        result = []

        for group_vals, group_df in df.groupby(group_col):
                group_sorted = group_df.sort_values(by=total_col)
                n = len(group_sorted)
                bucket_size = max(1, n // max_buckets)

                group_sorted['bucket'] = (np.arange(n) // bucket_size)

                agg_dict = {
                    **{f"{col}": (col, "sum") for col in sum_cols},
                    **{f"{col}": (col, "first") for col in first_cols},
                    f"{total_col}": (total_col, "max"),
                    f"{bucket_col}_min": (bucket_col, "min"),
                    f"{bucket_col}_max": (bucket_col, "max"),
                    f"{bucket_col}": (bucket_col, "max"),
                    f"{group_2_col}": (group_2_col, "unique")
                }
                agg_df = group_sorted.groupby('bucket').agg(**agg_dict).reset_index()

                asset_count = group_sorted.groupby('bucket')[asset_id_col].count().reset_index(drop=True)
                agg_df[asset_id_col] = asset_count

                agg_df[asset_name_col] = agg_df.apply(
                    lambda row: f"assets with {bucket_col} from {row[f'{bucket_col}_min']:.2f} to {row[f'{bucket_col}_max']:.2f}",
                    axis=1
                )

                if isinstance(group_col, list):
                    for col_name, val in zip(group_col, group_vals if isinstance(group_vals, tuple) else (group_vals,)):
                        agg_df[col_name] = val
                else:
                    agg_df[group_col] = group_vals

                result.append(agg_df)

        final_df = pd.concat(result).reset_index(drop=True)
        return final_df

    # clean df
    df = gdf_asset.copy()
    df['asset_value'] = 1
    df['net_reduction_potential'] = df['net_reduction_potential'].fillna(0)
    df[selected_y] = pd.to_numeric(df[selected_y], errors='coerce') 
    df[selected_color] = df[selected_color].apply(lambda x: False if pd.isna(x)==True else x)
    df['color'] = df[selected_color].map(dict_color[selected_color])

    num_sectors = df['subsector'].nunique()

    # threshold
    if threshold == '':
        threshold = df[selected_y].max() + 1
    else:
        threshold = float(threshold)

    # set up conditions
    cond0 = {
        'label': True,
        'label_distance': 0.003,
        'label_distance_scalar': 20,
        'label_limit': 0.2,
        'sort_order': [False,True]
    }
    for k,v in cond0.items():
        if k not in cond: cond[k] = v

    # change values based on x-axis
    if selected_x == 'count':
        selected_x = 'asset_value'
        if selected_group == 'asset':
            x_axis_title = 'Number of Assets'
        elif selected_group == 'country':
            x_axis_title = 'Number of Country-Sectors'
        elif selected_group == 'strategy_name':
            x_axis_title = 'Number of ERS strategies'
    elif selected_x == 'emissions_quantity':
        x_axis_title = 'Total Emissions (t of CO2e)'
    elif selected_x == 'net_reduction_potential':
        x_axis_title = 'Emissions Reduction Potential (t of CO2e)'
    elif selected_x == 'activity':
        x_axis_title = f'Activity ({df['activity_units'].iloc[-1]})'

    # change values based on y-axis
    if selected_y == 'emissions_quantity':
        y_axis_title = 'Total Emissions (t of CO2e)'
        ascending_order = False
    elif selected_y == 'net_reduction_potential':
        y_axis_title = 'Emissions Reduction Potential (t of CO2e)'
        ascending_order = False
    elif selected_y == 'emissions_factor':
        y_axis_title = 'Emissions Factor (t of CO2e / Activity)'
        ascending_order = False
    elif selected_y == 'asset_difficulty_score':
        y_axis_title = 'Difficulty Score (1-10)'
        ascending_order = True

    # update chart based off selected_group - cumulative sum activity
    if selected_group == 'country':
        df = df.groupby(
            [selected_list, 'iso3_country', 'country_name', 'continent', 'eu', 'oecd', 
             'unfccc_annex', 'developed_un', 'em_finance', 'sector', 'subsector', 'color']).agg(
                 activity=('activity', 'sum'),
                 emissions_quantity=('emissions_quantity', 'sum'),
                 net_reduction_potential=('net_reduction_potential', 'sum'),
                 asset_value=('asset_value', 'first'),
                 emissions_factor=('emissions_factor', 'median'),
                 asset_difficulty_score=('asset_difficulty_score', 'median')).reset_index()
        df['emissions_factor'] = np.where(df['activity'].isna(), df['emissions_factor'], df['emissions_quantity'] / df['activity'])                                                                                  
        
    if selected_group == 'strategy_name':
        df = df.groupby(
            [selected_list, 'sector', 'subsector', 'strategy_name', 'color']).agg(
                activity=('activity', 'sum'),
                emissions_quantity=('emissions_quantity', 'sum'),
                net_reduction_potential=('net_reduction_potential', 'sum'),
                asset_value=('asset_value', 'first'),
                emissions_factor=('emissions_factor', 'median'),
                asset_difficulty_score=('asset_difficulty_score', 'median')).reset_index()
        df['emissions_factor'] = np.where(df['activity'].isna(), df['emissions_factor'], df['emissions_quantity'] / df['activity'])

    # apply sector weights
    sector_weighted_scores = df.groupby('sector').apply(weighted_avg, x_col=selected_x, y_col=selected_y)
    df['sector'] = pd.Categorical(df['sector'], categories=sector_weighted_scores.index, ordered=True)
    # sort data by sector
    df = df.sort_values(['sector', selected_y], ascending=[True, ascending_order]).reset_index(drop=True)
    # find cumulative values, separate positive + negative values
    df['cum_pos'] = df[selected_x].where(df[selected_x] > 0, 0).cumsum().fillna(0)
    df['cum_neg'] = df[selected_x].where(df[selected_x] < 0, 0)[::-1].cumsum()[::-1]
    last_neg_cum = df.loc[df[selected_x]<0, 'cum_neg'].iloc[-1] if (df[selected_x]<0).any() else 0
    df.loc[df[selected_x] >= 0, 'cum_neg'] = last_neg_cum
    df['value_cum'] = df['cum_pos'] + df['cum_neg']
    df['value_cum'] = df['value_cum'].fillna(0)

    # add new row
    new_row = {}
    first_row = df.iloc[0]
    for col in df.columns:
        if col == selected_color:
            new_row[col] = first_row[selected_color]
        elif col == selected_list: 
            new_row[col] = ""
        elif col == 'sector': 
            new_row[col] = first_row['sector'] 
        elif pd.api.types.is_numeric_dtype(df[col]):
            new_row[col] = 0
        else:
            new_row[col] = first_row[col]
    df = pd.concat([pd.DataFrame([new_row], columns=df.columns), df], ignore_index=True)

    # create a selected_df based on highlighted assets
    selected_df = df[df[selected_list].isin(selected_assets)].copy()

    if selected_group == 'asset':

        # limit number of assets plotted
        if num_sectors > 3 or len(df) > 5000:
            subset_df = bucket_and_aggregate(df, 'sector', 'subsector', 'value_cum', selected_y, 'asset_id', 'asset_name', ['emissions_quantity', 'net_reduction_potential'], ['color'])
            subset_df['asset_type'] = 'N/A'
            subset_df['country_name'] = 'Aggregated'
            asset_id_txt = 'Total Assets:'
        else:
            subset_df = df.copy()
            asset_id_txt = 'Asset ID:'

        # set up formatting
        hover_id = 'asset_id'
        hover_name = 'asset_name'
        #asset highlights
        highlight_hover_text = [
            (
                f"{subsector}<br>"
                f"{country}<br>"
                f"<i>{asset_id_txt} {hover_val}</i><br>"
                f"{hover_val2}<br>"
                f"Asset Type: {type}<br>"
                f"{selected_y}: {yval}<br><br>"
                f"Emissions: {emissions:,.0f}<br>"
                f"Reduction: {reduction:,.0f}"
            )
            for subsector, country, hover_val, hover_val2, type, yval, emissions, reduction in zip(
                selected_df['subsector'],
                selected_df['country_name'],
                selected_df[hover_id],
                selected_df[hover_name],
                selected_df['asset_type'],
                selected_df[selected_y],
                selected_df['emissions_quantity'],
                selected_df['net_reduction_potential']
            )
        ]

    elif selected_group == 'country':
        # set up formatting
        hover_id = 'iso3_country'
        hover_name = 'country_name'
        #asset highlights
        highlight_hover_text = [
            (
                f"{subsector}<br>"
                f"<i>{hover_val}</i><br>"
                f"{hover_val2}<br>"
                f"{selected_y}: {yval}<br><br>"
                f"Emissions: {emissions:,.0f}<br>"
                f"Reduction: {reduction:,.0f}"
            )
            for subsector, hover_val, hover_val2,  yval, emissions, reduction in zip(
                selected_df['subsector'],
                selected_df[hover_id],
                selected_df[hover_name],
                selected_df[selected_y],
                selected_df['emissions_quantity'],
                selected_df['net_reduction_potential']
            )
        ]
        subset_df = df.copy()

    elif selected_group == 'strategy_name':
        # set up formatting
        hover_id = 'subsector'
        hover_name = 'strategy_name'
        #asset highlights
        highlight_hover_text = [
            (
                f"{sector}<br>"
                f"<i>{hover_val}</i><br>"
                f"{hover_val2}<br>"
                f"{selected_y}: {yval}<br><br>"
                f"Emissions: {emissions:,.0f}<br>"
                f"Reduction: {reduction:,.0f}"
            )
            for sector, hover_val, hover_val2, yval, emissions, reduction in zip(
                selected_df['sector'],
                selected_df[hover_id],
                selected_df[hover_name],
                selected_df[selected_y],
                selected_df['emissions_quantity'],
                selected_df['net_reduction_potential']
            )
        ]
        subset_df = df.copy()

    # create the fig
    fig = go.Figure()

    # calculate metrics for formatting
    x_min, x_max = min(df['value_cum']), max(df['value_cum'])
    y_min = df[selected_y].min()
    y_max = df[selected_y].max()
    y_offset = (y_max) * 0.01
    y_range_quantile = 0.99
    
    for i in range(1, len(subset_df)):
        if selected_group == 'asset':
            hover_text = (
                f"{subset_df['subsector'][i]}<br>"
                f"{subset_df['country_name'][i]}<br>"
                f"<i>{asset_id_txt} {subset_df[hover_id][i]}</i><br>"
                f"{subset_df[hover_name][i]}</i><br>"
                f"Asset Type: {subset_df['asset_type'][i]}</i><br>"
                f"{selected_y}: {round(subset_df[selected_y][i], 2)}</i><br><br>"
                f"Total Emissions: {round(subset_df['emissions_quantity'][i]):,.0f}<br>"
                f"Total Reductions: {round(subset_df['net_reduction_potential'][i]):,.0f}"
            )
        elif selected_group == 'country':
            hover_text = (
                f"{subset_df['subsector'][i]}<br>"
                f"{subset_df[hover_id][i]}<br>"
                f"{subset_df[hover_name][i]}<br>"
                f"{selected_y}: {round(subset_df[selected_y][i], 2)}</i><br><br>"
                f"Emissions: {round(subset_df['emissions_quantity'][i]):,.0f}<br>"
                f"Reduction: {round(subset_df['net_reduction_potential'][i]):,.0f}"
            )
        elif selected_group == 'strategy_name':
            hover_text = (
                f"{subset_df['sector'][i]}<br>"
                f"{subset_df[hover_id][i]}<br>"
                f"{subset_df[hover_name][i]}<br>"
                f"{selected_y}: {round(subset_df[selected_y][i], 2)}</i><br><br>"
                f"Emissions: {round(subset_df['emissions_quantity'][i]):,.0f}<br>"
                f"Reduction: {round(subset_df['net_reduction_potential'][i]):,.0f}"
            )

        color_value = subset_df['color'][i] 
        y_vals = [subset_df[selected_y].iloc[i-1], subset_df[selected_y].iloc[i]]
        if all(y <= threshold for y in y_vals):
            fill_col = hex_to_rgba(color_value, 0.9)
        else:
            if fill:
                fill_col = hex_to_rgba(color_value, 0.9)
            else:
                fill_col = 'rgba(0,0,0,0)'
        

        # if subset_df['sector'].iloc[i] != subset_df['sector'].iloc[i-1]:
        #     continue

        fig.add_trace(go.Scatter(
            x=[subset_df['value_cum'].iloc[i-1], subset_df['value_cum'].iloc[i]],
            y=[subset_df[selected_y].iloc[i], subset_df[selected_y].iloc[i]],
            fill='tozeroy',
            fillcolor=fill_col,
            line=dict(color=f'{color_value}', width=4),
            mode='lines',
            name=f'{subset_df[hover_name][i]}',
            line_shape='linear',
            legendgroup=f'{color_value}',
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text,
            hoverlabel=dict(
                bgcolor='white',
                font=dict(color=color_value, size=14))))

    #add line for threshold if needed
    if threshold != (df[selected_y].max() + 1):
        fig.add_hline(
            y=threshold,
            line=dict(color='gray', width=2, dash='dot'))
    
    fig.add_trace(go.Scatter(
        x=selected_df['value_cum'] - selected_df[selected_x] / 2,
        y=selected_df[selected_y],
        mode='markers',
        marker=dict(size=8, color='#A94442', symbol='diamond'),
        name="Selected Assets",
        hoverinfo='text',
        hovertext=highlight_hover_text,
        hoverlabel=dict(
            bgcolor="white",
            font=dict(color='#A94442', size=14))))
    
    fig.add_trace(go.Scatter(
        x=selected_df['value_cum'] - selected_df[selected_x] / 2,
        y=selected_df[selected_y] + y_offset,
        mode='text',
        hoverinfo=None,
        text=selected_df[hover_name],
        textposition="top right",
        textfont=dict(size=14, color="#A94442"),
        showlegend=False))
    
    # fig.update_yaxes(showgrid=False, zeroline=False)
    
    # add custom legend items
    for color_label, color_value in dict_color[selected_color].items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color_value, size=10),
            name=f'{color_label}', 
            showlegend=True
        ))

    # format plot layout
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        showlegend=True,
        legend=dict(x=1.08, y=0.97, xanchor='right', yanchor='top'),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=20, b=50),
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor='lightgrey',
            gridcolor='lightgrey',
            range=[min(df['value_cum']), math.ceil(max(df['value_cum']))*1.1]),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgrey',
            gridcolor='lightgrey',
            range=[min(df[selected_y]), df[selected_y].quantile(y_range_quantile) * 1.05]),
        height=700
    
    )

    # add line to the plot if overflowing information
    for line_name, line_y in dict_lines['outlier'].items():
        fig.add_shape(
            type='line',
            x0=x_min, x1=x_max,
            y0=line_y, y1=line_y,
            line=dict(color='#444546', width=1, dash='dash'),
        )
        ax_y_max = max(df[selected_y])
        text_y = line_y
        if line_y + 0.015 * ax_y_max > ax_y_max:
            text_y = line_y - 0.015 * ax_y_max
        elif line_y - 0.015 * ax_y_max < 0:
            text_y = line_y + 0.015 * ax_y_max
        fig.add_annotation(
            x=x_max + 1, 
            y=text_y,
            text=line_name,
            showarrow=False,
            font=dict(size=12, color='#444546'),
            align='left',
            xanchor='left',
            yanchor='middle',
            xref='x',
            yref='y'
        )

    # create csv to download the data
    if selected_group == 'asset':
        df_csv = df.iloc[1:][['asset_id', 'asset_name', 'asset_type', 'iso3_country', 'country_name', selected_color, 'subsector', selected_x, selected_y]].sort_values(['subsector', selected_y]).rename(columns={'net_reduction_potential': 'reduction_potential'}).to_csv(index=False).encode('utf-8')
    if selected_group == 'country':
        df_csv = df.iloc[1:][['iso3_country', 'country_name', selected_color, 'subsector', selected_x, selected_y]].sort_values(['subsector', selected_y]).rename(columns={'net_reduction_potential': 'reduction_potential'}).to_csv(index=False).encode('utf-8')
    if selected_group == 'strategy_name':
        df_csv = df.iloc[1:][['sector', 'subsector', 'strategy_name', selected_x, selected_y]].sort_values(['subsector', selected_y]).rename(columns={'net_reduction_potential': 'reduction_potential'}).to_csv(index=False).encode('utf-8')
    return fig, df_csv

def make_asset_url(row):
    url_root = 'https://climatetrace.org/explore/#'
    admin_value = f"{row['asset_name']}"
    params = {
        'admin': admin_value,
        'gas': 'co2e',
        'year': '2024',
        'timeframe': '100',
        'sector': '',
        'asset': str(row['asset_id'])
    }
    encoded = urllib.parse.urlencode(params, safe=':,._()-')
    full_url = f"{url_root}{encoded}"
    return full_url

def make_country_url(row):
    url_root = 'https://climatetrace.org/explore/#'
    admin_value = f"{row['country_name']} ({row['iso3_country']}):1:{row['iso3_country']}:country"
    params = {
        'admin': admin_value,
        'gas': 'co2e',
        'year': '2024',
        'timeframe': '100',
        'sector': '',
        'asset': ''
    }
    encoded = urllib.parse.urlencode(params, safe=':,._()-')
    full_url = f"{url_root}{encoded}"

    return full_url

def make_state_url(row):
    url_root = 'https://climatetrace.org/explore/#'
    if pd.isna(row['gadm_1']):
        return np.nan
    admin_value = f"{row['gadm_1_name']}-- {row['iso3_country']}:{row['gid_1']}:{row['gadm_1']}:state"
    params = {
        'admin': admin_value,
        'gas': 'co2e',
        'year': '2024',
        'timeframe': '100',
        'sector': '',
        'asset': ''
    }
    encoded = urllib.parse.urlencode(params, safe=':,._()-')
    full_url = f"{url_root}{encoded}"
    return full_url

def make_county_url(row):
    url_root = 'https://climatetrace.org/explore/#'
    if pd.isna(row['gadm_2']):
        return np.nan
    admin_value = f"{row['gadm_2_name']}-- {row['gadm_1_name']}-- {row['iso3_country']}:{row['gid_2']}:{row['gadm_2']}:county"
    params = {
        'admin': admin_value,
        'gas': 'co2e',
        'year': '2024',
        'timeframe': '100',
        'sector': '',
        'asset': ''
    }
    encoded = urllib.parse.urlencode(params, safe=':,._()-')
    full_url = f"{url_root}{encoded}"
    return full_url

def return_sector_type(sector):
    if sector in ['fossil-fuel-operations', 'manufacturing', 'mineral-extraction', 'power', 'waste']:
        sector_type = 'asset'
    elif sector in ['agriculture', 'buildings', 'fluorinated-gases', 'forestry-and-land-use', 'transportation']:
        sector_type = 'raster'
    return sector_type



def get_reduction_induction_json(df_stacked_bar, df_induced):
    def safe_format_number(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "0"
        return format_number_short(x)

    summary = []

    for _, row in df_stacked_bar.iterrows():
        sector = row["sector"]

        # gather inductions
        sector_inductions = df_induced[df_induced["receiving_sector"] == sector]
        inductions_list = []
        for _, ind in sector_inductions.iterrows():
            if ind["induced_emissions"] is not None and not math.isnan(ind["induced_emissions"]):
                inductions_list.append({
                    "inducing_sector": ind["inducing_sector"],
                    "induced_emissions": ind["induced_emissions"],
                    "formatted": safe_format_number(ind["induced_emissions"])
                })

        # core values
        asset_reductions = row.get("emissions_reduced_at_asset", 0) or 0
        reduction_potential = row.get("emissions_reduction_potential", 0) or 0
        static_emissions = row.get("static_emissions_q", 0) or 0

        asset_reductions_fmt = safe_format_number(asset_reductions)
        reduction_potential_fmt = safe_format_number(reduction_potential)
        static_emissions_fmt = safe_format_number(static_emissions)

        # total inductions
        total_inductions = sum(
            ind["induced_emissions"] for ind in inductions_list if ind["induced_emissions"] is not None
        )
        total_inductions_fmt = safe_format_number(total_inductions)
        total_inductions_color = "red" if total_inductions >= 0 else "green"

        # build hover text lines
        hover_lines = [f"<b>{sector}</b>"]

        # Asset reductions (moved down with space)
        hover_lines.append("&nbsp;")
        hover_lines.append(
            #f"<span style='color:green; font-size:13px; font-weight:bold'>{asset_reductions_fmt}</span> "
            f"<span style='color:green; font-size:13px; font-weight:bold'>&nbsp;{asset_reductions_fmt}</span>&nbsp;&nbsp;&nbsp;<span style='font-size:13px; font-weight:bold'>Asset Reductions</span>"
        )

        # blank line before inductions
        hover_lines.append("&nbsp;")

        # Inductions
        if inductions_list:
            hover_lines.append(
                f"<span style='color:{total_inductions_color}; font-size:13px; font-weight:bold'>{total_inductions_fmt}</span> "
                f"<span style='font-size:13px; font-weight:bold'>&nbsp;&nbsp;Net Inductions:</span>"
            )
            for ind in inductions_list:
                val = ind["induced_emissions"]
                if val is None or math.isnan(val):
                    continue
                if val >= 0:
                    hover_lines.append(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<span style='color:red'>{ind['formatted']}</span>) <i>{ind['inducing_sector']}</i>"
                    )
                else:
                    hover_lines.append(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(<span style='color:green'>{ind['formatted']}</span>) <i>{ind['inducing_sector']}</i>"
                    )
        else:
            hover_lines.append(
                f"<span style='color:green; font-size:13px; font-weight:bold'>&nbsp;0&nbsp;&nbsp;</span> "
                f"<span style='font-size:13px; font-weight:bold'>Net Inductions</span>"
            )

        # formula separator
        hover_lines.append("────────────────────")

        # Net consequential reductions (bigger, bold, first thing your eyes see)
        hover_lines.append(
            f"<span style='color:green; font-size:14px; font-weight:bold'>✅ {reduction_potential_fmt} Net Reduction Opportunity</span>"
        )

        # join with <br>
        hover_text = "<br>".join(hover_lines)

        summary.append({
            "sector": sector,
            "asset_reductions": asset_reductions,
            "asset_reductions_formatted": asset_reductions_fmt,
            "reduction_potential": reduction_potential,
            "reduction_potential_formatted": reduction_potential_fmt,
            "static_emissions": static_emissions,
            "static_emissions_formatted": static_emissions_fmt,
            "inductions": inductions_list,
            "hover_text": hover_text
        })

    return summary

def format_emissions(value):
    if value >= 1_000_000_000:
        scaled_value = value / 1_000_000_000
        return f"{scaled_value:,.1f} BtCO\u2082e"
    elif value >= 1_000_000:
        scaled_value = value / 1_000_000
        return f"{scaled_value:,.1f} MtCO\u2082e" 
    else:
        return f"{value:,.0f} tCO\u2082e" 

def get_consequetial_hover_text(df_induced):

    # Format values up front
    df_induced["formatted_asset_reductions"] = df_induced["emissions_reduced_at_asset"].apply(
        lambda x: format_number_short(x) if pd.notnull(x) else None
    )
    df_induced["formatted_reduction_potential"] = df_induced["emissions_reduction_potential"].apply(
        lambda x: format_number_short(x) if pd.notnull(x) else None
    )
    df_induced["formatted_induced_emissions"] = df_induced["induced_emissions"].apply(
        lambda x: format_number_short(x) if pd.notnull(x) else None
    )

    # Build induction lists grouped by receiving sector
    inductions_grouped = (
        df_induced.dropna(subset=["inducing_sector", "induced_emissions"])
        .groupby("receiving_sector")
        .apply(lambda g: [
            {
                "inducing_sector": row["inducing_sector"],
                "induced_emissions": row["formatted_induced_emissions"],
            }
            for _, row in g.iterrows()
        ])
        .to_dict()
    )

    # Attach inductions safely (fallback to empty list)
    df_summary = (
        df_induced.drop_duplicates(subset=["sector"])
        .assign(inductions=lambda d: d["sector"].map(inductions_grouped).apply(lambda x: x if isinstance(x, list) else []))
    )

    hover_texts = []
    for _, row in df_induced.iterrows():
        parts = []
        
        # Asset reductions
        if pd.notnull(row["formatted_asset_reductions"]) and row["formatted_asset_reductions"] not in ["0", None]:
            parts.append(f"+ Asset Reductions: {row['formatted_asset_reductions']} tCO₂e")
        
        # Inductions
        sector_inductions = df_induced[
            (df_induced["receiving_sector"] == row["sector"]) & 
            (df_induced["induced_emissions"].notnull())
        ]
        for _, ind in sector_inductions.iterrows():
            if ind["formatted_induced_emissions"] not in ["0", None]:
                parts.append(f"+ {ind['inducing_sector']}: {ind['formatted_induced_emissions']} tCO₂e")
        
        # Combine into hover text
        formula = "<br>".join(parts)
        text = (
            f"<b>{row['sector']}</b><br>"
            f"{formula}<br>"
            f"= <b style='color:green'>{row['formatted_reduction_potential']} tCO₂e</b>"
        )
        hover_texts.append(text)

    return hover_texts

# ------------- THIS IS TO ISOLATE TABS FROM RECOMPUTE -------------
def mark_ro_recompute():
    st.session_state.needs_recompute_reduction_opportunities = True

def mark_ac_recompute():
    st.session_state.needs_recompute_abatement_curve = True
    if not st.session_state["selected_region_AC"]:
        st.session_state["selected_region_AC"] = ["Global"]
    st.session_state["selected_assets"] = []

def mark_mt_recompute():
    st.session_state.needs_recompute_monthly_trends = True