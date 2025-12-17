import streamlit as st
import duckdb
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import plotly.express as px
from config import CONFIG
from utils.utils import *
from utils.queries import *


def show_ownership_module():
    st.markdown(
        """
        <style>
        /* Hide the sidebar completely */
        section[data-testid="stSidebar"] {
            display: none;
        }
        /* Hide the sidebar collapse/expand arrow */
        [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    ##### SET UP -------
    con = duckdb.connect()
    annual_asset_path = CONFIG['annual_asset_path']
    gadm_0_path = CONFIG['gadm_0_path']
    ownership_path = CONFIG['asset_ownership_path']
    
    sector_color_map, sector_line_map = define_color_lines('sector')

    ##### IMPORT DATA -------
    
    # import ownership + emissions data
    query_df_ownership = get_ownership_sql(annual_asset_path, ownership_path)
    df_ownership = con.execute(query_df_ownership).df()
    query_ct_emissions = get_gadm_emissions_sql(gadm_0_path)
    df_gadm_emissions = con.execute(query_ct_emissions).df()

    # calculate country-level and gadm-level emissions factors
    df_gadm_emissions = df_gadm_emissions[(df_gadm_emissions['activity'].notna()) & (df_gadm_emissions['emissions_quantity'].notna())]
    df_global_emissions = df_gadm_emissions.groupby(['subsector']).agg(activity=('activity', 'sum'), emissions_quantity=('emissions_quantity', 'sum')).reset_index()
    df_global_emissions['ef_global'] = np.where(df_global_emissions['activity'] != 0, df_global_emissions['emissions_quantity'] / df_global_emissions['activity'], np.nan)
    df_global_emissions['ef_global'] = np.where(df_global_emissions['ef_global'] == 0, np.nan, df_global_emissions['ef_global'])
    df_global_emissions = df_global_emissions[['subsector', 'ef_global']]
    df_gadm_emissions['ef_country'] = np.where(df_gadm_emissions['activity'] != 0, df_gadm_emissions['emissions_quantity'] / df_gadm_emissions['activity'], np.nan)
    df_gadm_emissions['ef_country'] = np.where(np.isclose(df_gadm_emissions['ef_country'], 0, atol=1e-6), np.nan, df_gadm_emissions['ef_country'])
    df_gadm_emissions = df_gadm_emissions[['iso3_country', 'subsector', 'ef_country']]

    # format + clean ownership data
    df_ownership['parent_entity_id'] = df_ownership['parent_entity_id'].fillna('')
    df_ownership['parent_name'] = df_ownership['parent_name'].str.strip()
    df_ownership['parent_name'] = df_ownership['parent_name'].replace('unknown', '').fillna('')
    df_ownership['parent_lei'] = np.where(((df_ownership['parent_lei'] == 'not applicable') & (df_ownership['parent_entity_type'] == 'unknown entity')) |
                                        (df_ownership['parent_lei'] == 'not found'), '', df_ownership['parent_lei'])
    df_ownership['immediate_source_owner'] = df_ownership['immediate_source_owner'].replace('unknown', '').fillna('')
    
    # create keys to search by parent, immediate source, and source operator
    df_ownership['parent'] = np.where(df_ownership['parent_lei'] != '', 
                                      df_ownership['parent_entity_id'].str.strip() + ': ' + df_ownership['parent_name'].str.strip() + ' (' + df_ownership['parent_lei'].str.strip() + ')',
                                      df_ownership['parent_entity_id'].str.strip() + ': ' + df_ownership['parent_name'].str.strip())
    df_ownership['parent'] = np.where(df_ownership['parent'] == ': ', 'Unknown parent', df_ownership['parent'])
    df_ownership['immediate source'] = df_ownership['immediate_source_owner_entity_id'].str.strip() + ': ' + df_ownership['immediate_source_owner'].str.strip()
    df_ownership['immediate source'] = np.where(df_ownership['immediate source'] == ': ', 'Unknown immediate source', df_ownership['immediate source'])
    df_ownership['source operator'] = df_ownership['source_operator_id'].str.strip() + ': ' + df_ownership['source_operator'].str.strip()
    df_ownership['source operator'] = np.where(df_ownership['source operator'] == ': ', 'Unknown source operator', df_ownership['source operator'])
    
    # calculate ownership emissions factors
    df_ownership['activity'] = df_ownership['activity'].astype(float)
    df_ownership['ef_asset'] = df_ownership['emissions_quantity'].div(df_ownership['activity'].where(df_ownership['activity'] != 0, np.nan))

    ##### DROPDOWN MENU: KEYS FOR SEARCHING -------

    ownership_list = \
        pd.concat([df_ownership['parent'], df_ownership['immediate source'], df_ownership['source operator']], ignore_index=True)
    ownership_list = ownership_list.drop_duplicates().sort_values().tolist()

    owner_col, loc_col = st.columns(2)

     # select relevant owners
    with owner_col:
        selected_owners_user = st.multiselect(
            "Select Owners (by Entity ID, Name, or LEI)",
            options=ownership_list,
            default=[]
        )


    with loc_col:
        # select relevant locations
        if not selected_owners_user:
            loc_options = df_ownership['iso3_country'].drop_duplicates().sort_values()
        else:
            loc_options = df_ownership[(df_ownership['parent'].isin(selected_owners_user)) | (df_ownership['immediate source'].isin(selected_owners_user)) | 
                                       (df_ownership['source operator'].isin(selected_owners_user))]['iso3_country'].drop_duplicates().sort_values()
        selected_location_user = st.multiselect(
            "Select Locations",
            options=loc_options,
            default=[]
        )

    # filter based on selection
    if not selected_owners_user:
        if not selected_location_user:
            df_selected = df_ownership.copy()
        else:
            df_selected = df_ownership[(df_ownership['iso3_country'].isin(selected_location_user))].copy()
    else:
        if not selected_location_user:
            df_selected = df_ownership[((df_ownership['parent'].isin(selected_owners_user)) | (df_ownership['immediate source'].isin(selected_owners_user)) | 
                                    (df_ownership['source operator'].isin(selected_owners_user)))].copy()
        else:
            df_selected = df_ownership[((df_ownership['parent'].isin(selected_owners_user)) | (df_ownership['immediate source'].isin(selected_owners_user)) | 
                                        (df_ownership['source operator'].isin(selected_owners_user))) & (df_ownership['iso3_country'].isin(selected_location_user))].copy()
    df_selected = df_selected.sort_values('emissions_quantity', ascending=False).reset_index()
    
    ##### SUMMARY INFO -------
    st.markdown("### Ownership Analysis")
    st.markdown(
    f"""
    <div style="text-align:left; font-size:22px; margin-top:5px;">
        <b>Total Emissions:</b> {format_emissions(df_selected.drop_duplicates('asset_id')['emissions_quantity'].sum())} <br> 
        <b>Total Reductions:</b> {format_emissions(df_selected.drop_duplicates('asset_id')['net_reduction_potential'].sum())} <br> 
        <b>Number of Sectors:</b> {df_selected['subsector'].nunique()} <br> 
        <b>Number of Assets:</b> {df_selected['asset_id'].nunique():,.0f} <br> 
    </div>
    """,
    unsafe_allow_html=True)

    st.markdown("""<br><br>""", unsafe_allow_html=True)

    ##### ASSET MAP + COUNTRY CHART -------
    country_col, map_col = st.columns([0.4, 0.6])

    # create bar chart based off country breakdown
    with country_col:
        st.markdown("#### By Countries")
        bar_data = df_selected.groupby('iso3_country', as_index=False)['emissions_quantity'].sum()
        # add row for sum of all countries
        total_row = pd.DataFrame({"iso3_country": ['Total'], "emissions_quantity": [bar_data['emissions_quantity'].sum()]})
        bar_data = pd.concat([bar_data, total_row])
        bar_data['color_group'] = bar_data['iso3_country'].apply(lambda x: 'Total' if x == 'Total' else 'Country')
        color_map = {'Country': '#3B7A72', 'Total': 'grey'}
        fig_bar = px.bar(
            bar_data,
            x='iso3_country',
            y='emissions_quantity',
            color='color_group',
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with map_col:
        st.markdown("#### Asset Map")
    # get country information
        df_map = df_selected.groupby(['iso3_country']).agg(num_assets=('asset_id', 'size')).reset_index()
        
        # get asset information
        df_assets = df_selected[['sector', 'subsector', 'asset_id', 'asset_name', 'lat_lon']].copy()
        df_assets['geometry'] = gpd.GeoSeries.from_wkt(df_assets['lat_lon'])
        df_assets['lon'] = df_assets['geometry'].apply(lambda p: p.x)
        df_assets['lat'] = df_assets['geometry'].apply(lambda p: p.y)

        # map countries
        fig = px.choropleth(df_map,
                            locations="iso3_country",
                            hover_name="iso3_country",
                            hover_data={"num_assets": True, "iso3_country": False},
                            labels={"num_assets": "Number of Assets"},
                            color='num_assets',
                            color_continuous_scale=px.colors.sequential.Teal)
        
        fig.update_layout(geo=dict(bgcolor='rgba(240, 240, 240, 1)'))

        # map assets
        fig.add_scattergeo(
            lat=df_assets['lat'],
            lon=df_assets['lon'],
            text=df_assets['asset_name'],
            mode="markers",
            marker=dict(size=4, color=df_assets['sector'].map(sector_color_map['sector']), opacity=0.7),
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Asset ID: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            customdata=np.stack([df_assets['asset_id'], df_assets['subsector']], axis=-1))
        st.plotly_chart(fig, use_container_width=True)


    ##### VISUALIZATIONS -------
    sector_col, subsector_col = st.columns([0.4, 0.6])

    # create pie chart based off sector breakdown
    with sector_col:
        st.markdown("#### By Sectors")
        fig_pie = px.pie(
            df_selected,
            values='emissions_quantity',
            names='sector',
            color='sector',
            color_discrete_map=sector_color_map['sector']
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # create table based off subsector breakdown
    with subsector_col:
        st.markdown("#### By Subsectors")
        df_subsectors = df_selected.groupby(['sector', 'subsector']).agg(total_emissions=('emissions_quantity', 'sum'),
                                                                         total_reductions=('net_reduction_potential', 'sum'),
                                                                         number_of_assets=('asset_id', 'nunique')).reset_index()
        st.dataframe(
            df_subsectors,
            use_container_width=True
        )

    
    ##### DATA TABLE -------

    st.markdown("###")
    st.markdown("### Top Assets Information")
    # add caveat
    st.markdown(
    """
    <div style="text-align:left; font-size:16px; margin-top:10px;">
        <i>Note: selected sectors have null activity value due to data license agreement (e.g. oil-and-gas-production/transport)</i>
    </div>
    """,
    unsafe_allow_html=True)

    # create table
    df_table = df_selected[['asset_id', 'asset_name', 'subsector', 'asset_type', 'iso3_country', 'activity_units', 'activity', 'emissions_quantity', 'net_reduction_potential', 'ef_asset']].drop_duplicates().head(1000)
    df_table = df_table.merge(df_gadm_emissions, how='left', on=['iso3_country', 'subsector']).merge(df_global_emissions, how='left', on=['subsector'])
    st.dataframe(
        df_table,
        use_container_width=True,
        height=600,
        column_config={"activity": st.column_config.NumberColumn(format="localized"),
                       "emissions_quantity": st.column_config.NumberColumn(format="localized"),
                       "net_reduction_potential": st.column_config.NumberColumn(format="localized"),
                       "ef_asset": st.column_config.NumberColumn(format="localized"),
                       "ef_country": st.column_config.NumberColumn(format="localized"),
                       "ef_global": st.column_config.NumberColumn(format="localized")}
    )