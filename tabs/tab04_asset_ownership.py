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
    df_ownership['parent_entity_id'] = df_ownership['parent_entity_id'].str.strip().fillna('')
    df_ownership['parent_name'] = df_ownership['parent_name'].str.strip()
    df_ownership['parent_name'] = df_ownership['parent_name'].replace('unknown', 'Unknown').fillna('Unknown')
    df_ownership['parent_lei'] = df_ownership['parent_lei'].str.strip()
    df_ownership['parent_lei'] = np.where(((df_ownership['parent_lei'] == 'not applicable') & (df_ownership['parent_entity_type'] == 'unknown entity')) |
                                        (df_ownership['parent_lei'] == 'not found'), '', df_ownership['parent_lei'])
    df_ownership['immediate_source_owner'] = df_ownership['immediate_source_owner'].str.strip().replace('unknown', 'Unknown').fillna('Unknown')
    df_ownership['immediate_source_owner_entity_id'] = df_ownership['immediate_source_owner_entity_id'].str.strip()
    df_ownership['source_operator'] = df_ownership['source_operator'].replace('unknown', 'Unknown').fillna('Unknown')
    df_ownership['source_operator_id'] = df_ownership['source_operator_id'].str.strip()
    
    # create keys to search by parent, immediate source, and source operator
    df_entity_id = df_ownership.copy()

    df_entity_id = pd.concat([df_entity_id[['parent_entity_id', 'parent_name']].rename(columns={'parent_entity_id': 'entity_id', 'parent_name': 'entity_key'}),
                              df_entity_id[['immediate_source_owner_entity_id', 'immediate_source_owner']].rename(columns={'immediate_source_owner_entity_id': 'entity_id', 'immedate_source_owner': 'entity_key'}),
                              df_entity_id[['source_operator_id', 'source_operator']].rename(columns={'source_operator_id': 'entity_id', 'source_operator': 'entity_key'})])
    
    df_entity_id['entity_key'] = df_entity_id['entity_key'].fillna('Unknown').astype(str)
    df_entity_id = df_entity_id.sort_values('entity_id').drop_duplicates(subset='entity_id', keep='first')
    df_entity_id['entity_key'] = np.where(df_entity_id['entity_id'] == '', 'Unknown', df_entity_id['entity_key'])
    dict_entity_id = df_entity_id.set_index('entity_id')['entity_key'].to_dict()

    df_ownership['parent_name'] = df_ownership['parent_entity_id'].map(dict_entity_id)
    df_ownership['immediate_source_operator'] = df_ownership['immediate_source_owner_entity_id'].map(dict_entity_id)
    df_ownership['source_operator'] = df_ownership['source_operator_id'].map(dict_entity_id)
    
    # calculate ownership emissions factors
    df_ownership['activity'] = df_ownership['activity'].astype(float)
    df_ownership['parent_emissions_quantity'] = np.where(df_ownership['parent_overall_share_pct'].isna(), df_ownership['emissions_quantity'],
                                                         df_ownership['parent_overall_share_pct'] * df_ownership['emissions_quantity'] / 100)
    df_ownership['parent_net_reduction_potential'] = np.where(df_ownership['parent_overall_share_pct'].isna(), df_ownership['net_reduction_potential'],
                                                         df_ownership['parent_overall_share_pct'] * df_ownership['net_reduction_potential'] / 100)
    df_ownership['ef_asset'] = df_ownership['emissions_quantity'].div(df_ownership['activity'].where(df_ownership['activity'] != 0, np.nan))

    ##### DROPDOWN MENU: KEYS FOR SEARCHING -------

    ownership_list = \
        pd.concat([df_ownership['parent_name'], df_ownership['immediate_source_owner'], df_ownership['source_operator']], ignore_index=True)
    ownership_list = ownership_list.drop_duplicates().sort_values().tolist()

    for key in ['selected_owners', 'selected_locations', 'selected_subsectors']:
        if key not in st.session_state:
            st.session_state[key] = []

    def is_owner_match(row, owners):
        return (row['parent_name'] in owners or 
                row['immediate_source_owner'] in owners or 
                row['source_operator'] in owners)

    def update_owner_options():
        """Callback: regenerate owner options when location/subsector change"""
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_locations:
            mask &= df_ownership['iso3_country'].isin(st.session_state.selected_locations)
        if st.session_state.selected_subsectors:
            mask &= df_ownership['subsector'].isin(st.session_state.selected_subsectors)
        
        owner_options = (pd.concat([df_ownership[mask]['parent_name'], 
                                df_ownership[mask]['immediate_source_owner'], 
                                df_ownership[mask]['source_operator']], ignore_index=True)
                        .drop_duplicates().sort_values().tolist())
        
        st.session_state.selected_owners = [o for o in st.session_state.selected_owners if o in owner_options]

    def update_location_options():
        """Callback: regenerate location options when owner/subsector change"""
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_owners:
            mask &= df_ownership.apply(lambda row: is_owner_match(row, st.session_state.selected_owners), axis=1)
        if st.session_state.selected_subsectors:
            mask &= df_ownership['subsector'].isin(st.session_state.selected_subsectors)
        loc_options = df_ownership[mask]['iso3_country'].drop_duplicates().sort_values().tolist()
        
        st.session_state.selected_locations = [l for l in st.session_state.selected_locations if l in loc_options]

    def update_subsector_options():
        """Callback: regenerate subsector options when owner/location change"""
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_owners:
            mask &= df_ownership.apply(lambda row: is_owner_match(row, st.session_state.selected_owners), axis=1)
        if st.session_state.selected_locations:
            mask &= df_ownership['iso3_country'].isin(st.session_state.selected_locations)
        subsector_options = df_ownership[mask]['subsector'].drop_duplicates().sort_values().tolist()
        
        st.session_state.selected_subsectors = [s for s in st.session_state.selected_subsectors if s in subsector_options]

    owner_col, loc_col, subsector_col = st.columns(3)

    # compute current options 
    with owner_col:
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_locations:
            mask &= df_ownership['iso3_country'].isin(st.session_state.selected_locations)
        if st.session_state.selected_subsectors:
            mask &= df_ownership['subsector'].isin(st.session_state.selected_subsectors)
        
        owner_options = (pd.concat([df_ownership[mask]['parent_name'], 
                                df_ownership[mask]['immediate_source_owner'], 
                                df_ownership[mask]['source_operator']], ignore_index=True)
                        .drop_duplicates().sort_values().tolist())
        
        st.multiselect("Select Owners", options=owner_options,
                    default=st.session_state.selected_owners, 
                    key="owners_multiselect",
                    on_change=lambda: (st.session_state.update(selected_owners=st.session_state.owners_multiselect), 
                                    update_location_options(), update_subsector_options()))

    with loc_col:
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_owners:
            mask &= df_ownership.apply(lambda row: is_owner_match(row, st.session_state.selected_owners), axis=1)
        if st.session_state.selected_subsectors:
            mask &= df_ownership['subsector'].isin(st.session_state.selected_subsectors)
        
        loc_options = df_ownership[mask]['iso3_country'].drop_duplicates().sort_values().tolist()
        
        st.multiselect("Select Locations", options=loc_options,
                    default=st.session_state.selected_locations,
                    key="locations_multiselect",
                    on_change=lambda: (st.session_state.update(selected_locations=st.session_state.locations_multiselect), 
                                    update_owner_options(), update_subsector_options()))

    with subsector_col:
        mask = pd.Series(True, index=df_ownership.index)
        if st.session_state.selected_owners:
            mask &= df_ownership.apply(lambda row: is_owner_match(row, st.session_state.selected_owners), axis=1)
        if st.session_state.selected_locations:
            mask &= df_ownership['iso3_country'].isin(st.session_state.selected_locations)
        
        subsector_options = df_ownership[mask]['subsector'].drop_duplicates().sort_values().tolist()
        
        st.multiselect("Select Subsector", options=subsector_options,
                    default=st.session_state.selected_subsectors,
                    key="subsectors_multiselect",
                    on_change=lambda: (st.session_state.update(selected_subsectors=st.session_state.subsectors_multiselect), 
                                    update_owner_options(), update_location_options()))

    # filter assets df based on selection
    mask = pd.Series(True, index=df_ownership.index)
    if st.session_state.selected_owners:
        mask &= df_ownership.apply(lambda row: is_owner_match(row, st.session_state.selected_owners), axis=1)
    if st.session_state.selected_locations:
        mask &= df_ownership['iso3_country'].isin(st.session_state.selected_locations)
    if st.session_state.selected_subsectors:
        mask &= df_ownership['subsector'].isin(st.session_state.selected_subsectors)

    df_selected = df_ownership[mask].copy()    
    df_selected = df_selected.sort_values('parent_emissions_quantity', ascending=False).reset_index()
    df_selected = df_selected.drop_duplicates('asset_id')
    
    ##### SUMMARY INFO -------
    emissions_box, reductions_box, sectors_box, asset_box = st.columns(4)
    st.markdown("""<br>""", unsafe_allow_html=True)

    with emissions_box:
        bordered_metric("Total Emissions", format_emissions(df_selected['parent_emissions_quantity'].sum()))

    with reductions_box:
        bordered_metric("Total Reductions", format_emissions(df_selected['parent_net_reduction_potential'].sum()), value_color='#6AAD89')

    with sectors_box:
        bordered_metric("Number of Sectors", df_selected['subsector'].nunique())

    with asset_box:
        bordered_metric("Number of Assets", f"{df_selected['asset_id'].nunique():,}")

    st.markdown("""<br>""", unsafe_allow_html=True)

    ##### ASSET MAP + COUNTRY CHART -------
    map_chart, country_chart, sector_chart, = st.columns(3)

    # create map of assets
    with map_chart:
        # get country information
        df_map = df_selected.groupby(['iso3_country']).agg(num_assets=('asset_id', 'nunique')).reset_index().copy()
        
        # get asset information
        df_assets = df_selected[['sector', 'subsector', 'asset_id', 'asset_name', 'lat_lon']].copy()
        df_assets['geometry'] = gpd.GeoSeries.from_wkt(df_assets['lat_lon'])
        df_assets['lon'] = df_assets['geometry'].apply(lambda p: p.x)
        df_assets['lat'] = df_assets['geometry'].apply(lambda p: p.y)

        # map countries
        fig = px.choropleth(df_map,
                            title='Asset Locations and Sector',
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
        st.plotly_chart(fig)

    # create bar chart based off country breakdown
    with country_chart:
        bar_data = df_selected.groupby('iso3_country', as_index=False)[['parent_net_reduction_potential', 'parent_emissions_quantity']].sum().sort_values('parent_emissions_quantity', ascending=False).copy()
        country_totals = bar_data.groupby('iso3_country')['parent_emissions_quantity'].sum().sort_values(ascending=False)
        sorted_countries = country_totals.index.tolist()
        bar_data['remaining_emissions'] = bar_data['parent_emissions_quantity'] - bar_data['parent_net_reduction_potential']
        bar_data = bar_data.melt(id_vars='iso3_country', value_vars=['remaining_emissions', 'parent_net_reduction_potential'], var_name='emissions_breakdown', value_name='parent_emissions')
        
        fig_bar = px.bar(
            bar_data,
            x='iso3_country',
            y='parent_emissions',
            title='Emissions by Country (tCO2e)',
            color='emissions_breakdown',
            color_discrete_map={'parent_net_reduction_potential': '#6AAD89', 'remaining_emissions': '#707070'},
            barmode='stack'
        )

        fig_bar.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="category"
            )
        )

        if len(bar_data) > 10:
            top10_range = [
                sorted_countries.index(sorted_countries[0]) - 0.5,
                sorted_countries.index(sorted_countries[9]) + 0.5,
            ]
            fig_bar.update_xaxes(range=top10_range, type="category")


        st.plotly_chart(fig_bar, use_container_width=True)


    # create pie chart based off sector breakdown
    with sector_chart:
        fig_pie = px.pie(
            df_selected,
            title='Emissions by Sector (%)',
            values='parent_emissions_quantity',
            names='sector',
            color='sector',
            color_discrete_map=sector_color_map['sector']
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # create table based off subsector breakdown
    st.markdown("#### Sector Summary")
    df_subsectors = df_selected.groupby(['sector', 'subsector']).agg(total_emissions=('parent_emissions_quantity', 'sum'),
                                                                     total_reductions=('parent_net_reduction_potential', 'sum'),
                                                                     number_of_assets=('asset_id', 'nunique')).reset_index().sort_values('total_emissions', ascending=False).reset_index(drop=True)
    
    numeric_cols = df_subsectors.select_dtypes(include="number").columns
    df_subsectors = df_subsectors.style.format({col: "{:,.0f}" for col in numeric_cols})
    
    st.dataframe(
        df_subsectors,
        use_container_width=True
    )

    
    ##### DATA TABLE -------

    st.markdown("###")
    st.markdown("### Top-Emitting Assets Information")
    # add caveat
    st.markdown(
    """
    <div style="text-align:left; font-size:16px; margin-top:10px;">
        <i>Note: selected sectors have null activity value due to data license agreement (e.g. oil-and-gas-production/transport)</i>
    </div>
    """,
    unsafe_allow_html=True)

    # create table
    df_table = df_selected[['asset_id', 'asset_name', 'subsector', 'asset_type', 'parent_name', 'parent_overall_share_pct', 'immediate_source_owner', 'source_operator', 'iso3_country', 'activity_units', 'activity', 'emissions_quantity', 'parent_emissions_quantity', 'net_reduction_potential', 'parent_net_reduction_potential', 'ef_asset']].drop_duplicates().head(500)
    df_table = df_table.merge(df_gadm_emissions, how='left', on=['iso3_country', 'subsector']).merge(df_global_emissions, how='left', on=['subsector'])

    numeric_cols = df_table.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != "asset_id"]
    df_table = df_table.style.format({col: "{:,.0f}" for col in numeric_cols})

    st.dataframe(
        df_table,
        use_container_width=True,
        height=600,
        column_config={"ef_asset": st.column_config.NumberColumn(format="localized"),
                       "ef_country": st.column_config.NumberColumn(format="localized"),
                       "ef_global": st.column_config.NumberColumn(format="localized")}
    )
