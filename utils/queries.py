import pandas as pd
import io
import streamlit as st
import html
import numpy as np


'''
This builds SQL for the pie chart on Reduction Opportunities tab. This will
behave the same regardless of whether the user has Climate TRACE Solutions or
Percentile Benchmarking selected as their reduction method.

Returns: country_sql

Type: string (SQL)
'''
def build_country_sql(table, where_sql):
    
    country_sql = f"""
        SELECT 
            year,
            sector,
            SUM(emissions_quantity) AS country_emissions_quantity
        FROM '{table}'
        
        {where_sql}
        
        GROUP BY year, sector
        
        ORDER BY sector
    """

    return country_sql

'''
This builds SQL for the stacked bar chart Reduction Opportunities tab. This will
dynamically pull in the correct data depending on the selected reduction method,
which could be "Climate TRACE Solutions" or "Percentile Benchmarking"

Returns: sector_reduction_sql_string

Type: string (SQL)
'''
def build_sector_reduction_sql(use_ct_ers,
                                annual_asset_path,
                                dropdown_join,
                                reduction_where_sql,
                                percentile_path=None,
                                percentile_col=None,
                                selected_proportion=None,
                                benchmark_join=None
                            ):
    
    if use_ct_ers is True:
        sector_reduction_sql_string = f'''
            WITH sector_mapping as (
                SELECT distinct sector
                    , subsector

                FROM '{annual_asset_path}'
            ),
            
            induced as (
                SELECT sector
                    , sum(induced_emissions) as induced_emissions

                FROM (
                        SELECT
                            sector,
                            sum(induced_emissions) AS induced_emissions
                        FROM (
                            SELECT distinct asset_id
                                , sector_mapping.sector
                                , induced_sector_1 AS induced_subsector
                                , induced_sector_1_induced_emissions AS induced_emissions
                            FROM '{annual_asset_path}' ae
                            INNER JOIN sector_mapping
                                on sector_mapping.subsector = ae.induced_sector_1
                            {dropdown_join}

                            {reduction_where_sql}
                        )  

                        group by sector
                        
                        UNION ALL
                            
                        SELECT
                            sector,
                            sum(induced_emissions) AS induced_emissions
                        FROM (
                            SELECT distinct asset_id
                                , sector_mapping.sector
                                , induced_sector_2 AS induced_subsector
                                , induced_sector_2_induced_emissions AS induced_emissions
                            FROM '{annual_asset_path}' ae
                            INNER JOIN sector_mapping
                                on sector_mapping.subsector = ae.induced_sector_2
                            {dropdown_join}

                            {reduction_where_sql}
                        )  

                        group by sector
                            
                        UNION ALL
                            
                        SELECT
                            sector,
                            sum(induced_emissions) AS induced_emissions
                        FROM (
                            SELECT distinct asset_id
                                , sector_mapping.sector
                                , induced_sector_3 AS induced_subsector
                                , induced_sector_3_induced_emissions AS induced_emissions
                            FROM '{annual_asset_path}' ae
                            INNER JOIN sector_mapping
                                on sector_mapping.subsector = cast(ae.induced_sector_3 as varchar)
                            {dropdown_join}

                            {reduction_where_sql}
                        )  

                        group by sector
                    )

                GROUP BY sector
            ),

            asset_reductions as (
                SELECT sector
                    , sum(emissions_quantity) emissions_quantity
                    , sum(emissions_reduced_at_asset) emissions_reduced_at_asset
            
                FROM (
                    SELECT asset_id
                        , sector
                        , subsector
                        , sum(emissions_quantity) emissions_quantity
                        , emissions_reduced_at_asset

                    FROM '{annual_asset_path}' ae
                    {dropdown_join}

                    {reduction_where_sql}

                    GROUP BY asset_id
                        , sector
                        , subsector
                        , emissions_reduced_at_asset
                ) asset

                GROUP BY sector
            )

            SELECT 
                COALESCE(ar.sector, induced.sector) AS sector,
                ar.emissions_quantity,
                induced.induced_emissions,
                ar.emissions_reduced_at_asset,
                
                CASE 
                    WHEN COALESCE(induced.induced_emissions, 0) > COALESCE(ar.emissions_reduced_at_asset, 0)
                    THEN COALESCE(induced.induced_emissions, 0) - COALESCE(ar.emissions_reduced_at_asset, 0)
                    ELSE 0 
                END AS induced_emissions,
                
                CASE 
                    WHEN COALESCE(induced.induced_emissions, 0) < COALESCE(ar.emissions_reduced_at_asset, 0)
                    THEN COALESCE(ar.emissions_reduced_at_asset, 0) - COALESCE(induced.induced_emissions, 0)
                    ELSE 0 
                END AS emissions_reduction_potential

            FROM asset_reductions ar
            FULL OUTER JOIN induced
                on induced.sector = ar.sector
        '''

            # print(sector_reduction_sql_string)
    
    else:
        sector_reduction_sql_string = f'''
            SELECT 
                sector,
                SUM(emissions_quantity) AS emissions_quantity,
                SUM(emissions_reduction_potential) AS emissions_reduction_potential

            FROM (
                SELECT 
                    ae.asset_id,
                    ae.sector,
                    ae.subsector,
                    ae.iso3_country,
                    ae.country_name,

                    CASE 
                        WHEN BOOL_OR(ae.activity_is_temporal) 
                            THEN SUM(ae.activity) 
                            ELSE AVG(ae.activity) 
                    END AS activity,

                    AVG(ae.ef_12_moer) AS ef_12_moer,

                    CASE 
                        WHEN AVG(ae.ef_12_moer) IS NULL 
                            THEN SUM(ae.emissions_quantity)
                        ELSE (
                            CASE 
                                WHEN BOOL_OR(ae.activity_is_temporal) 
                                    THEN SUM(ae.activity) 
                                    ELSE AVG(ae.activity) 
                            END
                        ) * AVG(ae.ef_12_moer)
                    END AS emissions_quantity,

                    GREATEST(
                        0,
                        CASE 
                            WHEN AVG(ae.ef_12_moer) IS NULL 
                            THEN (
                                SUM(ae.emissions_quantity)
                                - (
                                    CASE 
                                        WHEN BOOL_OR(ae.activity_is_temporal)
                                            THEN SUM(ae.activity * pct.{percentile_col})
                                            ELSE AVG(ae.activity * pct.{percentile_col})
                                    END
                                )
                            ) * ({selected_proportion} / 100.0)

                            ELSE (
                                (
                                    CASE 
                                        WHEN BOOL_OR(ae.activity_is_temporal)
                                            THEN SUM(ae.activity) * AVG(ae.ef_12_moer)
                                            ELSE AVG(ae.activity) * AVG(ae.ef_12_moer)
                                    END
                                )
                                - (
                                    CASE 
                                        WHEN BOOL_OR(ae.activity_is_temporal)
                                            THEN SUM(ae.activity * pct.{percentile_col})
                                            ELSE AVG(ae.activity * pct.{percentile_col})
                                    END
                                )
                            ) * ({selected_proportion} / 100.0)
                        END
                    ) AS emissions_reduction_potential

                FROM '{annual_asset_path}' ae
                LEFT JOIN '{percentile_path}' pct
                    ON ae.subsector = pct.original_inventory_sector
                    AND ae.asset_type_2 = pct.asset_type
                    {benchmark_join}
                {dropdown_join}

                {reduction_where_sql}

                GROUP BY 
                    ae.asset_id,
                    ae.sector,
                    ae.subsector,
                    ae.iso3_country,
                    ae.country_name
            ) asset_level

            GROUP BY sector
        '''
        # print(sector_reduction_sql_string)

    
    return sector_reduction_sql_string


def build_sector_induction_sql(annual_asset_path,
                                dropdown_join,
                                reduction_where_sql
                            ):

    sector_induction_sql_string = f'''
        WITH sector_mapping AS (
            SELECT DISTINCT sector, subsector
            FROM '{annual_asset_path}'
        ),

        induced_raw AS (
            SELECT
                ae.asset_id,
                ae.sector AS inducing_sector,
                sm.sector  AS receiving_sector,
                ae.induced_sector_1_induced_emissions AS induced_emissions
            FROM '{annual_asset_path}' ae
            INNER JOIN sector_mapping sm 
                ON sm.subsector = ae.induced_sector_1
            {dropdown_join}

            {reduction_where_sql}

            UNION ALL
            
            SELECT
                ae.asset_id,
                ae.sector,
                sm.sector,
                ae.induced_sector_2_induced_emissions
            FROM '{annual_asset_path}' ae
            INNER JOIN sector_mapping sm 
                ON sm.subsector = ae.induced_sector_2
            {dropdown_join}

            {reduction_where_sql}

            UNION ALL
            
            SELECT
                ae.asset_id,
                ae.sector,
                sm.sector,
                ae.induced_sector_3_induced_emissions 
            FROM '{annual_asset_path}' ae
            INNER JOIN sector_mapping sm 
                ON sm.subsector = CAST(ae.induced_sector_3 AS VARCHAR)
            {dropdown_join}

            {reduction_where_sql}
        ),

        induced_annual_per_asset AS (
            SELECT
                asset_id,
                inducing_sector,
                receiving_sector,
                MAX(induced_emissions) AS induced_emissions_annual
            
            FROM induced_raw
            
            GROUP BY asset_id
                , inducing_sector
                , receiving_sector
        )

        SELECT
            receiving_sector,
            inducing_sector,        
            SUM(induced_emissions_annual) AS induced_emissions
        
        FROM induced_annual_per_asset
        
        GROUP BY inducing_sector, 
            receiving_sector

        ORDER BY receiving_sector
    '''

    return sector_induction_sql_string


'''
Builds a query that inserts data into the 4th sentence within the 
"A Possible Emissions Reduction Plan" text block on the Reduction 
Opportunities tab.

Returns: sentence_4_query

Type: string (SQL)
'''
def build_sentence_4_sql(table, where_sql, include_sectors):

    sentence_4_query = f"""
        with sector as (
            select sector
                , sum(emissions_quantity) sector_emissions_quantity

            from '{table}'

            {where_sql}
                and lower(sector) <> 'power'
                and sector in ({include_sectors})

            group by sector
        ),

        subsector as (
            select sector
                , subsector
                , sum(emissions_quantity) subsector_emissions_quantity

            from '{table}'

            {where_sql}
                and lower(sector) <> 'power'
                and sector in ({include_sectors})

            group by sector
                , subsector
        ),

        agg as (
            select subsector.sector
                , subsector.subsector
                , sum(subsector.subsector_emissions_quantity) subsector_emissions_quantity

            from subsector
            inner join sector
                on sector.sector = subsector.sector

            group by subsector.sector
                , subsector.subsector

            having (sum(subsector.subsector_emissions_quantity) / sum(sector.sector_emissions_quantity)) >= 0.05
        ),
        
        subsector_rank as (
            select *
                , row_number() over (partition by sector order by subsector_emissions_quantity desc) as subsector_rank
            from agg
        )

        select *
        from subsector_rank
        where subsector_rank.subsector_rank <= 2

    """
    
    return sentence_4_query


'''
Builds SQL query for the asset table at the bottom of the page. It will
dynamically determine whether or not to use Climate TRACE Solutions, or
the Percentile Benchmarking method based on user selection.

Returns: asset_table_query

Type: string (SQL)
'''
def build_asset_reduction_sql(use_ct_ers,
                                annual_asset_path,
                                dropdown_join,
                                reduction_where_sql,
                                sorting_preference,
                                exclude_forestry,
                                percentile_path=None,
                                percentile_col=None,
                                selected_proportion=None,
                                benchmark_join=None,
                                ):
    
    if sorting_preference == 'Net Reduction Potential':
        sorting_col = 'total_emissions_reduced_per_year'
    elif sorting_preference == 'Asset Reduction Potential':
        sorting_col = 'emissions_reduction_potential'
    elif sorting_preference == 'Asset Annual Emissions':
        sorting_col = 'emissions_quantity'

    if not reduction_where_sql:
        reduction_where_sql = f"""Where lower(ae.asset_type) <> 'biomass'"""

    if exclude_forestry:
        reduction_where_sql += " and sector <> 'forestry-and-land-use' "

    if use_ct_ers is True:

        asset_table_query = f"""
            SELECT asset_id,
                asset_name,
                iso3_country,
                country_name,
                sector,
                subsector,
                asset_type,
                strategy_name,
                SUM(emissions_quantity) AS emissions_quantity,
                coalesce(emissions_reduced_at_asset,0) AS emissions_reduction_potential,
                coalesce(total_emissions_reduced_per_year,0) AS total_emissions_reduced_per_year
            
            FROM '{annual_asset_path}' ae
            {dropdown_join}

            {reduction_where_sql}
            
            GROUP BY asset_id,
                asset_name,
                iso3_country,
                country_name,
                sector,
                subsector,
                asset_type,
                strategy_name,
                coalesce(emissions_reduced_at_asset,0),
                coalesce(total_emissions_reduced_per_year,0)
            
            ORDER BY 
                {sorting_col} DESC
            
            LIMIT 100
        """
    
    else:

        asset_table_query = f"""
            SELECT asset_id 
                , asset_name
                , iso3_country
                , country_name
                , sector
                , subsector
                , asset_type
                , emissions_quantity
                , emissions_reduction_potential

            FROM (
                SELECT ae.asset_id,
                    ae.asset_name,
                    ae.asset_type,
                    ae.iso3_country,
                    ae.country_name,
                    ae.sector,
                    ae.subsector,
                    ae.asset_type,
                    
                    -- Adjust emissions_quantity with SUM vs AVG activity rule
                    CASE 
                        WHEN AVG(ae.ef_12_moer) IS NULL 
                            THEN SUM(ae.emissions_quantity)
                        ELSE (
                            CASE 
                                WHEN BOOL_OR(ae.activity_is_temporal) 
                                    THEN SUM(ae.activity)
                                    ELSE AVG(ae.activity)
                            END
                        ) * AVG(ae.ef_12_moer)
                    END AS emissions_quantity,

                    -- Emissions reduction potential with same rule
                    GREATEST(
                        0,
                        CASE 
                            WHEN pct.{percentile_col} IS NULL THEN 0
                            WHEN AVG(ae.ef_12_moer) IS NULL 
                                THEN (
                                    SUM(ae.emissions_quantity) 
                                    - (
                                        CASE 
                                            WHEN BOOL_OR(ae.activity_is_temporal)
                                                THEN SUM(ae.activity * pct.{percentile_col})
                                                ELSE AVG(ae.activity * pct.{percentile_col})
                                        END
                                    )
                                ) * ({selected_proportion} / 100.0)

                            ELSE (
                                (
                                    CASE 
                                        WHEN BOOL_OR(ae.activity_is_temporal)
                                            THEN SUM(ae.activity) * AVG(ae.ef_12_moer)
                                            ELSE AVG(ae.activity) * AVG(ae.ef_12_moer)
                                    END
                                )
                                - (
                                    CASE 
                                        WHEN BOOL_OR(ae.activity_is_temporal)
                                            THEN SUM(ae.activity * pct.{percentile_col})
                                            ELSE AVG(ae.activity * pct.{percentile_col})
                                    END
                                )
                            ) * ({selected_proportion} / 100.0)
                        END
                    ) AS emissions_reduction_potential,

                    ROW_NUMBER() OVER (
                        ORDER BY 
                            GREATEST(
                                0,
                                CASE 
                                    WHEN pct.{percentile_col} IS NULL THEN 0
                                    WHEN AVG(ae.ef_12_moer) IS NULL 
                                        THEN (
                                            SUM(ae.emissions_quantity) 
                                            - (
                                                CASE 
                                                    WHEN BOOL_OR(ae.activity_is_temporal)
                                                        THEN SUM(ae.activity * pct.{percentile_col})
                                                        ELSE AVG(ae.activity * pct.{percentile_col})
                                                END
                                            )
                                        ) * ({selected_proportion} / 100.0)
                                    ELSE (
                                        (
                                            CASE 
                                                WHEN BOOL_OR(ae.activity_is_temporal)
                                                    THEN SUM(ae.activity) * AVG(ae.ef_12_moer)
                                                    ELSE AVG(ae.activity) * AVG(ae.ef_12_moer)
                                            END
                                        )
                                        - (
                                            CASE 
                                                WHEN BOOL_OR(ae.activity_is_temporal)
                                                    THEN SUM(ae.activity * pct.{percentile_col})
                                                    ELSE AVG(ae.activity * pct.{percentile_col})
                                            END
                                        )
                                    ) * ({selected_proportion} / 100.0)
                                END
                            ) DESC
                    ) AS rank
                
                FROM '{annual_asset_path}' ae
                LEFT JOIN '{percentile_path}' pct
                    ON ae.subsector = pct.original_inventory_sector
                    AND ae.asset_type_2 = pct.asset_type
                    {benchmark_join}
                {dropdown_join}

                {reduction_where_sql}
                
                GROUP BY ae.asset_id,
                    ae.asset_name,     
                    ae.iso3_country,           
                    ae.country_name,
                    ae.sector,
                    ae.subsector,
                    ae.asset_type,
                    pct.{percentile_col}
            ) assets

            ORDER BY {sorting_col} DESC
            LIMIT 100
        """
    
    return asset_table_query

'''
ABATEMENT CURVE TAB
- find_sector_assets_sql()
- These are the SQL queries to find all assets and their GADM information within selected sectors and year. 

Returns: query_sector_assets_sql

Type: string (SQL)
'''

def find_sector_assets_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_sector_assets_sql = f'''
        SELECT 
            ae.year,
            ae.asset_id,
            ae.asset_name,
            ae.asset_type,
            ae.iso3_country,
            ae.country_name,
            ae.balancing_authority_region,
            ae.continent,
            ae.eu,
            ae.oecd,
            ae.unfccc_annex,
            ae.developed_un,
            ae.em_finance,
            ae.sector,
            ae.subsector,
            ae.reduction_q_type,
            gadm0.gid_0,
            ae.gadm_1,
            gadm1.gid_1,
            gadm1.gadm_1_name,
            ae.gadm_2,
            gadm2.gid_2,
            gadm2.gadm_2_name,
            ae.activity_units,
            ae.strategy_name,
            SUM(ae.activity) AS activity,
            SUM(ae.capacity) AS capacity,
            SUM(ae.emissions_quantity) AS emissions_quantity,
            SUM(ae.emissions_quantity) / SUM(ae.activity) AS emissions_factor,
            ae.emissions_reduced_at_asset AS asset_reduction_potential,
            ae.total_emissions_reduced_per_year AS net_reduction_potential,
            ae.asset_difficulty_score,
            (ae.iso3_country || ': ' || ae.asset_name || ' (' || CAST(ae.asset_id AS TEXT) || ')') AS selected_asset_list,
            (ae.iso3_country || ': ' || ae.subsector) AS selected_country_list,
            (ae.sector || ': ' || ae.subsector) AS selected_subsector_list,
            (ae.subsector || ': ' || ae.strategy_name) AS selected_strategy_list
        FROM '{annual_asset_path}' ae
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_0, 
                iso3_country 
            FROM '{gadm_0_path}'
            WHERE try_cast(gid AS INTEGER) IS NOT NULL
        ) gadm0 ON ae.iso3_country = gadm0.iso3_country
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_1,
                gadm_id AS gadm_1,
                gadm_1_corrected_name AS gadm_1_name
            FROM '{gadm_1_path}'
        ) gadm1 ON ae.gadm_1 = gadm1.gadm_1
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_2,
                gadm_2_id AS gadm_2,
                gadm_2_corrected_name AS gadm_2_name
            FROM '{gadm_2_path}'
        ) gadm2 ON ae.gadm_2 = gadm2.gadm_2
        LEFT JOIN (
            SELECT DISTINCT
                city_id, 
                city_name
            FROM '{city_path}') city ON regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g') = city.city_id
        WHERE 
            ae.subsector IN ({formatted_subsectors})
            AND ae.year = {selected_year}
            AND ae.reduction_q_type = 'asset'
            AND {geography_filters_clause}
            AND ae.total_emissions_reduced_per_year IS NOT NULL
        GROUP BY
            ae.year,
            ae.asset_id,
            ae.asset_name,
            ae.asset_type,
            ae.iso3_country,
            ae.country_name,
            ae.balancing_authority_region,
            ae.continent,
            ae.eu,
            ae.oecd,
            ae.unfccc_annex,
            ae.developed_un,
            ae.em_finance,
            ae.sector,
            ae.subsector,
            ae.reduction_q_type,
            gadm0.gid_0,
            ae.gadm_1,
            gadm1.gid_1,
            gadm1.gadm_1_name,
            ae.gadm_2,
            gadm2.gid_2,
            gadm2.gadm_2_name,
            ae.activity_units,
            ae.strategy_name,
            ae.emissions_reduced_at_asset,
            ae.total_emissions_reduced_per_year,
            ae.asset_difficulty_score
        ORDER BY ae.total_emissions_reduced_per_year DESC;
    '''
    return query_sector_assets_sql

'''
ABATEMENT CURVE TAB
- summarize_totals_sql()
- This is the SQL query to total summary information for assets, emissions, ERS, etc. 

Returns: query_total_sql

Type: string (SQL)
'''

def summarize_totals_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_total_sql = f'''
        WITH summary_by_asset AS (
            SELECT
                ae.iso3_country,
                ae.balancing_authority_region,
                ae.asset_id,
                ae.subsector,
                ae.strategy_name,
                SUM(ae.emissions_quantity) AS emissions_sum,
                ae.total_emissions_reduced_per_year
            FROM '{annual_asset_path}' ae
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_0, 
                    iso3_country 
                FROM '{gadm_0_path}'
                WHERE try_cast(gid AS INTEGER) IS NOT NULL
            ) gadm0 ON ae.iso3_country = gadm0.iso3_country
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_1,
                    gadm_id AS gadm_1,
                    gadm_1_corrected_name AS gadm_1_name
                FROM '{gadm_1_path}'
            ) gadm1 ON ae.gadm_1 = gadm1.gadm_1
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_2,
                    gadm_2_id AS gadm_2,
                    gadm_2_corrected_name AS gadm_2_name
                FROM '{gadm_2_path}'
            ) gadm2 ON ae.gadm_2 = gadm2.gadm_2
            LEFT JOIN (
                SELECT DISTINCT
                    city_id, 
                    city_name
                FROM '{city_path}') city ON regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g') = city.city_id
            WHERE subsector IN ({formatted_subsectors})
            AND year = {selected_year}
            AND reduction_q_type = 'asset'
            AND {geography_filters_clause}
            AND ae.total_emissions_reduced_per_year IS NOT NULL
            GROUP BY ae.iso3_country, ae.balancing_authority_region, ae.asset_id, ae.subsector, ae.strategy_name, ae.total_emissions_reduced_per_year
        )
        SELECT
            COUNT(DISTINCT strategy_name) AS total_ers,
            COUNT(DISTINCT asset_id) AS total_assets,
            COUNT(DISTINCT subsector) AS total_subsectors,
            COUNT(DISTINCT iso3_country) AS total_countries,
            COUNT(DISTINCT balancing_authority_region) AS total_ba
        FROM summary_by_asset;
    '''
    return query_total_sql

'''
ABATEMENT CURVE TAB
- summarize_reductions_sql()
- This is the SQL query to total summary information for reductions (remainders + asset). 

Returns: query_reductions_sql

Type: string (SQL)
'''

def summarize_reductions_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_reductions_sql = f'''
        WITH summary_by_asset AS (
            SELECT
                ae.iso3_country,
                ae.balancing_authority_region,
                ae.asset_id,
                ae.strategy_name,
                SUM(ae.emissions_quantity) AS emissions_sum,
                ae.total_emissions_reduced_per_year
            FROM '{annual_asset_path}' ae
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_0, 
                    iso3_country 
                FROM '{gadm_0_path}'
                WHERE try_cast(gid AS INTEGER) IS NOT NULL
            ) gadm0 ON ae.iso3_country = gadm0.iso3_country
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_1,
                    gadm_id AS gadm_1,
                    gadm_1_corrected_name AS gadm_1_name
                FROM '{gadm_1_path}'
            ) gadm1 ON ae.gadm_1 = gadm1.gadm_1
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_2,
                    gadm_2_id AS gadm_2,
                    gadm_2_corrected_name AS gadm_2_name
                FROM '{gadm_2_path}'
            ) gadm2 ON ae.gadm_2 = gadm2.gadm_2
            LEFT JOIN (
                SELECT DISTINCT
                    city_id, 
                    city_name
                FROM '{city_path}') city ON regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g') = city.city_id
            WHERE subsector IN ({formatted_subsectors})
            AND year = {selected_year}
            AND {geography_filters_clause}
            AND ae.total_emissions_reduced_per_year IS NOT NULL
            GROUP BY ae.iso3_country, ae.balancing_authority_region, ae.asset_id, ae.strategy_name, ae.total_emissions_reduced_per_year
        )
        SELECT
            SUM(total_emissions_reduced_per_year) AS total_reductions,
        FROM summary_by_asset;
    '''
    return query_reductions_sql

'''
ABATEMENT CURVE TAB
- summarize_emissions_sql()
- This is the SQL query to totals for emissions + reductions based on selected geography + subsectors. 

Returns: query_emissions_sql

Type: string (SQL)
'''

def summarize_emissions_sql(path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_emissions_sql = f'''
        SELECT 
            year,
            sector,
            SUM(emissions_quantity) AS emissions_quantity
        FROM '{path}'
        WHERE subsector IN ({formatted_subsectors})
            AND year = {selected_year}
            AND {geography_filters_clause}
        GROUP BY year, sector
        ORDER BY sector
    '''
    return query_emissions_sql

'''
ABATEMENT CURVE TAB
- summarize_ers_sql(), create_table_assets_sql()
- These are the SQL queries to summarize ERS information into two tables - overview of all solutions and top opportunities

Returns: query_ers_sql, query_table_assets_sql

Type: string (SQL)
'''

def summarize_ers_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_ers_sql = f'''
        WITH asset_level AS (
            SELECT asset_id,
                subsector,
                strategy_name,
                strategy_description,
                mechanism,
                SUM(emissions_quantity) AS total_emissions_quantity,
                MAX(emissions_reduced_at_asset) AS emissions_reduced_at_asset,
                MAX(total_emissions_reduced_per_year) AS total_emissions_reduced_per_year 
            FROM '{annual_asset_path}' ae
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_0, 
                    iso3_country 
                FROM '{gadm_0_path}'
                WHERE try_cast(gid AS INTEGER) IS NOT NULL
            ) gadm0 ON ae.iso3_country = gadm0.iso3_country
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_1,
                    gadm_id AS gadm_1,
                    gadm_1_corrected_name AS gadm_1_name
                FROM '{gadm_1_path}'
            ) gadm1 ON ae.gadm_1 = gadm1.gadm_1
            LEFT JOIN (
                SELECT DISTINCT
                    gid AS gid_2,
                    gadm_2_id AS gadm_2,
                    gadm_2_corrected_name AS gadm_2_name
                FROM '{gadm_2_path}'
            ) gadm2 ON ae.gadm_2 = gadm2.gadm_2
            LEFT JOIN (
                SELECT DISTINCT
                    city_id, 
                    city_name
                FROM '{city_path}') city ON regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g') = city.city_id
            WHERE subsector IN ({formatted_subsectors})
            AND year = {selected_year}
            AND reduction_q_type = 'asset'
            AND {geography_filters_clause}
            AND ae.total_emissions_reduced_per_year IS NOT NULL
            GROUP BY asset_id, subsector, strategy_name, strategy_description, mechanism
        )
        SELECT subsector,
            strategy_name,
            strategy_description,
            mechanism,
            COUNT(distinct asset_id) AS assets_impacted,
            ROUND(SUM(total_emissions_quantity), 0) AS emissions_quantity,
            ROUND(SUM(total_emissions_reduced_per_year), 0) AS total_reduction_potential
        FROM asset_level
        GROUP BY subsector, strategy_name, strategy_description, mechanism
        ORDER BY total_reduction_potential DESC
    '''

    return query_ers_sql


def create_table_assets_sql(annual_asset_path, gadm_0_path, gadm_1_path, gadm_2_path, city_path, selected_subsector, selected_year, geography_filters_clause):
    formatted_subsectors = ', '.join(f"'{subsector}'" for subsector in selected_subsector)
    query_table_assets_sql = f'''
        SELECT 
            ae.subsector,
            ae.year,
            ae.asset_id,
            ae.asset_name,
            ae.iso3_country,
            ae.country_name,
            gadm0.gid_0,
            ae.gadm_1,
            gadm1.gid_1,
            gadm1.gadm_1_name,
            ae.gadm_2,
            gadm2.gid_2,
            gadm2.gadm_2_name,
            ae.activity_units,
            ae.strategy_name,
            ae.strategy_description,
            ae.mechanism,
            SUM(ae.activity) AS activity,
            SUM(ae.capacity) AS capacity,
            ROUND(SUM(ae.emissions_quantity), 0) AS "emissions_quantity (t CO2e)",
            SUM(ae.emissions_quantity) / NULLIF(SUM(ae.activity), 0) AS emissions_factor,
            ROUND(ae.emissions_reduced_at_asset) AS "asset_reduction_potential (t CO2e)",
            ROUND(ae.total_emissions_reduced_per_year) AS "reduction_potential (t CO2e)",
            ae.asset_difficulty_score
        FROM '{annual_asset_path}' ae
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_0, 
                iso3_country 
            FROM '{gadm_0_path}'
            WHERE try_cast(gid AS INTEGER) IS NOT NULL
            ) gadm0 ON ae.iso3_country = gadm0.iso3_country
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_1,
                gadm_id AS gadm_1,
                gadm_1_corrected_name AS gadm_1_name
            FROM '{gadm_1_path}'
            ) gadm1 ON ae.gadm_1 = gadm1.gadm_1
        LEFT JOIN (
            SELECT DISTINCT
                gid AS gid_2,
                gadm_2_id AS gadm_2,
                gadm_2_corrected_name AS gadm_2_name
            FROM '{gadm_2_path}'
        ) gadm2 ON ae.gadm_2 = gadm2.gadm_2
        LEFT JOIN (
            SELECT DISTINCT
                city_id, 
                city_name
            FROM '{city_path}') city ON regexp_replace(ae.ghs_fua[1], '[{{}}]', '', 'g') = city.city_id
        WHERE 
            subsector IN ({formatted_subsectors})
            AND year = {selected_year}
            AND reduction_q_type = 'asset'
            AND {geography_filters_clause}
            AND ae.total_emissions_reduced_per_year IS NOT NULL
        GROUP BY
            ae.subsector,
            ae.year,
            ae.asset_id,
            ae.asset_name,
            ae.iso3_country,
            ae.country_name,
            gadm0.gid_0,
            ae.gadm_1,
            gadm1.gid_1,
            gadm1.gadm_1_name,
            ae.gadm_2,
            gadm2.gid_2,
            gadm2.gadm_2_name,
            ae.activity_units,
            ae.strategy_name,
            ae.strategy_description,
            ae.mechanism,
            ae.emissions_reduced_at_asset,
            ae.total_emissions_reduced_per_year,
            ae.asset_difficulty_score
        ORDER BY ae.asset_difficulty_score
        LIMIT 200
    '''

    return query_table_assets_sql

def create_heatmap_sql(country_selected_bool,
                       state_selected_bool,
                       g20_bool,
                       region_condition,
                       selected_state_province,
                       annual_asset_path,
                       sum_column,
                       gadm_1_path=None,
                       gadm_2_path=None,
                       ):
    
    # agriculture
    # buildings
    # fluorinated-gases
    # fossil-fuel-operations
    # manufacturing
    # mineral-extraction
    # power
    # transportation
    # waste
    # forestry-and-land-use

    if state_selected_bool:
        iso3_country = region_condition['column_value']

        sector_total_join = f'''
            INNER JOIN (
                SELECT distinct gadm_2_id gadm_id

                FROM '{gadm_2_path}'

                where iso3_country = '{iso3_country}'
                    and gadm_1_corrected_name = '{selected_state_province}'
            ) gadm_2
                on a.gadm_2 = gadm_2.gadm_id
        '''

        table_join = f'''
            INNER JOIN (
                SELECT distinct gadm_2_id gadm_id
                    , gadm_2_corrected_name gadm_2_name

                FROM '{gadm_2_path}'

                where iso3_country = '{iso3_country}'
                    and gadm_1_corrected_name = '{selected_state_province}'
            ) gadm_2
                on a.gadm_2 = gadm_2.gadm_id
        '''

        field = 'gadm_2.gadm_2_name '
    
    elif country_selected_bool:
        iso3_country = region_condition['column_value']

        sector_total_join = f'''
            INNER JOIN (
                SELECT distinct gadm_id

                FROM '{gadm_1_path}'

                where iso3_country = '{iso3_country}'
            ) gadm_1
                on a.gadm_1 = gadm_1.gadm_id
        '''

        table_join = f'''
            INNER JOIN (
                SELECT distinct gadm_id
                    , gadm_1_corrected_name gadm_1_name

                FROM '{gadm_1_path}'

                where iso3_country = '{iso3_country}'
            ) gadm_1
                on a.gadm_1 = gadm_1.gadm_id
        '''

        field = 'gadm_1.gadm_1_name '

    elif g20_bool:
        sector_total_join = f'''
            WHERE a.g20 = true
        '''

        table_join = f'''
            WHERE a.g20 = true
        '''

        field = "country_name "
    
    else:
        sector_total_join = ""
        table_join = ""
        field = "country_name "

    if sum_column == "Reduction Potential":
        alias = 'total_reduction_potential'
    else:
        alias = 'total_emissions_quantity'


    sector_summary = f'''
        select cast('Total' as string) as Region
            , sum(case when a.sector = 'agriculture' then a.{sum_column} else 0 end) as agriculture
            , sum(case when a.sector = 'buildings' then a.{sum_column} else 0 end) as buildings
            , sum(case when a.sector = 'fluorinated-gases' then a.{sum_column} else 0 end) as fluorinated_gases
            , sum(case when a.sector = 'fossil-fuel-operations' then a.{sum_column} else 0 end) as fossil_fuel_operations
            , sum(case when a.sector = 'manufacturing' then a.{sum_column} else 0 end) as manufacturing
            , sum(case when a.sector = 'mineral-extraction' then a.{sum_column} else 0 end) as mineral_extraction
            , sum(case when a.sector = 'power' then a.{sum_column} else 0 end) as power
            , sum(case when a.sector = 'transportation' then a.{sum_column} else 0 end) as transportation
            , sum(case when a.sector = 'waste' then a.{sum_column} else 0 end) as waste
            , sum(case when a.sector <> 'forestry-and-land-use' then a.{sum_column} else 0 end) as total_exc_forestry
            , sum(case when a.sector = 'forestry-and-land-use' then a.{sum_column} else 0 end) forestry_and_land_use
            , sum(a.{sum_column}) as {alias}
            , count(distinct a.asset_id) asset_count

        from (
            select distinct a.asset_id
                , a.gadm_1
                , a.gadm_2
                , a.sector
                , a.{sum_column}
                , a.most_granular
            
            from '{annual_asset_path}' a
            {sector_total_join}
        ) a

        where most_granular is true
        
    '''

    table_summary = f'''
        select {field} as Region
            , sum(case when sector = 'agriculture' then {sum_column} else 0 end) as agriculture
            , sum(case when sector = 'buildings' then {sum_column} else 0 end) as buildings
            , sum(case when sector = 'fluorinated-gases' then {sum_column} else 0 end) as fluorinated_gases
            , sum(case when sector = 'fossil-fuel-operations' then {sum_column} else 0 end) as fossil_fuel_operations
            , sum(case when sector = 'manufacturing' then {sum_column} else 0 end) as manufacturing
            , sum(case when sector = 'mineral-extraction' then {sum_column} else 0 end) as mineral_extraction
            , sum(case when sector = 'power' then {sum_column} else 0 end) as power
            , sum(case when sector = 'transportation' then {sum_column} else 0 end) as transportation
            , sum(case when sector = 'waste' then {sum_column} else 0 end) as waste
            , sum(case when sector <> 'forestry-and-land-use' then {sum_column} else 0 end) as total_exc_forestry
            , sum(case when sector = 'forestry-and-land-use' then {sum_column} else 0 end) forestry_and_land_use
            , sum({sum_column}) as {alias}
            , count(distinct asset_id) asset_count

        from (
            select distinct asset_id
                , gadm_1
                , gadm_2
                , g20
                , sector
                , country_name
                , {sum_column}
            
            from '{annual_asset_path}'

            where most_granular is true
        ) a

        {table_join}

        group by {field}

        order by {alias} desc
        '''

    heatmap_sql = {
        "sector_summary": sector_summary,
        "table_summary": table_summary
    }

    return heatmap_sql


# def create_heatmap_sql(country_selected_bool,
#                        state_selected_bool,
#                        g20_bool,
#                        region_condition,
#                        selected_state_province,
#                        annual_asset_path,
#                        gadm_1_path=None,
#                        gadm_2_path=None,
#                        selected_metric='reductions'):
    
#     # agriculture
#     # buildings
#     # fluorinated-gases
#     # fossil-fuel-operations
#     # manufacturing
#     # mineral-extraction
#     # power
#     # transportation
#     # waste
#     # forestry-and-land-use

#     if state_selected_bool:
#         iso3_country = region_condition['column_value']

#         sector_total_join = f'''
#             INNER JOIN (
#                 SELECT distinct gadm_2_id gadm_id

#                 FROM '{gadm_2_path}'

#                 where iso3_country = '{iso3_country}'
#                     and gadm_1_corrected_name = '{selected_state_province}'
#             ) gadm_2
#                 on a.gadm_2 = gadm_2.gadm_id
#         '''

#         table_join = f'''
#             INNER JOIN (
#                 SELECT distinct gadm_2_id gadm_id
#                     , gadm_2_corrected_name gadm_2_name

#                 FROM '{gadm_2_path}'

#                 where iso3_country = '{iso3_country}'
#                     and gadm_1_corrected_name = '{selected_state_province}'
#             ) gadm_2
#                 on a.gadm_2 = gadm_2.gadm_id
#         '''

#         field = 'gadm_2.gadm_2_name '
    
#     elif country_selected_bool:
#         iso3_country = region_condition['column_value']

#         sector_total_join = f'''
#             INNER JOIN (
#                 SELECT distinct gadm_id

#                 FROM '{gadm_1_path}'

#                 where iso3_country = '{iso3_country}'
#             ) gadm_1
#                 on a.gadm_1 = gadm_1.gadm_id
#         '''

#         table_join = f'''
#             INNER JOIN (
#                 SELECT distinct gadm_id
#                     , gadm_1_corrected_name gadm_1_name

#                 FROM '{gadm_1_path}'

#                 where iso3_country = '{iso3_country}'
#             ) gadm_1
#                 on a.gadm_1 = gadm_1.gadm_id
#         '''

#         field = 'gadm_1.gadm_1_name '

#     elif g20_bool:
#         sector_total_join = f'''
#             WHERE a.g20 = true
#         '''

#         table_join = f'''
#             WHERE a.g20 = true
#         '''

#         field = "country_name "
    
#     else:
#         sector_total_join = ""
#         table_join = ""
#         field = "country_name "

#     if selected_metric == 'reductions':
#         metric = 'total_emissions_reduced_per_year'
#         metric_total = 'total_reduction_potential'
#     else:
#         metric = 'emissions_quantity'
#         metric_total = 'total_emissions'

#     sector_summary = f'''
#         select cast('Total' as string) as Region
#             , sum(case when a.sector = 'agriculture' then a.{metric} else 0 end) as agriculture
#             , sum(case when a.sector = 'buildings' then a.{metric} else 0 end) as buildings
#             , sum(case when a.sector = 'fluorinated-gases' then a.{metric} else 0 end) as fluorinated_gases
#             , sum(case when a.sector = 'fossil-fuel-operations' then a.{metric} else 0 end) as fossil_fuel_operations
#             , sum(case when a.sector = 'manufacturing' then a.{metric} else 0 end) as manufacturing
#             , sum(case when a.sector = 'mineral-extraction' then a.{metric} else 0 end) as mineral_extraction
#             , sum(case when a.sector = 'power' then a.{metric} else 0 end) as power
#             , sum(case when a.sector = 'transportation' then a.{metric} else 0 end) as transportation
#             , sum(case when a.sector = 'waste' then a.{metric} else 0 end) as waste
#             , sum(case when a.sector <> 'forestry-and-land-use' then a.{metric} else 0 end) as total_exc_forestry
#             , sum(case when a.sector = 'forestry-and-land-use' then a.{metric} else 0 end) forestry_and_land_use
#             , sum(a.{metric}) {metric_total}
#             , count(distinct a.asset_id) asset_count

#         from (
#             select distinct a.asset_id
#                 , a.gadm_1
#                 , a.gadm_2
#                 , a.sector
#                 , a.{metric}
#                 , a.most_granular
            
#             from '{annual_asset_path}' a
#             {sector_total_join}
#         ) a

#         where most_granular is true
        
#     '''

#     table_summary = f'''
#         select {field} as Region
#             , sum(case when sector = 'agriculture' then {metric} else 0 end) as agriculture
#             , sum(case when sector = 'buildings' then {metric} else 0 end) as buildings
#             , sum(case when sector = 'fluorinated-gases' then {metric} else 0 end) as fluorinated_gases
#             , sum(case when sector = 'fossil-fuel-operations' then {metric} else 0 end) as fossil_fuel_operations
#             , sum(case when sector = 'manufacturing' then {metric} else 0 end) as manufacturing
#             , sum(case when sector = 'mineral-extraction' then {metric} else 0 end) as mineral_extraction
#             , sum(case when sector = 'power' then {metric} else 0 end) as power
#             , sum(case when sector = 'transportation' then {metric} else 0 end) as transportation
#             , sum(case when sector = 'waste' then {metric} else 0 end) as waste
#             , sum(case when sector <> 'forestry-and-land-use' then {metric} else 0 end) as total_exc_forestry
#             , sum(case when sector = 'forestry-and-land-use' then {metric} else 0 end) forestry_and_land_use
#             , sum({metric}) {metric_total}
#             , count(distinct asset_id) asset_count

#         from (
#             select distinct asset_id
#                 , gadm_1
#                 , gadm_2
#                 , g20
#                 , sector
#                 , country_name
#                 , {metric}
            
#             from '{annual_asset_path}'

#             where most_granular is true
#         ) a

#         {table_join}

#         group by {field}

#         order by {metric_total} desc
#         '''

#     heatmap_sql = {
#         "sector_summary": sector_summary,
#         "table_summary": table_summary
#     }

#     return heatmap_sql

'''
This is ad-hoc for Ting to download subsector reductions by country by given percentile
'''
def build_subsector_reduction_percentile_download(
                                    annual_asset_path,
                                    dropdown_join,
                                    reduction_where_sql,
                                    table,
                                    where_sql,
                                    percentile_path,
                                    percentile_col,
                                    selected_proportion,
                                    benchmark_join,
                                    exclude_forestry
                                ):
    
    if exclude_forestry:
        forestry_where = f'''
            where coalesce(a.sector, ce.sector) <> 'forestry-and-land-use' 
        '''

    else:
        forestry_where = ""
    
    subsector_reduction_sql_string = f'''
            SELECT 
                coalesce(a.sector, ce.sector) sector,
                coalesce(a.subsector, ce.subsector) subsector,
                sum(ce.country_emissions_quantity) as country_emissions_quantity,
                SUM(a.asset_emissions_quantity) AS asset_emissions_quantity,
                SUM(a.emissions_reduction_potential) AS emissions_reduction_potential

            
            from (

                select sector
                    , subsector
                    , sum(emissions_quantity) asset_emissions_quantity
                    , sum(emissions_reduction_potential) emissions_reduction_potential

                FROM (
                    SELECT 
                        ae.asset_id,
                        ae.sector,
                        ae.subsector,
                        ae.iso3_country,
                        ae.country_name,

                        CASE 
                            WHEN BOOL_OR(ae.activity_is_temporal) 
                                THEN SUM(ae.activity) 
                                ELSE AVG(ae.activity) 
                        END AS activity,

                        AVG(ae.ef_12_moer) AS ef_12_moer,

                        CASE 
                            WHEN AVG(ae.ef_12_moer) IS NULL 
                                THEN SUM(ae.emissions_quantity)
                            ELSE (
                                CASE 
                                    WHEN BOOL_OR(ae.activity_is_temporal) 
                                        THEN SUM(ae.activity) 
                                        ELSE AVG(ae.activity) 
                                END
                            ) * AVG(ae.ef_12_moer)
                        END AS emissions_quantity,

                        GREATEST(
                            0,
                            CASE 
                                WHEN AVG(ae.ef_12_moer) IS NULL 
                                THEN (
                                    SUM(ae.emissions_quantity)
                                    - (
                                        CASE 
                                            WHEN BOOL_OR(ae.activity_is_temporal)
                                                THEN SUM(ae.activity * pct.{percentile_col})
                                                ELSE AVG(ae.activity * pct.{percentile_col})
                                        END
                                    )
                                ) * ({selected_proportion} / 100.0)

                                ELSE (
                                    (
                                        CASE 
                                            WHEN BOOL_OR(ae.activity_is_temporal)
                                                THEN SUM(ae.activity) * AVG(ae.ef_12_moer)
                                                ELSE AVG(ae.activity) * AVG(ae.ef_12_moer)
                                        END
                                    )
                                    - (
                                        CASE 
                                            WHEN BOOL_OR(ae.activity_is_temporal)
                                                THEN SUM(ae.activity * pct.{percentile_col})
                                                ELSE AVG(ae.activity * pct.{percentile_col})
                                        END
                                    )
                                ) * ({selected_proportion} / 100.0)
                            END
                        ) AS emissions_reduction_potential

                    FROM '{annual_asset_path}' ae
                    LEFT JOIN '{percentile_path}' pct
                        ON ae.subsector = pct.original_inventory_sector
                        AND ae.asset_type_2 = pct.asset_type
                        {benchmark_join}
                    {dropdown_join}

                    {reduction_where_sql}

                    GROUP BY 
                        ae.asset_id,
                        ae.sector,
                        ae.subsector,
                        ae.iso3_country,
                        ae.country_name
                ) asset_level

                group by sector
                    , subsector
            ) a
            
            full outer join (
                SELECT 
                    year,
                    sector,
                    subsector,
                    SUM(emissions_quantity) AS country_emissions_quantity
                FROM '{table}'
                
                {where_sql}
                
                GROUP BY year, sector, subsector
            ) ce
                on ce.sector = a.sector
                and ce.subsector = a.subsector

            {forestry_where}

            GROUP BY coalesce(a.sector, ce.sector),
                coalesce(a.subsector, ce.subsector)

            order by coalesce(a.sector, ce.sector),
                coalesce(a.subsector, ce.subsector)
        '''
        # print(sector_reduction_sql_string)

    
    return subsector_reduction_sql_string

def get_ownership_sql(annual_asset_path, ownership_path):
    query_ownership_sql = f'''
    SELECT
        ao.asset_id,
        ae.asset_name,
        ae.asset_type,
        ae.sector,
        ae.subsector,
        ae.lat_lon,
        ae.iso3_country,
        ae.gadm_1,
        ae.gadm_2,
        ao.parent_name,
        ao.parent_entity_id,
        ao.parent_entity_type,
        ao.parent_lei,
        ao.parent_registration_country,
        ao.parent_headquarter_country,
        ao.immediate_source_owner,
        ao.immediate_source_owner_entity_id,
        ao.source_operator,
        ao.source_operator_id,
        ao.overall_share_percent,
        SUM(ae.emissions_quantity) AS emissions_quantity,
        SUM(ae.activity) AS activity,
        ae.activity_units,
        coalesce(total_emissions_reduced_per_year,0) AS net_reduction_potential
    FROM '{ownership_path}' ao
    LEFT JOIN '{annual_asset_path}' ae
        ON ao.asset_id = ae.asset_id
    WHERE
        ae.year = 2024
        AND ae.reduction_q_type = 'asset'
    GROUP BY
        ao.asset_id,
        ae.asset_name,
        ae.asset_type,
        ae.sector,
        ae.subsector,
        ae.lat_lon,
        ae.iso3_country,
        ae.gadm_1,
        ae.gadm_2,
        ao.parent_name,
        ao.parent_entity_id,
        ao.parent_entity_type,
        ao.parent_lei,
        ao.parent_registration_country,
        ao.parent_headquarter_country,
        ao.immediate_source_owner,
        ao.immediate_source_owner_entity_id,
        ao.source_operator,
        ao.source_operator_id,
        ao.overall_share_percent,
        ae.activity_units,
        coalesce(total_emissions_reduced_per_year,0)
    ORDER BY
        ae.sector,
        ae.subsector,
        ae.iso3_country,
        ao.asset_id;
    '''
    return query_ownership_sql

def get_gadm_emissions_sql(gadm_0_path):
    query_ct_emissions = f'''
    SELECT
        ge.iso3_country,
        ge.subsector,
        SUM(ge.asset_activity) AS activity,
        SUM(ge.asset_emissions) AS emissions_quantity,
    FROM '{gadm_0_path}' ge
    WHERE
        ge.year = 2024
        AND ge.gas = 'co2e_100yr'
    GROUP BY
        ge.iso3_country,
        ge.subsector
    ORDER BY
        ge.iso3_country,
        ge.subsector;
    '''
    return query_ct_emissions
