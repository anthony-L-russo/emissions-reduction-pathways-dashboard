'''
THIS IS WORK IN PROGRESS TO CREATE A SCRIPT THAT WILL LOAD ALL DATA INTO APP
'''


import duckdb
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

user = quote_plus(os.getenv("CLIMATETRACE_USER"))
password = quote_plus(os.getenv("CLIMATETRACE_PASS"))
host = os.getenv("CLIMATETRACE_HOST")
port = os.getenv("CLIMATETRACE_PORT")
database = os.getenv("CLIMATETRACE_DB")

postgres_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

con = duckdb.connect()

con.execute(f'''
    INSTALL postgres;
    LOAD postgres;

    CREATE TABLE asset_emissions_parquet AS
    select extract(year from g1e.start_time) as year 
        , g1e.gadm_id
        , gb.admin_level
        , g1e.iso3_country
        , gb.corrected_name
        , asch.sector
        , g1e.original_inventory_sector
        , sum(asset_activity) asset_activity
        , sum(asset_emissions) asset_emissions
        , sum(remainder_activity) remainder_activity
        , sum(remainder_emissions) remainder_emissions
        , sum(asset_emissions) + sum(remainder_emissions) as emissions_quantity

    from postgres_scan('{postgres_url}', 'public', 'gadm_1_emissions') g1e
    inner join (
        select distinct gadm_id
            , corrected_name
            , admin_level
        from gadm_boundaries 
        where admin_level = 1
    ) as gb
        on g1e.gadm_id = gb.gadm_id
    left join (
        select distinct sector
            , subsector
        from asset_schema
    ) asch
        on cast(asch.subsector as varchar) = cast(g1e.original_inventory_sector as varchar)

    where g1e.gas = 'co2e_100yr'
        and extract(year from start_time) = {year}

    group by extract(year from g1e.start_time) 
        , g1e.gadm_id
        , gb.admin_level
        , g1e.iso3_country
        , gb.corrected_name
        , asch.sector
        , g1e.original_inventory_sector
''')














def get_gadm_2_query(year):
    
    gadm_2_query = f'''
        select extract(year from ge.start_time) as year 
            , ge.gadm_id
            , gb.admin_level
            , ge.iso3_country
            , gb.corrected_name
            , asch.sector
            , ge.original_inventory_sector
            , sum(asset_activity) asset_activity
            , sum(asset_emissions) asset_emissions
            , sum(remainder_activity) remainder_activity
            , sum(remainder_emissions) remainder_emissions
            , sum(asset_emissions) + sum(remainder_emissions) as emissions_quantity

        from gadm_emissions as ge
        inner join (
            select distinct gadm_id
                , corrected_name
                , admin_level
            from gadm_boundaries 
            where admin_level = 2
        ) as gb
            on ge.gadm_id = gb.gadm_id
        left join (
            select distinct sector
                , subsector
            from asset_schema
        ) asch
            on cast(asch.subsector as varchar) = cast(ge.original_inventory_sector as varchar)

        where ge.gas = 'co2e_100yr'
            and extract(year from start_time) = {year}

        group by extract(year from ge.start_time) 
            , ge.gadm_id
            , gb.admin_level
            , ge.iso3_country
            , gb.corrected_name
            , asch.sector
            , ge.original_inventory_sector
    '''

    return gadm_2_query








# ------------------ EXAMPLE QUERY BELOW ------------------
print("Running asset-level query and writing to parquet file, this may take a while...")
con.execute(f"""
    INSTALL postgres;
    LOAD postgres;

    CREATE TABLE asset_emissions_parquet AS
    SELECT ae.iso3_country,
        ae.original_inventory_sector,
        ae.start_time,
        ae.gas,
        sch.sector,
        ca.name as country_name,
        ca.continent,
        ca.unfccc_annex,
        ca.em_finance,
        ca.eu,
        ca.oecd,
        ca.developed_un,
        ae.release,
        sum(emissions_quantity) emissions_quantity,
        sum(activity) activity,
        sum(emissions_quantity) / sum(activity) weighted_average_emissions_factor
    
    FROM postgres_scan('{postgres_url}', 'public', 'asset_emissions') ae
    LEFT JOIN postgres_scan('{postgres_url}', 'public', 'country_analysis') ca
        ON CAST(ca.iso3_country AS VARCHAR) = CAST(ae.iso3_country AS VARCHAR)
    LEFT JOIN (
        SELECT DISTINCT sector, subsector FROM postgres_scan('{postgres_url}', 'public', 'asset_schema')
    ) sch
        ON CAST(sch.subsector AS VARCHAR) = CAST(ae.original_inventory_sector AS VARCHAR)
    
    WHERE ae.start_time >= (DATE '{max_date}' - INTERVAL '36 months')
      AND ae.gas = 'co2e_100yr'
      AND ae.most_granular = TRUE
    
    GROUP BY ae.iso3_country,
        ae.original_inventory_sector,
        ae.start_time,
        ae.gas,
        sch.sector,
        ca.name,
        ca.continent,
        ca.unfccc_annex,
        ca.em_finance,
        ca.eu,
        ca.oecd,
        ca.developed_un,
        ae.release;

    COPY asset_emissions_parquet TO '{parquet_path}' (FORMAT PARQUET);
""")
con.close()
