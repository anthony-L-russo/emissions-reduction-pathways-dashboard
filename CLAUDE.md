# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Instructions for Claude

- Always prefer DuckDB queries over Pandas when working with large parquet files
- Never modify files under /data/raw
- If connected to the postgres database never run WRITE, UPDATE, or OVERWRITE commands. Your use of postgres should be READ ONLY
- Streamlit apps must remain single-page unless explicitly requested
- Use existing color palette in config/theme.py
- Avoid adding new dependencies without asking

## Project Overview

This is an internal Climate TRACE dashboard built with Streamlit for analyzing emissions reduction pathways. The application visualizes emissions data across sectors, regions, and assets, providing multiple analytical views including abatement curves, heat maps, ownership analysis, and monthly trends.

**Tech Stack**: Python 3.12, Streamlit, DuckDB, PostgreSQL, Pandas, Plotly, Geopandas

**Deployment**:
- Production: Fly.io (`main` branch)
- Staging: Streamlit Cloud (`stage` branch)

## Development Setup

### Environment Setup

1. Create and activate virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure PostgreSQL credentials (required for data pipeline):
   ```bash
   export CLIMATETRACE_USER="your_username"
   export CLIMATETRACE_PASS="your_password"
   export CLIMATETRACE_HOST="your_postgres_host"
   export CLIMATETRACE_PORT="your_port"
   export CLIMATETRACE_DB="your_database_name"
   ```

### Running the Application

```bash
# Run locally
streamlit run app.py

# The app will be available at http://localhost:8501
```

## Architecture

### Application Structure

```
├── app.py                  # Landing page with navigation cards
├── pages/                  # Streamlit multi-page app structure
│   ├── 0_Home.py
│   ├── 1_Sector_Reduction_Pathways.py
│   ├── 2_Abatement_Curve.py
│   ├── 3_Heat_Map.py
│   ├── 4_Monthly_Trends.py
│   ├── 5_Ownership.py
│   └── 6_Global_Trends.py
├── tabs/                   # Reusable visualization components
│   ├── tab01_emissions_reduction_tab.py
│   ├── tab02_abatement_curve_tab.py
│   ├── tab03_monthly_dashboard_tab.py
│   ├── tab04_asset_ownership.py
│   ├── tab05_abatement_curve_demo.py
│   ├── tab06_reduction_heatmap.py
│   └── tab07_monthly_trends_v2.py  # Backs 6_Global_Trends.py
├── utils/                  # Helper modules
│   ├── queries.py         # SQL query builders
│   ├── run_sql.py         # PostgreSQL connection
│   ├── utils.py           # General utilities
│   └── header.py          # App header rendering
├── config.py              # Central configuration
└── data/                  # Parquet files organized by type
```

### Data Flow

1. **PostgreSQL Database**: Source data is queried via `utils/run_sql.py` using environment variables for credentials
2. **Local Parquet Files**: Pre-processed data stored in `data/` subdirectories, queried via DuckDB
3. **Config-Driven Paths**: All data paths defined in `config.py` using glob patterns for parquet files

### Key Design Patterns

- **Query Building**: Dynamic SQL generation in `utils/queries.py` based on user selections (region, sector, year)
- **Region Mapping**: `utils.py` contains `map_region_condition()` which maps UI selections to database columns/values. It returns a dict with `column_name`, `column_value`, and optionally `is_subregion: True` for UN subregion selections. Subregions require a JOIN against `data/country_region_mapping/country_region_mapping.parquet` — they cannot be filtered with a simple `column = value` clause. Pass `subregion_list=CONFIG['subregion_options']` to enable subregion detection.
- **`is_country()` limitation**: `utils.py:is_country()` only has a hardcoded exclusion list (continents, blocs). Subregion names are not in that list so `is_country()` incorrectly returns `True` for them. Check for `is_subregion` from `map_region_condition()` instead when subregion awareness is needed.
- **DuckDB Caching** (`tab07`): All DuckDB calls go through three module-level `@st.cache_data(ttl=3600)` helpers — `_cached_df(sql)`, `_cached_fetchone(sql)`, `_cached_fetchall(sql)`. The SQL string is the cache key. Use this pattern in any new tab that queries parquet files to avoid re-running queries on every widget interaction.
- **Streamlit Pages**: Each page in `pages/` corresponds to a different analytical view

## Data Pipeline

### Monthly Data Updates

The data pipeline is executed via `data/refresh_data.ipynb` and typically runs ~1.5 hours.

**Prerequisites**:
1. Production tables must be frozen
2. Data Fusion must complete Monthly Statistics process
3. Reductions tables must be ready: `reductions_data_fusion`, `gadm_reductions_data_fusion`, `city_reductions_data_fusion`

**Required Local Folders** (gitignored):
- `data/zzz_archive` - Holds previous versions of parquet files
- `data/zzz_landing_zone` - Temporary workspace for CSV to parquet conversion

**Process**:
1. Download statistics CSVs from Data Fusion:
   - `country_subsector_emissions_statistics_XXXXXX.csv`
   - `country_subsector_emissions_totals_XXXXXX.csv`
   - `gadm_1_emissions_statistics_XXXXXX.csv`

2. Place CSVs into `data/zzz_landing_zone/`

3. Run `data/refresh_data.ipynb` (all cells)

4. Validate outputs and test locally

5. Commit and deploy following branching strategy

## Deployment

### Branching Strategy

**Two types of merges**:

1. **Full Merge** (new features, bug fixes, code changes):
   - Feature branch → `stage`
   - Test on Streamlit Cloud
   - `stage` → `main`
   - Deploy to Fly.io

2. **Data-Only Merge** (monthly data releases):
   - Create branch off `stage`: `data-update-VX.X`
   - Merge into `stage`
   - **Cherry-pick** the data commit into `main` (do NOT merge `stage` into `main`)
   - This prevents pulling unrelated `stage` code into production

### Deploying to Stage

1. Merge changes into `stage` branch

2. Reboot the Streamlit app:
   - Via Streamlit Cloud dashboard, or
   - Within the app's "Manage App" drawer

3. Test at: https://climate-trace-sandbox-stage.streamlit.app/

### Deploying to Production

1. Ensure `main` branch is updated (via full merge or cherry-pick)

2. Authenticate with Fly.io:
   ```bash
   fly auth login
   ```

3. Deploy:
   ```bash
   fly deploy
   ```
   (Reads configuration from `fly.toml`)

4. Verify deployment:
   ```bash
   fly status
   fly logs
   ```

5. Test at: https://climate-trace-emissions-reduction-pathways-beta.fly.dev/

### Fly.io Configuration

- **App name**: `climate-trace-emissions-reduction-pathways-beta`
- **Region**: Dallas (dfw)
- **Resources**: 4GB RAM, 2 shared CPUs
- **Port**: 8501 (Streamlit default)
- **Auto-scaling**: Enabled (min 0 machines)

## Common Tasks

### Adding a New Visualization

1. If it's a new page:
   - Create file in `pages/` following naming convention: `N_Page_Name.py`
   - Number prefix determines order in sidebar
   - Add navigation card to `app.py` if needed

2. If it's a reusable component:
   - Create in `tabs/` with descriptive name
   - Import and use in relevant pages

### Modifying SQL Queries

- Query builders are in `utils/queries.py`
- Follow existing pattern of building dynamic SQL based on parameters
- Use f-strings for table paths and WHERE clauses
- Database connection handled by `utils/run_sql.py`

### Adding New Data Sources

1. Update `config.py` with new path pattern
2. Add data processing to `data/refresh_data.ipynb`
3. Update relevant query builders in `utils/queries.py`

### Working with DuckDB

DuckDB is used to query parquet files directly:
```python
import duckdb
con = duckdb.connect()
result = con.execute(sql_query).df()
```

Paths from `config.py` can be used directly in SQL:
```python
from config import CONFIG
sql = f"SELECT * FROM '{CONFIG['asset_path']}'"
```

## Important Notes

- **No test suite**: This project does not have automated tests. Validate changes manually via local testing and staging deployment.

- **Data privacy**: This is an internal tool. All pages include disclaimer about data being for internal Climate TRACE use only.

- **Version tracking**: The app displays Climate TRACE release version (e.g., V5.2.0) pulled from data files via `utils.utils.get_release_version()`.

- **Caffeinate during data pipeline**: The notebook can run for 1.5 hours. Use `caffeinate -dims` to prevent laptop sleep.

- **Docker**: Multi-stage build defined in `Dockerfile` for efficient production images.

- **Streamlit config**: `.streamlit/config.toml` sets headless mode, port 8501, disables CORS for deployment, and sets `[client] showSidebarNavigation = false`. All pages also set `initial_sidebar_state="collapsed"` in `st.set_page_config()` — both are required to prevent the sidebar from flashing on page navigation.

- **Known syntax error**: `utils/utils.py` ~line 1005 has a nested-quote f-string bug inside `plot_abatement_curve()` — outer quotes are single quotes but the expression uses `df['activity_units']`. This function is used by the abatement curve tabs, not tab07.
