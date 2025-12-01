# 🚀 Data Pipeline — Setup & Run Guide

This document explains how to set up the local environment, prepare required folders, run the data pipeline, and push updated data into the staging environment.

---

## 1.&nbsp;&nbsp; 📁&nbsp; Local Setup<br>  

### 1.1&nbsp;&nbsp;&nbsp;Configure Postgres Credentials via Environmental Variables

The pipeline reads credentials from environment variables. Ensure you have the following variables set in your shell (`.zshrc`, `.bashrc`, or VSCode environment settings):

```bash
export CLIMATETRACE_USER="your_username"
export CLIMATETRACE_PASS="your_password"
export CLIMATETRACE_HOST="your_postgres_host"
export CLIMATETRACE_PORT="your_port"
export CLIMATETRACE_DB="your_database_name"
```

### 1.2&nbsp;&nbsp;&nbsp;Add Required Local Folders 

Inside the repository's `data/` directory, manually create the following 2 folders:
```
data/
  zzz_archive
  zzz_landing_zone
```

- `zzz_archive`: This is a holding area for previous versions of parquet files. Whenever the pipeline is run, existing files are moved to this folder, and provides a way to restore older data versions if needed.
- `zzz_landing_zone`: Temporary workspace where the pipeline writes intermediate parquet files before processing them further and organizing them into their final folder destination. It is also temporary storage for the statistics files that data fusion creates. The script will convert these files into parquets, and place them in the appropriate location.

Each of these 3 folders should be visbile within the `.gitignore`, and none of their contents should ever be committed to github.

---

## 2.&nbsp;&nbsp; 🏃&nbsp; Running the Pipeline
<br>
⚠️🚨🚨 WARNING 🚨🚨⚠️<br>
This step should only be started when: 

- The production tables have been frozen
- Data Fusion has completed the Monthly Statistics process
- Data Fusion has indicated that the reductions tables are ready. This includes: `reductions_data_fusion`, `gadm_reductions_data_fusion`, and `city_reductions_data_fusion`.

Running this process before the above is complete will result in outdated/incorrect data.<br>


### 2.1&nbsp;&nbsp;&nbsp;Create a Branch off `stage`
```
git checkout stage
git pull
git checkout -b data-update-VX.X.X
```
Example branch name: `data-update-V5.2.0`

### 2.2&nbsp;&nbsp;&nbsp;Download the latest statstics files and place them into the `zzz_landing_zone` folder. The statistics files you need are:
  ```
  country_subsector_emissions_statistics_XXXXXX.csv
  country_subsector_emissions_totals_XXXXXX.csv
  gadm_1_emissions_statistics_XXXXXX.csv
  ```
  There is no need to change the file names/dates, just drag and drop into `zzz_landing_zone` folder and the code will handle them!

### 2.3&nbsp;&nbsp;&nbsp;Run the Notebook
  - Open the `refresh_data.ipynb` file within the `data/` folder
  - Execute the entire file (run all cells). The expected runtime is ~1.5 hours
  - If you need to step away from your laptop while it runs, run `caffeinate -dims` within your command line to prevent your laptop from going to sleep. Just remember to disable this command when you're done.

### 2.4&nbsp;&nbsp;&nbsp;Validate Output
  - After completion, new parquet files should appear in the appropriate folders
  - Scroll through the notebook and verify no cells failed or produced an error
  - Run the Steamlit app locally to ensure modules load and the Climate TRACE version updates correctly

    Example, if running for V5.2.0, this is your expected text at top of Abatement Curve module:
     ```
     The data in this dashboard is from Climate TRACE release V5.2.0 (excluding forestry), covering 740 million assets globally.
     ```
  - If everything loads without errors, proceed to push.

### 2.5&nbsp;&nbsp;&nbsp;Commit and Deploy to Stage
  - Push your branch:
     ```
     git add .
     git commit -m "Updating data for V5.2.0"
     git push --set-upstream origin data-update-V5.2.0
     ```
  - Merge branch into `stage` branch. If using the github UI, be careful, it usually defaults to try to merge into the `main` branch. Make sure you set it to merge into `stage`

### 2.6&nbsp;&nbsp;&nbsp;Reboot the `stage` App in Streamlit 
  - It can be rebooted by signing into [Streamlit](https://streamlit.io/) or directly in the [stage UI](https://climate-trace-sandbox-stage.streamlit.app/) (as long as you are already signed into Streamlit and have push access to this repo).
  - Test new data in its staging environment
  - Use Monthly Trends module for Monthly Press Release

### 2.7&nbsp;&nbsp;&nbsp;For deployment into prod/main via Fly.io, see the Deployment Instructions and start at the [Data Only Merge section](https://github.com/anthony-L-russo/emissions-reduction-pathways-dashboard/tree/stage?tab=readme-ov-file#2%EF%B8%8F%E2%83%A3-data-only-merge-monthly-data-releases)
      






