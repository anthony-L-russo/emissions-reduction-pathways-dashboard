# Deployment Instructions

---

## 🧩 Setup (Before Deployment)

Before deploying or merging any changes, make sure you’re properly signed into Streamlit and Fly.io.

### 1️⃣ Streamlit Setup

1. Go to Streamlit Cloud: https://share.streamlit.io
2. Log in or sign up using your WattTime email, the same email you use in the Github account that pushes to this repo.
3. Once logged in, you should see the `emissions-reduction-pathways-dashboard` listed under your apps.
4. Confirm that you can open the app and access the “Manage App” drawer in the bottom-right corner. This drawer allows you to reboot the app when needed.

⚠️ Only users with push access to the GitHub repo will be able to reboot the app or see the management options.


### 2️⃣ Fly.io Setup

Fly.io handles production deployments for the app. Follow these steps to get set up locally:

1. **Install Fly CLI**
   - **macOS (if you have Homebrew):**  `brew install flyctl`
   - **macOS (without Homebrew):**  `curl -L https://fly.io/install.sh | sh`
   - After running the install script, verify the command: `which fly`
   - If you see an error or no path is returned, add Fly CLI to your PATH: `export PATH="$HOME/.fly/bin:$PATH"`
   - Then verify again: `which fly`

2. **Authenticate with Fly.io**
   - Once installed, log in to your Fly.io account: `fly auth login`
   - This will open a browser where you can sign in. Use your **WattTime credentials** and confirm you have access to the `climate-trace` organization and the app `climate-trace-emissions-reduction-pathways-beta`.
   - Verify you’re logged in: `fly auth whoami`. You should see your Fly.io username or email listed.


---

## 🧭 Merge Types

Before deploying, determine which type of merge is needed. This ensures we use the correct merge strategy for `stage` and `main`.

### 1️⃣ Full Merge (New Feature Releases)

Use this when you are deploying **new features, UI updates, logic changes, bug fixes**, or anything that affects the app beyond `/data/`.

Workflow:
- Merge feature branches → `stage`
- Reboot and test the staging app
- Merge `stage` → `main`
- See deployment directions

### 2️⃣ Data-Only Merge (Monthly Data Releases)

Use this when only updating files in the `/data/` directory and no code changes are required. This merge type should be used for monthly data releases.

**Important:** Do not merge `stage` into `main` during data-only updates. Instead, merge the branch once into `stage`, and cherry-pick its commit into `main`.

#### Steps:

1. Create a data update branch off `stage`
   - `git checkout stage`
   - `git pull origin stage`
   - `git checkout -b data-update-<VX.X>`
      - Example Branch: `data-update-V5.1`

2. [Run data pipeline](https://github.com/WattTime/emissions-reduction-pathways-dashboard/tree/stage/data#readme), then commit/push:
   - `git add data/`
   - `git commit -m "Data update: <description>"`
   - `git push origin data-update-<VX.X>`

3. Merge this branch into `stage`
   - `git checkout stage`
   - `git pull origin stage`
   - `git merge data-update-<VX.X>`
   - `git push origin stage`

4. Reboot the staging app in Streamlit Cloud to load the updated data

5. Cherry-pick the exact same commit into `main`

   First, find the commit hash from the data-update branch:
   - `git checkout data-update-<VX.X>`
   - `git log --oneline`

   The commit hash is the short alphanumeric code at the start of the line.  
   Example output:  
      - `3f9a2b1  Data update: refreshed 11-2025 data for V5.1`  
   In this example, the commit hash is: **3f9a2b1**

   Now apply that commit to `main`:
   - `git checkout main`
   - `git pull origin main`
   - `git cherry-pick <commit-hash>`
   - `git push origin main`

⚠️ This avoids pulling unrelated `stage` code into `main` when only data needs updating.

---



## 🚀 Deploying to Stage

1. Make sure all appropriate commits are merged into the `stage` branch.

2. In Streamlit Cloud or within the staging version of the app, navigate to the **Reboot** option:
   - If using Streamlit Cloud, click the **three dots** next to the app name and select **Reboot**.
   - If you are already inside the staging app, open the **Manage App** drawer in the bottom-right corner, click the **three dots**, and select **Reboot app**. This will pull in the latest commits from the `stage` branch into the UI.

3. This will pull in the latest commits from the `stage` branch into the UI. Stress-test any new features in the staging environment to confirm stability.
   - Stage Link: [https://climate-trace-sandbox-stage.streamlit.app/](https://climate-trace-sandbox-stage.streamlit.app/)
   
   💡 If it works (or crashes) in `stage`, it will likely behave the same in `main`.


---


## 🌍 Deploying to Production (main)

1. Choose the correct merge strategy above, and complete the steps.

2. On your local machine, run:
   - `git checkout main`
   - `git pull origin main`. Run or test the app locally to ensure everything works as expected.

3. Run deployment command: `fly deploy`
   - This will run for ~5 minutes. Fly will automatically read the existing `fly.toml` configuration file to build and deploy the latest version.

5. Verify the deployment:  
   - `fly status`  
   - `fly logs`
   - Visit the production app to ensure new data/features are reflected: https://climate-trace-emissions-reduction-pathways-beta.fly.dev/

---


