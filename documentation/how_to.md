# How to run the automation

## Option 1: via Github Workflow
1. Add your secret keys such as Reddit username and password to Github "Secrets and Variables" settings.

2. Rename the folder `github_workflows` to `.github/workflows` to activate the yaml files in your github Actions.

Note that the free version of Github is only limited to 2000 minutes of workflow per month and does not have GPU support. The LLMs in the sentiment automation pipeline require GPU support and are unable to run on the free version of Github Action.


## Option 2: via Bash Script
1. Install conda in specific folder in your environment, if not installed. For example:
```bash
# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# set it up in a folder of your choice
bash miniconda.sh -b -p /home/group1/ENTER   
```

2. Modify the configuration variables in `start_scipt.sh` to match your file paths: 
```bash
CONDA_PATH="/home/group1/ENTER/bin/conda"   # Path to your conda installed
API_KEYS_FILE="$BASE_DIR/../API_Keys.txt"   # Create a txt file to store the secret keys
GIT_BRANCH="main" # Set the branch name you want to pull from and push to
LOG_FILE="$BASE_DIR/../main_branch.log"     # Create a log file to track logs
```

3. A virtual environment will be created when `start_scipt.sh` is called, if it does not exist. Otherwise, you can also manually create the virtual environment and install dependencies.
```bash
# create virtual environment
/home/group1/ENTER/bin/conda create -n finrl_stock_pipeline_env python=3.10 -y

# activate conda
eval "$(/home/group1/ENTER/bin/conda shell.bash hook)"
conda activate finrl_stock_pipeline_env

# install dependencies
pip install -r "scrape/requirements.txt"
pip install -r "sentiment/requirements.txt"
pip install -r "finrl/requirements.txt"
pip install -r "dashboard/requirements.txt" 
```

4. Enter this command to open the crontab.
```bash
crontab -e
```

5. Insert the following commands in the crontab to automate the pipelines at their scheduled time.
```bash
# Scraper: every 13 hours
0 */13 * * * /bin/bash /home/group1/finrl-deepseek-stock-prediction/start_script.sh scrape

# Sentiment: every Tues-Sat at 10:00 AM UTC
0 10 * * 2-6 /bin/bash /home/group1/finrl-deepseek-stock-prediction/start_script.sh sentiment 

# FinRL: every Thursday at 12:00 PM UTC
0 12 * * 4 /bin/bash /home/group1/finrl-deepseek-stock-prediction/start_script.sh finrl 
```

Note: The LLMs in the sentiment automation pipeline require GPU support in your device for the bash script to execute successfully.