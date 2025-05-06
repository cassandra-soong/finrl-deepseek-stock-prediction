#!/bin/bash

# --- Configuration ---
CONDA_ENV_NAME="finrl_stock_pipeline_env"
PYTHON_VERSION="3.10"
# Assumes script is run from the root of the git repository (e.g., finrl-deepseek-stock-prediction)
BASE_DIR=$(pwd)
API_KEYS_FILE="../API_Keys.txt" # Path to the keys file relative to BASE_DIR
TIMESTAMP_DIR="$BASE_DIR/.run_timestamps" # Directory to store last run times

# Git Configuration
GIT_BRANCH="ander_branch" # Set the branch name you want to pull from and push to
LOG_FILE="$BASE_DIR/pipeline_run.log" # Log file in the project directory

export CUDA_VISIBLE_DEVICES="3" # Set the GPU device to use (if applicable)

# --- Schedule Control Variables (Intervals in Minutes) ---
SCRAPE_INTERVAL_MINUTES=780    # Approx 13 hours
SENTIMENT_INTERVAL_MINUTES=1440 # Approx 24 hours
FINRL_INTERVAL_MINUTES=10080   # Approx 7 days

# --- Timestamp File Paths ---
mkdir -p "$TIMESTAMP_DIR" # Ensure the directory exists
LAST_RUN_SCRAPE_FILE="$TIMESTAMP_DIR/.last_run_scrape"
LAST_RUN_SENTIMENT_FILE="$TIMESTAMP_DIR/.last_run_sentiment"
LAST_RUN_FINRL_FILE="$TIMESTAMP_DIR/.last_run_finrl"

# --- Logging Function ---
log_message() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# --- Initial Setup ---
log_message "--- Starting Pipeline Execution ---"
log_message "Project Directory (BASE_DIR): $BASE_DIR"
log_message "Git Branch: $GIT_BRANCH"
log_message "Log File: $LOG_FILE"
log_message "API Keys File: $API_KEYS_FILE"
log_message "Timestamp Directory: $TIMESTAMP_DIR"
log_message "Scrape Interval: $SCRAPE_INTERVAL_MINUTES minutes"
log_message "Sentiment Interval: $SENTIMENT_INTERVAL_MINUTES minutes"
log_message "FinRL Interval: $FINRL_INTERVAL_MINUTES minutes"

# --- Load API Keys & Secrets from File ---
log_message ">>> Loading API keys from '$API_KEYS_FILE'..."
if [ ! -f "$API_KEYS_FILE" ]; then
    log_message "ERROR: API keys file not found at '$API_KEYS_FILE'. Exiting."
    echo "Error: API keys file not found at '$API_KEYS_FILE'." >&2
    exit 1
fi
set -a # Export variables sourced from the file
# shellcheck source=/dev/null
source "$API_KEYS_FILE"
set +a
log_message "API keys loaded and exported."

# --- Check if a key was loaded (example) ---
if [ -z "$NEWSAPI_KEY" ]; then # Make sure NEWSAPI_KEY is defined in your API_Keys.txt
    log_message "ERROR: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'. Exiting."
    echo "Error: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'." >&2
    exit 1 # It's critical for the scraper
fi

# --- Function to check if a stage should run ---
should_run_stage() {
    local stage_name="$1"
    local interval_minutes="$2"
    local timestamp_file="$3"
    local current_time
    current_time=$(date +%s)
    local last_run_time=0
    local interval_seconds=$((interval_minutes * 60))

    if [ -f "$timestamp_file" ]; then
        last_run_time=$(cat "$timestamp_file")
    fi

    local time_diff=$((current_time - last_run_time))

    log_message ">>> Checking schedule for $stage_name..."
    echo "--- Checking schedule for $stage_name ---" # VERBOSE
    log_message "    Current Time: $current_time ($(date -d @"$current_time"))"
    log_message "    Last Run Time: $last_run_time ($(date -d @"$last_run_time" 2>/dev/null || echo 'Never'))"
    log_message "    Time Difference: $time_diff seconds"
    log_message "    Required Interval: $interval_seconds seconds ($interval_minutes minutes)"

    if [ "$last_run_time" -eq 0 ] || [ "$time_diff" -ge "$interval_seconds" ]; then
        log_message "    Decision: Time interval met or first run. $stage_name should run."
        echo "Decision: Time interval met or first run. $stage_name will run." # VERBOSE
        return 0 # Success (should run)
    else
        log_message "    Decision: Time interval not met. Skipping $stage_name."
        echo "Decision: Time interval not met. Skipping $stage_name for this cycle." # VERBOSE
        return 1 # Failure (should not run)
    fi
}

# --- Function to Add, Commit, and Push changes to Git ---
git_commit_and_push() {
    local stage_name_for_commit="$1"
    local commit_msg_prefix="$2" 

    log_message "Attempting Git operations for stage: $stage_name_for_commit"
    echo "--- Attempting Git operations for stage: $stage_name_for_commit ---" # VERBOSE
    ( 
        set -e 
        cd "$BASE_DIR" 

        log_message "Adding all relevant changes to Git staging area..."
        echo "Git: Adding all relevant changes to staging area (git add -A)..." # VERBOSE
        git add -A 
        
        if git diff --staged --quiet; then
            log_message "No changes staged for commit after $stage_name_for_commit."
            echo "Git: No changes staged for commit after $stage_name_for_commit." # VERBOSE
        else
            log_message "Committing changes for $stage_name_for_commit..."
            local COMMIT_MSG
            COMMIT_MSG="$commit_msg_prefix - Automated run $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Git: Committing changes with message: '$COMMIT_MSG'" # VERBOSE
            git commit -m "$COMMIT_MSG"
            log_message "Pushing changes to Git branch '$GIT_BRANCH'..."
            echo "Git: Pushing changes to branch '$GIT_BRANCH'..." # VERBOSE
            git push origin "$GIT_BRANCH"
            log_message "Git push for $stage_name_for_commit successful."
            echo "Git: Push for $stage_name_for_commit successful." # VERBOSE
        fi
        log_message "Git operations for $stage_name_for_commit finished."
        echo "--- Git operations for $stage_name_for_commit finished. ---" # VERBOSE
    ) || {
        log_message "ERROR: Git operations (add/commit/push) failed for stage: $stage_name_for_commit."
        echo "ERROR: Git operations (add/commit/push) failed for stage: $stage_name_for_commit. Check $LOG_FILE for details." >&2 
        return 1 
    }
    return 0 
}

# --- Create/Activate Conda Environment ---
log_message ">>> Setting up/Activating Conda environment '$CONDA_ENV_NAME'..."
echo "--- Setting up/Activating Conda environment '$CONDA_ENV_NAME' ---" 
if ! conda env list | grep -q "^$CONDA_ENV_NAME\s"; then
    log_message "Creating new Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    echo "Creating new Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..." 
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    log_message ">>> Installing dependencies..."
    echo "Conda env created. Installing dependencies..." 
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    pip install --upgrade pip
    log_message "Installing requirements for scrape..."
    echo "Installing requirements from scrape/requirements.txt..." 
    pip install -r "$BASE_DIR/scrape/requirements.txt"
    log_message "Installing requirements for sentiment..."
    echo "Installing requirements from sentiment/requirements.txt..." 
    pip install -r "$BASE_DIR/sentiment/requirements.txt"
    log_message "Installing requirements for finrl..."
    echo "Installing requirements from finrl/requirements.txt..." 
    pip install -r "$BASE_DIR/finrl/requirements.txt"
    conda deactivate
    log_message "Dependencies installed. Conda environment '$CONDA_ENV_NAME' created and populated."
    echo "All dependencies installed. Conda environment '$CONDA_ENV_NAME' populated." 
else
    log_message "Conda environment '$CONDA_ENV_NAME' already exists."
    echo "Conda environment '$CONDA_ENV_NAME' already exists." 
fi

log_message "Activating Conda environment '$CONDA_ENV_NAME' for script execution..."
echo "Activating Conda environment '$CONDA_ENV_NAME' for script execution..." 
eval "$(conda shell.bash hook)" 
conda activate "$CONDA_ENV_NAME"
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    log_message "ERROR: Failed to activate Conda environment '$CONDA_ENV_NAME'. Exiting."
    echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'. Exiting." >&2
    exit 1
fi
log_message "Conda environment '$CONDA_ENV_NAME' activated."
echo "Conda environment '$CONDA_ENV_NAME' activated." 

# --- Git Pull ---
log_message "Attempting Git pull from origin '$GIT_BRANCH'..."
echo "--- Attempting Git pull from origin '$GIT_BRANCH' ---"

GIT_PULL_SEQUENCE_FAILED=false
(
    set -e 
    cd "$BASE_DIR"

    log_message "Stashing local changes to TRACKED files (if any)..."
    echo "Git: Stashing local changes to TRACKED files (using git stash push --keep-index)..."
    # MODIFIED: Removed --include-untracked.
    # This will only stash changes to tracked files (staged or unstaged).
    # Ignored files like pipeline_run.log will be left alone by stash.
    # If no tracked files changed, "No local changes to save" is expected.
    git stash push --keep-index -m "Auto-stash (tracked files) before pull by pipeline_run.sh"

    log_message "Pulling from origin '$GIT_BRANCH' with rebase..."
    echo "Git: Pulling from origin '$GIT_BRANCH' with rebase..."
    git pull origin "$GIT_BRANCH" --rebase

    log_message "Attempting to pop stashed changes (if any)..."
    echo "Git: Attempting to pop stashed changes (if any)..."
    # Try to pop. If no stash was created, this will state "No stash found." and not be an error.
    # If a real conflict occurs during pop, `set -e` will cause the subshell to exit.
    if ! git stash pop; then
        # This block might be reached if 'git stash pop' itself has an issue but doesn't exit with error (e.g. "No stash found" prints to stderr but exits 0)
        # or if set -e is not catching a specific type of pop failure.
        # A more robust check for "No stash found" could be added if necessary.
        # For now, any message from `git stash pop` indicating failure (other than "No stash found") is a concern.
        log_message "INFO: Stash pop command executed. It might have failed due to a conflict or reported 'No stash found'. Manual check of 'git status' may be needed if issues arise."
        echo "Git INFO: Stash pop command executed. Check 'git status' if unexpected behavior occurs."
    else
        log_message "Stash popped successfully." # Or no stash was present.
        echo "Git: Stash popped successfully (or no stash was present)."
    fi
    log_message "Git pull and stash pop sequence completed in subshell."
) || {
    log_message "ERROR: Git operations (stash/pull/pop) sequence FAILED in subshell. Check git status. Continuing with current local state."
    echo "ERROR: Git operations (stash/pull/pop) sequence FAILED in subshell. Check $LOG_FILE and git status. Continuing with current local state." >&2
    GIT_PULL_SEQUENCE_FAILED=true
}

if [ "$GIT_PULL_SEQUENCE_FAILED" = true ]; then
    log_message "Git pull sequence encountered an error. The script will continue, but please check repository state and logs."
    echo "--- Git pull sequence encountered an error. Check logs and git status. ---"
else
    log_message "Git pull sequence finished." # This implies success or non-critical pop issue like 'no stash'.
    echo "--- Git pull sequence finished. ---"
fi

# --- Execute Stages Based on Interval Checks ---
# (The rest of the script for Scraper, Sentiment, FinRL, and Deactivate Conda Env remains the same as your provided version)

# (1) Scraper
if should_run_stage "Scraper" "$SCRAPE_INTERVAL_MINUTES" "$LAST_RUN_SCRAPE_FILE"; then
    log_message ">>> (1/3) Running news scraping script..."
    echo ">>> (1/3) Starting News Scraping script (scrape/scrape_script.py)..."
    cd "$BASE_DIR/scrape" || { log_message "ERROR: Failed to cd to $BASE_DIR/scrape. Skipping scraper."; echo "ERROR: Failed to cd to $BASE_DIR/scrape. Skipping scraper." >&2; }
    python scrape_script.py
    exit_code=$?
    cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; exit 1; }

    if [ $exit_code -eq 0 ]; then
        log_message "Scraping finished successfully."
        echo "News Scraping script (scrape/scrape_script.py) completed successfully."
        date +%s > "$LAST_RUN_SCRAPE_FILE" 
        log_message "Updated last run time for Scraper."
        git_commit_and_push "Scraper" "Scraper"
    else
        log_message "ERROR: Scraping script failed with exit code $exit_code."
        echo "ERROR: News Scraping script (scrape/scrape_script.py) failed with exit code $exit_code. Check $LOG_FILE." >&2
    fi
fi

# (2) Sentiment Analysis
if should_run_stage "Sentiment Analysis" "$SENTIMENT_INTERVAL_MINUTES" "$LAST_RUN_SENTIMENT_FILE"; then
    log_message ">>> (2/3) Interval met for Sentiment Analysis. Checking prerequisites..."
    echo ">>> (2/3) Checking prerequisites for Sentiment Analysis..."
    if [ -f "$BASE_DIR/scrape/news.csv" ]; then
        log_message "Input 'scrape/news.csv' found. Running Sentiment Analysis..."
        echo "Input 'scrape/news.csv' found. Starting Sentiment Analysis script (sentiment/main.py)..."
        mkdir -p "$BASE_DIR/sentiment"
        cp "$BASE_DIR/scrape/news.csv" "$BASE_DIR/sentiment/news.csv"
        
        cd "$BASE_DIR/sentiment" || { log_message "ERROR: Failed to cd to $BASE_DIR/sentiment. Skipping sentiment analysis."; echo "ERROR: Failed to cd to $BASE_DIR/sentiment. Skipping sentiment analysis." >&2; }
        python main.py
        exit_code=$?
        cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; exit 1; }

        if [ $exit_code -eq 0 ]; then
            log_message "Sentiment analysis finished successfully."
            echo "Sentiment Analysis script (sentiment/main.py) completed successfully."
            date +%s > "$LAST_RUN_SENTIMENT_FILE"
            log_message "Updated last run time for Sentiment Analysis."
            git_commit_and_push "Sentiment Analysis" "Sentiment"
        else
            log_message "ERROR: Sentiment analysis script failed with exit code $exit_code."
            echo "ERROR: Sentiment Analysis script (sentiment/main.py) failed with exit code $exit_code. Check $LOG_FILE." >&2
        fi
    else
        log_message "ERROR: Cannot run Sentiment Analysis. Input 'scrape/news.csv' not found."
        echo "Skipping Sentiment Analysis: Input 'scrape/news.csv' not found."
        log_message "Skipping Sentiment Analysis run for this cycle."
    fi
fi

# (3) FinRL
if should_run_stage "FinRL" "$FINRL_INTERVAL_MINUTES" "$LAST_RUN_FINRL_FILE"; then
    log_message ">>> (3/3) Interval met for FinRL. Checking prerequisites..."
    echo ">>> (3/3) Checking prerequisites for FinRL..."
    if [ -f "$BASE_DIR/sentiment/aggregated_risk_scores.csv" ]; then
        log_message "Input 'sentiment/aggregated_risk_scores.csv' found. Running FinRL..."
        echo "Input 'sentiment/aggregated_risk_scores.csv' found. Starting FinRL script (finrl/main.py)..."
        mkdir -p "$BASE_DIR/finrl"
        cp "$BASE_DIR/sentiment/aggregated_risk_scores.csv" "$BASE_DIR/finrl/aggregated_risk_scores.csv"
        
        cd "$BASE_DIR/finrl" || { log_message "ERROR: Failed to cd to $BASE_DIR/finrl. Skipping FinRL."; echo "ERROR: Failed to cd to $BASE_DIR/finrl. Skipping FinRL." >&2; }
        python main.py
        exit_code=$?
        cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; exit 1; }

        if [ $exit_code -eq 0 ]; then
            log_message "FinRL process finished successfully."
            echo "FinRL script (finrl/main.py) completed successfully."
            date +%s > "$LAST_RUN_FINRL_FILE"
            log_message "Updated last run time for FinRL."
            git_commit_and_push "FinRL" "FinRL"
        else
            log_message "ERROR: FinRL script failed with exit code $exit_code."
            echo "ERROR: FinRL script (finrl/main.py) failed with exit code $exit_code. Check $LOG_FILE." >&2
        fi
    else
        log_message "ERROR: Cannot run FinRL. Input 'sentiment/aggregated_risk_scores.csv' not found."
        echo "Skipping FinRL: Input 'sentiment/aggregated_risk_scores.csv' not found."
        log_message "Skipping FinRL run for this cycle."
    fi
fi

# --- Deactivate Conda Environment ---
log_message ">>> Deactivating Conda environment..."
echo "--- Deactivating Conda environment '$CONDA_DEFAULT_ENV' ---" 
conda deactivate

log_message "--- Pipeline execution check finished ---"
echo "--- Pipeline execution check finished ---" 
exit 0