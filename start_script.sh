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
  # Ensure log directory exists (though LOG_FILE is in BASE_DIR here)
  # mkdir -p "$(dirname "$LOG_FILE")" # Not strictly needed if LOG_FILE is in BASE_DIR
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# --- Initial Setup ---
# Navigate to project directory (already there due to BASE_DIR=$(pwd))
# cd "$BASE_DIR" || { echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to navigate to initial project directory $BASE_DIR. Exiting." >> "$LOG_FILE"; exit 1; }

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
if [ -z "$NEWSAPI_KEY" ]; then
    log_message "ERROR: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'. Exiting."
    echo "Error: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'." >&2
    exit 1
fi

# --- Function to check if a stage should run ---
should_run_stage() {
    local stage_name="$1"
    local interval_minutes="$2"
    local timestamp_file="$3"
    local current_time=$(date +%s)
    local last_run_time=0
    local interval_seconds=$((interval_minutes * 60))

    if [ -f "$timestamp_file" ]; then
        last_run_time=$(cat "$timestamp_file")
    fi

    local time_diff=$((current_time - last_run_time))

    log_message ">>> Checking schedule for $stage_name..."
    log_message "    Current Time: $current_time ($(date -d @$current_time))"
    log_message "    Last Run Time: $last_run_time ($(date -d @$last_run_time 2>/dev/null || echo 'Never'))"
    log_message "    Time Difference: $time_diff seconds"
    log_message "    Required Interval: $interval_seconds seconds ($interval_minutes minutes)"

    if [ "$last_run_time" -eq 0 ] || [ "$time_diff" -ge "$interval_seconds" ]; then
        log_message "    Decision: Time interval met or first run. $stage_name should run."
        return 0 # Success (should run)
    else
        log_message "    Decision: Time interval not met. Skipping $stage_name."
        return 1 # Failure (should not run)
    fi
}

# --- Function to Add, Commit, and Push changes to Git ---
git_commit_and_push() {
    local stage_name_for_commit="$1"
    local commit_msg_prefix="$2" # e.g., "Scraper", "Sentiment Analysis"

    log_message "Attempting Git operations for stage: $stage_name_for_commit"
    ( # Use subshell for git operations to isolate potential errors and ensure cd back
        set -e # Exit subshell on error
        cd "$BASE_DIR" # Ensure we are in the correct directory

        log_message "Adding all changes to Git staging area..."
        git add -A
        # Check if there are changes to commit
        if git diff --staged --quiet; then
            log_message "No changes staged for commit after $stage_name_for_commit."
        else
            log_message "Committing changes for $stage_name_for_commit..."
            local COMMIT_MSG="$commit_msg_prefix - Automated run $(date '+%Y-%m-%d %H:%M:%S')"
            git commit -m "$COMMIT_MSG"
            log_message "Pushing changes to Git branch '$GIT_BRANCH'..."
            git push origin "$GIT_BRANCH"
            log_message "Git push for $stage_name_for_commit successful."
        fi
        log_message "Git operations for $stage_name_for_commit finished."
    ) || {
        log_message "ERROR: Git operations (add/commit/push) failed for stage: $stage_name_for_commit."
        # Decide if this failure should stop the whole script.
        # For now, it just logs and continues.
        # To exit, uncomment the next line:
        # exit 1
        return 1 # Indicate failure
    }
    return 0 # Indicate success
}


# --- Create/Activate Conda Environment ---
log_message ">>> Setting up/Activating Conda environment '$CONDA_ENV_NAME'..."
if ! conda env list | grep -q "^$CONDA_ENV_NAME\s"; then
    log_message "Creating new Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    log_message ">>> Installing dependencies..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    pip install --upgrade pip
    log_message "Installing requirements for scrape..."
    pip install -r "$BASE_DIR/scrape/requirements.txt"
    log_message "Installing requirements for sentiment..."
    pip install -r "$BASE_DIR/sentiment/requirements.txt"
    log_message "Installing requirements for finrl..."
    pip install -r "$BASE_DIR/finrl/requirements.txt"
    # Deactivate after install, will activate again for script execution
    conda deactivate
    log_message "Dependencies installed. Conda environment '$CONDA_ENV_NAME' created and populated."
else
    log_message "Conda environment '$CONDA_ENV_NAME' already exists."
fi

log_message "Activating Conda environment '$CONDA_ENV_NAME' for script execution..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    log_message "ERROR: Failed to activate Conda environment '$CONDA_ENV_NAME'. Exiting."
    echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'." >&2
    exit 1
fi
log_message "Conda environment '$CONDA_ENV_NAME' activated."

# --- Git Pull ---
log_message "Attempting Git pull from origin '$GIT_BRANCH'..."
echo "--- Attempting Git pull from origin '$GIT_BRANCH' ---" # VERBOSE

# Temporarily move pipeline_run.log if it exists
LOG_FILE_TEMP_NAME=""
ORIGINAL_LOG_EXISTS=false
if [ -f "$LOG_FILE" ]; then
    ORIGINAL_LOG_EXISTS=true
    LOG_FILE_TEMP_NAME="${LOG_FILE}.temp_pull_$(date +%s)_$$" # Added PID for more uniqueness
    # Logging this message to the log file before it's moved.
    log_message "Temporarily moving current $LOG_FILE to $LOG_FILE_TEMP_NAME during git pull operations."
    echo "Git: Temporarily moving $LOG_FILE to $LOG_FILE_TEMP_NAME" # VERBOSE
    mv "$LOG_FILE" "$LOG_FILE_TEMP_NAME"
fi

( # Use subshell for git operations
    set -e # Exit subshell on error
    cd "$BASE_DIR" # Ensure we are in the correct directory

    log_message "Stashing local changes (if any)... (pipeline_run.log is temporarily moved)" # Log will go to new/empty log if created by pull
    echo "Git: Stashing local changes (if any)... (pipeline_run.log is out of the way)" # VERBOSE
    # Now that pipeline_run.log is moved, --include-untracked is safer for other potential untracked files.
    git stash push --keep-index --include-untracked -m "Auto-stash before pull by pipeline_run.sh"
    
    log_message "Pulling from origin '$GIT_BRANCH' with rebase..." # Log will go to new/empty log
    echo "Git: Pulling from origin '$GIT_BRANCH' with rebase..." # VERBOSE
    git pull origin "$GIT_BRANCH" --rebase
    
    log_message "Attempting to pop stashed changes..." # Log will go to new/empty log
    echo "Git: Attempting to pop stashed changes..." # VERBOSE
    if ! git stash pop; then
        log_message "INFO: No stash to pop, or stash pop resulted in conflicts that need manual resolution, or stash pop failed for other reasons. Check git status." # Log will go to new/empty log
        echo "Git INFO: No stash to pop, or pop failed (may require manual conflict resolution). Check git status." # VERBOSE
    else
        log_message "Stash popped successfully." # Log will go to new/empty log
        echo "Git: Stash popped successfully." # VERBOSE
    fi
    log_message "Git pull sequence completed in subshell." # Log will go to new/empty log
) || {
    # This log message might go to a new log file if the subshell created one, or it won't be logged if LOG_FILE doesn't exist yet.
    log_message "ERROR: Git pull sequence failed. Check for conflicts or other issues. Continuing with current local state."
    echo "ERROR: Git pull sequence failed. Check current logs and git status. Continuing with current local state." >&2 # VERBOSE
}

# Restore pipeline_run.log
if [ "$ORIGINAL_LOG_EXISTS" = true ]; then # Only attempt restoration if it originally existed
    if [ -f "$LOG_FILE_TEMP_NAME" ]; then
        echo "Git: Restoring $LOG_FILE from $LOG_FILE_TEMP_NAME" # VERBOSE
        # If $LOG_FILE was recreated by the pull (e.g., if it was briefly untracked but committed on another branch by mistake)
        # and now exists, we append the temp log's content to it, then remove temp.
        # Otherwise, just move temp back.
        if [ -f "$LOG_FILE" ]; then
            echo "Git: $LOG_FILE exists, appending content from $LOG_FILE_TEMP_NAME." # VERBOSE
            cat "$LOG_FILE_TEMP_NAME" >> "$LOG_FILE"
            rm "$LOG_FILE_TEMP_NAME"
        else
            mv "$LOG_FILE_TEMP_NAME" "$LOG_FILE"
        fi
        log_message "Restored $LOG_FILE from temporary file. Git operations finished." # This goes to the restored/appended log
    else
        # This case should be rare if ORIGINAL_LOG_EXISTS was true.
        # The log_message here will create a new LOG_FILE if it doesn't exist.
        log_message "INFO: Temporary log file $LOG_FILE_TEMP_NAME was not found for restoration, though original log existed. A new log file may have been created by git pull if it was tracked."
        echo "Git INFO: Temporary log file $LOG_FILE_TEMP_NAME not found for restoration." # VERBOSE
    fi
else
    log_message "Git operations finished. No pre-existing log file was moved." # Creates new log if it doesn't exist
fi
echo "--- Git pull sequence and log restoration finished. ---" # VERBOSE

# --- Execute Stages Based on Interval Checks ---

# (1) Scraper
if should_run_stage "Scraper" "$SCRAPE_INTERVAL_MINUTES" "$LAST_RUN_SCRAPE_FILE"; then
    log_message ">>> (1/3) Running news scraping script..."
    cd "$BASE_DIR/scrape" || { log_message "ERROR: Failed to cd to $BASE_DIR/scrape. Skipping scraper."; }
    python scrape_script.py
    exit_code=$?
    cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; exit 1; }

    if [ $exit_code -eq 0 ]; then
        log_message "Scraping finished successfully."
        date +%s > "$LAST_RUN_SCRAPE_FILE" # Update timestamp on success
        log_message "Updated last run time for Scraper."
        # Attempt to commit and push changes
        git_commit_and_push "Scraper" "Scraper"
    else
        log_message "ERROR: Scraping script failed with exit code $exit_code."
    fi
fi

# (2) Sentiment Analysis
if should_run_stage "Sentiment Analysis" "$SENTIMENT_INTERVAL_MINUTES" "$LAST_RUN_SENTIMENT_FILE"; then
    log_message ">>> (2/3) Interval met for Sentiment Analysis. Checking prerequisites..."
    if [ -f "$BASE_DIR/scrape/news.csv" ]; then
        log_message "Input 'scrape/news.csv' found. Running Sentiment Analysis..."
        # Ensure the target directory exists
        mkdir -p "$BASE_DIR/sentiment"
        cp "$BASE_DIR/scrape/news.csv" "$BASE_DIR/sentiment/news.csv"
        
        cd "$BASE_DIR/sentiment" || { log_message "ERROR: Failed to cd to $BASE_DIR/sentiment. Skipping sentiment analysis."; }
        python main.py
        exit_code=$?
        cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; exit 1; }

        if [ $exit_code -eq 0 ]; then
            log_message "Sentiment analysis finished successfully."
            date +%s > "$LAST_RUN_SENTIMENT_FILE" # Update timestamp on success
            log_message "Updated last run time for Sentiment Analysis."
            # Attempt to commit and push changes
            git_commit_and_push "Sentiment Analysis" "Sentiment"
        else
            log_message "ERROR: Sentiment analysis script failed with exit code $exit_code."
        fi
    else
        log_message "ERROR: Cannot run Sentiment Analysis. Input 'scrape/news.csv' not found."
        log_message "Skipping Sentiment Analysis run for this cycle."
    fi
fi

# (3) FinRL
if should_run_stage "FinRL" "$FINRL_INTERVAL_MINUTES" "$LAST_RUN_FINRL_FILE"; then
    log_message ">>> (3/3) Interval met for FinRL. Checking prerequisites..."
    if [ -f "$BASE_DIR/sentiment/aggregated_risk_scores.csv" ]; then
        log_message "Input 'sentiment/aggregated_risk_scores.csv' found. Running FinRL..."
        # Ensure the target directory exists
        mkdir -p "$BASE_DIR/finrl"
        cp "$BASE_DIR/sentiment/aggregated_risk_scores.csv" "$BASE_DIR/finrl/aggregated_risk_scores.csv"
        
        cd "$BASE_DIR/finrl" || { log_message "ERROR: Failed to cd to $BASE_DIR/finrl. Skipping FinRL."; }
        python main.py
        exit_code=$?
        cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; exit 1; }

        if [ $exit_code -eq 0 ]; then
            log_message "FinRL process finished successfully."
            date +%s > "$LAST_RUN_FINRL_FILE" # Update timestamp on success
            log_message "Updated last run time for FinRL."
            # Attempt to commit and push changes
            git_commit_and_push "FinRL" "FinRL"
        else
            log_message "ERROR: FinRL script failed with exit code $exit_code."
        fi
    else
        log_message "ERROR: Cannot run FinRL. Input 'sentiment/aggregated_risk_scores.csv' not found."
        log_message "Skipping FinRL run for this cycle."
    fi
fi

# --- Deactivate Conda Environment ---
log_message ">>> Deactivating Conda environment..."
conda deactivate

log_message "--- Pipeline execution check finished ---"
exit 0

