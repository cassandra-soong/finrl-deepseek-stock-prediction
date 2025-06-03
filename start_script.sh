#!/bin/bash
 
# --- Configuration ---
CONDA_PATH="/home/group1/ENTER/bin/conda" 
CONDA_ENV_NAME="finrl_stock_pipeline_env"
PYTHON_VERSION="3.10"

# Assumes script is run from the root of the git repository (e.g., finrl-deepseek-stock-prediction)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
API_KEYS_FILE="$BASE_DIR/../API_Keys.txt"  # Path to the keys file relative to BASE_DIR

# Git Configuration
GIT_BRANCH="main" # Set the branch name you want to pull from and push to
LOG_FILE="$BASE_DIR/../main_branch.log" # Log file in the project directory

export CUDA_VISIBLE_DEVICES="0" # Set the GPU device to use (if applicable)

# --- Logging Function ---
log_message() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# --- Argument Check for Stage ---
if [ -z "$1" ]; then
    log_message "ERROR: No stage specified. Usage: $0 <scrape|sentiment|finrl>"
    echo "ERROR: No stage specified." >&2
    echo "Usage: $0 <scrape|sentiment|finrl>" >&2
    exit 1
fi
STAGE_TO_RUN="$1"

# --- Initial Setup ---
log_message "--- Starting Pipeline Execution for stage: $STAGE_TO_RUN ---"
log_message "Project Directory (BASE_DIR): $BASE_DIR"
log_message "Git Branch: $GIT_BRANCH"
log_message "Log File: $LOG_FILE"
log_message "API Keys File: $API_KEYS_FILE"

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
if [ "$STAGE_TO_RUN" = "scrape" ] && [ -z "$NEWSAPI_KEY" ]; then # Make sure NEWSAPI_KEY is defined in your API_Keys.txt
    log_message "ERROR: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'. Exiting as it's critical for the scraper."
    echo "Error: NEWSAPI_KEY was not found after loading '$API_KEYS_FILE'. Exiting as it's critical for the scraper." >&2
    exit 1 # It's critical for the scraper
fi

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

# Initialize Conda - IMPORTANT: Ensure conda is findable.
# If conda is not in PATH for cron, you might need to source the main conda.sh or use absolute path to conda binary.
if [ -f "/home/group1/ENTER/etc/profile.d/conda.sh" ]; then
    source "/home/group1/ENTER/etc/profile.d/conda.sh"
else
    log_message "WARNING: Conda profile script not found at /home/group1/ENTER/etc/profile.d/conda.sh. Conda commands might fail."
    echo "WARNING: Conda profile script not found. Conda commands might fail."
fi


# Check if environment exists
if ! conda env list | grep -qE "^$CONDA_ENV_NAME\s"; then
    log_message "Creating new Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    echo "Creating new Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y

    log_message ">>> Installing dependencies..."
    echo "Conda env created. Installing dependencies..."

    # Temporarily activate to install packages
    # Use eval with conda shell hook for robust activation
    eval "$($CONDA_PATH shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"

    pip install --upgrade pip
    pip install -r "$BASE_DIR/scrape/requirements.txt"
    pip install -r "$BASE_DIR/sentiment/requirements.txt"
    pip install -r "$BASE_DIR/finrl/requirements.txt"
    pip install -r "$BASE_DIR/dashboard/requirements.txt" 
    conda deactivate

    log_message "Dependencies installed. Conda environment '$CONDA_ENV_NAME' created and populated."
else
    log_message "Conda environment '$CONDA_ENV_NAME' already exists."
    echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi

# Activate for script execution
log_message "Activating Conda environment '$CONDA_ENV_NAME' for script execution..."
echo "Activating Conda environment '$CONDA_ENV_NAME' for script execution..."
eval "$($CONDA_PATH shell.bash hook)" # Use eval with conda shell hook
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
    git stash push --keep-index -m "Auto-stash (tracked files) before pull by pipeline_run.sh"

    log_message "Pulling from origin '$GIT_BRANCH' with rebase..."
    echo "Git: Pulling from origin '$GIT_BRANCH' with rebase..."
    git pull origin "$GIT_BRANCH" --rebase

    log_message "Attempting to pop stashed changes (if any)..."
    echo "Git: Attempting to pop stashed changes (if any)..."
    if ! git stash pop; then
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
    log_message "Git pull sequence finished."
    echo "--- Git pull sequence finished. ---"
fi

# --- Execute Specific Stage ---

case "$STAGE_TO_RUN" in
    scrape)
        log_message ">>> (1/3) Running news scraping script (STAGE: scrape)..."
        echo ">>> (1/3) Starting News Scraping script (scrape/scrape_script.py)..."
        cd "$BASE_DIR/scrape" || { log_message "ERROR: Failed to cd to $BASE_DIR/scrape. Skipping scraper."; echo "ERROR: Failed to cd to $BASE_DIR/scrape. Skipping scraper." >&2; conda deactivate; exit 1; }
        python scrape_script.py
        exit_code=$?
        cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; conda deactivate; exit 1; }

        if [ $exit_code -eq 0 ]; then
            log_message "Scraping finished successfully."
            echo "News Scraping script (scrape/scrape_script.py) completed successfully."
            # Timestamp file for interval logic is removed
            git_commit_and_push "Scraper" "Scraper"
        else
            log_message "ERROR: Scraping script failed with exit code $exit_code."
            echo "ERROR: News Scraping script (scrape/scrape_script.py) failed with exit code $exit_code. Check $LOG_FILE." >&2
        fi
        ;;

    sentiment)
        log_message ">>> (2/3) Running Sentiment Analysis (STAGE: sentiment)..."
        echo ">>> (2/3) Checking prerequisites for Sentiment Analysis..."
        if [ -f "$BASE_DIR/scrape/news.csv" ]; then
            log_message "Input 'scrape/news.csv' found. Running Sentiment Analysis..."
            echo "Input 'scrape/news.csv' found. Starting Sentiment Analysis script (sentiment/main.py)..."
            mkdir -p "$BASE_DIR/sentiment"

            cd "$BASE_DIR/sentiment" || { log_message "ERROR: Failed to cd to $BASE_DIR/sentiment. Skipping sentiment analysis."; echo "ERROR: Failed to cd to $BASE_DIR/sentiment. Skipping sentiment analysis." >&2; conda deactivate; exit 1; }
            python main.py
            exit_code=$?
            cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; conda deactivate; exit 1; }

            if [ $exit_code -eq 0 ]; then
                log_message "Sentiment analysis finished successfully."
                echo "Sentiment Analysis script (sentiment/main.py) completed successfully."
                # Timestamp file for interval logic is removed
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
        ;;

    finrl)
        log_message ">>> (3/3) Running FinRL (STAGE: finrl)..."
        echo ">>> (3/3) Checking prerequisites for FinRL..."
        if [ -f "$BASE_DIR/sentiment/aggregated_risk_scores.csv" ]; then
            log_message "Input 'sentiment/aggregated_risk_scores.csv' found. Running FinRL..."
            echo "Input 'sentiment/aggregated_risk_scores.csv' found. Starting FinRL script (finrl/main.py)..."
            mkdir -p "$BASE_DIR/finrl"

            cd "$BASE_DIR/finrl" || { log_message "ERROR: Failed to cd to $BASE_DIR/finrl. Skipping FinRL."; echo "ERROR: Failed to cd to $BASE_DIR/finrl. Skipping FinRL." >&2; conda deactivate; exit 1; }
            python main.py
            exit_code=$?
            cd "$BASE_DIR" || { log_message "ERROR: Failed to cd back to $BASE_DIR. Critical error."; echo "CRITICAL ERROR: Failed to cd back to $BASE_DIR. Exiting." >&2; conda deactivate; exit 1; }

            if [ $exit_code -eq 0 ]; then
                log_message "FinRL process finished successfully."
                echo "FinRL script (finrl/main.py) completed successfully."
                # Timestamp file for interval logic is removed
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
        ;;

    *)
        log_message "ERROR: Invalid stage '$STAGE_TO_RUN' provided. Valid stages are: scrape, sentiment, finrl."
        echo "ERROR: Invalid stage '$STAGE_TO_RUN' provided." >&2
        echo "Valid stages are: scrape, sentiment, finrl." >&2
        conda deactivate
        exit 1
        ;;
esac

# --- Deactivate Conda Environment ---
log_message ">>> Deactivating Conda environment..."
echo "--- Deactivating Conda environment '$CONDA_DEFAULT_ENV' ---"
conda deactivate

log_message "--- Pipeline execution for stage '$STAGE_TO_RUN' finished ---"
echo "--- Pipeline execution for stage '$STAGE_TO_RUN' finished ---"
exit 0