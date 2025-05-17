#!/bin/bash

# Define the model repository details
HF_ORG="meta-llama"
HF_MODEL="Llama-4-Scout-17B-16E-Instruct"
HF_BRANCH="main" # Or the specific branch you need

# Define your Hugging Face API token
# IMPORTANT: Replace "YOUR_HF_TOKEN" with your actual Hugging Face User Access Token.
# You can generate one in your Hugging Face account settings: Settings -> Access Tokens.
# Make sure you have accepted the model's terms on the Hugging Face website first!
HF_TOKEN="" # <-- REPLACE THIS WITH YOUR TOKEN

# --- IMPORTANT ---
# Before running this script:
# 1. Go to the model page on Hugging Face (https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct).
# 2. Log in to your Hugging Face account.
# 3. Read and accept the model's terms and conditions.
# 4. Generate a User Access Token in your Hugging Face account settings (with 'read' role or higher).
# 5. Replace "YOUR_HF_TOKEN" above with the token you generated.
# --- -------------

# Define the base URL for downloading
BASE_URL="https://huggingface.co/${HF_ORG}/${HF_MODEL}/resolve/${HF_BRANCH}/"

# Define the target directory to save the model files
TARGET_DIR="${HF_MODEL}-${HF_BRANCH}"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Change to the target directory
cd "$TARGET_DIR" || { echo "Error: Could not change directory to $TARGET_DIR"; exit 1; }

# List of files to download (based on the image provided)
# You might need to adjust this list if the repository content changes
FILES=(
    ".gitattributes"
    "README.md"
    "config.json"
    "generation_config.json"
    # Add the sharded safetensors files (adjust the range if needed)
    $(printf "model-%05d-of-00050.safetensors\n" {1..50})
    "model.safetensors.index.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
)

# Download each file using curl with authentication
for file in "${FILES[@]}"; do
    DOWNLOAD_URL="${BASE_URL}${file}"
    OUTPUT_FILE="$file" # Save with the original filename

    echo "Downloading: ${file}"

    # Use curl with the Authorization header for token authentication
    # -L: Follow redirects
    # -C -: Continue/resume transfer (like wget -c)
    # -H: Add custom header (Authorization with Bearer token)
    # -o: Write output to a local file
    curl -L -C - -H "Authorization: Bearer ${HF_TOKEN}" "$DOWNLOAD_URL" -o "$OUTPUT_FILE"

    # Check the exit status of curl
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to download ${file}"
        echo "Curl command failed. Make sure your HF_TOKEN is correct, you have accepted the model terms on HF website, and you have internet connectivity."
        # Optionally add retry logic here
    fi
done

echo "Download process finished."
echo "Model files are saved in the directory: $(pwd)"
