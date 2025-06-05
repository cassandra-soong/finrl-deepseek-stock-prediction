
# 📰 Nvidia Financial News Scraper

This project is a Python-based web scraping tool that collects financial news articles and Reddit posts related to **NVIDIA (NVDA)** from multiple online sources. It is designed to extract relevant information such as timestamps, full article text, sources, and links and store it in a structured CSV file. The script runs automatically every 13 hours, making it ideal for future sentiment analysis.

---

## 🔍 Purpose

The purpose of this project is to gather relevant financial news and Reddit posts about **NVIDIA (NVDA)** for **sentiment analysis**. The sentiment data will be used to influence the behavior of a **FinRL agent**. By incorporating real-time market sentiment from various sources, this project aims to improve the agent’s trading strategies.

---

## 📦 Features

### ➤ **Multi-source scraping**  
- **Reddit** using the PRAW API  
- **RSS Feeds** (e.g., Yahoo Finance, Investopedia)  
- **NewsAPI** for media outlets (Forbes, Fool, NDTV, etc.)

### ➤ **Keyword filtering**  
Ensures only relevant content is included by scanning both titles and article body for keywords like `nvidia`, `nvda`, and `stock`.

### ➤ **Duplicate detection with hashing**  
Previously seen articles are stored in a `seen_hashes.json` file to avoid duplication.

### ➤ **Data**

The `news.csv` file contains the following information:

- **Date and Timestamp**: The date and time the article or post was published.
- **Title**: The title of the article or Reddit post.
- **Full Article Text**: The body text of the article or Reddit post.
- **Source**: Indicates whether the data comes from **News** or **Reddit**.
- **Specific Source**: The specific news service or subreddit where the content was sourced from.

### ➤ **Automated Execution with bash script**  
 - Runs every every 13 hours.
 - Setup of automated run time in `crontab -e`.
 - Previously automated on **Github Actions**.

---

### Install Dependencies

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

Make sure you are in the right directory:

```bash
cd scrape
```

To run the script, run the following command:

```bash
python scrape_script.py
```

To run the scraper part in the automated bash script in the event of manual run or rerun from failed executions, run the following command:

```bash
bash start_script.sh scrape
```
