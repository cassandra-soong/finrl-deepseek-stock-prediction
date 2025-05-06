import csv
import os
import json
import hashlib
import time
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import praw # Ensure praw is installed
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import tldextract

# --------------------------------------------------------------------------------------
REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT"),
    "username": os.getenv("REDDIT_USERNAME"), # Optional for read-only but good practice
    "password": os.getenv("REDDIT_PASSWORD")   # Optional for read-only
}
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket", "finance"]
QUERIES = ["nvidia", "NVDA"] # Keywords to search for within subreddits
REDDIT_SCORE_MIN = 50
REDDIT_LOOKBACK_DAYS = 2
SEARCH_LIMIT_PER_QUERY = 100 # Limit how many posts to check per query to avoid very long runs

RSS_URLS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    "https://feeds-api.dotdashmeredith.com/v1/rss/google/f8466ec3-5044-46bc-94b7-2df65f770eff"
]
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWS_DOMAINS = 'fool.com,etfdailynews.com,marketplace.org,forbes.com,denverpost.com,bostonherald.com,globenewswire.com,ndtv.com,nbcnews.com'

today = datetime.now()
two_days_ago = today - timedelta(days=REDDIT_LOOKBACK_DAYS) # Sync with Reddit lookback for NewsAPI as well
FROM_DATE = two_days_ago.strftime('%Y-%m-%dT00:00:00')
TO_DATE = today.strftime('%Y-%m-%dT23:59:59')

COMBINED_OUTPUT_FILE = "news.csv"
SEEN_HASH_FILE = "seen_hashes.json"

# --------------------------------------------------------------------------------------

def compute_hash(title, link):
    if title is None or link is None:
        return None
    return hashlib.md5(f"{str(title)}{str(link)}".encode('utf-8')).hexdigest()

def load_seen_hashes():
    if os.path.exists(SEEN_HASH_FILE):
        try:
            with open(SEEN_HASH_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {SEEN_HASH_FILE}. Starting with an empty set.")
            return set()
    return set()

def save_seen_hashes(hashes):
    with open(SEEN_HASH_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(hashes), f, indent=2)

def contains_relevant_keywords(title, content):
    relevant_keywords = ['nvidia', 'nvda', 'stock'] # Keep this focused
    # Ensure title and content are strings before lowercasing
    title_str = str(title).lower() if title else ""
    content_str = str(content).lower() if content else ""
    return any(keyword in title_str or keyword in content_str for keyword in relevant_keywords)

# --------------------------------------------------------------------------------------

def extract_specific_source(link):
    if pd.isna(link) or not isinstance(link, str) or not link.strip():
        return 'Unknown'
    try:
        domain_info = tldextract.extract(link)
        # Use top_domain_under_public_suffix instead of deprecated registered_domain
        full_domain = domain_info.top_domain_under_public_suffix.lower() if domain_info.top_domain_under_public_suffix else ""

        if 'reddit.com' in full_domain:
            match = re.search(r'reddit\.com/r/([^/]+)/', link)
            return match.group(1).capitalize() if match else 'Reddit' # Capitalize subreddit name
        if 'yahoo.com' in full_domain and 'finance.yahoo.com' in link.lower():
            return 'Yahoo Finance'
        return domain_info.domain.capitalize() if domain_info.domain else 'Unknown'
    except Exception as e:
        # print(f"    Error extracting specific source from link '{link}': {e}") # Can be too verbose
        return 'Unknown'

def preprocess_new_entries(df):
    print("  Preprocessing new entries...")
    df['Full Text'] = df['Full Text'].apply(lambda x: str(x).replace('\n', ' ').replace('\r', ' ') if pd.notna(x) else '')
    df['SpecificSource'] = df['Link'].apply(extract_specific_source)
    print("  Preprocessing complete.")
    return df

# --------------------------------------------------------------------------------------

def scrape_reddit():
    print("\n--- Starting Reddit Scraper ---")
    # Check for essential PRAW config keys
    essential_keys = ["client_id", "client_secret", "user_agent"]
    if not all(REDDIT_CONFIG.get(key) for key in essential_keys):
        print("  ERROR: Reddit API client_id, client_secret, or user_agent is not set in environment variables.")
        print("  Skipping Reddit scraping.")
        return []

    try:
        reddit = praw.Reddit(**REDDIT_CONFIG)
        # Perform a simple read-only check to ensure PRAW is working
        print(f"  PRAW instance created. Read-only status: {reddit.read_only}")
        # You can try a simple fetch, e.g., reddit.subreddit("popular").hot(limit=1)
        # but be mindful of API rate limits if done frequently.
    except Exception as e:
        print(f"  ERROR: Failed to initialize PRAW. Check Reddit credentials and API status: {e}")
        return []

    min_time = datetime.utcnow() - timedelta(days=REDDIT_LOOKBACK_DAYS)
    min_timestamp = min_time.timestamp()
    results = []
    total_posts_checked = 0
    total_posts_collected = 0

    print(f"  Looking for posts in the last {REDDIT_LOOKBACK_DAYS} days with a minimum score of {REDDIT_SCORE_MIN}.")
    print(f"  Target Subreddits: {SUBREDDITS}")
    print(f"  Search Queries: {QUERIES}")

    for sub_name in SUBREDDITS:
        print(f"\n  Scraping Subreddit: r/{sub_name}")
        try:
            subreddit = reddit.subreddit(sub_name)
            for query in QUERIES:
                print(f"    Searching for query: '{query}' in r/{sub_name} (limit {SEARCH_LIMIT_PER_QUERY} posts)")
                posts_checked_this_query = 0
                try:
                    for post in subreddit.search(query, sort="new", limit=SEARCH_LIMIT_PER_QUERY):
                        total_posts_checked += 1
                        posts_checked_this_query += 1
                        # print(f"      Checking post: '{post.title[:70]}...' (Score: {post.score}, UTC: {post.created_utc})")

                        if post.created_utc < min_timestamp:
                            # print(f"        SKIP (Too Old): Post '{post.id}' from {datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d')}")
                            # Assuming search results are sorted by new, we can break early for this query
                            # However, PRAW's search might not strictly guarantee chronological order across all scenarios.
                            # To be safe, continue checking but log it.
                            # For very active subreddits and broad queries, breaking might be efficient.
                            # For now, let's continue but this is a point of optimization if needed.
                            continue 
                        if post.score < REDDIT_SCORE_MIN:
                            # print(f"        SKIP (Low Score): Post '{post.id}' score {post.score} < {REDDIT_SCORE_MIN}")
                            continue
                        if not post.selftext or not post.selftext.strip():
                            # print(f"        SKIP (No Text): Post '{post.id}' has no selftext.")
                            continue

                        # If all checks pass:
                        dt = datetime.utcfromtimestamp(post.created_utc)
                        print(f"      COLLECTED (r/{sub_name}, Q: '{query}'): '{post.title[:60]}...' (Score: {post.score})")
                        results.append({
                            "Date and Timestamp": dt.strftime('%Y-%m-%d %H:%M:%S'),
                            "Title": post.title,
                            "Full Text": post.selftext.strip().replace('\n', ' '), # Clean newlines
                            "Source": "Reddit", # Generic source
                            "Link": f"https://www.reddit.com{post.permalink}" # Ensure full URL
                        })
                        total_posts_collected += 1
                except Exception as e:
                    print(f"    ERROR searching query '{query}' in r/{sub_name}: {e}")
                if posts_checked_this_query == 0:
                    print(f"    No posts found for query '{query}' in r/{sub_name} matching search criteria within the set limit.")

        except Exception as e:
            print(f"  ERROR accessing subreddit r/{sub_name}: {e}")
            
    print(f"\n--- Reddit Scraper Finished ---")
    print(f"  Total posts checked across all subreddits/queries: {total_posts_checked}")
    print(f"  Total relevant posts collected from Reddit: {total_posts_collected}")
    return results

def scrape_rss():
    print("\n--- Starting RSS Feed Scraper ---")
    results = []
    item_count = 0
    for feed_url in RSS_URLS:
        print(f"  Processing RSS feed: {feed_url}")
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            print(f"    WARNING: Feed at {feed_url} may be malformed. Error: {feed.bozo_exception}")
        
        entries_in_feed = len(feed.entries)
        print(f"    Found {entries_in_feed} entries in feed.")
        for i, entry in enumerate(feed.entries):
            link = entry.get('link')
            title = entry.get('title', 'N/A')
            # print(f"      Processing RSS item {i+1}/{entries_in_feed}: '{title[:70]}...'")

            # Ensure link is valid for domain checking
            if not link or not isinstance(link, str):
                # print(f"        SKIP (Invalid Link): Link is '{link}'")
                continue

            pub_dt_parsed = entry.get('published_parsed')
            if pub_dt_parsed:
                pub_dt = datetime.fromtimestamp(time.mktime(pub_dt_parsed))
            else:
                pub_dt = datetime.utcnow()
            timestamp = pub_dt.strftime('%Y-%m-%d %H:%M:%S')

            def get_full_content_rss(url_to_fetch, entry_title): # Renamed for clarity
                time.sleep(1) # Reduced sleep from 2 to 1 for slightly faster RSS
                try:
                    r = requests.get(url_to_fetch, headers=REQUEST_HEADERS, timeout=10) # Reduced timeout
                    r.raise_for_status() # Check for HTTP errors
                    soup = BeautifulSoup(r.content, 'html.parser')
                    
                    # Try more specific selectors first, then more general ones
                    # This list can be expanded based on common news site structures
                    selectors = [
                        {'tag': 'article'}, # Semantic HTML5 tag
                        {'tag': 'div', 'class_': 'article-body'}, # Common class
                        {'tag': 'div', 'id': 'articleBody'}, # Common ID
                        {'tag': 'div', 'class_': 'story-content'},
                        {'tag': 'div', 'class_': 'entry-content'},
                        {'tag': 'div', 'class_': 'post-content'},
                        {'tag': 'div', 'class_': 'caas-body'}, # For Yahoo News
                        {'tag': 'div', 'class_': 'body yf-3qln1o'} # For Yahoo Finance (from original)
                    ]
                    article_text = ""
                    for selector_dict in selectors:
                        tag = selector_dict.get('tag', 'div')
                        attrs = {k: v for k, v in selector_dict.items() if k != 'tag'}
                        article_content_element = soup.find(tag, attrs)
                        if article_content_element:
                            paragraphs = article_content_element.find_all('p')
                            if paragraphs:
                                article_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                                if len(article_text.strip()) > 100: # Basic check for meaningful content
                                    # print(f"          Extracted content using: {selector_dict}")
                                    break # Found good content
                    
                    if not article_text.strip() or len(article_text.strip()) < 100 : # If specific selectors failed
                        # Fallback: get all paragraphs from the body if no specific container worked well
                        all_paragraphs = soup.find_all('p')
                        if all_paragraphs:
                            article_text = '\n'.join(p.get_text(strip=True) for p in all_paragraphs[:15]) # Limit length for generic
                            # print("          Extracted content using generic <p> tags fallback.")

                    return article_text.replace('\n', ' ') if article_text.strip() else "Article body not found or too short."

                except requests.exceptions.Timeout:
                    # print(f"        ERROR (Timeout): Fetching content from {url_to_fetch}")
                    return "Error: Timeout fetching content"
                except requests.exceptions.RequestException as e:
                    # print(f"        ERROR (RequestException): Fetching {url_to_fetch} - {e}")
                    return f"Error fetching content: {str(e)}"
                except Exception as e:
                    # print(f"        ERROR (Other): Processing {url_to_fetch} - {e}")
                    return f"Error processing content: {str(e)}"

            content = get_full_content_rss(link, title)

            if contains_relevant_keywords(title, content) and "Error:" not in content and "Article body not found" not in content :
                print(f"      COLLECTED (RSS: {feed_url.split('/')[-1][:20]}...): '{title[:60]}...'")
                results.append({
                    "Date and Timestamp": timestamp,
                    "Title": title,
                    "Full Text": content,
                    "Source": "News", # Generic source
                    "Link": link
                })
                item_count += 1
            # else:
                # print(f"        SKIP (Not relevant/No content): '{title[:60]}...'")

    print(f"\n--- RSS Feed Scraper Finished ---")
    print(f"  Total relevant items collected from RSS: {item_count}")
    return results

def scrape_newsapi():
    print("\n--- Starting NewsAPI Scraper ---")
    if not NEWSAPI_KEY:
        print("  ERROR: NEWSAPI_KEY not set in environment variables. Skipping NewsAPI.")
        return []
        
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    results = []
    item_count = 0
    
    query_str = ' OR '.join(f'"{q}"' for q in QUERIES) # e.g., "nvidia" OR "NVDA"
    print(f"  Querying NewsAPI for: '{query_str} stock' from {FROM_DATE} to {TO_DATE}, domains: {NEWS_DOMAINS or 'all'}")

    try:
        all_articles_data = client.get_everything(
            q=f'({query_str}) AND stock', # More specific query
            language='en',
            sort_by='publishedAt', # 'relevancy' could also be an option
            page_size=100, # Max allowed by NewsAPI for 'everything' is 100
            page=1, # Fetch first page
            from_param=FROM_DATE,
            to=TO_DATE,
            domains=NEWS_DOMAINS if NEWS_DOMAINS else None
        )
    except Exception as e:
        print(f"  ERROR calling NewsAPI: {e}")
        return []
        
    articles = all_articles_data.get('articles', [])
    total_api_results = all_articles_data.get('totalResults', 0)
    print(f"  NewsAPI returned {len(articles)} articles on page 1 (total available: {total_api_results}).")

    def fetch_full_article_newsapi(url_to_fetch, article_title): # Renamed for clarity
        # print(f"      Fetching NewsAPI article: '{article_title[:70]}...' from {url_to_fetch}")
        time.sleep(1) 
        try:
            r = requests.get(url_to_fetch, headers=REQUEST_HEADERS, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, 'html.parser')
            # Similar selector logic as RSS, can be fine-tuned
            selectors = [
                {'tag': 'article'}, 
                {'tag': 'div', 'class_': 'article-body'},
                {'tag': 'div', 'id': 'articleBody'},
                {'tag': 'div', 'class_': 'b6Cr_ article-body-container'}, # From original
                {'tag': 'div', 'class_': 'entry'}, # From original
                {'tag': 'div', 'class_': 'Story_body__ZYOg0 userContent'}, # From original
                {'tag': 'div', 'class_': 'zox-post-body-wrap'}, # From original
                {'tag': 'div', 'class_': 'sp_txt'} # From original
            ]
            article_text = ""
            for selector_dict in selectors:
                tag = selector_dict.get('tag', 'div')
                attrs = {k: v for k, v in selector_dict.items() if k != 'tag'}
                article_content_element = soup.find(tag, attrs)
                if article_content_element:
                    paragraphs = article_content_element.find_all('p')
                    if paragraphs:
                        article_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)
                        if len(article_text.strip()) > 100:
                            # print(f"          Extracted content using: {selector_dict}")
                            break
            
            if not article_text.strip() or len(article_text.strip()) < 100 :
                all_paragraphs = soup.find_all('p')
                if all_paragraphs:
                    article_text = '\n'.join(p.get_text(strip=True) for p in all_paragraphs[:15])
                    # print("          Extracted content using generic <p> tags fallback.")
            
            return article_text.replace('\n', ' ') if article_text.strip() else "Full article text not available or too short."

        except requests.exceptions.Timeout:
            # print(f"        ERROR (Timeout): Fetching NewsAPI article from {url_to_fetch}")
            return "Error: Timeout fetching article content"
        except requests.exceptions.RequestException as e:
            # print(f"        ERROR (RequestException): Fetching {url_to_fetch} - {e}")
            return f"Error fetching article: {str(e)}"
        except Exception as e:
            # print(f"        ERROR (Other): Processing {url_to_fetch} - {e}")
            return f"Error processing article: {str(e)}"

    for i, article in enumerate(articles):
        title = article.get('title', 'N/A')
        url = article.get('url')
        
        if not title or title == 'N/A' or not url:
            # print(f"    Skipping NewsAPI article {i+1} due to missing title or URL.")
            continue
            
        # print(f"    Processing NewsAPI article {i+1}/{len(articles)}: '{title[:70]}...'")
        content = fetch_full_article_newsapi(url, title)
        
        published_at_str = article.get('publishedAt')
        if published_at_str:
            try:
                dt = datetime.strptime(published_at_str, '%Y-%m-%dT%H:%M:%SZ')
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                # print(f"      Warning: Could not parse date '{published_at_str}'. Using current UTC.")
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        else:
            # print(f"      Warning: Missing 'publishedAt'. Using current UTC.")
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        if contains_relevant_keywords(title, content) and "Error:" not in content and "Full article text not available" not in content:
            print(f"      COLLECTED (NewsAPI): '{title[:60]}...'")
            results.append({
                "Date and Timestamp": timestamp,
                "Title": title,
                "Full Text": content,
                "Source": "News", # Generic source
                "Link": url
            })
            item_count += 1
        # else:
            # print(f"        SKIP (Not relevant/No content): '{title[:60]}...'")
            
    print(f"\n--- NewsAPI Scraper Finished ---")
    print(f"  Total relevant items collected from NewsAPI: {item_count}")
    return results

# --------------------------------------------------------------------------------------

def main():
    print("======== Starting Main Scraping Process ========")
    all_data = []
    seen_hashes = load_seen_hashes()
    print(f"  Loaded {len(seen_hashes)} seen_hashes from '{SEEN_HASH_FILE}'.")

    # Order of scrapers can be changed if needed
    scraper_functions = [scrape_reddit, scrape_newsapi, scrape_rss]
    
    total_new_items_globally = 0

    for scraper_func in scraper_functions:
        # Each scraper function now prints its own start/end and progress
        try:
            data = scraper_func() # This will print progress within the function
            
            if data: # Ensure data is not None and is a list
                new_items_this_scraper = 0
                for item_idx, item_data in enumerate(data): # Renamed to item_data
                    # print(f"  Processing item {item_idx+1}/{len(data)} from {scraper_func.__name__}...") # Can be verbose
                    title = item_data.get('Title')
                    link = item_data.get('Link')

                    if not title or not link:
                        # print(f"    WARNING: Item missing title or link, skipping: {str(item_data)[:100]}")
                        continue
                        
                    item_hash = compute_hash(title, link)
                    if item_hash is None: # Should not happen if title/link are present
                        # print(f"    WARNING: Could not compute hash for item, skipping: {title[:30]}")
                        continue

                    if item_hash not in seen_hashes:
                        all_data.append(item_data)
                        seen_hashes.add(item_hash)
                        new_items_this_scraper += 1
                        total_new_items_globally +=1
                        # print(f"    Added new unique item #{total_new_items_globally} (Title: '{title[:50]}...')")
                # print(f"  {scraper_func.__name__} added {new_items_this_scraper} new unique items to collection.")
            # else:
                # print(f"  No data returned by {scraper_func.__name__}.")
        except Exception as e:
            print(f"  CRITICAL ERROR during execution of {scraper_func.__name__}: {e}")
            import traceback
            traceback.print_exc()


    if all_data:
        print(f"\n  Collected {len(all_data)} new unique items globally. Processing for CSV output...")
        new_data_df = pd.DataFrame(all_data)
        
        # Ensure 'tldextract' version compatibility for registered_domain if error persists
        # For new tldextract: domain_info.top_domain_under_public_suffix
        # The extract_specific_source function has been updated already.
        new_data_df = preprocess_new_entries(new_data_df) 
        
        COLUMN_ORDER = ['Date and Timestamp', 'Title', 'Full Text', 'Source', 'SpecificSource', 'Link']
        # Ensure all columns exist, add missing ones as empty
        for col in COLUMN_ORDER:
            if col not in new_data_df.columns:
                new_data_df[col] = "" 
        new_data_df = new_data_df[COLUMN_ORDER]

        print("  Converting 'Date and Timestamp' to datetime and sorting...")
        new_data_df['Date and Timestamp'] = pd.to_datetime(new_data_df['Date and Timestamp'])
        new_data_df = new_data_df.sort_values(by='Date and Timestamp', ascending=True)
        
        print("  Dropping duplicates by 'Title' (keeping first by date)...")
        initial_rows_dedup = len(new_data_df)
        new_data_df = new_data_df.drop_duplicates(subset=['Title'], keep='first')
        print(f"    Removed {initial_rows_dedup - len(new_data_df)} duplicate titles. {len(new_data_df)} items remain.")

        output_file_exists = os.path.exists(COMBINED_OUTPUT_FILE)
        print(f"  Saving {len(new_data_df)} items to '{COMBINED_OUTPUT_FILE}' (mode: {'append' if output_file_exists else 'write new'})...")
        try:
            new_data_df.to_csv(COMBINED_OUTPUT_FILE, mode='a' if output_file_exists else 'w', header=not output_file_exists, index=False, sep=';', quoting=csv.QUOTE_ALL)
            print(f"  Successfully saved items to '{COMBINED_OUTPUT_FILE}'.")
        except Exception as e:
            print(f"  ERROR saving data to CSV '{COMBINED_OUTPUT_FILE}': {e}")

        save_seen_hashes(seen_hashes)
        print(f"  Saved {len(seen_hashes)} total seen_hashes to '{SEEN_HASH_FILE}'.")
    else:
        print("\n  No new unique data collected from any source in this run.")

    print("======== Main Scraping Process Finished ========")

if __name__ == "__main__":
    main()