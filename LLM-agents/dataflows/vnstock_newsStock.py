import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from vnstock_news.core.crawler import Crawler
from vnstock_data import Trading
import json

def _fetch_cafef_urls(symbol: str, look_back_months: int = 3, max_articles: int = 10) -> list[dict]:
    """Fetch news URLs from cafef.vn for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'FPT', 'ACB').
        look_back_months: Filter articles from the last N months.
        max_articles: Maximum articles to fetch.
        
    Returns:
        List of dicts with 'date' and 'href' keys.
    """
    cutoff_date = datetime.now() - relativedelta(months=look_back_months)
    
    url = "https://cafef.vn/du-lieu/Ajax/Events_RelatedNews_New.aspx"
    params = {
        "symbol": symbol,
        "floorID": 0,
        "configID": 0,
        "PageIndex": 1,
        "PageSize": max_articles,
        "Type": 2
    }
    
    articles = []
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for li in soup.select("ul.News_Title_Link li"):
            date_tag = li.find("span", class_="timeTitle")
            a_tag = li.find("a", class_="docnhanhTitle")
            
            if date_tag and a_tag:
                date_str = date_tag.get_text(strip=True)
                href = "https://cafef.vn" + a_tag["href"]
                
                # Try to parse date and filter by cutoff
                try:
                    # cafef uses format like "10/11/2024"
                    article_date = datetime.strptime(date_str, "%d/%m/%Y")
                    if article_date >= cutoff_date:
                        articles.append({"date": date_str, "href": href})
                        if len(articles) >= max_articles:
                            break
                except ValueError:
                    # If date parsing fails, include anyway
                    articles.append({"date": date_str, "href": href})
                    if len(articles) >= max_articles:
                        break
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ cafef cho {symbol}: {str(e)}")
    
    return articles


def _fetch_vietstock_urls(symbol: str, look_back_months: int = 3, max_articles: int = 10) -> list[dict]:
    """Fetch news URLs from vietstock.vn for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'FPT', 'ACB').
        look_back_months: Filter articles from the last N months.
        max_articles: Maximum articles to fetch.
        
    Returns:
        List of dicts with 'date' and 'href' keys.
    """
    cutoff_date = datetime.now() - relativedelta(months=look_back_months)
    from_date = cutoff_date.strftime("%d/%m/%Y")
    to_date = datetime.now().strftime("%d/%m/%Y")
    
    url = "https://finance.vietstock.vn/View/PagingNewsContent"
    payload = {
        "view": 1,
        "code": symbol,
        "type": 1,
        "fromDate": from_date,
        "toDate": to_date,
        "channelID": -1,
        "page": 1,
        "pageSize": max_articles
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0"
    }
    
    articles = []
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        rows = soup.select("table tbody tr")
        for tr in rows[:max_articles]:
            date_tag = tr.find("td", class_="col-date")
            a_tag = tr.find("a", class_="text-link")
            
            if date_tag and a_tag:
                date_str = date_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                
                # Convert // to https://
                if href.startswith("//"):
                    href = "https:" + href
                
                articles.append({"date": date_str, "href": href})
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ vietstock cho {symbol}: {str(e)}")
    
    return articles


def get_news_stock(ticker: str, look_back_months: int = 3) -> dict[str, str] | str:
    """Fetch and crawl stock news from cafef.vn and vietstock.vn.
    
    Retrieves URLs from both sources, crawls article content using vnstock's Crawler,
    filters articles to last N months, limits to 10 articles per source, and handles
    crawling errors gracefully.
    
    Args:
        ticker: Stock symbol (e.g., 'FPT', 'ACB').
        look_back_months: Filter articles from the last N months (default: 3).
        
    Returns:
        Dictionary with article URLs as keys and article content/details as values.
        Returns error message string if crawling fails entirely.
    """
    news_dict = {}
    
    # Initialize crawlers
    cafef_crawler = Crawler("cafef")
    vietstock_crawler = Crawler("vietstock")
    
    # Fetch URLs from both sources
    cafef_articles = _fetch_cafef_urls(ticker, look_back_months, max_articles=10)
    vietstock_articles = _fetch_vietstock_urls(ticker, look_back_months, max_articles=10)
    
    all_articles = cafef_articles + vietstock_articles
    
    if not all_articles:
        return f"Không có bài viết nào cho {ticker} trong {look_back_months} tháng gần nhất."
    
    # Crawl article details
    for article in all_articles:
        href = article.get("href")
        if not href:
            continue
        
        try:
            # Determine which crawler to use based on URL
            if "cafef" in href.lower():
                crawler = cafef_crawler
            else:
                crawler = vietstock_crawler
            
            details = crawler.get_article_details(href)
            
            # Store article with URL as key and details as value
            if details:
                news_dict[href] = details
        except Exception as e:
            # Skip articles that cannot be crawled
            print(f"Lỗi khi đọc bài viết từ {href}: {str(e)}")
            continue
    
    if not news_dict:
        return f"Không thể đọc được bất kỳ bài viết nào cho {ticker}."
    
    return news_dict


def get_insider_transactions(ticker: str, look_back_months: int = 3) -> pd.DataFrame | str:
    """Fetch insider deal transactions for a given stock symbol.
    
    Retrieves insider trading data from the last N months using vnstock_data.
    
    Args:
        ticker: Stock symbol (e.g., 'MSN', 'FPT', 'ACB').
        look_back_months: Filter transactions from the last N months (default: 3).
        
    Returns:
        DataFrame with insider transaction details.
        Returns error message string if no data is found or retrieval fails.
    """
    try:
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        cutoff_date = datetime.now() - relativedelta(months=look_back_months)
        start_date = cutoff_date.strftime('%Y-%m-%d')
        
        # Initialize Trading object
        trading = Trading(symbol=ticker, source='VCI')
        
        # Fetch insider deal data
        df = trading.insider_deal(start=start_date, end=end_date)
        
        if df.empty:
            return f"Không có giao dịch nội bộ nào cho {ticker} từ {start_date} đến {end_date}."
        
        return df
    except Exception as e:
        error_msg = f"Lỗi khi lấy dữ liệu giao dịch nội bộ cho {ticker}: {str(e)}"
        print(error_msg)
        return error_msg


# Test hàm
if __name__ == "__main__":
    ticker = "FPT"
    
    # Test get_news_stock
    print("=" * 70)
    print("Test get_news_stock()")
    print("=" * 70)
    result = get_news_stock(ticker, look_back_months=3)
    
    if isinstance(result, dict):
        output_file = f"{ticker}_news.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu {len(result)} bài viết vào file: {output_file}")
    else:
        print(result)
    
    # Test get_insider_transactions
    print("\n" + "=" * 70)
    print("Test get_insider_transactions()")
    print("=" * 70)
    insider_result = get_insider_transactions(ticker, look_back_months=3)
    
    if isinstance(insider_result, pd.DataFrame):
        print(f"Tổng số giao dịch: {len(insider_result)}")
        print(insider_result.head())
    else:
        print(insider_result)
