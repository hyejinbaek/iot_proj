# 네이버 뉴스 기사 크롤링
# 경제, 정치, IT/과학
# data-useragent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
# https://news.naver.com/section/100 -> 정치
# https://news.naver.com/section/101 -> 경제
# https://news.naver.com/section/105 -> IT/과학

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


# 뉴스 카테고리별 URL
category_url = {
    "policy": "https://news.naver.com/main/list.naver?mode=LSD&mid=shm&sid1=100",  # 정치
    "economy": "https://news.naver.com/main/list.naver?mode=LSD&mid=shm&sid1=101",  # 경제
    "it": "https://news.naver.com/main/list.naver?mode=LSD&mid=shm&sid1=105"      # IT/과학
}

def crawl_naver_news(category, pages=5):
    """
    네이버 뉴스 특정 카테고리를 크롤링합니다.
    
    Args:
        category (str): 크롤링할 카테고리 (economy, policy, it).
        pages (int): 크롤링할 페이지 수.
    
    Returns:
        pd.DataFrame: 크롤링한 뉴스 데이터프레임.
    """
    base_url = category_url[category]
    news_data = []

    for page in range(1, pages + 1):
        url = f"{base_url}&page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 기사 리스트 추출
        articles = soup.select("ul.type06_headline li") + soup.select("ul.type06 li")
        
        for article in articles:
            try:
                title_tag = article.select_one("a")
                title = title_tag.get_text(strip=True)
                link = title_tag['href']
                
                date_tag = article.select_one(".date")
                date = date_tag.get_text(strip=True) if date_tag else "unknown"
                
                company_tag = article.select_one(".writing")
                company = company_tag.get_text(strip=True) if company_tag else "unknown"
                
                # 본문 내용 크롤링
                content = crawl_article_content(link)
                
                news_data.append([date, category, company, title, content, link])
            except Exception as e:
                print(f"Error parsing article: {e}")
        
        time.sleep(1)  # 서버 부하 방지
        
    columns = ['date', 'category', 'company', 'title', 'content', 'url']
    return pd.DataFrame(news_data, columns=columns)

def crawl_article_content(url):
    """
    네이버 뉴스 기사 본문을 크롤링합니다.
    
    Args:
        url (str): 기사 URL.
    
    Returns:
        str: 기사 본문 내용.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 여러 선택자를 확인
        content = soup.select_one("#articleBodyContents")  # 기존 선택자
        if not content:
            content = soup.select_one("#newsct_article")  # 변경된 선택자
        
        if content:
            return content.get_text(strip=True).replace("\n", " ").replace("\t", " ")
        else:
            return "Content not found"
    except Exception as e:
        print(f"Error fetching article content: {e}")
        return "Content unavailable"


# CSV 파일로 저장
def save_news_to_csv():
    for category in category_url.keys():
        print(f"Crawling category: {category}")
        df = crawl_naver_news(category, pages=5)
        output_path = f"./data/naver_news/{category}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {category} news to {output_path}")

# 실행
if __name__ == "__main__":
    save_news_to_csv()
        