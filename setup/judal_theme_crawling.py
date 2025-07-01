import requests
from bs4 import BeautifulSoup
import time # 요청 사이에 지연을 주기 위해 import
import random
import pandas as pd
import re
import os
import sys

def get_theme_links(url):
    # 브라우저처럼 보이도록 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    # 세션 생성으로 쿠키 유지
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status() # HTTP 오류가 발생하면 예외를 발생시킵니다.
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러 발생: {e}")
        print(f"응답 상태 코드: {response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"URL에 접근하는 중 오류 발생: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    theme_data = []
    # 'list-group list-group-flush collapse navbar-collapse' 클래스를 가진 div 찾기
    theme_div = soup.find('div', class_='list-group list-group-flush collapse navbar-collapse')

    if theme_div:
        # '테마별 종목' <span> 태그를 찾아서 그 이후의 링크를 파싱
        start_parsing = False
        for element in theme_div.children:
            if element.name == 'div' and 'disabled' in element.get('class', []) and '테마별 종목' in element.get_text():
                start_parsing = True
                continue
            
            if start_parsing and element.name == 'a' and 'list-group-item-sub' in element.get('class', []):
                theme_url = element.get('href')
                theme_name_with_count = element.find('span').get_text(strip=True)
                
                # (숫자) 제거
                theme_name = re.sub(r'\(\d+\)', '', theme_name_with_count).strip()
                
                theme_data.append({'theme': theme_name, 'theme_url': theme_url})
    return theme_data

def get_theme_stocks(theme_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    session = requests.Session()
    session.headers.update(headers)

    stock_names = []
    try:
        response = session.get(theme_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러 발생: {e} for URL: {theme_url}")
        print(f"응답 상태 코드: {response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"URL에 접근하는 중 오류 발생: {e} for URL: {theme_url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # tbody 태그 내의 tr 태그들을 찾습니다.
    tbody = soup.find('tbody', class_='text-right')
    if tbody:
        trs = tbody.find_all('tr')
        for tr in trs:
            # 각 tr 내의 th 태그를 찾습니다.
            th = tr.find('th', scope='row')
            if th:
                # th 태그 내의 b 태그에서 종목명을 추출합니다.
                b_tag = th.find('b', style="font-size:.95em")
                if b_tag:
                    stock_names.append(b_tag.get_text(strip=True))
    return stock_names

def main():
    base_url = "https://www.judal.co.kr/"
    
    # 1. 테마 링크 및 이름 크롤링
    theme_links = get_theme_links(base_url)
    
    all_data = []
    
    # 2. 각 테마 URL을 순회하며 종목명 크롤링
    for theme_info in theme_links:
        theme_name = theme_info['theme']
        if theme_name == '테마없음' :
            continue
        
        theme_url = theme_info['theme_url']
        print(f"크롤링 중: {theme_name} ({theme_url})")
        
        stocks = get_theme_stocks(theme_url)
        
        if stocks:
            for stock_name in stocks:
                all_data.append({'theme': theme_name, 'theme_stock': stock_name})
        else:
            all_data.append({'theme': theme_name, 'theme_stock': '종목 없음'}) # 해당 테마에 종목이 없는 경우
        
        time.sleep(random.uniform(0.01, 0.03)) # 서버 부하를 줄이기 위해 랜덤 지연

    # DataFrame 생성 및 Excel 저장
    df = pd.DataFrame(all_data)
    print(df)
    
    # Excel file path
    excel_file_path = 'datas/judal_theme.xlsx'
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    output_excel_path = os.path.join(modpath, excel_file_path)
    
    # datas 폴더 생성 (없으면)
    datas_dir = os.path.dirname(output_excel_path)
    if not os.path.exists(datas_dir):
        os.makedirs(datas_dir)
        print(f"'{datas_dir}' 폴더를 생성했습니다.")
    
    df.to_excel(output_excel_path, index=False)
    print(f"데이터가 {output_excel_path} 파일에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()