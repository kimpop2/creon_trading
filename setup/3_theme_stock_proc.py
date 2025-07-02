import pandas as pd
import pymysql
# from konlpy.tag import Komoran # 이 스크립트에서는 Komoran을 사용하지 않으므로 주석 처리
from collections import defaultdict
import os.path
import sys
import re
import textwrap
import time
from datetime import datetime, date, timedelta
import json # JSON 데이터 처리를 위해 추가

# 프로젝트 루트 경로를 sys.path에 추가 (data 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 실제 TraderManager, DBManager, APIManager 클래스 임포트
# 실제 프로젝트 구조에 맞게 경로를 수정하세요.
from manager.trader_manager import TraderManager

# 로거 정의 (실제 사용 시 logger 라이브러리를 임포트하고 설정)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# MariaDB 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'backtest_db', # 실제 사용하는 DB명으로 변경하세요
    'charset': 'utf8mb4'
}

# --- 새로운 스코어 계산을 위한 상수 정의 ---
LOOKBACK_DAYS = 100 # 최근 시세 데이터를 조회할 기간 (일) 아래보다 15일 앞서야야
RECENT_DAILY_THEME_DAYS = 90 # daily_theme에서 최근 언급된 기간 (일)

# 각 점수 요소의 가중치 (총합 100)
WEIGHT_PRICE_TREND = 40
WEIGHT_TRADING_VOLUME = 30
WEIGHT_VOLATILITY = 10
WEIGHT_THEME_MENTION = 20

# --- UTILITY FUNCTIONS ---
def print_query_execution_time(start_time, end_time, record_count, table_name=""):
    """쿼리 실행 시간, 레코드 건수, 레코드당 실행 시간을 출력하는 함수"""
    execution_time = end_time - start_time
    time_per_record = execution_time / record_count if record_count > 0 else 0
    print(f"[{table_name}] 쿼리 실행 시간: {execution_time:.4f} 초")
    print(f"[{table_name}] 처리된 레코드 건수: {record_count} 건")
    print(f"[{table_name}] 레코드 1건당 실행 시간: {time_per_record:.6f} 초")
    print(f"[{table_name}] Data insertion/update completed")

def initialize_database_tables(cursor, connection):
    """
    필요한 데이터베이스 테이블(theme_class, theme_stock, daily_theme, word_dic, theme_word_relevance)을 생성합니다.
    """
    print("\n--- 데이터베이스 테이블 초기화 시작 ---")
    
    # 1. theme_class 테이블 생성 (사용자 제공 스키마 반영)
    create_theme_class_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `theme_class` (
            `theme_id` INT(11) NOT NULL AUTO_INCREMENT,
            `theme` VARCHAR(30) NOT NULL ,
            `theme_class` VARCHAR(30) NULL ,
            `theme_synonyms` JSON NULL ,
            `theme_hit` INT(11) NULL DEFAULT '0',
            `theme_score` INT(11) NULL DEFAULT '0',
            `momentum_score` DECIMAL(10,4) NULL DEFAULT '0.00',
            `theme_desc` VARCHAR(200) NULL DEFAULT NULL ,
            PRIMARY KEY (`theme_id`) USING BTREE,
            INDEX `idx_theme` (`theme`) USING BTREE
        );
    """)
    cursor.execute(create_theme_class_table_query)
    print("theme_class 테이블 존재 확인 및 필요시 생성 완료.")

    # 2. theme_stock 테이블 생성
    create_theme_stock_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `theme_stock` (
            `theme_id` INT(11) NOT NULL,
            `stock_code` VARCHAR(7) NOT NULL ,
            `stock_score` DECIMAL(10,4) NULL DEFAULT '0.0000', -- DECIMAL로 변경
            PRIMARY KEY (`theme_id`, `stock_code`) USING BTREE,
            INDEX `idx_ts_theme` (`theme_id`) USING BTREE,
            INDEX `idx_ts_stock_code` (`stock_code`) USING BTREE
        );
    """)
    cursor.execute(create_theme_stock_table_query)
    print("theme_stock 테이블 존재 확인 및 필요시 생성 완료.")

    # 3. daily_theme 테이블 생성 (reason_nouns JSON 컬럼 포함)
    create_daily_theme_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS `daily_theme` (
            `date` DATE NOT NULL,
            `market` VARCHAR(8) NOT NULL ,
            `stock_code` VARCHAR(7) NOT NULL ,
            `stock_name` VARCHAR(25) NOT NULL ,
            `rate` DECIMAL(5,2) NULL DEFAULT '0.00',
            `amount` INT(11) NULL DEFAULT '0',
            `reason` VARCHAR(250) NOT NULL ,
            `reason_nouns` JSON NULL , -- JSON 타입 컬럼
            `theme` VARCHAR(250) NULL DEFAULT NULL ,
            PRIMARY KEY (`date`, `market`, `stock_code`) USING BTREE
        );
    """)
    cursor.execute(create_daily_theme_table_query)
    print("daily_theme 테이블 존재 확인 및 필요시 생성 완료.")

    # 4. word_dic 테이블 생성
    create_word_dic_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS word_dic (
            word VARCHAR(25) NOT NULL,
            freq INT NULL DEFAULT NULL,
            cumul_rate DECIMAL(10,2) NULL DEFAULT NULL,
            avg_rate DECIMAL(10,2) NULL DEFAULT NULL,
            PRIMARY KEY (word)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_word_dic_table_query)
    print("word_dic 테이블 존재 확인 및 필요시 생성 완료.")
    
    # 5. theme_word_relevance 테이블 생성
    create_theme_word_relevance_table_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS theme_word_relevance (
            theme VARCHAR(30) NOT NULL,
            word VARCHAR(25) NOT NULL,
            relevance_score DECIMAL(10,4) NULL DEFAULT 0,
            num_occurrences INT NULL DEFAULT 0,
            avg_stock_rate_in_theme DECIMAL(10,2) NULL DEFAULT 0,
            PRIMARY KEY (theme, word),
            INDEX idx_twr_theme (theme),
            INDEX idx_twr_word (word)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    cursor.execute(create_theme_word_relevance_table_query)
    print("theme_word_relevance 테이블 존재 확인 및 필요시 생성 완료.")

    connection.commit()
    print("--- 데이터베이스 테이블 초기화 완료 ---")


def calculate_price_trend_score(df_stock_data):
    """
    주가 추세 점수 계산: 최근 5일, 20일 이동평균선 정배열 및 누적 수익률 반영
    DataFrame은 반드시 'close' 컬럼을 포함해야 합니다.
    """
    if len(df_stock_data) < 20: # 20일 이동평균을 위한 최소 데이터
        return 0

    df_stock_data = df_stock_data.sort_values(by='date').copy()
    
    # 이동평균선 계산
    df_stock_data['MA5'] = df_stock_data['close'].rolling(window=5).mean()
    df_stock_data['MA20'] = df_stock_data['close'].rolling(window=20).mean()

    latest_data = df_stock_data.iloc[-1]
    
    score = 0
    
    # 1. 이동평균선 정배열 가점 (5일 > 20일)
    if latest_data['MA5'] > latest_data['MA20']:
        score += 30 # 기본 정배열 가점

    # 2. 최근 5일 누적 수익률
    if len(df_stock_data) >= 5:
        recent_5_days_data = df_stock_data.tail(5)
        start_price_5 = recent_5_days_data.iloc[0]['close']
        end_price_5 = recent_5_days_data.iloc[-1]['close']
        if start_price_5 > 0:
            return_5_day = ((end_price_5 - start_price_5) / start_price_5) * 100
            if return_5_day > 0: # 5일 수익률이 양수
                score += min(return_5_day, 50) * 0.5 # 최대 25점 (수익률 50%까지 반영)
    
    # 3. 최근 20일 누적 수익률 (장기 추세)
    if len(df_stock_data) >= 20:
        recent_20_days_data = df_stock_data.tail(20)
        start_price_20 = recent_20_days_data.iloc[0]['close']
        end_price_20 = recent_20_days_data.iloc[-1]['close']
        if start_price_20 > 0:
            return_20_day = ((end_price_20 - start_price_20) / start_price_20) * 100
            if return_20_day > 0: # 20일 수익률이 양수
                score += min(return_20_day, 100) * 0.2 # 최대 20점 (수익률 100%까지 반영)

    return min(max(score, 0), 100) # 0점에서 100점 사이로 정규화

def calculate_trading_volume_score(df_stock_data):
    """
    거래대금 점수 계산: 최근 거래대금 증가율 및 절대적인 규모 반영
    DataFrame은 반드시 'trading_value' 컬럼을 포함해야 합니다.
    """
    if len(df_stock_data) < 20: # 최소 20일 데이터 필요
        return 0
    
    df_stock_data = df_stock_data.sort_values(by='date').copy()

    recent_volume = df_stock_data['trading_value'].tail(5).mean() # 최근 5일 평균 거래대금
    avg_volume = df_stock_data['trading_value'].mean() # 전체 기간 평균 거래대금

    score = 0
    # 1. 최근 5일 평균 거래대금 증가율
    if avg_volume > 0:
        volume_growth_rate = ((recent_volume - avg_volume) / avg_volume) * 100
        if volume_growth_rate > 0: # 증가했으면 가점
            score += min(volume_growth_rate * 0.2, 50) # 증가율 250%까지 반영하여 최대 50점

    # 2. 절대적인 거래대금 규모 (예: 100억 이상 10점, 500억 이상 20점, 1000억 이상 30점)
    # 기준은 시장 상황에 따라 조정 필요
    if recent_volume >= 10_000_000_000: # 100억
        score += 10
    if recent_volume >= 50_000_000_000: # 500억
        score += 10
    if recent_volume >= 100_000_000_000: # 1000억
        score += 10 # 총 30점
    
    return min(max(score, 0), 100) # 0점에서 100점 사이로 정규화

def calculate_volatility_score(df_stock_data):
    """
    변동성 점수 계산: ATR 기반 변동성 및 일중 변동폭 반영
    DataFrame은 반드시 'high', 'low', 'close' 컬럼을 포함해야 합니다.
    """
    if len(df_stock_data) < 10: # ATR 계산을 위한 최소 데이터
        return 0

    df_stock_data = df_stock_data.sort_values(by='date').copy()
    
    # ATR (Average True Range) 계산
    # TR (True Range) = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    high_low = df_stock_data['high'] - df_stock_data['low']
    high_prev_close = abs(df_stock_data['high'] - df_stock_data['close'].shift(1))
    low_prev_close = abs(df_stock_data['low'] - df_stock_data['close'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.rolling(window=10).mean().iloc[-1] # 최근 10일 ATR
    
    latest_close = df_stock_data.iloc[-1]['close']
    
    score = 0
    # ATR 비율 (종가 대비)
    if latest_close > 0:
        atr_ratio = (atr / latest_close) * 100 # ATR이 종가의 몇 %인지
        score += min(atr_ratio * 5, 50) # ATR 비율 10%까지 반영하여 최대 50점

    # 최근 5일 평균 일중 변동폭 (고가-저가) 비율
    if len(df_stock_data) >= 5:
        df_stock_data['daily_range_ratio'] = ((df_stock_data['high'] - df_stock_data['low']) / df_stock_data['close']) * 100
        avg_daily_range_ratio = df_stock_data['daily_range_ratio'].tail(5).mean()
        score += min(avg_daily_range_ratio * 2, 50) # 일중 변동폭 비율 25%까지 반영하여 최대 50점

    return min(max(score, 0), 100) # 0점에서 100점 사이로 정규화

def calculate_theme_mention_score(mention_count_in_period):
    """
    테마 언급 빈도 점수 계산: 최근 한달 내 언급 횟수에 비례
    """
    # 언급 횟수가 많을수록 점수 상승, 최대 점수 제한 (예: 5회 이상 언급 시 100점)
    # 언급 횟수 1회: 20점, 2회: 40점, 3회: 60점, 4회: 80점, 5회 이상: 100점
    score = min(mention_count_in_period * 20, 100) 
    return max(0, score) # 0점에서 100점 사이로 정규화


# --- MAIN EXECUTION BLOCK ---
try:
    # MariaDB에 연결
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # 실제 DBManager, APIManager 인스턴스를 생성하여 TraderManager에 전달
    trader_manager = TraderManager()
    
    # 1. 모든 필요한 테이블 생성/확인
    initialize_database_tables(cursor, connection)
    connection.commit() # initialize_database_tables 내부에서 commit을 하지만, 명시적으로 한번 더

    # --- 0. theme_stock 테이블 초기화 (매일 재계산 전에) ---
    print("\n--- theme_stock 테이블 초기화 ---")
    start_time_reset_ts = time.time()
    cursor.execute("UPDATE theme_stock SET stock_score = 0")
    connection.commit()
    end_time_reset_ts = time.time()
    print(f"theme_stock 테이블 stock_score 초기화 완료 ({end_time_reset_ts - start_time_reset_ts:.4f} 초)")

    ## 1. theme_class에서 테마 동의어 및 테마 이름 사전 로드
    print("\n--- 테마 동의어 및 테마 이름 사전 로딩 (theme_class 테이블에서) ---")
    start_time_load_themes = time.time()
    
    # theme_class 테이블에서 theme_id, theme, theme_synonyms (JSON) 로드
    cursor.execute("SELECT theme_id, theme, theme_synonyms FROM theme_class")
    theme_class_records = cursor.fetchall()

    theme_id_to_name_map = {}
    theme_synonym_map = {} # {synonym: theme_id}

    for theme_id, theme_name, synonyms_json in theme_class_records:
        theme_id_to_name_map[theme_id] = theme_name.strip()
        
        # 주 테마명도 동의어로 포함
        theme_synonym_map[theme_name.strip()] = theme_id

        # JSON 동의어 파싱
        if synonyms_json:
            try:
                # DB에서 가져온 JSON은 bytes일 수 있으므로 decode
                if isinstance(synonyms_json, bytes):
                    synonyms_json = synonyms_json.decode('utf-8')
                
                parsed_synonyms = json.loads(synonyms_json)
                if isinstance(parsed_synonyms, list):
                    for syn in parsed_synonyms:
                        if isinstance(syn, str) and syn.strip():
                            theme_synonym_map[syn.strip()] = theme_id # 이미 존재하면 덮어쓰기 (하나의 동의어는 하나의 theme_id에 매핑)
            except json.JSONDecodeError as e:
                logger.warning(f"경고: theme_id {theme_id}의 theme_synonyms JSON 파싱 오류: {e}. 원본 JSON: {synonyms_json[:50]}...")
            except Exception as e:
                logger.warning(f"경고: theme_id {theme_id}의 theme_synonyms 처리 중 예상치 못한 오류: {e}")

    # 정규식 컴파일을 위한 동의어 리스트 (더 긴 것 먼저 매칭되도록 정렬)
    sorted_synonyms = sorted(list(theme_synonym_map.keys()), key=len, reverse=True)
    theme_synonym_regex_map = [ 
        (synonym, theme_synonym_map[synonym], re.compile(r'\b' + re.escape(synonym) + r'\b', re.IGNORECASE)) 
        for synonym in sorted_synonyms 
    ]

    end_time_load_themes = time.time()
    print(f"테마 동의어/이름 사전 로드 완료. 총 {len(theme_synonym_map)}개 동의어 ({end_time_load_themes - start_time_load_themes:.4f} 초)")

    ## 2. daily_theme 테이블 처리 및 대상 종목 필터링
    print("\n--- daily_theme 테이블 처리 및 대상 종목 필터링 ---")
    start_time_process_daily = time.time()

    # 현재 날짜 기준
    today = date.today() 
    one_month_ago = today - timedelta(days=RECENT_DAILY_THEME_DAYS)

    # 필터링된 종목 및 해당 종목에 연결된 theme_id 집합
    # {stock_code: {'theme_ids': set(), 'mention_count': int}} 형태로 저장
    target_stock_theme_mentions = {} 
    
    # daily_theme 업데이트용 데이터
    daily_theme_updates = []

    # 최근 기간의 daily_theme 레코드만 로드
    # reason_nouns 컬럼 대신 reason 컬럼을 사용합니다.
    cursor.execute("""
        SELECT date, market, stock_code, stock_name, rate, reason, theme 
        FROM daily_theme 
        WHERE date >= %s
    """, (one_month_ago,))
    recent_daily_theme_records = cursor.fetchall()
    print(f"최근 {RECENT_DAILY_THEME_DAYS}일간 daily_theme 레코드 {len(recent_daily_theme_records)}개 로드.")

    # daily_theme 레코드 처리 (theme 컬럼 업데이트 및 필터링)
    for record in recent_daily_theme_records:
        dt_date, dt_market, dt_stock_code, dt_stock_name, dt_rate, dt_reason, dt_existing_theme_str = record

        found_theme_ids_for_record = set()
        found_theme_names_for_record = set()

        # reason 텍스트에서 정규식을 사용하여 테마 동의어 매칭
        current_reason_text = str(dt_reason)
        for synonym, theme_id, synonym_regex in theme_synonym_regex_map:
            if synonym_regex.search(current_reason_text):
                found_theme_ids_for_record.add(theme_id)
                if theme_id in theme_id_to_name_map:
                    found_theme_names_for_record.add(theme_id_to_name_map[theme_id])
        
        # daily_theme 업데이트용 데이터 준비
        current_themes_in_db = set()
        if dt_existing_theme_str:
            current_themes_in_db.update([t.strip() for t in dt_existing_theme_str.split(',') if t.strip()])
        
        new_theme_list = sorted(list(current_themes_in_db | found_theme_names_for_record))
        new_theme_string = ", ".join(new_theme_list)
        
        if new_theme_string != dt_existing_theme_str:
            daily_theme_updates.append((new_theme_string, dt_date, dt_market, dt_stock_code))

        # stock_score 계산을 위한 대상 종목 필터링 (rate 10% 이상)
        # 이 시점에서 종목과 관련된 모든 theme_id를 저장
        if dt_rate is not None and dt_rate >= 10 and found_theme_ids_for_record:
            if dt_stock_code not in target_stock_theme_mentions:
                target_stock_theme_mentions[dt_stock_code] = {
                    'theme_ids': set(),
                    'mention_count': 0,
                }
            
            target_stock_theme_mentions[dt_stock_code]['theme_ids'].update(found_theme_ids_for_record)
            target_stock_theme_mentions[dt_stock_code]['mention_count'] += 1
    
    end_time_process_daily = time.time()
    print(f"daily_theme 레코드 처리 완료: {len(target_stock_theme_mentions)}개 종목 필터링 ({end_time_process_daily - start_time_process_daily:.4f} 초)")

    ## daily_theme 테이블 배치 업데이트
    if daily_theme_updates:
        print("\n--- daily_theme 테이블 업데이트 ---")
        start_time_update_daily = time.time()
        update_daily_theme_query = textwrap.dedent("""
            UPDATE daily_theme
            SET theme = %s
            WHERE date = %s AND market = %s AND stock_code = %s
        """)
        cursor.executemany(update_daily_theme_query, daily_theme_updates)
        connection.commit()
        end_time_update_daily = time.time()
        print_query_execution_time(start_time_update_daily, end_time_update_daily, len(daily_theme_updates), "daily_theme (update)")
    else:
        print("daily_theme 테이블에 업데이트할 레코드가 없습니다.")

    ## 3. 필터링된 종목들의 시세 데이터 로드 (TraderManager 사용) 및 stock_score 계산
    print("\n--- 대상 종목 시세 데이터 로드 및 stock_score 계산 ---")
    start_time_calculate_scores = time.time()
    
    all_stock_codes_to_process = list(target_stock_theme_mentions.keys())
    
    theme_stock_score_updates = {} # { (theme_id, stock_code): final_score }

    if all_stock_codes_to_process:
        # 각 종목별로 cache_daily_ohlcv 메서드 호출
        for stock_code in all_stock_codes_to_process:
            # from_date는 LOOKBACK_DAYS 이전으로 설정
            from_date_ohlcv = today - timedelta(days=LOOKBACK_DAYS)
            to_date_ohlcv = today # 오늘까지의 데이터

            # TraderManager를 통해 일별 시세 데이터 가져오기
            # cache_daily_ohlcv는 인덱스가 날짜인 DataFrame을 반환
            df_stock_data = trader_manager.cache_daily_ohlcv(stock_code, from_date_ohlcv, to_date_ohlcv)
            # 3. 'trading_value' (거래대금) 컬럼 생성
            # 만약 API에서 trading_value를 주지 않는다면, 종가 * 거래량으로 계산
            if 'trading_value' not in df_stock_data.columns:
                df_stock_data['trading_value'] = df_stock_data['close'] * df_stock_data['volume']
                logger.debug(f"종목 {stock_code}: 'trading_value' 컬럼이 없어 'close * volume'으로 생성했습니다.")

            # cache_daily_ohlcv가 반환하는 DataFrame의 컬럼명을 확인하고 필요한 경우 변경
            # 일반적으로 'close', 'volume', 'trading_value', 'high', 'low' 등을 가정
            # 만약 컬럼명이 다르다면 여기서 rename 처리 필요
            # 예: df_stock_data = df_stock_data.rename(columns={'종가': 'close', '거래량': 'volume'})
            
            if df_stock_data.empty or len(df_stock_data) < 20: # 20일 이동평균을 위한 최소 데이터 확인
                logger.debug(f"종목 {stock_code}: 시세 데이터 부족 ({len(df_stock_data)}개), 점수 계산 스킵.")
                continue

            # Pandas DataFrame 인덱스를 'date' 컬럼으로 변경하여 함수 호환성을 높임
            # cache_daily_ohlcv에서 날짜가 인덱스로 온다고 가정했으므로, 이를 'date' 컬럼으로 리셋
            df_stock_data = df_stock_data.reset_index().rename(columns={'index': 'date'})
            df_stock_data['date'] = pd.to_datetime(df_stock_data['date']).dt.date # datetime.date 객체로 변환

            # 각 점수 계산
            price_trend_score = calculate_price_trend_score(df_stock_data)
            trading_volume_score = calculate_trading_volume_score(df_stock_data)
            volatility_score = calculate_volatility_score(df_stock_data)
            
            # 테마 언급 빈도 점수 (daily_theme 필터링에서 얻은 정보 활용)
            mention_data = target_stock_theme_mentions[stock_code]
            theme_mention_score = calculate_theme_mention_score(mention_data['mention_count'])

            # 최종 stock_score 계산 (가중치 합산)
            final_stock_score = (
                (price_trend_score * WEIGHT_PRICE_TREND) +
                (trading_volume_score * WEIGHT_TRADING_VOLUME) +
                (volatility_score * WEIGHT_VOLATILITY) +
                (theme_mention_score * WEIGHT_THEME_MENTION)
            ) / 100 # 총합을 100으로 나눔 (가중치 총합이 100이므로)
            
            # 소수점 둘째 자리까지 반올림
            final_stock_score = round(final_stock_score, 2)

            # 해당 종목에 연결된 모든 테마 ID에 대해 점수 업데이트 준비
            for theme_id in mention_data['theme_ids']:
                key = (theme_id, stock_code)
                theme_stock_score_updates[key] = final_stock_score 
    else:
        print("stock_score를 계산할 대상 종목이 없습니다.")

    end_time_calculate_scores = time.time()
    print(f"stock_score 계산 완료. 총 {len(theme_stock_score_updates)}개 (테마,종목) 쌍 업데이트 예정 ({end_time_calculate_scores - start_time_calculate_scores:.4f} 초)")

    ## 4. theme_stock 테이블 배치 업데이트 (새로운 스코어로 교체)
    if theme_stock_score_updates:
        print("\n--- theme_stock 테이블 업데이트 ---")
        start_time_update_ts = time.time()

        theme_stock_data_to_replace = [
            (theme_id, stock_code, score)
            for (theme_id, stock_code), score in theme_stock_score_updates.items()
        ]

        replace_theme_stock_query = textwrap.dedent("""
            REPLACE INTO theme_stock (theme_id, stock_code, stock_score)
            VALUES (%s, %s, %s)
        """)
        cursor.executemany(replace_theme_stock_query, theme_stock_data_to_replace)
        connection.commit()
        end_time_update_ts = time.time()
        print_query_execution_time(start_time_update_ts, end_time_update_ts, len(theme_stock_data_to_replace), "theme_stock (replace)")
    else:
        print("theme_stock 테이블에 업데이트할 유효한 레코드가 없습니다.")

    ## 5. 최종 결과 출력: stock_score 상위 종목들
    print("\n--- 주식 점수 (Stock Score) 상위 종목 ---")
    start_time_print_results = time.time()

    # theme_stock, stock_info, theme_class를 조인하여 필요한 정보 가져오기
    select_top_stocks_query = textwrap.dedent("""
        SELECT 
            ts.stock_code, 
            si.stock_name, 
            tc.theme, 
            ts.stock_score
        FROM 
            theme_stock ts
        JOIN 
            stock_info si ON ts.stock_code = si.stock_code
        JOIN 
            theme_class tc ON ts.theme_id = tc.theme_id
        WHERE
            ts.stock_score > 0 -- 점수가 0보다 큰 종목만 출력
        ORDER BY 
            ts.stock_score DESC
        LIMIT 80; -- 상위 80개 종목만 출력 (원래 20개였으나, 80개로 변경)
    """)
    
    cursor.execute(select_top_stocks_query)
    top_stocks = cursor.fetchall()

    if top_stocks:
        print(f"{'순위':<4} | {'종목코드':<8} | {'종목명':<15} | {'테마':<20} | {'합계점수':<8}")
        print("-" * 70)
        for i, row in enumerate(top_stocks):
            stock_code, stock_name, theme_name, stock_score = row
            print(f"{i+1:<4} | {stock_code:<8} | {stock_name:<15} | {theme_name:<20} | {stock_score:<8.2f}")
    else:
        print("현재 계산된 stock_score가 있는 종목이 없습니다.")

    end_time_print_results = time.time()
    print(f"\n최종 결과 출력 완료 ({end_time_print_results - start_time_print_results:.4f} 초)")

except pymysql.MySQLError as e:
    if 'connection' in locals() and connection:
        connection.rollback()
    print(f"MariaDB 오류 발생: {e}")
except Exception as e:
    if 'connection' in locals() and connection:
        connection.rollback()
    print(f"예상치 못한 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 오류 스택 트레이스 출력
finally:
    if 'connection' in locals() and connection:
        cursor.close()
        connection.close()
        print("MariaDB 연결 해제.")