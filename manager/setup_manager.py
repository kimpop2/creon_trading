import pandas as pd
import pymysql
import os.path
import sys
import re
import textwrap
import time
from datetime import datetime, date, timedelta
import json
import logging
from collections import defaultdict

# 프로젝트 루트 경로를 sys.path에 추가 (manager 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from manager.backtest_manager import BacktestManager

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetupManager:
    """
    MariaDB 데이터베이스의 테마, 종목, 단어 관련 데이터를 관리하고 점수를 계산하는 클래스.
    3_, 4_, 6_ 스크립트의 기능을 통합하여 제공합니다.
    """
    
    # --- 상수 정의 (필요에 따라 클래스 속성 또는 __init__ 인자로 관리) ---
    LOOKBACK_DAYS_OHLCV = 100 # to_date를 기준 주식의 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 조회

    # 각 점수 요소의 가중치 (총합 100)
    WEIGHT_PRICE_TREND = 40     # 주가 추세 점수(가격 이동평균선 정배열 여부, 누적 수익률)의 가중치
    WEIGHT_TRADING_VOLUME = 30  # 거래대금 점수(최근 거래대금 증가율 및 절대적인 규모)의 가중치
    WEIGHT_VOLATILITY = 10      # 변동성 점수(ATR 기반 변동성 및 일중 변동폭)의 가중치
    WEIGHT_THEME_MENTION = 20   # 테마 언급 빈도 점수(최근 한 달 내 언급 횟수)의 가중치

    # word_dic 클리닝 임계값 (금요일에만 적용)
    WORD_DIC_MIN_GLOBAL_FREQ = 5    # word_dic에서 단어의 최소 전역 빈도입니다. 이 값 미만의 단어는 제거
    # 빈도 상위 특정 비율에 해당하는 단어 중, 특정 조건(영향력 등)을 만족하면 제거될 수 있는 단어 그룹을 식별하는 데 사용되는 백분위수
    WORD_DIC_MAX_GLOBAL_FREQ_PERCENTILE = 0.99 
    # 빈도가 높은 일반적인 단어의 평균 관련 점수(avg_rate)가 이 편차 값보다 낮을 경우 제거될 수 있는 임계값
    WORD_DIC_MAX_COMMON_WORD_AVG_RATE_DEVIATION = 0.5 
    WORD_DIC_BLACKLIST_WORDS = ['기자', '사진', '뉴시스', '연합뉴스', '머니투데이', '코스피', '코스닥', '지수', '시장', '증시', '개장', '폐장', '마감', '시가총액', '뉴스']

    # theme_word_relevance 계산 임계값
    THEME_WORD_RELEVANCE_MIN_OCCURRENCES = 4 # (테마, 단어) 쌍의 최소 발생 빈도
    DAILY_THEME_MAX_RATE_THRESHOLD = 30.0 # daily_theme rate가 이 값을 초과하면 분석에서 제외

    def __init__(self, from_date: date = None, to_date: date = None, setup_parameters: dict = None):
        """
        SetupManager를 초기화합니다.
        :param from_date: 데이터 처리 시작 날짜 (기본값: 90일 전)
        :param to_date: 데이터 처리 종료 날짜 (기본값: 오늘)
        :param setup_parameters: 설정 파라미터를 담은 딕셔너리
        """
        self.db_config = {
            'host': DB_HOST,
            'port': DB_PORT,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'database': DB_NAME,
            'charset': 'utf8mb4'
        }
        self.connection = None
        self.cursor = None
        self.backtest_manager = BacktestManager() # BacktestManager 인스턴스 초기화

        # 처리 기간 설정
        self.to_date = to_date if to_date else date.today()
        self.from_date = from_date if from_date else self.to_date - timedelta(days=90) 

        # setup_parameters 설정
        if setup_parameters is None:
            setup_parameters = {}

        self.LOOKBACK_DAYS_OHLCV = setup_parameters.get('lookback_days_ohlcv', self.LOOKBACK_DAYS_OHLCV)
        self.WEIGHT_PRICE_TREND = setup_parameters.get('weight_price_trend', self.WEIGHT_PRICE_TREND)
        self.WEIGHT_TRADING_VOLUME = setup_parameters.get('weight_trading_volume', self.WEIGHT_TRADING_VOLUME)
        self.WEIGHT_VOLATILITY = setup_parameters.get('weight_volatility', self.WEIGHT_VOLATILITY)
        self.WEIGHT_THEME_MENTION = setup_parameters.get('weight_theme_mention', self.WEIGHT_THEME_MENTION)

        logger.info(f"SetupManager 초기화 완료. 처리 기간: {self.from_date} ~ {self.to_date}")
        logger.info(f"설정 파라미터: LOOKBACK_DAYS_OHLCV={self.LOOKBACK_DAYS_OHLCV}, "
                    f"WEIGHT_PRICE_TREND={self.WEIGHT_PRICE_TREND}, "
                    f"WEIGHT_TRADING_VOLUME={self.WEIGHT_TRADING_VOLUME}, "
                    f"WEIGHT_VOLATILITY={self.WEIGHT_VOLATILITY}, "
                    f"WEIGHT_THEME_MENTION={self.WEIGHT_THEME_MENTION}")

    def _connect_db(self):
        """데이터베이스에 연결하고 커서를 생성합니다."""
        if self.connection is None or not self.connection.open:
            try:
                self.connection = pymysql.connect(**self.db_config)
                self.cursor = self.connection.cursor()
                logger.info("MariaDB 연결 성공.")
            except pymysql.MySQLError as e:
                logger.error(f"MariaDB 연결 오류: {e}")
                raise
        else:
            self.cursor = self.connection.cursor()

    def _close_db(self):
        """데이터베이스 연결과 커서를 닫습니다."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
        logger.info("MariaDB 연결 해제.")

    def _print_processing_summary(self, start_time, end_time, total_items, process_name=""):
        """처리 시간, 항목 수, 그리고 항목당 실행 시간을 출력합니다."""
        processing_time = end_time - start_time
        time_per_item = processing_time / total_items if total_items > 0 else 0
        logger.debug(f"[{process_name}] 총 처리 시간: {processing_time:.4f} 초")
        logger.debug(f"[{process_name}] 처리된 항목 수: {total_items} 개")
        logger.debug(f"[{process_name}] 항목 1개당 실행 시간: {time_per_item:.6f} 초")
        logger.debug(f"[{process_name}] 작업 완료.")

    def _initialize_database_tables(self):
        """
        필요한 모든 데이터베이스 테이블(theme_class, theme_stock, daily_theme, word_dic, theme_word_relevance, stock_info)을 생성합니다.
        """
        logger.info("\n--- 데이터베이스 테이블 초기화 시작 ---")
        try:
            create_stock_info_table_query = textwrap.dedent("""
                CREATE TABLE IF NOT EXISTS `stock_info` (
                    `stock_code` VARCHAR(7) NOT NULL,
                    `stock_name` VARCHAR(25) NOT NULL,
                    `market` VARCHAR(8) NULL,
                    `sector` VARCHAR(50) NULL,
                    `industry` VARCHAR(50) NULL,
                    `listing_date` DATE NULL,
                    `ipo_price` INT(11) NULL,
                    `market_cap` BIGINT(20) NULL,
                    PRIMARY KEY (`stock_code`) USING BTREE,
                    UNIQUE INDEX `stock_name_UNIQUE` (`stock_name`) USING BTREE
                );
            """)
            self.cursor.execute(create_stock_info_table_query)
            logger.info("stock_info 테이블 존재 확인 및 필요시 생성 완료.")

            create_theme_class_table_query = textwrap.dedent("""
                CREATE TABLE IF NOT EXISTS `theme_class` (
                    `theme_id` VARCHAR(36) NOT NULL,
                    `theme` VARCHAR(30) NOT NULL,
                    `theme_class` VARCHAR(30) NULL,
                    `theme_synonyms` JSON NULL,
                    `theme_hit` INT(11) NULL DEFAULT '0',
                    `theme_score` INT(11) NULL DEFAULT '0',
                    `momentum_score` DECIMAL(10,4) NULL DEFAULT '0.00',
                    `theme_desc` VARCHAR(200) NULL DEFAULT NULL,
                    PRIMARY KEY (`theme_id`) USING BTREE,
                    UNIQUE INDEX `theme_UNIQUE` (`theme`),
                    INDEX `idx_theme` (`theme`) USING BTREE
                );
            """)
            self.cursor.execute(create_theme_class_table_query)
            logger.info("theme_class 테이블 존재 확인 및 필요시 생성 완료.")

            create_theme_stock_table_query = textwrap.dedent("""
                CREATE TABLE IF NOT EXISTS `theme_stock` (
                    `theme_id` VARCHAR(36) NOT NULL,
                    `stock_code` VARCHAR(7) NOT NULL,
                    `stock_score` DECIMAL(10,4) DEFAULT 0.0000,
                    PRIMARY KEY (`theme_id`, `stock_code`) USING BTREE,
                    INDEX `idx_ts_stock_code` (`stock_code`),
                    INDEX `idx_ts_theme_id` (`theme_id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            self.cursor.execute(create_theme_stock_table_query)
            logger.info("theme_stock 테이블 존재 확인 및 필요시 생성 완료.")

            create_daily_theme_table_query = textwrap.dedent("""
                CREATE TABLE IF NOT EXISTS `daily_theme` (
                    `date` DATE NOT NULL,
                    `market` VARCHAR(8) NOT NULL,
                    `stock_code` VARCHAR(7) NOT NULL,
                    `stock_name` VARCHAR(25) NOT NULL,
                    `rate` DECIMAL(5,2) NULL DEFAULT '0.00',
                    `amount` INT(11) NULL DEFAULT '0',
                    `reason` VARCHAR(250) NOT NULL,
                    `reason_nouns` JSON NULL,
                    `theme` VARCHAR(250) NULL DEFAULT NULL,
                    PRIMARY KEY (`date`, `market`, `stock_code`) USING BTREE
                );
            """)
            self.cursor.execute(create_daily_theme_table_query)
            logger.info("daily_theme 테이블 존재 확인 및 필요시 생성 완료.")

            create_word_dic_table_query = textwrap.dedent("""
                CREATE TABLE IF NOT EXISTS word_dic (
                    word VARCHAR(25) NOT NULL,
                    freq INT NULL DEFAULT NULL,
                    cumul_rate DECIMAL(10,2) NULL DEFAULT NULL,
                    avg_rate DECIMAL(10,2) NULL DEFAULT NULL,
                    PRIMARY KEY (word)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            self.cursor.execute(create_word_dic_table_query)
            logger.info("word_dic 테이블 존재 확인 및 필요시 생성 완료.")
            
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
            self.cursor.execute(create_theme_word_relevance_table_query)
            logger.info("theme_word_relevance 테이블 존재 확인 및 필요시 생성 완료.")

            self.connection.commit()
            logger.info("--- 데이터베이스 테이블 초기화 완료 ---")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"테이블 초기화 중 오류 발생: {e}")
            raise

    def _load_stock_names(self):
        """stock_info 테이블에서 모든 종목명(stock_name)을 로드합니다."""
        logger.info("\n--- stock_info 테이블에서 종목명 로드 중 ---")
        start_time = time.time()
        self.cursor.execute("SELECT stock_name FROM stock_info WHERE stock_name IS NOT NULL AND stock_name != ''")
        stock_names = {row[0] for row in self.cursor.fetchall()}
        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(stock_names), "종목명 로드")
        return stock_names

    def _load_theme_synonyms(self):
        """theme_class에서 테마 ID, 이름, 동의어(JSON)를 로드하고 매핑을 생성합니다."""
        logger.info("\n--- 테마 동의어 및 테마 이름 사전 로딩 (theme_class 테이블에서) ---")
        start_time = time.time()
        
        self.cursor.execute("SELECT theme_id, theme, theme_synonyms FROM theme_class")
        theme_class_records = self.cursor.fetchall()

        theme_id_to_name_map = {}
        theme_synonym_map = {} # {synonym: theme_id}

        for theme_id, theme_name, synonyms_json in theme_class_records:
            theme_id_to_name_map[theme_id] = theme_name.strip()
            theme_synonym_map[theme_name.strip()] = theme_id

            if synonyms_json:
                try:
                    if isinstance(synonyms_json, bytes):
                        synonyms_json = synonyms_json.decode('utf-8')
                    parsed_synonyms = json.loads(synonyms_json)
                    if isinstance(parsed_synonyms, list):
                        for syn in parsed_synonyms:
                            if isinstance(syn, str) and syn.strip():
                                theme_synonym_map[syn.strip()] = theme_id
                except json.JSONDecodeError as e:
                    logger.warning(f"경고: theme_id {theme_id}의 theme_synonyms JSON 파싱 오류: {e}. 원본 JSON: {synonyms_json[:50]}...")
                except Exception as e:
                    logger.warning(f"경고: theme_id {theme_id}의 theme_synonyms 처리 중 예상치 못한 오류: {e}")

        sorted_synonyms = sorted(list(theme_synonym_map.keys()), key=len, reverse=True)
        theme_synonym_regex_map = [ 
            (synonym, theme_synonym_map[synonym], re.compile(r'\b' + re.escape(synonym) + r'\b', re.IGNORECASE)) 
            for synonym in sorted_synonyms 
        ]
        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(theme_synonym_map), "테마 동의어 로드")
        return theme_id_to_name_map, theme_synonym_map, theme_synonym_regex_map

    def _process_daily_theme_for_stock_scores(self):
        """
        daily_theme 테이블을 처리하여 종목별 테마 언급 정보와 daily_theme 업데이트 데이터를 준비합니다.
        지정된 from_date와 to_date 기간 내의 데이터를 처리합니다.
        """
        logger.info(f"\n--- daily_theme 테이블 처리 및 대상 종목 필터링 ({self.from_date} ~ {self.to_date}) ---")
        start_time = time.time()

        theme_id_to_name_map, _, theme_synonym_regex_map = self._load_theme_synonyms()

        target_stock_theme_mentions = {} # {stock_code: {'theme_ids': set(), 'mention_count': int}}
        daily_theme_updates = []

        self.cursor.execute("""
            SELECT date, market, stock_code, stock_name, rate, reason, theme 
            FROM daily_theme 
            WHERE date >= %s AND date <= %s
        """, (self.from_date, self.to_date))
        recent_daily_theme_records = self.cursor.fetchall()
        logger.info(f"daily_theme 레코드 {len(recent_daily_theme_records)}개 로드.")

        for record in recent_daily_theme_records:
            dt_date, dt_market, dt_stock_code, dt_stock_name, dt_rate, dt_reason, dt_existing_theme_str = record

            found_theme_ids_for_record = set()
            found_theme_names_for_record = set()

            current_reason_text = str(dt_reason)
            for synonym, theme_id, synonym_regex in theme_synonym_regex_map:
                if synonym_regex.search(current_reason_text):
                    found_theme_ids_for_record.add(theme_id)
                    if theme_id in theme_id_to_name_map:
                        found_theme_names_for_record.add(theme_id_to_name_map[theme_id])
            
            current_themes_in_db = set()
            if dt_existing_theme_str:
                current_themes_in_db.update([t.strip() for t in dt_existing_theme_str.split(',') if t.strip()])
            
            new_theme_list = sorted(list(current_themes_in_db | found_theme_names_for_record))
            new_theme_string = ", ".join(new_theme_list)
            
            if new_theme_string != dt_existing_theme_str:
                daily_theme_updates.append((new_theme_string, dt_date, dt_market, dt_stock_code))

            if dt_rate is not None and float(dt_rate) < self.DAILY_THEME_MAX_RATE_THRESHOLD and found_theme_ids_for_record:
                if dt_stock_code not in target_stock_theme_mentions:
                    target_stock_theme_mentions[dt_stock_code] = {
                        'theme_ids': set(),
                        'mention_count': 0,
                    }
                target_stock_theme_mentions[dt_stock_code]['theme_ids'].update(found_theme_ids_for_record)
                target_stock_theme_mentions[dt_stock_code]['mention_count'] += 1
        
        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(recent_daily_theme_records), "daily_theme 처리 및 필터링")
        return target_stock_theme_mentions, daily_theme_updates

    def _update_daily_theme_table(self, daily_theme_updates):
        """daily_theme 테이블의 theme 컬럼을 업데이트합니다."""
        if daily_theme_updates:
            logger.info("\n--- daily_theme 테이블 업데이트 ---")
            start_time = time.time()
            update_daily_theme_query = textwrap.dedent("""
                UPDATE daily_theme
                SET theme = %s
                WHERE date = %s AND market = %s AND stock_code = %s
            """)
            self.cursor.executemany(update_daily_theme_query, daily_theme_updates)
            self.connection.commit()
            end_time = time.time()
            self._print_processing_summary(start_time, end_time, len(daily_theme_updates), "daily_theme (update)")
        else:
            logger.info("daily_theme 테이블에 업데이트할 레코드가 없습니다.")

    def _calculate_price_trend_score(self, df_stock_data):
        """주가 추세 점수 계산: 최근 5일, 20일 이동평균선 정배열 및 누적 수익률 반영"""
        if len(df_stock_data) < 20: return 0
        
        df_stock_data = df_stock_data.sort_values(by='date').copy()
        df_stock_data['MA5'] = df_stock_data['close'].rolling(window=5).mean()
        df_stock_data['MA20'] = df_stock_data['close'].rolling(window=20).mean()
        
        latest_data = df_stock_data.iloc[-1]
        
        score = 0
        
        if latest_data['MA5'] > latest_data['MA20']:
            score += 30
        
        if len(df_stock_data) >= 5:
            start_price_5 = df_stock_data.tail(5).iloc[0]['close']
            end_price_5 = df_stock_data.tail(5).iloc[-1]['close']
            if start_price_5 > 0:
                return_5_day = ((end_price_5 - start_price_5) / start_price_5) * 100
                if return_5_day > 0:
                    score += min(return_5_day, 50) * 0.5 
        
        if len(df_stock_data) >= 20:
            start_price_20 = df_stock_data.tail(20).iloc[0]['close']
            end_price_20 = df_stock_data.tail(20).iloc[-1]['close']
            if start_price_20 > 0:
                return_20_day = ((end_price_20 - start_price_20) / start_price_20) * 100
                if return_20_day > 0:
                    score += min(return_20_day, 100) * 0.2 
        
        return min(max(score, 0), 100) 

    def _calculate_trading_volume_score(self, df_stock_data):
        """거래대금 점수 계산: 최근 거래대금 증가율 및 절대적인 규모 반영"""
        if len(df_stock_data) < 20: return 0
        
        df_stock_data = df_stock_data.sort_values(by='date').copy()
        
        recent_volume = df_stock_data['trading_value'].tail(5).mean()
        avg_volume = df_stock_data['trading_value'].mean()
        
        score = 0
        
        if avg_volume > 0:
            volume_growth_rate = ((recent_volume - avg_volume) / avg_volume) * 100
            if volume_growth_rate > 0:
                score += min(volume_growth_rate * 0.2, 50) 
        
        if recent_volume >= 10_000_000_000: score += 10 
        if recent_volume >= 50_000_000_000: score += 10 
        if recent_volume >= 100_000_000_000: score += 10 
        
        return min(max(score, 0), 100) 

    def _calculate_volatility_score(self, df_stock_data):
        """변동성 점수 계산: ATR 기반 변동성 및 일중 변동폭 반영"""
        if len(df_stock_data) < 10: return 0
        
        df_stock_data = df_stock_data.sort_values(by='date').copy()
        
        high_low = df_stock_data['high'] - df_stock_data['low']
        high_prev_close = abs(df_stock_data['high'] - df_stock_data['close'].shift(1))
        low_prev_close = abs(df_stock_data['low'] - df_stock_data['close'].shift(1))
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
        atr = tr.rolling(window=10).mean().iloc[-1]
        
        latest_close = df_stock_data.iloc[-1]['close']
        score = 0
        
        if latest_close > 0:
            atr_ratio = (atr / latest_close) * 100 
            score += min(atr_ratio * 5, 50) 
                                             
        if len(df_stock_data) >= 5:
            df_stock_data['daily_range_ratio'] = ((df_stock_data['high'] - df_stock_data['low']) / df_stock_data['close']) * 100
            avg_daily_range_ratio = df_stock_data['daily_range_ratio'].tail(5).mean()
            score += min(avg_daily_range_ratio * 2, 50) 
                                                        
        return min(max(score, 0), 100) 

    def _calculate_theme_mention_score(self, mention_count_in_period):
        """테마 언급 빈도 점수 계산: 최근 한달 내 언급 횟수에 비례"""
        score = min(mention_count_in_period * 20, 100) 
        return max(0, score) 

    def _calculate_stock_scores(self, target_stock_theme_mentions):
        """
        필터링된 종목들의 시세 데이터를 로드하고 stock_score를 계산합니다.
        시세 데이터는 self.to_date를 기준으로 LOOKBACK_DAYS_OHLCV 만큼 이전부터 가져옵니다.
        """
        logger.info("\n--- 대상 종목 시세 데이터 로드 및 stock_score 계산 ---")
        start_time = time.time()
        
        all_stock_codes_to_process = list(target_stock_theme_mentions.keys())
        theme_stock_score_updates = {} # { (theme_id, stock_code): final_score }

        if not all_stock_codes_to_process:
            logger.info("stock_score를 계산할 대상 종목이 없습니다.")
            return theme_stock_score_updates

        for stock_code in all_stock_codes_to_process:
            from_date_ohlcv = self.to_date - timedelta(days=self.LOOKBACK_DAYS_OHLCV)
            to_date_ohlcv = self.to_date

            df_stock_data = self.backtest_manager.cache_daily_ohlcv(stock_code, from_date_ohlcv, to_date_ohlcv)
            
            if 'trading_value' not in df_stock_data.columns:
                df_stock_data['trading_value'] = df_stock_data['close'] * df_stock_data['volume']
                logger.debug(f"종목 {stock_code}: 'trading_value' 컬럼이 없어 'close * volume'으로 생성했습니다.")
            
            df_stock_data = df_stock_data.reset_index().rename(columns={'index': 'date'})
            df_stock_data['date'] = pd.to_datetime(df_stock_data['date']).dt.date

            if df_stock_data.empty or len(df_stock_data) < 20: 
                logger.debug(f"종목 {stock_code}: 시세 데이터 부족 ({len(df_stock_data)}개), 점수 계산 스킵.")
                continue

            price_trend_score = self._calculate_price_trend_score(df_stock_data)
            trading_volume_score = self._calculate_trading_volume_score(df_stock_data)
            volatility_score = self._calculate_volatility_score(df_stock_data)
            
            mention_data = target_stock_theme_mentions[stock_code]
            theme_mention_score = self._calculate_theme_mention_score(mention_data['mention_count'])

            final_stock_score = (
                (price_trend_score * self.WEIGHT_PRICE_TREND) +
                (trading_volume_score * self.WEIGHT_TRADING_VOLUME) +
                (volatility_score * self.WEIGHT_VOLATILITY) +
                (theme_mention_score * self.WEIGHT_THEME_MENTION)
            ) / 100
            
            final_stock_score = round(final_stock_score, 2)

            for theme_id in mention_data['theme_ids']:
                key = (theme_id, stock_code)
                theme_stock_score_updates[key] = final_stock_score
        
        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(all_stock_codes_to_process), "stock_score 계산")
        return theme_stock_score_updates

    def _update_theme_stock_table(self, theme_stock_score_updates):
        """theme_stock 테이블을 새로운 스코어로 업데이트합니다."""
        logger.info("\n--- theme_stock 테이블 초기화 ---")
        start_time_reset_ts = time.time()
        self.cursor.execute("UPDATE theme_stock SET stock_score = 0")
        self.connection.commit()
        end_time_reset_ts = time.time()
        logger.info(f"theme_stock 테이블 stock_score 초기화 완료 ({end_time_reset_ts - start_time_reset_ts:.4f} 초)")


        if theme_stock_score_updates:
            logger.info("\n--- theme_stock 테이블 업데이트 ---")
            start_time = time.time()

            theme_stock_data_to_replace = [
                (theme_id, stock_code, score)
                for (theme_id, stock_code), score in theme_stock_score_updates.items()
            ]

            replace_theme_stock_query = textwrap.dedent("""
                REPLACE INTO theme_stock (theme_id, stock_code, stock_score)
                VALUES (%s, %s, %s)
            """)
            self.cursor.executemany(replace_theme_stock_query, theme_stock_data_to_replace)
            self.connection.commit()
            end_time = time.time()
            self._print_processing_summary(start_time, end_time, len(theme_stock_data_to_replace), "theme_stock (replace)")
        else:
            logger.info("theme_stock 테이블에 업데이트할 유효한 레코드가 없습니다.")

    def _print_top_stock_scores(self) -> pd.DataFrame:
        """
        stock_score 상위 종목들을 DataFrame으로 반환하고 내용을 로깅합니다.
        """
        logger.info("\n--- 주식 점수 (Stock Score) 상위 종목 조회 ---")
        start_time = time.time()

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
                ts.stock_score > 0
            ORDER BY 
                ts.stock_score DESC
            LIMIT 80;
        """)
        
        self.cursor.execute(select_top_stocks_query)
        top_stocks_raw = self.cursor.fetchall()

        if top_stocks_raw:
            df = pd.DataFrame(top_stocks_raw, columns=['stock_code', 'stock_name', 'theme', 'stock_score'])
            # 종목코드를 인덱스로 설정하고 중복 제거 (정렬된 상태이므로 첫 번째(가장 높은 점수) 유지)
            df = df.drop_duplicates(subset=['stock_code']).set_index('stock_code')
            logger.info("\n--- 주식 점수 (Stock Score) 상위 종목 DataFrame 내용 ---")
            logger.info(df)
        else:
            df = pd.DataFrame(columns=['stock_code', 'stock_name', 'theme', 'stock_score']).set_index('stock_code')
            logger.info("현재 계산된 stock_score가 있는 종목이 없습니다.")

        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(top_stocks_raw), "최종 결과 조회")
        return df

    def _process_daily_theme_and_update_word_dic(self, stock_names_set):
        """
        daily_theme 테이블에서 데이터를 읽어 미리 추출된 명사를 활용하여 word_dic 테이블에 누적/업데이트합니다.
        지정된 from_date와 to_date 기간 내의 데이터를 처리합니다.
        """
        logger.info(f"\n--- daily_theme DB 데이터 분석 및 word_dic 업데이트 시작 ({self.from_date} ~ {self.to_date}) ---")
        start_total_time = time.time()

        self.cursor.execute("""
            SELECT date, market, stock_code, stock_name, rate, reason_nouns, theme
            FROM daily_theme
            WHERE date >= %s AND date <= %s
        """, (self.from_date, self.to_date))
        daily_theme_records = self.cursor.fetchall() 
        total_records_loaded = len(daily_theme_records)
        logger.info(f"daily_theme 테이블에서 총 {total_records_loaded}개 레코드 로드.")

        if not daily_theme_records:
            logger.info("daily_theme 테이블에 유효한 데이터가 없어 word_dic 분석을 진행할 수 없습니다.")
            return False

        noun_counts = defaultdict(int)
        noun_weighted_sum = defaultdict(float) 
        processed_reason_count = 0
        
        logger.info("daily_theme 테이블 각 레코드의 'reason_nouns' 컬럼 명사 추출 중 (사전 추출된 명사 사용)...")
        start_processing_records_time = time.time()

        for dt_date, dt_market, dt_stock_code, dt_stock_name, dt_rate, dt_reason_nouns_json, dt_existing_theme in daily_theme_records:
            try:
                rate = float(dt_rate) 
                if rate > self.DAILY_THEME_MAX_RATE_THRESHOLD: 
                    continue
            except (ValueError, TypeError):
                continue

            nouns = []
            if dt_reason_nouns_json:
                try:
                    if isinstance(dt_reason_nouns_json, bytes): 
                        dt_reason_nouns_json = dt_reason_nouns_json.decode('utf-8')
                    nouns = json.loads(dt_reason_nouns_json)
                    if not isinstance(nouns, list): 
                        nouns = []
                except json.JSONDecodeError:
                    logger.warning(f"경고: reason_nouns JSON 파싱 오류. 레코드 건너뜀: {dt_reason_nouns_json[:50]}...")
                    continue
            
            if not nouns: continue 

            for noun in nouns:
                cleaned_noun = str(noun).strip().replace(" ", "") 
                
                if (len(cleaned_noun) == 1 or
                    re.match(r'^[-\d]', cleaned_noun) or
                    cleaned_noun in stock_names_set):
                    continue

                noun_counts[cleaned_noun] += 1
                noun_weighted_sum[cleaned_noun] += rate
            processed_reason_count += 1
        
        end_processing_records_time = time.time()
        self._print_processing_summary(start_processing_records_time, end_processing_records_time,
                                       processed_reason_count, "명사 추출 및 통계 계산 (word_dic)")

        data_to_update_word_dic = []
        for noun, count in noun_counts.items():
            cumul_rate = noun_weighted_sum[noun]
            avg_rate = cumul_rate / count

            if count > 0 and abs(avg_rate) > 1: 
                data_to_update_word_dic.append((noun, count, round(cumul_rate, 2), round(avg_rate, 2)))

        logger.info(f"word_dic 업데이트를 위한 최종 데이터 {len(data_to_update_word_dic)}개 준비 완료.")

        if data_to_update_word_dic:
            logger.info("\n--- word_dic 테이블 업데이트/삽입 시작 ---")
            start_db_update_time = time.time()
            replace_word_dic_query = textwrap.dedent("""
                REPLACE INTO word_dic (word, freq, cumul_rate, avg_rate)
                VALUES (%s, %s, %s, %s)
            """)
            self.cursor.executemany(replace_word_dic_query, data_to_update_word_dic)
            self.connection.commit()
            end_db_update_time = time.time()
            self._print_processing_summary(start_db_update_time, end_db_update_time,
                                           len(data_to_update_word_dic), "word_dic DB 업데이트")
        else:
            logger.info("word_dic 테이블에 업데이트/삽입할 데이터가 없습니다.")

        end_total_time = time.time()
        self._print_processing_summary(start_total_time, end_total_time, total_records_loaded, "전체 daily_theme DB 분석 및 word_dic 업데이트")
        return True

    def _clean_word_dic(self):
        """
        word_dic 테이블을 클리닝합니다. 이 함수는 금요일에만 호출됩니다.
        """
        logger.info(f"\n--- word_dic 클리닝 시작 ---")
        start_time = time.time()

        try:
            if self.WORD_DIC_BLACKLIST_WORDS:
                placeholders = ', '.join(['%s'] * len(self.WORD_DIC_BLACKLIST_WORDS))
                delete_blacklist_query = f"DELETE FROM word_dic WHERE word IN ({placeholders})"
                self.cursor.execute(delete_blacklist_query, self.WORD_DIC_BLACKLIST_WORDS)
                logger.info(f"블랙리스트 단어 {self.cursor.rowcount}개 제거 완료.")
                self.connection.commit()

            delete_low_freq_query = f"DELETE FROM word_dic WHERE freq < %s"
            self.cursor.execute(delete_low_freq_query, (self.WORD_DIC_MIN_GLOBAL_FREQ,))
            logger.info(f"최소 빈도 {self.WORD_DIC_MIN_GLOBAL_FREQ} 미만 단어 {self.cursor.rowcount}개 제거 완료.")
            self.connection.commit()

            self.cursor.execute("SELECT SUM(freq) FROM word_dic")
            total_freq_result = self.cursor.fetchone()
            total_freq = total_freq_result[0] if total_freq_result and total_freq_result[0] else 0

            if total_freq == 0:
                logger.info("word_dic이 비어있거나 데이터가 없어 고빈도 단어 처리를 건너낍니다.")
                return

            self.cursor.execute("SELECT word, freq, avg_rate FROM word_dic ORDER BY freq DESC")
            all_words_sorted_by_freq = self.cursor.fetchall()

            freq_cutoff_index = int(len(all_words_sorted_by_freq) * (1 - self.WORD_DIC_MAX_GLOBAL_FREQ_PERCENTILE))
            if freq_cutoff_index >= len(all_words_sorted_by_freq): 
                freq_cutoff_index = len(all_words_sorted_by_freq) - 1
            
            high_freq_threshold = all_words_sorted_by_freq[freq_cutoff_index][1] if all_words_sorted_by_freq else 0
            
            deleted_high_freq_count = 0
            words_to_delete = []
            for word, freq, avg_rate in all_words_sorted_by_freq:
                if freq >= high_freq_threshold and abs(float(avg_rate)) < self.WORD_DIC_MAX_COMMON_WORD_AVG_RATE_DEVIATION:
                    words_to_delete.append(word)
            
            if words_to_delete:
                placeholders = ', '.join(['%s'] * len(words_to_delete))
                delete_high_freq_query = f"DELETE FROM word_dic WHERE word IN ({placeholders})"
                self.cursor.executemany(delete_high_freq_query, words_to_delete)
                deleted_high_freq_count = self.cursor.rowcount
                self.connection.commit()
            logger.info(f"최고 빈도 & 낮은 영향력 단어 {deleted_high_freq_count}개 제거 완료.")
            self.connection.commit()

        except Exception as e:
            self.connection.rollback()
            logger.error(f"word_dic 클리닝 중 오류 발생: {e}")
            raise
        finally:
            end_time = time.time()
            self._print_processing_summary(start_time, end_time, 0, "word_dic 클리닝")

    def _load_theme_stock_mapping(self):
        """theme_stock 테이블에서 종목 코드와 테마 매핑을 로드합니다."""
        logger.info("\n--- theme_stock 테이블에서 종목-테마 매핑 로드 중 ---")
        start_time = time.time()
        query = textwrap.dedent("""
            SELECT 
                ts.stock_code, 
                tc.theme AS theme_name
            FROM 
                theme_stock ts
            JOIN 
                theme_class tc ON ts.theme_id = tc.theme_id;
        """)
        self.cursor.execute(query)
        
        stock_to_themes = defaultdict(list)
        for stock_code, theme in self.cursor.fetchall():
            stock_to_themes[stock_code].append(theme)
        
        end_time = time.time()
        self._print_processing_summary(start_time, end_time, len(stock_to_themes), "종목-테마 매핑 로드")
        return stock_to_themes

    def _calculate_and_update_theme_word_relevance(self, stock_names_set, stock_to_themes_map):
        """
        daily_theme 테이블에서 데이터를 읽어 미리 추출된 명사를 활용하고,
        theme_stock 매핑을 활용하여 theme_word_relevance 테이블을 계산하고 업데이트합니다.
        지정된 from_date와 to_date 기간 내의 데이터를 처리합니다.
        """
        logger.info(f"\n--- theme_word_relevance 계산 및 업데이트 시작 ({self.from_date} ~ {self.to_date}) ---")
        start_total_time = time.time()

        self._truncate_theme_word_relevance()

        self.cursor.execute("""
            SELECT date, stock_code, rate, reason_nouns
            FROM daily_theme
            WHERE date >= %s AND date <= %s
        """, (self.from_date, self.to_date))
        daily_theme_records = self.cursor.fetchall()
        total_records_loaded = len(daily_theme_records)
        logger.info(f"daily_theme 테이블에서 총 {total_records_loaded}개 레코드 로드.")

        if not daily_theme_records:
            logger.info("daily_theme 테이블에 유효한 데이터가 없어 theme_word_relevance 계산을 진행할 수 없습니다.")
            return False

        theme_word_aggr = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sum_rate': 0.0}))

        processed_record_count = 0
        
        logger.info("daily_theme 레코드 분석 및 테마-단어 관련성 집계 중...")
        start_processing_records_time = time.time()

        for dt_date, dt_stock_code, dt_rate, dt_reason_nouns_json in daily_theme_records:
            try:
                rate = float(dt_rate) 
                if rate > self.DAILY_THEME_MAX_RATE_THRESHOLD: 
                    continue
            except (ValueError, TypeError):
                continue

            nouns = []
            if dt_reason_nouns_json:
                try:
                    if isinstance(dt_reason_nouns_json, bytes):
                        dt_reason_nouns_json = dt_reason_nouns_json.decode('utf-8')
                    nouns = json.loads(dt_reason_nouns_json)
                    if not isinstance(nouns, list): 
                        nouns = []
                except json.JSONDecodeError:
                    logger.warning(f"경고: reason_nouns JSON 파싱 오류. 레코드 건너_theme_word_relevance: {dt_reason_nouns_json[:50]}...")
                    continue
            
            if not nouns: continue

            associated_themes = stock_to_themes_map.get(dt_stock_code, [])
            if not associated_themes: continue 

            for noun in nouns:
                cleaned_noun = str(noun).strip().replace(" ", "") 
                
                if (len(cleaned_noun) == 1 or
                    re.match(r'^[-\d]', cleaned_noun) or
                    cleaned_noun in stock_names_set):
                    continue

                for theme in associated_themes:
                    theme_word_aggr[theme][cleaned_noun]['count'] += 1
                    theme_word_aggr[theme][cleaned_noun]['sum_rate'] += rate

            processed_record_count += 1
        
        end_processing_records_time = time.time()
        self._print_processing_summary(start_processing_records_time, end_processing_records_time,
                                       processed_record_count, "테마-단어 관련성 데이터 집계")

        data_to_update_theme_word_relevance = []
        for theme, words_data in theme_word_aggr.items():
            for word, metrics in words_data.items():
                num_occurrences = metrics['count']
                sum_rate = metrics['sum_rate']

                if num_occurrences < self.THEME_WORD_RELEVANCE_MIN_OCCURRENCES: 
                    continue
                
                avg_stock_rate_in_theme = sum_rate / num_occurrences
                
                relevance_score = num_occurrences * (1 + abs(avg_stock_rate_in_theme / 100.0))
                
                data_to_update_theme_word_relevance.append((
                    theme, 
                    word, 
                    round(relevance_score, 4), 
                    num_occurrences, 
                    round(avg_stock_rate_in_theme, 2)
                ))
        
        logger.info(f"theme_word_relevance 업데이트를 위한 최종 데이터 {len(data_to_update_theme_word_relevance)}개 준비 완료.")

        if data_to_update_theme_word_relevance:
            logger.info("\n--- theme_word_relevance 테이블 업데이트/삽입 시작 ---")
            start_db_update_time = time.time()
            
            insert_theme_word_relevance_query = textwrap.dedent("""
                INSERT INTO theme_word_relevance (theme, word, relevance_score, num_occurrences, avg_stock_rate_in_theme)
                VALUES (%s, %s, %s, %s, %s)
            """)
            self.cursor.executemany(insert_theme_word_relevance_query, data_to_update_theme_word_relevance)
            self.connection.commit()
            end_db_update_time = time.time()
            self._print_processing_summary(start_db_update_time, end_db_update_time,
                                           len(data_to_update_theme_word_relevance), "theme_word_relevance DB 업데이트")
        else:
            logger.info("theme_word_relevance 테이블에 업데이트/삽입할 데이터가 없습니다.")

        end_total_time = time.time()
        self._print_processing_summary(start_total_time, end_total_time, total_records_loaded, "전체 theme_word_relevance 계산 및 업데이트")
        return True

    def _truncate_theme_word_relevance(self):
        """theme_word_relevance 테이블을 비웁니다."""
        try:
            self.cursor.execute("TRUNCATE TABLE theme_word_relevance")
            self.connection.commit()
            logger.info("기존 theme_word_relevance 데이터 삭제 완료.")
        except Exception as e:
            logger.error(f"theme_word_relevance 테이블 TRUNCATE 중 오류 발생: {e}")
            self.connection.rollback()
            raise


    def run_all_processes(self, from_date: date = None, to_date: date = None):
        """
        SetupManager의 모든 데이터 처리 및 점수 계산 작업을 순차적으로 실행합니다.
        :param from_date: 데이터 처리 시작 날짜 (기본값: 객체 초기화 시 설정된 값)
        :param to_date: 데이터 처리 종료 날짜 (기본값: 객체 초기화 시 설정된 값)
        """
        if from_date:
            self.from_date = from_date
        if to_date:
            self.to_date = to_date

        logger.info(f"--- SetupManager의 모든 처리 작업 시작 (기간: {self.from_date} ~ {self.to_date}) ---")
        overall_start_time = time.time()

        try:
            self._connect_db()
            self._initialize_database_tables()
            stock_names = self._load_stock_names()

            _, _, _ = self._load_theme_synonyms()
            target_stock_theme_mentions, daily_theme_updates = \
                self._process_daily_theme_for_stock_scores() 
            self._update_daily_theme_table(daily_theme_updates)
            
            theme_stock_score_updates = self._calculate_stock_scores(target_stock_theme_mentions)
            self._update_theme_stock_table(theme_stock_score_updates)

            self._process_daily_theme_and_update_word_dic(stock_names) 
            
            current_system_date = date.today()
            if current_system_date.weekday() == 4: 
                logger.info(f"\n[알림] 오늘은 금요일이므로 word_dic 클리닝을 시작합니다. (오늘 날짜: {current_system_date})")
                self._clean_word_dic()
            else:
                logger.info(f"\n[알림] 오늘은 금요일이 아니므로 word_dic 클리닝을 건너킵니다. (오늘 날짜: {current_system_date}, 오늘 요일: {current_system_date.weekday()})")

            stock_to_themes = self._load_theme_stock_mapping()
            self._calculate_and_update_theme_word_relevance(stock_names, stock_to_themes) 

            # 최종 결과 출력 (DataFrame 반환)
            top_stocks_df = self._print_top_stock_scores()
            # 필요에 따라 반환된 DataFrame을 추가 처리하거나 사용
            logger.info("SetupManager 작업 완료. Top Stock Scores DataFrame이 반환되었습니다.")

        except Exception as e:
            logger.critical(f"SetupManager 전체 실행 중 치명적인 오류 발생: {e}", exc_info=True)
            if self.connection:
                self.connection.rollback()
        finally:
            self._close_db()
            overall_end_time = time.time()
            self._print_processing_summary(overall_start_time, overall_end_time, 0, "SetupManager 전체 실행")
            logger.info("--- SetupManager의 모든 처리 작업 완료 ---")


if __name__ == "__main__":

    # 예시 1: 기본 기간 (초기화 시 설정된 90일) 및 기본 파라미터로 실행
    logger.info("\n--- 기본 기간 및 기본 파라미터로 SetupManager 실행 ---")
    manager_default = SetupManager()
    manager_default.run_all_processes()

    # 예시 2: 특정 기간을 지정하여 실행 (예: 2024년 1월 1일부터 2024년 6월 30일까지) 및 특정 파라미터 사용
    logger.info("\n--- 특정 기간 (2025-01-01 ~ 2025-06-30) 및 특정 파라미터로 SetupManager 실행 ---")
    specific_from_date = date(2025, 1, 1)
    specific_to_date = date(2025, 6, 30)
    custom_params = {
        'lookback_days_ohlcv': 120,
        'weight_price_trend': 50,
        'weight_trading_volume': 20,
        'weight_volatility': 15,
        'weight_theme_mention': 15
    }
    manager_custom = SetupManager(setup_parameters=custom_params)
    manager_custom.run_all_processes(from_date=specific_from_date, to_date=specific_to_date)

    # 예시 3: 최근 30일 데이터만, 일부 파라미터만 변경
    logger.info("\n--- 최근 30일 기간 및 일부 파라미터 변경으로 SetupManager 실행 ---")
    recent_30_days_from = date.today() - timedelta(days=30)
    recent_30_days_to = date.today()
    partial_params = {
        'weight_price_trend': 60,
        'weight_theme_mention': 10
    }
    manager_partial = SetupManager(setup_parameters=partial_params)
    manager_partial.run_all_processes(from_date=recent_30_days_from, to_date=recent_30_days_to)