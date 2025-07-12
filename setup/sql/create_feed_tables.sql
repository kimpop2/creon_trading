-- create_feed_tables.sql


-- ohlcv_minute 테이블 생성 (1분봉 데이터)
-- PriceFeed 모듈에서 수집
CREATE TABLE IF NOT EXISTS ohlcv_minute (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    datetime DATETIME NOT NULL COMMENT '데이터 시점 (날짜 및 시간)',
    open DECIMAL(18, 2) NOT NULL COMMENT '시가',
    high DECIMAL(18, 2) NOT NULL COMMENT '고가',
    low DECIMAL(18, 2) NOT NULL COMMENT '저가',
    close DECIMAL(18, 2) NOT NULL COMMENT '종가',
    volume BIGINT NOT NULL COMMENT '거래량',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_ohlcv_minute (stock_code, datetime),
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code) ON DELETE CASCADE
) COMMENT='분봉 OHLCV 데이터';


-- market_volume 테이블 생성 (시장별 거래대금)
-- PriceFeed 모듈에서 수집
CREATE TABLE IF NOT EXISTS market_volume (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    market_type VARCHAR(10) NOT NULL COMMENT '시장 구분 (KOSPI/KOSDAQ)',
    date DATE NOT NULL COMMENT '날짜',
    time TIME NOT NULL COMMENT '시간',
    total_amount BIGINT NOT NULL COMMENT '총 거래대금 (단위: 원)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_market_volume (market_type, date, time)
) COMMENT='시장별 실시간/평균 거래대금';

-- news_raw 테이블 생성 (원본 뉴스 및 텔레그램 메시지)
-- NewsFeed, TelegramFeed 모듈에서 수집
CREATE TABLE IF NOT EXISTS news_raw (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    source VARCHAR(50) NOT NULL COMMENT '출처 (예: Creon News, Telegram, Creon Announcement)',
    datetime DATETIME NOT NULL COMMENT '수집 시점',
    title VARCHAR(500) NOT NULL COMMENT '기사/메시지 제목',
    content TEXT COMMENT '원본 내용',
    url VARCHAR(500) COMMENT '원본 URL (선택 사항)',
    related_stocks VARCHAR(255) COMMENT '관련 종목 코드 (JSON 배열 또는 쉼표 구분)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    INDEX idx_news_raw_datetime (datetime),
    INDEX idx_news_raw_source (source)
) COMMENT='원본 뉴스 및 메시지 데이터';

-- investor_trends 테이블 생성 (투자자 매매 동향)
-- InvesterFeed 모듈에서 수집
CREATE TABLE IF NOT EXISTS investor_trends (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    date DATE NOT NULL COMMENT '일자',
    time TIME NOT NULL COMMENT '시간 (잠정치 기준)',
    current_price DECIMAL(18, 2) COMMENT '현재가',
    volume_total BIGINT COMMENT '총 거래량',
    net_foreign BIGINT COMMENT '외국인 순매수 수량/금액',
    net_institutional BIGINT COMMENT '기관계 순매수 수량/금액',
    net_insurance_etc BIGINT COMMENT '보험/기타금융 순매수 수량/금액',
    net_trust BIGINT COMMENT '투신 순매수 수량/금액',
    net_bank BIGINT COMMENT '은행 순매수 수량/금액',
    net_pension BIGINT COMMENT '연기금 순매수 수량/금액',
    net_gov_local BIGINT COMMENT '국가/지자체 순매수 수량/금액',
    net_other_corp BIGINT COMMENT '기타법인 순매수 수량/금액',
    data_type VARCHAR(10) NOT NULL COMMENT '수량 또는 금액 데이터 (예: 수량, 금액)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_investor_trends (stock_code, date, time, data_type),
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code) ON DELETE CASCADE,
    INDEX idx_investor_trends_date_time (date, time)
) COMMENT='투자자별 매매 동향 데이터';

-- news_summaries 테이블 생성 (NLP 분석을 통한 뉴스 요약 및 감성)
-- NLP_Analysis 모듈에서 처리
CREATE TABLE IF NOT EXISTS news_summaries (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    original_news_id BIGINT NOT NULL COMMENT '원본 news_raw 테이블 ID',
    summary TEXT NOT NULL COMMENT 'AI/NLP 요약 내용',
    sentiment_score DECIMAL(5, 2) COMMENT '감성 점수 (예: -1.0 ~ 1.0)',
    processed_at DATETIME NOT NULL COMMENT '처리 완료 시점',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    FOREIGN KEY (original_news_id) REFERENCES news_raw(id) ON DELETE CASCADE,
    INDEX idx_news_summaries_processed_at (processed_at)
) COMMENT='뉴스 요약 및 감성 분석 결과';

-- thematic_stocks 테이블 생성 (발굴된 테마 및 관련 종목)
-- NLP_Analysis 모듈에서 처리
CREATE TABLE IF NOT EXISTS thematic_stocks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    theme_name VARCHAR(255) NOT NULL COMMENT '테마 키워드',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    analysis_date DATE NOT NULL COMMENT '분석 날짜',
    relevance_score DECIMAL(5, 2) COMMENT '테마와 종목 간의 관련성 점수',
    mention_count INT COMMENT '뉴스에서 언급된 빈도',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_thematic_stocks (theme_name, stock_code, analysis_date),
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code) ON DELETE CASCADE,
    INDEX idx_thematic_stocks_theme_date (theme_name, analysis_date)
) COMMENT='테마별 관련 종목 정보';

-- daily_universe 테이블 생성 (일별 매매 대상 유니버스 종목 및 점수)
-- NLP_Analysis 모듈에서 처리 (setup_daily_universe.py)
CREATE TABLE IF NOT EXISTS daily_universe (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '고유 ID',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    date DATE NOT NULL COMMENT '유니버스 선정 날짜',
    total_score DECIMAL(10, 4) NOT NULL COMMENT '계산된 총 점수 (가격 추세, 거래량, 테마 등)',
    score_detail JSON COMMENT '점수 세부 내용 (예: {"price_trend": 0.5, "volume": 0.3})',
    is_selected BOOLEAN DEFAULT FALSE COMMENT '매매 유니버스에 포함 여부',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_daily_universe (stock_code, date),
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code) ON DELETE CASCADE,
    INDEX idx_daily_universe_date (date)
) COMMENT='일일 매매 유니버스 선정 결과';
