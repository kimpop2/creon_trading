

-- stock_info 테이블 생성 (종목 기본 정보 및 최신 재무 데이터)
CREATE TABLE IF NOT EXISTS stock_info (
    stock_code VARCHAR(10) PRIMARY KEY COMMENT '종목 코드 (예: A005930)',
    stock_name VARCHAR(100) NOT NULL COMMENT '종목명',
    market_type VARCHAR(20) COMMENT '시장 구분 (예: KOSPI, KOSDAQ)',
    sector VARCHAR(100) COMMENT '섹터/업종',
    per DECIMAL(10, 2) COMMENT '주가수익비율 (최신)',
    pbr DECIMAL(10, 2) COMMENT '주가순자산비율 (최신)',
    eps DECIMAL(15, 2) COMMENT '주당순이익 (최신)',
    roe DECIMAL(10, 2) COMMENT '자기자본이익률 (최신)',
    debt_ratio DECIMAL(10, 2) COMMENT '부채비율 (최신)',
    sales BIGINT COMMENT '매출액 (최신, 백만 원 단위)',
    operating_profit BIGINT COMMENT '영업이익 (최신, 백만 원 단위)',
    net_profit BIGINT COMMENT '당기순이익 (최신, 백만 원 단위)',
    recent_financial_date DATE COMMENT '최신 재무 데이터의 기준 일자 (YYYY-MM-DD)',
    reg_date DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '등록일시',
    upd_date DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 일시'
) COMMENT='종목 기본 정보 및 재무 데이터';

-- market_calendar 테이블 생성
CREATE TABLE IF NOT EXISTS market_calendar (
    date DATE NOT NULL PRIMARY KEY,         -- 날짜 (YYYY-MM-DD), 기본 키
    is_holiday BOOLEAN NOT NULL DEFAULT FALSE, -- 공휴일 여부 (TRUE: 공휴일, FALSE: 영업일)
    description VARCHAR(4000)                  -- 공휴일 또는 특이사항 설명
) COMMENT='주식시장 캘린더';

-- day_price 테이블 생성 (일봉 주가 데이터)
CREATE TABLE IF NOT EXISTS daily_price (
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    date DATE NOT NULL COMMENT '거래일',
    open DECIMAL(18, 2) NOT NULL COMMENT '시가',
    high DECIMAL(18, 2) NOT NULL COMMENT '고가',
    low DECIMAL(18, 2) NOT NULL COMMENT '저가',
    close DECIMAL(18, 2) NOT NULL COMMENT '종가',
    volume BIGINT NOT NULL COMMENT '거래량',
    trading_value BIGINT COMMENT '거래대금 (선택적)',
    change_rate DECIMAL(10, 4) COMMENT '대비 (직전일 대비 변화율, 백분율)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',

    -- 기본 키 정의: 종목 코드와 날짜의 조합으로 고유성을 보장하는 복합 기본 키
    PRIMARY KEY (stock_code, date)
)
COMMENT='일봉 주가 데이터'
PARTITION BY RANGE (YEAR(date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p2027 VALUES LESS THAN (2028),
    PARTITION p2028 VALUES LESS THAN (2029),
    PARTITION p2029 VALUES LESS THAN (2030),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- minute_price 테이블 생성 (분봉 주가 데이터)
CREATE TABLE IF NOT EXISTS minute_price (
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    datetime DATETIME NOT NULL COMMENT '거래시각각',
    open DECIMAL(18, 2) NOT NULL COMMENT '시가',
    high DECIMAL(18, 2) NOT NULL COMMENT '고가',
    low DECIMAL(18, 2) NOT NULL COMMENT '저가',
    close DECIMAL(18, 2) NOT NULL COMMENT '종가',
    volume BIGINT NOT NULL COMMENT '거래량',
    trading_value BIGINT COMMENT '거래대금 (선택적)',
    change_rate DECIMAL(10, 4) COMMENT '대비 (직전일 대비 변화율, 백분율)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',

    -- 기본 키 정의: 종목 코드와 시각(날짜+시각)의 조합으로 고유성을 보장하는 복합 기본 키
    PRIMARY KEY (stock_code, datetime)
)
COMMENT='분봉 주가 데이터'
PARTITION BY RANGE (YEAR(datetime)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p2027 VALUES LESS THAN (2028),
    PARTITION p2028 VALUES LESS THAN (2029),
    PARTITION p2029 VALUES LESS THAN (2030),
    PARTITION pmax VALUES LESS THAN MAXVALUE
);

-- 일별 퀀트 팩터 데이터를 저장하는 테이블
CREATE TABLE IF NOT EXISTS daily_factors (
    `date` DATE NOT NULL COMMENT '기준일자',
    `stock_code` VARCHAR(10) NOT NULL COMMENT '종목코드',

    -- 1. 수급 (Supply/Demand)
    `foreigner_net_buy` BIGINT COMMENT '외국인 순매수(수량) | MarketEye: 118',
    `institution_net_buy` BIGINT COMMENT '기관 순매수(수량) | MarketEye: 120',
    `program_net_buy` BIGINT COMMENT '프로그램 순매수(수량) | MarketEye: 116',
    `trading_intensity` FLOAT COMMENT '체결강도 | MarketEye: 24',

    -- 2. 위험/관심도 (Risk/Sentiment)
    `credit_ratio` FLOAT COMMENT '신용잔고율(%) | MarketEye: 126',
    `short_volume` BIGINT COMMENT '공매도 수량 | MarketEye: 127',
    `trading_value` BIGINT COMMENT '거래대금(원) | MarketEye: 11',

    -- 3. 가치평가 (Valuation)
    `per` FLOAT COMMENT '주가수익비율(배) | MarketEye: 67',
    `pbr` FLOAT COMMENT '주가순자산비율(배) | 계산값: 현재가(4)/BPS(89)',
    `psr` FLOAT COMMENT '주가매출비율(배) | 계산값: 현재가(4)/SPS(123)',
    `dividend_yield` FLOAT COMMENT '배당수익률(%) | MarketEye: 74',

    -- 4. 추세 및 상대강도 (Trend & Relative Strength)
    `relative_strength` FLOAT COMMENT '시장 대비 상대강도 점수 |계산값: 종목수익률 - 지수수익률',
    `beta_coefficient` FLOAT COMMENT '베타계수 | MarketEye: 150',
    `days_since_52w_high` INT COMMENT '52주 신고가 경신 후 경과일 | 계산값',
    `dist_from_ma20` FLOAT COMMENT '20일 이동평균선 이격도(%) | 계산값',

    -- 5. 변동성 (Volatility)
    `historical_volatility_20d` FLOAT COMMENT '20일 역사적 변동성 | 계산값',

    -- 6. 성장성 (Growth)
    `q_revenue_growth_rate` FLOAT COMMENT '분기 매출액 증가율(%) | MarketEye: 97',
    `q_op_income_growth_rate` FLOAT COMMENT '분기 영업이익 증가율(%) | MarketEye: 98',

    PRIMARY KEY (`date`, `stock_code`)
) COMMENT '일별 퀀트 팩터 데이터';

-- 추가 인덱스 정의 (PK에 포함된 인덱스 외에 필요시 추가)
-- PRIMARY KEY (stock_code, date)에 이미 (stock_code, date) 순서로 인덱스가 포함되므로,
-- 대부분의 조회는 이 PK 인덱스를 통해 효율적으로 처리됩니다.
-- 특정 컬럼 단독 조회에 대한 성능 개선이 필요하다면 아래 인덱스들을 고려할 수 있습니다.
-- CREATE INDEX idx_stock_price_date ON stock_price (date); -- 날짜 단독 조회 효율화
-- CREATE INDEX idx_stock_price_code ON stock_price (stock_code); -- 종목 코드 단독 조회 효율화

-- 외래 키(FK) 정의 (현재 이 두 테이블 간 직접적인 FK는 없음)
-- 다른 테이블과의 관계가 추가될 경우 여기에 ALTER TABLE ... ADD CONSTRAINT ... 구문을 추가합니다.