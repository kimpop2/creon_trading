-- backtesting/db/schema.sql
-- 데이터베이스 생성 (필요시 주석 해제 후 사용)
CREATE DATABASE IF NOT EXISTS backtest_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE backtest_db;

SET FOREIGN_KEY_CHECKS = 0; -- 외래 키 검사 일시 비활성화

DROP TABLE IF EXISTS minute_stock_data;
DROP TABLE IF EXISTS daily_stock_data;
DROP TABLE IF EXISTS stock_info;

SET FOREIGN_KEY_CHECKS = 1; -- 외래 키 검사 다시 활성화

-- stock_info 테이블: 종목 기본 정보 및 최신 재무 데이터 통합
CREATE TABLE IF NOT EXISTS stock_info (
    stock_code VARCHAR(10) PRIMARY KEY, -- 종목 코드 (예: A005930)
    stock_name VARCHAR(100) NOT NULL,   -- 종목명
    market_type VARCHAR(20),            -- 시장 구분 (예: KOSPI, KOSDAQ)
    sector VARCHAR(100),                -- 섹터/업종 (현재 Creon API에서 직접 제공되지 않아 NULL 가능)
    per DECIMAL(10, 2),                 -- 주가수익비율 (최신)
    pbr DECIMAL(10, 2),                 -- 주가순자산비율 (최신)
    eps DECIMAL(15, 2),                 -- 주당순이익 (최신)
    roe DECIMAL(10, 2),                 -- 자기자본이익률 (최신)
    debt_ratio DECIMAL(10, 2),          -- 부채비율 (최신)
    sales BIGINT,                       -- 매출액 (최신, 백만 원 단위)
    operating_profit BIGINT,            -- 영업이익 (최신, 백만 원 단위)
    net_profit BIGINT,                  -- 당기순이익 (최신, 백만 원 단위)
    recent_financial_date DATE,         -- 최신 재무 데이터의 기준 일자 (YYYY-MM-DD)
    reg_date DATETIME DEFAULT CURRENT_TIMESTAMP, -- 등록일시
    upd_date DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- 최종 업데이트 일시
);

-- daily_stock_data 테이블: 일별 주식 데이터 (OHLCV + α)
CREATE TABLE IF NOT EXISTS daily_stock_data (
    stock_code VARCHAR(10) NOT NULL,    -- 종목 코드
    date DATE NOT NULL,                 -- 날짜 (YYYY-MM-DD)
    open_price INT NOT NULL,            -- 시가
    high_price INT NOT NULL,            -- 고가
    low_price INT NOT NULL,             -- 저가
    close_price INT NOT NULL,           -- 종가
    volume BIGINT NOT NULL,             -- 거래량
    change_rate DECIMAL(10, 2),         -- 전일 대비 등락률 (%)
    trading_value BIGINT,               -- 거래대금 (옵션, Creon API에서 제공하지 않을 경우 NULL)
    PRIMARY KEY (stock_code, date),     -- 종목코드와 날짜 조합을 기본 키로
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code)
        ON DELETE CASCADE ON UPDATE CASCADE
);

-- minute_stock_data 테이블: 분별 주식 데이터 (OHLCV)
CREATE TABLE IF NOT EXISTS minute_stock_data (
    stock_code VARCHAR(10) NOT NULL,    -- 종목 코드
    datetime DATETIME NOT NULL,         -- 날짜 및 시간 (YYYY-MM-DD HH:MM:SS)
    open_price INT NOT NULL,            -- 시가
    high_price INT NOT NULL,            -- 고가
    low_price INT NOT NULL,             -- 저가
    close_price INT NOT NULL,           -- 종가
    volume BIGINT NOT NULL,             -- 거래량
    PRIMARY KEY (stock_code, datetime), -- 종목코드와 시간 조합을 기본 키로
    FOREIGN KEY (stock_code) REFERENCES stock_info(stock_code)
        ON DELETE CASCADE ON UPDATE CASCADE
);