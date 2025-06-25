

-- trader_run 테이블 생성 (자동매매 실행 정보를 저장)
CREATE TABLE IF NOT EXISTS trader_run (
    run_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '자동매매 실행 고유 ID',
    start_date DATE NOT NULL COMMENT '자동매매 시작일',
    end_date DATE NOT NULL COMMENT '자동매매 종료일',
    initial_capital DECIMAL(18, 2) NOT NULL COMMENT '초기 투자 자본',
    final_capital DECIMAL(18, 2) COMMENT '최종 자본',
    total_profit_loss DECIMAL(18, 2) COMMENT '총 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률 (예: 0.15 = 15%)',
    max_drawdown DECIMAL(10, 4) COMMENT '최대 낙폭 (예: 0.10 = 10%)',
    strategy_daily VARCHAR(255) COMMENT '일봉 전략 이름 (NULL 허용)',
    strategy_minute VARCHAR(255) COMMENT '분봉 전략 이름 (NULL 허용)',
    params_json_daily JSON COMMENT '일봉 전략 파라미터 JSON',
    params_json_minute JSON COMMENT '분봉 전략 파라미터 JSON',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시각'
) COMMENT='자동매매 실행 정보';

-- trader_run 인덱스 (조회 효율을 위해)
CREATE INDEX idx_trader_run_date ON trader_run (start_date, end_date);


-- trader_trade 테이블 생성 (개별 거래 내역을 저장)
CREATE TABLE IF NOT EXISTS trader_trade (
    trade_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '거래 고유 ID',
    run_id BIGINT NOT NULL COMMENT '자동매매 실행 ID (FK)',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    trade_type ENUM('BUY', 'SELL') NOT NULL COMMENT '거래 유형 (매수/매도)',
    trade_price DECIMAL(18, 2) NOT NULL COMMENT '거래 가격',
    trade_quantity INT NOT NULL COMMENT '거래 수량',
    trade_amount DECIMAL(18, 2) NOT NULL COMMENT '거래 금액 (가격 * 수량)',
    trade_datetime DATETIME NOT NULL COMMENT '거래 시각',
    commission DECIMAL(18, 2) COMMENT '거래 수수료',
    tax DECIMAL(18, 2) COMMENT '거래세',
    realized_profit_loss DECIMAL(18, 2) COMMENT '실현 손익',
    entry_trade_id BIGINT COMMENT '매수 시 연결된 trade_id (매도 거래인 경우)',
    UNIQUE KEY uk_trade_run_stock_datetime (run_id, stock_code, trade_datetime) -- 특정 자동매매 실행 내에서 종목/시각별 유니크
) COMMENT='자동매매 개별 거래 내역';

-- trader_trade 인덱스 (조회 효율을 위해)
CREATE INDEX idx_trader_trade_run_id ON trader_trade (run_id);
CREATE INDEX idx_trader_trade_stock_code ON trader_trade (stock_code);
CREATE INDEX idx_trader_trade_datetime ON trader_trade (trade_datetime);


-- trader_performance 테이블 생성 (일별 또는 특정 기간별 성능 지표를 저장)
CREATE TABLE IF NOT EXISTS trader_performance (
    performance_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '성능 지표 고유 ID',
    run_id BIGINT NOT NULL COMMENT '자동매매 실행 ID (FK)',
    date DATE NOT NULL COMMENT '해당 성능 기록 날짜',
    end_capital DECIMAL(18, 2) NOT NULL COMMENT '해당 날짜 기준 최종 자본',
    daily_return DECIMAL(10, 4) COMMENT '일일 수익률',
    daily_profit_loss DECIMAL(18, 2) COMMENT '일일 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    drawdown DECIMAL(10, 4) COMMENT '해당 날짜 기준 낙폭',
    UNIQUE KEY uk_performance_run_date (run_id, date) -- 특정 자동매매 실행 내에서 날짜별 유니크
) COMMENT='자동매매 일별/기간별 성능 지표';

-- trader_performance 인덱스 (조회 효율을 위해)
CREATE INDEX idx_trader_performance_run_id ON trader_performance (run_id);
CREATE INDEX idx_trader_performance_date ON trader_performance (date);


-- 모든 외래 키(FK) 정의
-- FK 정의는 항상 참조하는 테이블과 컬럼이 먼저 존재해야 합니다.
ALTER TABLE trader_trade
    ADD CONSTRAINT fk_trader_trade_run_id
    FOREIGN KEY (run_id) REFERENCES trader_run (run_id)
    ON DELETE CASCADE; -- 자동매매 실행 정보 삭제 시 관련 거래 내역도 삭제

ALTER TABLE trader_trade
    ADD CONSTRAINT fk_trader_trade_entry_trade_id
    FOREIGN KEY (entry_trade_id) REFERENCES trader_trade (trade_id)
    ON DELETE SET NULL; -- 매수 거래 삭제 시 연결 끊기 (NULL로 설정)

ALTER TABLE trader_performance
    ADD CONSTRAINT fk_trader_performance_run_id
    FOREIGN KEY (run_id) REFERENCES trader_run (run_id)
    ON DELETE CASCADE; -- 자동매매 실행 정보 삭제 시 관련 성능 지표도 삭제