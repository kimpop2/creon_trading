-- backtesting/db/sql/create_backtesting_schema.sql

-- backtest_run 테이블 생성 (백테스트 실행 요약 정보)
CREATE TABLE IF NOT EXISTS backtest_run (
    run_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '백테스트 실행 ID',
    start_date DATE NOT NULL COMMENT '백테스트 시작일',
    end_date DATE NOT NULL COMMENT '백테스트 종료일',
    initial_capital DECIMAL(18, 2) NOT NULL COMMENT '초기 자본금',
    final_capital DECIMAL(18, 2) COMMENT '최종 자본금',
    total_profit_loss DECIMAL(18, 2) COMMENT '총 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    max_drawdown DECIMAL(10, 4) COMMENT '최대 낙폭',
    strategy_daily VARCHAR(100) COMMENT '일봉 전략명',
    strategy_minute VARCHAR(100) COMMENT '분봉 전략명',
    params_json_daily JSON COMMENT '일봉 전략 파라미터 (JSON)',
    params_json_minute JSON COMMENT '분봉 전략 파라미터 (JSON)',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '생성 일시'
) COMMENT='백테스트 실행 정보';

-- backtest_trade 테이블 생성 (백테스트 개별 거래 내역)
CREATE TABLE IF NOT EXISTS backtest_trade (
    trade_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '거래 ID',
    run_id INT NOT NULL COMMENT '백테스트 실행 ID (FK)',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    trade_type ENUM('BUY', 'SELL') NOT NULL COMMENT '거래 유형 (매수/매도)',
    trade_price DECIMAL(18, 2) NOT NULL COMMENT '거래 가격',
    trade_quantity INT NOT NULL COMMENT '거래 수량',
    trade_amount DECIMAL(18, 2) NOT NULL COMMENT '거래 금액',
    trade_datetime DATETIME NOT NULL COMMENT '거래 시각',
    commission DECIMAL(18, 2) DEFAULT 0 COMMENT '거래 수수료',
    tax DECIMAL(18, 2) DEFAULT 0 COMMENT '거래 세금',
    realized_profit_loss DECIMAL(18, 2) DEFAULT 0 COMMENT '실현 손익',
    entry_trade_id INT COMMENT '매수 거래 ID (청산 거래의 경우)',

    -- 복합 Unique Key: 특정 백테스트 실행 내에서 동일 종목의 동일 시각 거래는 유일해야 함
    UNIQUE KEY (run_id, stock_code, trade_datetime), 

    -- 외래 키 제약 조건
    CONSTRAINT fk_backtest_trade_run_id FOREIGN KEY (run_id) REFERENCES backtest_run (run_id) ON DELETE CASCADE,
    CONSTRAINT fk_backtest_trade_entry_trade_id FOREIGN KEY (entry_trade_id) REFERENCES backtest_trade (trade_id) ON DELETE SET NULL
) COMMENT='백테스트 개별 거래 내역';

-- backtest_performance 테이블 생성 (백테스트 일별/기간별 성능 지표)
CREATE TABLE IF NOT EXISTS backtest_performance (
    performance_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '성능 지표 ID',
    run_id INT NOT NULL COMMENT '백테스트 실행 ID (FK)',
    date DATE NOT NULL COMMENT '날짜',
    end_capital DECIMAL(18, 2) NOT NULL COMMENT '최종 자본금',
    daily_return DECIMAL(10, 4) COMMENT '일일 수익률',
    daily_profit_loss DECIMAL(18, 2) COMMENT '일일 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    drawdown DECIMAL(10, 4) COMMENT '낙폭',

    -- 복합 Unique Key: 특정 백테스트 실행 내에서 동일 날짜의 성능 기록은 유일해야 함
    UNIQUE KEY (run_id, date),

    -- 외래 키 제약 조건
    CONSTRAINT fk_backtest_performance_run_id FOREIGN KEY (run_id) REFERENCES backtest_run (run_id) ON DELETE CASCADE
) COMMENT='백테스트 일별/기간별 성능 지표';