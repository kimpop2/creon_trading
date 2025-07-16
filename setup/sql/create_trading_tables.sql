-- create_trading_tables.sql

-- trading_log 테이블 생성 (자동매매 주문 및 체결 내역)
CREATE TABLE IF NOT EXISTS trading_log (
    log_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '로그 고유 ID',
    order_id VARCHAR(50) COMMENT '증권사 주문번호 (체결되지 않은 주문에도 존재)',
    original_order_id VARCHAR(50) COMMENT '원주문번호 (정정/취소 시 원주문 식별)',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    stock_name VARCHAR(100) NOT NULL COMMENT '종목명',
    trading_date DATE NOT NULL COMMENT '매매 일자',
    trading_time TIME NOT NULL COMMENT '매매 시각',
    order_type ENUM('buy', 'sell') NOT NULL COMMENT '주문 유형 (매수/매도)',
    order_price DECIMAL(18, 2) COMMENT '주문 가격 (지정가/시장가)',
    order_quantity INT NOT NULL COMMENT '주문 수량',
    filled_price DECIMAL(18, 2) COMMENT '체결 가격 (부분/전체 체결 시)',
    filled_quantity INT COMMENT '체결 수량 (부분/전체 체결 시)',
    unfilled_quantity INT COMMENT '미체결 수량',
    order_status ENUM('접수', '체결', '부분체결', '확인', '거부', '정정', '취소') NOT NULL COMMENT '주문 상태',
    commission DECIMAL(18, 2) COMMENT '거래 수수료',
    tax DECIMAL(18, 2) COMMENT '거래세',
    net_amount DECIMAL(18, 2) COMMENT '순 매매 금액 (체결가 * 수량 - 수수료 - 세금)',
    credit_type VARCHAR(20) COMMENT '신용 구분 (예: 신용, 현금)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각'
) COMMENT='자동매매 주문 및 체결 로그';

-- trading_log 인덱스
CREATE INDEX idx_trading_log_stock_code ON trading_log (stock_code);
CREATE INDEX idx_trading_log_trading_date ON trading_log (trading_date);
CREATE INDEX idx_trading_log_order_id ON trading_log (order_id);
CREATE INDEX idx_trading_log_original_order_id ON trading_log (original_order_id);


-- daily_portfolio 테이블 생성 (일별 포트폴리오 스냅샷)
CREATE TABLE IF NOT EXISTS daily_portfolio (
    portfolio_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '포트폴리오 스냅샷 고유 ID',
    record_date DATE NOT NULL UNIQUE COMMENT '기록 일자 (YYYY-MM-DD)',
    total_capital DECIMAL(18, 2) NOT NULL COMMENT '총 자본금 (현금 + 평가액)',
    cash_balance DECIMAL(18, 2) NOT NULL COMMENT '현금 잔고',
    total_asset_value DECIMAL(18, 2) NOT NULL COMMENT '총 평가 자산 (보유 종목 평가액 합계)',
    daily_profit_loss DECIMAL(18, 2) COMMENT '일일 손익 (평가손익 + 실현손익)',
    daily_return_rate DECIMAL(10, 4) COMMENT '일일 수익률',
    cumulative_profit_loss DECIMAL(18, 2) COMMENT '누적 손익',
    cumulative_return_rate DECIMAL(10, 4) COMMENT '누적 수익률',
    max_drawdown DECIMAL(10, 4) COMMENT '최대 낙폭 (시스템 운영 시작점 대비)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '기록 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각'
) COMMENT='자동매매 일별 포트폴리오 스냅샷';

-- daily_portfolio 인덱스
CREATE INDEX idx_daily_portfolio_record_date ON daily_portfolio (record_date);


-- current_positions 테이블 생성 (현재 보유 종목 현황)
CREATE TABLE IF NOT EXISTS current_positions (
    stock_code VARCHAR(10) PRIMARY KEY COMMENT '종목 코드',
    stock_name VARCHAR(100) NOT NULL COMMENT '종목명',
    quantity INT NOT NULL COMMENT '보유 수량',
    sell_avail_qty INT NOT NULL COMMENT '매도가능수량',
    avg_price DECIMAL(18, 2) NOT NULL COMMENT '평균 매입 단가',
    eval_profit_loss DECIMAL(18, 2) COMMENT '평가 손익',
    eval_return_rate DECIMAL(10, 4) COMMENT '평가 수익률',
    entry_date DATE COMMENT '최초 매수 일자',
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '마지막 업데이트 시각'
) COMMENT='현재 보유 종목 현황';

-- current_positions 인덱스
CREATE INDEX idx_current_positions_stock_name ON current_positions (stock_name);


-- daily_signals 테이블 생성 (일일 매매 신호 관리)
CREATE TABLE IF NOT EXISTS daily_signals (
    signal_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '신호 고유 ID',
    signal_date DATE NOT NULL COMMENT '신호 발생 일자',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    stock_name VARCHAR(100) NOT NULL COMMENT '종목명',
    signal_type ENUM('BUY', 'SELL', 'HOLD') NOT NULL COMMENT '신호 유형',
    strategy_name VARCHAR(100) COMMENT '신호를 발생시킨 전략 이름',
    target_price DECIMAL(18, 2) COMMENT '신호가 제시하는 목표 가격 (예: 매수 목표가, 손절 가격)',
    target_quantity INT COMMENT '신호가 제시하는 목표 수량',
    is_executed BOOLEAN DEFAULT FALSE COMMENT '신호 실행 여부',
    executed_order_id VARCHAR(50) COMMENT '실행된 경우 주문 ID',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '신호 생성 시각',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각',
    UNIQUE KEY uk_daily_signal_date_stock_code_type (signal_date, stock_code, signal_type) -- 특정 날짜, 종목, 신호 타입별 유니크
) COMMENT='일일 매매 신호';

-- daily_signals 인덱스
CREATE INDEX idx_daily_signals_date ON daily_signals (signal_date);
CREATE INDEX idx_daily_signals_stock_code ON daily_signals (stock_code);
CREATE INDEX idx_daily_signals_type ON daily_signals (signal_type);