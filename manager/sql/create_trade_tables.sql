

-- daily_signals 테이블 생성 (일봉 전략 매매 신호 정보)
CREATE TABLE IF NOT EXISTS daily_signals (
    signal_date DATE NOT NULL COMMENT '신호 생성일',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    strategy_name VARCHAR(50) NOT NULL COMMENT '전략명',
    signal_type VARCHAR(10) NOT NULL COMMENT '신호 유형 (BUY, SELL, HOLD)',
    target_price DECIMAL(15, 2) COMMENT '목표 가격 (매수/매도 시)',
    signal_strength DECIMAL(10, 4) COMMENT '신호 강도 (선택적)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, stock_code, strategy_name)
) COMMENT='일봉 전략 매매 신호';

-- trade_log 테이블 생성 (실시간 자동매매 주문 및 체결 로그)
CREATE TABLE IF NOT EXISTS trade_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '로그 ID',
    order_time DATETIME NOT NULL COMMENT '주문 시각',
    stock_code VARCHAR(10) NOT NULL COMMENT '종목 코드',
    order_type VARCHAR(10) NOT NULL COMMENT '주문 유형 (BUY, SELL)',
    order_price DECIMAL(15, 2) COMMENT '주문 가격',
    order_quantity INT NOT NULL COMMENT '주문 수량',
    executed_price DECIMAL(15, 2) COMMENT '체결 가격',
    executed_quantity INT COMMENT '체결 수량',
    commission DECIMAL(15, 2) COMMENT '수수료',
    tax DECIMAL(15, 2) COMMENT '세금',
    net_amount DECIMAL(15, 2) COMMENT '실제 금액 (수수료, 세금 포함)',
    order_status VARCHAR(20) NOT NULL COMMENT '주문 상태 (PENDING, FILLED, PARTIAL_FILLED, CANCELED, REJECTED)',
    creon_order_id VARCHAR(50) COMMENT '크레온 주문번호 (실제 주문 식별자)',
    message TEXT COMMENT '추가 메시지',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) COMMENT='실시간 자동매매 주문 및 체결 로그';

-- daily_portfolio_snapshot 테이블 생성 (일별 포트폴리오 스냅샷)
CREATE TABLE IF NOT EXISTS daily_portfolio_snapshot (
    snapshot_date DATE PRIMARY KEY COMMENT '스냅샷 날짜',
    cash DECIMAL(20, 2) NOT NULL COMMENT '현금 잔고',
    total_asset_value DECIMAL(20, 2) NOT NULL COMMENT '총 자산 가치 (현금 + 주식 평가액)',
    total_stock_value DECIMAL(20, 2) NOT NULL COMMENT '총 주식 평가액',
    profit_loss_rate DECIMAL(10, 4) COMMENT '일일 수익률',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) COMMENT='일별 포트폴리오 스냅샷';

-- current_positions 테이블 생성 (현재 보유 종목 포지션 (실시간 동기화 필요))
CREATE TABLE IF NOT EXISTS current_positions (
    stock_code VARCHAR(10) PRIMARY KEY COMMENT '종목 코드',
    current_size INT NOT NULL COMMENT '현재 보유 수량',
    average_price DECIMAL(15, 2) NOT NULL COMMENT '평균 매수 단가',
    entry_date DATE COMMENT '최초 진입일',
    highest_price_since_entry DECIMAL(15, 2) COMMENT '진입 이후 최고가 (손절 기준)',
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '최종 업데이트 시각'
) COMMENT='현재 보유 종목 포지션 (실시간 동기화 필요)';