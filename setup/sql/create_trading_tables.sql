
CREATE TABLE IF NOT EXISTS trading_run (
    model_id BIGINT NOT NULL PRIMARY KEY COMMENT '판단 기준이 된 HMM 모델 ID (FK)',
    start_date DATE NOT NULL COMMENT '해당 모델로 자동매매를 시작한 최초 날짜',
    end_date DATE NOT NULL COMMENT '해당 모델로 자동매매를 실행한 최종 날짜',
    initial_capital DECIMAL(18, 2) NOT NULL COMMENT '최초 시작 자본',
    final_capital DECIMAL(18, 2) COMMENT '최종 자본',
    total_profit_loss DECIMAL(18, 2) COMMENT '누적 총 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    max_drawdown DECIMAL(10, 4) COMMENT '기간 내 최대 낙폭',
    strategy_daily VARCHAR(255) COMMENT '사용한 일봉 전략 이름',
    params_json_daily JSON COMMENT '사용한 일봉 전략 파라미터 JSON',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '마지막 업데이트 시각',
    FOREIGN KEY (model_id) REFERENCES hmm_models (model_id) ON DELETE CASCADE
) COMMENT='자동매매 모델별 누적 실행 정보';


-- trading_trade 테이블 생성 (strategy_name 컬럼 추가)
CREATE TABLE IF NOT EXISTS trading_trade (
    trade_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    -- run_id 대신 model_id 와 trade_date로 식별
    model_id BIGINT NOT NULL COMMENT '판단 기준이 된 HMM 모델 ID',
    trade_date DATE NOT NULL COMMENT '거래일',
    strategy_name VARCHAR(255) NOT NULL COMMENT '거래를 실행한 전략 이름', -- 신규 컬럼
    stock_code VARCHAR(10) NOT NULL,
    trade_type ENUM('BUY', 'SELL') NOT NULL,
    trade_price DECIMAL(18, 2) NOT NULL,
    trade_quantity INT NOT NULL,
    trade_datetime DATETIME NOT NULL,
    commission DECIMAL(18, 2),
    tax DECIMAL(18, 2),
    realized_profit_loss DECIMAL(18, 2),
    -- 외래 키 제약조건 추가 가능
    FOREIGN KEY (model_id) REFERENCES hmm_models(model_id)
) COMMENT='자동매매 개별 거래 내역';

-- trading_performance 테이블 생성 (자동매매의 일별 자산 스냅샷을 저장)
CREATE TABLE IF NOT EXISTS trading_performance (
    performance_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '성능 기록 고유 ID',
    model_id BIGINT NOT NULL COMMENT '판단 기준이 된 HMM 모델 ID (FK)',
    date DATE NOT NULL COMMENT '기록 날짜',
    end_capital DECIMAL(18, 2) NOT NULL COMMENT '해당 날짜 기준 최종 자본',
    daily_return DECIMAL(10, 4) COMMENT '일일 수익률',
    daily_profit_loss DECIMAL(18, 2) COMMENT '일일 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    drawdown DECIMAL(10, 4) COMMENT '해당 날짜 기준 낙폭',
    UNIQUE KEY uk_performance_model_date (model_id, date), -- 특정 모델의 하루치 기록은 유일해야 함
    FOREIGN KEY (model_id) REFERENCES hmm_models (model_id) ON DELETE CASCADE
) COMMENT='자동매매 일별 성과 스냅샷';

