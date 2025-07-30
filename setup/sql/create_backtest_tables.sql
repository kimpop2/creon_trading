

-- backtest_run 테이블 생성 (백테스트 실행 정보를 저장)
CREATE TABLE IF NOT EXISTS backtest_run (
    run_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '백테스트 실행 고유 ID',
    start_date DATE NOT NULL COMMENT '백테스트 시작일',
    end_date DATE NOT NULL COMMENT '백테스트 종료일',
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
) COMMENT='백테스트 실행 정보';

-- backtest_run 인덱스 (조회 효율을 위해)
CREATE INDEX idx_backtest_run_date ON backtest_run (start_date, end_date);


-- backtest_trade 테이블 생성 (개별 거래 내역을 저장)
CREATE TABLE IF NOT EXISTS backtest_trade (
    trade_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '거래 고유 ID',
    run_id BIGINT NOT NULL COMMENT '백테스트 실행 ID (FK)',
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
    UNIQUE KEY uk_trade_run_stock_datetime (run_id, stock_code, trade_datetime) -- 특정 백테스트 실행 내에서 종목/시각별 유니크
) COMMENT='백테스트 개별 거래 내역';

-- backtest_trade 인덱스 (조회 효율을 위해)
CREATE INDEX idx_backtest_trade_run_id ON backtest_trade (run_id);
CREATE INDEX idx_backtest_trade_stock_code ON backtest_trade (stock_code);
CREATE INDEX idx_backtest_trade_datetime ON backtest_trade (trade_datetime);


-- backtest_performance 테이블 생성 (일별 또는 특정 기간별 성능 지표를 저장)
CREATE TABLE IF NOT EXISTS backtest_performance (
    performance_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '성능 지표 고유 ID',
    run_id BIGINT NOT NULL COMMENT '백테스트 실행 ID (FK)',
    date DATE NOT NULL COMMENT '해당 성능 기록 날짜',
    end_capital DECIMAL(18, 2) NOT NULL COMMENT '해당 날짜 기준 최종 자본',
    daily_return DECIMAL(10, 4) COMMENT '일일 수익률',
    daily_profit_loss DECIMAL(18, 2) COMMENT '일일 손익',
    cumulative_return DECIMAL(10, 4) COMMENT '누적 수익률',
    drawdown DECIMAL(10, 4) COMMENT '해당 날짜 기준 낙폭',
    UNIQUE KEY uk_performance_run_date (run_id, date) -- 특정 백테스트 실행 내에서 날짜별 유니크
) COMMENT='백테스트 일별/기간별 성능 지표';

-- backtest_performance 인덱스 (조회 효율을 위해)
CREATE INDEX idx_backtest_performance_run_id ON backtest_performance (run_id);
CREATE INDEX idx_backtest_performance_date ON backtest_performance (date);


-- HMM 모델 정보를 저장하는 테이블 
CREATE TABLE IF NOT EXISTS hmm_models (
    model_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '모델의 고유 식별자',
    model_name VARCHAR(255) NOT NULL UNIQUE COMMENT '사용자가 식별하기 쉬운 모델 이름',
    n_states INT NOT NULL COMMENT '모델의 숨겨진 상태(regime) 개수',
    observation_vars JSON NOT NULL COMMENT '학습에 사용된 관찰 변수 목록 (JSON 배열)',
    model_params_json JSON NOT NULL COMMENT '모델의 핵심 파라미터(전이행렬, 평균, 공분산 등)를 JSON 형태로 저장',
    training_start_date DATE NOT NULL COMMENT '모델 학습에 사용된 데이터의 시작일',
    training_end_date DATE NOT NULL COMMENT '모델 학습에 사용된 데이터의 종료일',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '모델 생성 시각'
) COMMENT='HMM(은닉 마코프 모델)의 파라미터 및 메타데이터 저장 테이블';
CREATE INDEX IF NOT EXISTS idx_hmm_models_model_name ON hmm_models (model_name);

-- 전략의 장세별 성과 프로파일을 저장하는 테이블
CREATE TABLE IF NOT EXISTS strategy_profiles (
    profile_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '프로파일의 고유 식별자',
    strategy_name VARCHAR(255) NOT NULL COMMENT '프로파일링 대상 전략의 이름(예: ''SMADaily'')',
    model_id BIGINT NOT NULL COMMENT '장세 구분에 사용된 HMM 모델의 ID (hmm_models 테이블 참조)',
    regime_id INT NOT NULL COMMENT '측정된 장세(regime)의 ID (예: 0, 1, 2, 3)',
    sharpe_ratio DOUBLE COMMENT '해당 장세에서의 샤프 지수',
    mdd DOUBLE COMMENT '해당 장세에서의 최대 낙폭 (Max Drawdown)',
    total_return DOUBLE COMMENT '해당 장세에서의 누적 수익률',
    win_rate DOUBLE COMMENT '해당 장세에서의 승률',
    num_trades INT COMMENT '해당 장세에서의 총 거래 횟수',
    profiling_start_date DATE NOT NULL COMMENT '프로파일링에 사용된 데이터의 시작일',
    profiling_end_date DATE NOT NULL COMMENT '프로파일링에 사용된 데이터의 종료일',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '프로파일 업데이트 시각',
    CONSTRAINT fk_profiles_model FOREIGN KEY (model_id) REFERENCES hmm_models (model_id),
    UNIQUE KEY uk_strategy_model_regime (strategy_name, model_id, regime_id)
) COMMENT='전략의 장세별 성과 프로파일 저장 테이블';

CREATE INDEX IF NOT EXISTS idx_strategy_profiles_strategy_model ON strategy_profiles (strategy_name, model_id);


-- 모든 외래 키(FK) 정의
-- FK 정의는 항상 참조하는 테이블과 컬럼이 먼저 존재해야 합니다.
ALTER TABLE backtest_trade
    ADD CONSTRAINT fk_backtest_trade_run_id
    FOREIGN KEY (run_id) REFERENCES backtest_run (run_id)
    ON DELETE CASCADE; -- 백테스트 실행 정보 삭제 시 관련 거래 내역도 삭제

ALTER TABLE backtest_trade
    ADD CONSTRAINT fk_backtest_trade_entry_trade_id
    FOREIGN KEY (entry_trade_id) REFERENCES backtest_trade (trade_id)
    ON DELETE SET NULL; -- 매수 거래 삭제 시 연결 끊기 (NULL로 설정)

ALTER TABLE backtest_performance
    ADD CONSTRAINT fk_backtest_performance_run_id
    FOREIGN KEY (run_id) REFERENCES backtest_run (run_id)
    ON DELETE CASCADE; -- 백테스트 실행 정보 삭제 시 관련 성능 지표도 삭제