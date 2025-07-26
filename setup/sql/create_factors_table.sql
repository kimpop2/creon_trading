-- setup/sql/create_factors_table.sql

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