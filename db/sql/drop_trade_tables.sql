-- backtesting/db/sql/drop_backtest_schema.sql

-- 1. 모든 외래 키(FK) 제약 조건 비활성화 및 삭제
-- SET GLOBAL foreign_key_checks = 0; -- 전체 세션에 영향을 주므로 권장하지 않습니다.
SET FOREIGN_KEY_CHECKS = 0; -- 현재 세션에만 영향을 주어 안전하게 FK 비활성화

-- 외래 키 제약 조건 삭제 (명시적으로 ADD CONSTRAINT 했던 이름으로 삭제)
ALTER TABLE backtest_trade DROP CONSTRAINT IF EXISTS fk_backtest_trade_run_id;
ALTER TABLE backtest_trade DROP CONSTRAINT IF EXISTS fk_backtest_trade_entry_trade_id;
ALTER TABLE backtest_performance DROP CONSTRAINT IF EXISTS fk_backtest_performance_run_id;

-- 2. 테이블 삭제 (삭제 순서는 FK 관계에 상관 없이 이제 가능)
DROP TABLE IF EXISTS backtest_performance;
DROP TABLE IF EXISTS backtest_trade;
DROP TABLE IF EXISTS backtest_run;

SET FOREIGN_KEY_CHECKS = 1; -- 작업 완료 후 FK 제약 조건 다시 활성화