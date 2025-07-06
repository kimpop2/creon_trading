-- 외래 키 제약 조건 일시 비활성화
-- 이 설정을 통해 테이블 간의 참조 순서에 관계없이 테이블을 삭제할 수 있습니다.
SET FOREIGN_KEY_CHECKS = 0;

-- 자동매매 관련 테이블들을 삭제합니다.
-- DROP TABLE IF EXISTS는 테이블이 존재하지 않아도 오류를 발생시키지 않습니다.
-- 자식 테이블을 먼저 삭제하고 부모 테이블을 나중에 삭제하는 것이 좋지만,
-- FOREIGN_KEY_CHECKS=0 이므로 순서는 덜 중요합니다.
-- sql/drop_trading_tables.sql
DROP TABLE IF EXISTS trading_log;
DROP TABLE IF EXISTS daily_portfolio;
DROP TABLE IF EXISTS current_positions;
DROP TABLE IF EXISTS daily_signals;

-- 외래 키 제약 조건 다시 활성화
SET FOREIGN_KEY_CHECKS = 1;