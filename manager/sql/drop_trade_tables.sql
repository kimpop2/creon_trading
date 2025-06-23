

-- 1. 모든 외래 키(FK) 제약 조건 비활성화 및 삭제
-- SET GLOBAL foreign_key_checks = 0; -- 전체 세션에 영향을 주므로 권장하지 않습니다.
SET FOREIGN_KEY_CHECKS = 0; -- 현재 세션에만 영향을 주어 안전하게 FK 비활성화


-- 2. 테이블 삭제 (삭제 순서는 FK 관계에 상관 없이 이제 가능)
DROP TABLE IF EXISTS current_positions;
DROP TABLE IF EXISTS daily_portfolio_snapshot;
DROP TABLE IF EXISTS trade_log;
DROP TABLE IF EXISTS daily_signals;

SET FOREIGN_KEY_CHECKS = 1; -- 작업 완료 후 FK 제약 조건 다시 활성화