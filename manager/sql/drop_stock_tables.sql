

-- 1. 모든 외래 키(FK) 제약 조건 비활성화
SET FOREIGN_KEY_CHECKS = 0; 

-- 현재 stock_info와 stock_price 테이블 간에는 직접적인 외래 키가 없습니다.
-- 만약 다른 테이블에서 이 테이블들을 참조한다면, 해당 FK를 먼저 DROP CONSTRAINT 해야 합니다.
-- 예를 들어: ALTER TABLE other_table DROP CONSTRAINT IF EXISTS fk_other_table_stock_code;

-- 2. 테이블 삭제
-- 참조 관계가 없는 테이블들이므로 삭제 순서는 중요하지 않으나, 일반적으로 알파벳 순서 또는 논리적 순서를 따릅니다.
DROP TABLE IF EXISTS market_calendar;
DROP TABLE IF EXISTS minute_price;
DROP TABLE IF EXISTS daily_price;
DROP TABLE IF EXISTS stock_info;

-- 작업 완료 후 FK 제약 조건 다시 활성화
SET FOREIGN_KEY_CHECKS = 1;
