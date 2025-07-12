-- drop_feed_tables.sql

-- 외래 키 제약 조건 일시 비활성화
-- 이 설정을 통해 테이블 간의 참조 순서에 관계없이 테이블을 삭제할 수 있습니다.
SET FOREIGN_KEY_CHECKS = 0;

-- Feed 관련 테이블들을 삭제합니다.
-- 자식 테이블을 먼저 삭제하고 부모 테이블을 나중에 삭제하는 것이 좋습니다.
-- DROP TABLE IF EXISTS daily_universe;
DROP TABLE IF EXISTS thematic_stocks;
DROP TABLE IF EXISTS news_summaries;
DROP TABLE IF EXISTS investor_trends;
DROP TABLE IF EXISTS news_raw;
DROP TABLE IF EXISTS market_volume;
DROP TABLE IF EXISTS ohlcv_minute;
-- DROP TABLE IF EXISTS stock_info; -- 가장 마지막에 삭제 (다른 테이블의 참조 대상)

-- 외래 키 제약 조건 다시 활성화
SET FOREIGN_KEY_CHECKS = 1;
