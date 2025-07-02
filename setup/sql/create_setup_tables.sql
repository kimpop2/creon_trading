
-- daily_theme 테이블 생성 (일별 테마 테이블블)
CREATE TABLE IF NOT EXISTS `daily_theme` (
	`date` DATE NOT NULL,
	`market` VARCHAR(8) NOT NULL ,
	`stock_code` VARCHAR(7) NOT NULL ,
	`stock_name` VARCHAR(25) NOT NULL ,
	`rate` DECIMAL(5,2) NULL DEFAULT '0.00',
	`amount` INT(11) NULL DEFAULT '0',
	`reason` VARCHAR(250) NOT NULL ,
	`reason_nouns` JSON NOT NULL ,
	`theme` VARCHAR(250) NULL DEFAULT NULL ,
	PRIMARY KEY (`date`, `market`, `stock_code`) USING BTREE
);

-- theme_class 테이블 생성 (테마 분류 테이블)
CREATE TABLE IF NOT EXISTS `theme_class` (
	`theme_id` INT(11) NOT NULL AUTO_INCREMENT,
	`theme` VARCHAR(30) NOT NULL ,
	`theme_class` VARCHAR(30) NULL ,
	`theme_synonyms` JSON NULL ,
	`theme_hit` INT(11) NULL DEFAULT '0',
	`theme_score` INT(11) NULL DEFAULT '0',
	`momentum_score` DECIMAL(10,4) NULL DEFAULT '0.0000',
	`theme_desc` VARCHAR(1000) NULL DEFAULT NULL ,
	PRIMARY KEY (`theme_id`) USING BTREE,
	-- UNIQUE INDEX `ak_theme_name_class` (`theme`, `theme_class`) USING BTREE,
	INDEX `idx_theme` (`theme`) USING BTREE
);


-- theme_stock 테이블 생성 (테마별 종목 테이블블)
CREATE TABLE IF NOT EXISTS `theme_stock` (
	`theme_id` INT(11) NOT NULL,
	`stock_code` VARCHAR(7) NOT NULL ,
	`stock_score` INT(11) NULL DEFAULT '0',
	PRIMARY KEY (`theme_id`, `stock_code`) USING BTREE,
	INDEX `idx_ts_theme` (`theme_id`) USING BTREE,
	INDEX `idx_ts_stock_code` (`stock_code`) USING BTREE
);


-- word_dic 테이블 생성 (단어 사전)
CREATE TABLE IF NOT EXISTS `word_dic` (
	`word` VARCHAR(30) NOT NULL ,
	`freq` INT(11) NULL DEFAULT NULL,
	`cumul_rate` DECIMAL(10,2) NULL DEFAULT NULL,
	`avg_rate` DECIMAL(10,2) NULL DEFAULT NULL,
	PRIMARY KEY (`word`) USING BTREE
);

/* theme_synonyms 테이블을 삭제 전 동의어를 theme_class 테이블로 옮기는 처리
UPDATE theme_class tc
SET tc.theme_synonyms = COALESCE(
    (SELECT JSON_ARRAYAGG(ts.theme_synonym)
     FROM theme_synonyms ts
     WHERE ts.theme_id = tc.theme_id
     GROUP BY ts.theme_id),
    '[]' -- 일치하는 동의어가 없는 경우 빈 JSON 배열로 설정
);
*/

-- 이 테이블 대신, theme_class 의 theme_class 컬럼을 theme_synomyms JSON 로 사용 예정
-- theme_synonyms 테이블 생성 (테마 동의어 )
/*
CREATE TABLE IF NOT EXISTS `theme_synonyms` (
	`theme_synonym` VARCHAR(30) NOT NULL ,
	`theme_id` INT(11) NOT NULL,
	UNIQUE INDEX `theme_id` (`theme_id`, `theme_synonym`) USING BTREE
);
*/
