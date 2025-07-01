
-- theme_class 테이블 생성 (종목 기본 정보 및 최신 재무 데이터)
CREATE TABLE `daily_theme` (
	`date` DATE NOT NULL,
	`market` VARCHAR(8) NOT NULL COLLATE 'utf8mb4_general_ci',
	`stock_code` VARCHAR(7) NOT NULL COLLATE 'utf8mb4_general_ci',
	`stock_name` VARCHAR(25) NOT NULL COLLATE 'utf8mb4_general_ci',
	`rate` DECIMAL(5,2) NULL DEFAULT '0.00',
	`amount` INT(11) NULL DEFAULT '0',
	`reason` VARCHAR(250) NOT NULL COLLATE 'utf8mb4_general_ci',
	`theme` VARCHAR(250) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
	PRIMARY KEY (`date`, `market`, `stock_code`) USING BTREE
)
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
;

-- theme_class 테이블 생성 (종목 기본 정보 및 최신 재무 데이터)
CREATE TABLE `theme_class` (
	`theme_id` INT(11) NOT NULL AUTO_INCREMENT,
	`theme` VARCHAR(30) NOT NULL COLLATE 'utf8mb4_general_ci',
	`theme_class` VARCHAR(30) NOT NULL COLLATE 'utf8mb4_general_ci',
	`theme_hit` INT(11) NULL DEFAULT '0',
	`theme_score` INT(11) NULL DEFAULT '0',
	`momentum_score` DECIMAL(10,2) NULL DEFAULT '0.0000',
	`theme_desc` VARCHAR(1000) NULL DEFAULT NULL COLLATE 'utf8mb4_general_ci',
	PRIMARY KEY (`theme_id`) USING BTREE,
	UNIQUE INDEX `ak_theme_name_class` (`theme`, `theme_class`) USING BTREE,
	INDEX `idx_theme` (`theme`) USING BTREE,
	INDEX `idx_theme_class` (`theme_class`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- theme_class 테이블 생성 (종목 기본 정보 및 최신 재무 데이터)
CREATE TABLE `theme_synonyms` (
	`theme_synonym` VARCHAR(30) NOT NULL COLLATE 'utf8mb4_general_ci',
	`theme_id` INT(11) NOT NULL,
	UNIQUE INDEX `theme_id` (`theme_id`, `theme_synonym`) USING BTREE
)
COLLATE='utf8mb4_unicode_ci'
ENGINE=InnoDB
;

-- word_dic 테이블 생성 (종목 기본 정보 및 최신 재무 데이터)
CREATE TABLE `word_dic` (
	`word` VARCHAR(25) NOT NULL COLLATE 'utf8mb4_general_ci',
	`freq` INT(11) NULL DEFAULT NULL,
	`cumul_rate` DECIMAL(10,2) NULL DEFAULT NULL,
	`avg_rate` DECIMAL(10,2) NULL DEFAULT NULL,
	PRIMARY KEY (`word`) USING BTREE
)
COLLATE='utf8mb4_general_ci'
ENGINE=InnoDB
;

-- 추가 인덱스 정의 (PK에 포함된 인덱스 외에 필요시 추가)
-- PRIMARY KEY (stock_code, date)에 이미 (stock_code, date) 순서로 인덱스가 포함되므로,
-- 대부분의 조회는 이 PK 인덱스를 통해 효율적으로 처리됩니다.
-- 특정 컬럼 단독 조회에 대한 성능 개선이 필요하다면 아래 인덱스들을 고려할 수 있습니다.
-- CREATE INDEX idx_stock_price_date ON stock_price (date); -- 날짜 단독 조회 효율화
-- CREATE INDEX idx_stock_price_code ON stock_price (stock_code); -- 종목 코드 단독 조회 효율화

-- 외래 키(FK) 정의 (현재 이 두 테이블 간 직접적인 FK는 없음)
-- 다른 테이블과의 관계가 추가될 경우 여기에 ALTER TABLE ... ADD CONSTRAINT ... 구문을 추가합니다.