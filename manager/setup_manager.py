# manager/setup_manager.py

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from korean_lunar_calendar import KoreanLunarCalendar
import json
from collections import defaultdict
# 프로젝트 루트 경로를 sys.path에 추가
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.data_manager import DataManager
from config.settings import UNIVERSE_CONFIGS # UNIVERSE_CONFIGS 임포트    
logger = logging.getLogger(__name__)

class SetupManager(DataManager):
    """
    초기 데이터 셋업 및 일일 데이터 업데이트를 담당하는 클래스.
    DataManager를 상속받아 데이터 처리의 기본 기능을 활용합니다.
    """
    

    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager):
        super().__init__(api_client, db_manager)
        self.UNIVERSE_CONFIGS = UNIVERSE_CONFIGS # 모듈의 전역 변수로 할당
    
        logger.info("SetupManager 초기화 완료.")

    def update_all_stock_info(self) -> bool:
        """
        [수정] CreonAPIClient 초기화 시 생성된 종목 딕셔너리를 사용하여
        DB의 stock_info 테이블을 업데이트합니다.
        """
        if not self.api_client or not self.api_client.is_connected():
            logger.error("API 클라이언트가 초기화되지 않아 종목 정보를 업데이트할 수 없습니다.")
            return False
        
        try:
            logger.info("API 클라이언트의 내부 캐시를 사용하여 모든 종목 정보 업데이트를 시작합니다.")
            
            stock_code_dict = self.api_client.stock_code_dic
            if not stock_code_dict:
                logger.warning("API 클라이언트에서 조회된 종목 정보가 없습니다.")
                return False
        
            stock_data_list = [{'stock_code': code, 'stock_name': name, 'market_type' : self.api_client.get_market_type(code)} for code, name in stock_code_dict.items()]
            
            logger.info(f"{len(stock_data_list)}개 종목의 정보를 DB에 저장합니다.")
            success = self.db_manager.save_stock_info(stock_data_list)
            
            if success:
                logger.info("stock_info 테이블 업데이트가 성공적으로 완료되었습니다.")
                self._load_stock_info_map()
            else:
                logger.error("stock_info 테이블 업데이트에 실패했습니다.")
                
            return success

        except Exception as e:
            logger.critical(f"종목 정보 업데이트 중 심각한 오류 발생: {e}", exc_info=True)
            return False

    def update_market_calendar(self, year: int) -> bool:
        """
        [최종 수정] 대한민국 공휴일 로직을 기반으로 market_calendar 테이블을 생성/업데이트합니다.
        (설정 파일 의존성 제거)
        """
        try:
            logger.info(f"{year}년도 시장 캘린더 자동 생성을 시작합니다.")

            # 1. 음력 변환기 및 휴일 세트 초기화
            lunar_cal = KoreanLunarCalendar()
            
            holidays = set()
            holiday_names = {}

            # 2. 고정 국경일 및 기본 휴장일 정의
            fixed_holidays = {
                (3, 1): '삼일절', (5, 1): '근로자의 날', (5, 5): '어린이날',
                (6, 6): '현충일', (8, 15): '광복절', (10, 3): '개천절',
                (10, 9): '한글날', (12, 25): '성탄절',
            }
            for (month, day), name in fixed_holidays.items():
                d = date(year, month, day)
                holidays.add(d)
                holiday_names[d] = name

            # 연초/연말 휴장일
            holidays.add(date(year, 1, 1))
            holiday_names[date(year, 1, 1)] = '신정'
            holidays.add(date(year, 12, 31))
            holiday_names[date(year, 12, 31)] = '연말 휴장일'

            # 3. 음력 기반 공휴일 계산
            lunar_cal.setLunarDate(year, 1, 1, False)
            seollal_day = datetime.strptime(lunar_cal.SolarIsoFormat(), '%Y-%m-%d').date()
            holidays.update([seollal_day - timedelta(days=1), seollal_day, seollal_day + timedelta(days=1)])
            holiday_names[seollal_day] = '설날'

            lunar_cal.setLunarDate(year, 8, 15, False)
            chuseok_day = datetime.strptime(lunar_cal.SolarIsoFormat(), '%Y-%m-%d').date()
            holidays.update([chuseok_day - timedelta(days=1), chuseok_day, chuseok_day + timedelta(days=1)])
            holiday_names[chuseok_day] = '추석'

            lunar_cal.setLunarDate(year, 4, 8, False)
            buddha_day = datetime.strptime(lunar_cal.SolarIsoFormat(), '%Y-%m-%d').date()
            holidays.add(buddha_day)
            holiday_names[buddha_day] = '부처님오신날'
            
            # 4. 대체 공휴일 계산
            substitute_holidays = set()
            for holiday in sorted(list(holidays)):
                if holiday.weekday() in [5, 6]:
                    substitute_day = holiday
                    while substitute_day.weekday() in [5, 6] or substitute_day in holidays:
                        substitute_day += timedelta(days=1)
                    substitute_holidays.add(substitute_day)
                    holiday_names[substitute_day] = holiday_names.get(holiday, '대체공휴일') + ' (대체)'
            
            all_holidays = holidays.union(substitute_holidays)
            logger.info(f"{year}년도 계산된 총 휴장일(대체공휴일 포함): {len(all_holidays)}일")

            # 5. 최종 캘린더 생성 및 DB 저장
            all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            calendar_df = pd.DataFrame(all_dates, columns=['date'])
            
            # [수정] day_of_week 컬럼을 명시적으로 추가
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek
            
            calendar_df['is_holiday'] = calendar_df['day_of_week'].isin([5, 6]).astype(int)
            
            holiday_dates_dt = pd.to_datetime(list(all_holidays))
            calendar_df.loc[calendar_df['date'].isin(holiday_dates_dt), 'is_holiday'] = 1
            
            calendar_df['description'] = '거래일'
            calendar_df.loc[calendar_df['day_of_week'] == 5, 'description'] = '토요일'
            calendar_df.loc[calendar_df['day_of_week'] == 6, 'description'] = '일요일'
            for dt, name in holiday_names.items():
                calendar_df.loc[calendar_df['date'] == pd.to_datetime(dt), 'description'] = name

            # [수정] 최종 데이터 생성 시 day_of_week 컬럼 제외
            calendar_data_list = calendar_df[['date', 'is_holiday', 'description']].to_dict('records')
            
            logger.info(f"{len(calendar_data_list)}일의 캘린더 정보를 DB에 저장합니다.")
            success = self.db_manager.save_market_calendar(calendar_data_list)

            if success:
                logger.info("market_calendar 테이블 업데이트가 성공적으로 완료되었습니다.")
            else:
                logger.error("market_calendar 테이블 업데이트에 실패했습니다.")
                
            return success

        except Exception as e:
            logger.critical(f"시장 캘린더 업데이트 중 심각한 오류 발생: {e}", exc_info=True)
            return False

    def update_stock_info_with_marketeye(self):
        """
        [신규 이동] MarketEye를 사용하여 stock_info 테이블의 모든 종목에 대한
        펀더멘털 및 수급 스냅샷 데이터를 업데이트합니다.
        """
        logger.info("MarketEye를 이용한 stock_info 상세 정보 업데이트 시작")
        try:
            stock_info_df = self.db_manager.fetch_stock_info()
            if stock_info_df.empty:
                logger.warning("DB에 종목 정보가 없어 MarketEye 업데이트를 건너뜁니다.")
                return False

            all_stock_codes = stock_info_df['stock_code'].tolist()
            stock_name_map = pd.Series(stock_info_df.stock_name.values, index=stock_info_df.stock_code).to_dict()
            market_type_map = pd.Series(stock_info_df.market_type.values, index=stock_info_df.stock_code).to_dict()
            sector_map = pd.Series(stock_info_df.sector.values, index=stock_info_df.stock_code).to_dict()

            marketeye_data_dict = self.api_client.get_market_eye_datas(all_stock_codes)
            
            if not marketeye_data_dict:
                logger.error("MarketEye 데이터 조회에 실패했습니다.")
                return False

            update_data_list = []
            for stock_code, data in marketeye_data_dict.items():

                update_data_list.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name_map.get(stock_code),
                    'market_type': market_type_map.get(stock_code),
                    'sector': sector_map.get(stock_code),
                    'recent_financial_date': data.get('recent_financial_date'),
                    'foreigner_net_buy': data.get('foreigner_net_buy'),
                    'institution_net_buy': data.get('institution_net_buy'),
                    'credit_ratio': data.get('credit_ratio'),
                    'short_volume': data.get('short_volume'),
                    'trading_value': data.get('trading_value'),
                    'beta_coefficient': data.get('beta_coefficient'),
                    'q_revenue_growth_rate': data.get('q_revenue_growth_rate'),
                    'q_op_income_growth_rate': data.get('q_op_income_growth_rate'),
                    'per': data.get('per'),
                    'pbr': (data['current_price'] / data['bps']) if data.get('bps') and data.get('bps') > 0 else None,
                    'dividend_yield': data.get('dividend_yield'),
                    'program_net_buy': data.get('program_net_buy'),
                    'sps': data.get('sps'),
                    'bps': data.get('bps'),
                    'trading_intensity': data.get('trading_intensity'),
                })
            
            if self.db_manager.save_stock_info_factors(update_data_list):
                 logger.info(f"총 {len(update_data_list)}개 종목의 MarketEye 스냅샷 정보 업데이트 완료.")
                 return True
            else:
                logger.error("stock_info 스냅샷 업데이트 실패.")
                return False
        except Exception as e:
            logger.critical(f"MarketEye 정보 업데이트 중 오류 발생: {e}", exc_info=True)
            return False


    def update_price_and_factors_for_recent_themes(self, days: int = 30) -> bool:
        """
        최근 N일간 daily_theme에 등록된 종목들에 대해
        최신 일봉(daily_price) 및 팩터(daily_factors) 데이터를 수집/업데이트합니다.
        """
        logger.info(f"최근 {days}일 내 daily_theme 등록 종목의 가격/팩터 업데이트 시작.")
        try:
            # 1. 최근 N일간의 대상 종목 코드 조회
            target_codes = self.db_manager.fetch_recent_theme_stocks(days)
            if not target_codes:
                logger.info("업데이트할 최근 테마 종목이 없습니다.")
                return True
            
            logger.info(f"총 {len(target_codes)}개 종목에 대한 데이터 업데이트를 진행합니다.")
            
            # 2. 각 종목에 대해 데이터 업데이트
            today = date.today()
            start_date = today - timedelta(days=days + 5) # 여유 기간을 두고 조회
            
            for i, stock_code in enumerate(target_codes):
                logger.info(f"  - 처리 중 ({i+1}/{len(target_codes)}): {stock_code} ({self.get_stock_name(stock_code)})")
                # DataManager의 캐싱 메서드 사용
                self.cache_daily_ohlcv(stock_code, start_date, today)
                self.cache_factors(start_date, today, stock_code) # SetupManager에 cache_factors 구현 필요
            
            logger.info("최근 테마 종목 가격/팩터 업데이트 완료.")
            return True
        except Exception as e:
            logger.error(f"최근 테마 종목 데이터 업데이트 중 오류 발생: {e}", exc_info=True)
            return False

    def calculate_theme_momentum_scores(self, data_period_days: int = 40) -> bool:
        """
        thematic_stocks.py의 process_daily_theme_data 로직을 수행하여
        테마별 모멘텀 점수를 계산하고 DB에 업데이트합니다.
        """
        logger.info(f"테마 모멘텀 점수 계산 시작 (최근 {data_period_days}일 데이터 기준)")

        try:
            # 데이터 로드
            start_date = date.today() - timedelta(days=data_period_days)
            daily_theme_records = self.db_manager.fetch_daily_theme_stock(start_date, date.today()) # 이 메서드는 db_manager에 추가 필요
            if not daily_theme_records:
                logger.warning("분석할 daily_theme 데이터가 없습니다.")
                return True

            theme_word_aggr = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sum_rate': 0.0}))
            
            # 데이터 집계
            for record in daily_theme_records:
                try:
                    # reason_nouns가 JSON 형태의 문자열이므로 파싱
                    nouns = json.loads(record['reason_nouns']) if record['reason_nouns'] else []
                    themes = record['theme'].split(',') if record['theme'] else []
                    rate = float(record['rate'])
                    
                    for theme_name in themes:
                        theme_name = theme_name.strip()
                        if not theme_name: continue
                        for noun in nouns:
                            theme_word_aggr[theme_name][noun]['count'] += 1
                            theme_word_aggr[theme_name][noun]['sum_rate'] += rate
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"레코드 처리 중 오류 발생 (건너뜀): {record} - {e}")
                    continue

            # 모멘텀 점수 계산
            theme_momentum_updates = []
            for theme_name, words_data in theme_word_aggr.items():
                momentum_score = sum(
                    metrics['count'] * abs(metrics['sum_rate'] / metrics['count'])
                    for metrics in words_data.values() if metrics['count'] > 0
                )
                if momentum_score > 0:
                    theme_momentum_updates.append((round(momentum_score, 4), theme_name))
            
            # DB 업데이트
            success = self.db_manager.update_theme_momentum_scores(theme_momentum_updates)
            
            logger.info(f"테마 모멘텀 점수 계산 완료.")
            return success

        except Exception as e:
            logger.error(f"테마 모멘텀 점수 계산 중 심각한 오류 발생: {e}", exc_info=True)
            return False


    def generate_universe_candidates(self, limit_themes: int, limit_stocks_per_theme: int) -> List[Dict[str, Any]]:
        """
        thematic_stocks.py의 get_actionable_insights 로직을 수행하여
        분석 및 저장을 위한 유니버스 후보군 데이터를 생성합니다.
        """
        logger.info("유니버스 후보군 생성 시작...")
        
        top_themes = self.db_manager.fetch_top_momentum_themes(limit=limit_themes)
        if not top_themes:
            logger.warning("상위 모멘텀 테마를 찾을 수 없습니다.")
            return []

        universe_candidates = []
        for theme_data in top_themes:
            theme_id = theme_data['theme_id']
            theme_class = theme_data['theme_class']
            
            related_stocks = self.db_manager.fetch_top_stocks_for_theme(theme_id, limit=limit_stocks_per_theme)
            
            if related_stocks:
                # 최근 reason 조회를 위해 stock_code 리스트 생성
                stock_codes_in_theme = [s['stock_code'] for s in related_stocks]
                recent_reasons = self.db_manager.fetch_latest_reasons_for_stocks(stock_codes_in_theme) # 이 메서드는 db_manager에 추가 필요

                theme_result = {
                    'theme_class': theme_class,
                    'broad_theme': theme_class, # 현재는 동일하게 설정
                    'momentum_score': theme_data['momentum_score'],
                    'recommended_stocks': []
                }
                for stock in related_stocks:
                    stock_code = stock['stock_code']
                    reason_data = recent_reasons.get(stock_code, {'rate': 0.0, 'reason': 'N/A'})
                    
                    theme_result['recommended_stocks'].append({
                        'stock_code': stock_code,
                        'stock_name': stock['stock_name'],
                        'stock_score': stock['stock_score'],
                        'recent_rate': reason_data['rate'],
                        'recent_reason': reason_data['reason']
                    })
                universe_candidates.append(theme_result)

        logger.info(f"유니버스 후보군 생성 완료. {len(universe_candidates)}개 테마, 총 {sum(len(t['recommended_stocks']) for t in universe_candidates)}개 종목.")
        return universe_candidates
    
# ============================================================

    def select_universe_initial_filter(self) -> set:
        """
        DB에 저장된 모든 종목을 대상으로 거래에 부적합한 종목(우선주, 스팩 등)을
        제외한 1차 후보군을 반환합니다.
        """
        all_stocks_df = self.db_manager.fetch_stock_info()
        if all_stocks_df.empty:
            logger.warning("DB에 종목 정보가 없어 초기 필터링을 진행할 수 없습니다.")
            return set()

        # 필터링 조건:
        # 1. 우선주 제외: stock_code 마지막 자리가 '5' 이상이거나, 이름이 '우'로 끝나는 경우
        # 2. 스팩 제외: stock_name에 '스팩'이 포함된 경우
        initial_candidates = all_stocks_df[
            ~all_stocks_df['stock_code'].str.endswith(('5', '6', '7', 'K', 'L', 'M')) &
            ~all_stocks_df['stock_name'].str.endswith('우') &
            ~all_stocks_df['stock_name'].str.contains('스팩')
        ]
        
        candidate_codes = set(initial_candidates['stock_code'].tolist())
        logger.info(f"초기 필터링 완료: 전체 {len(all_stocks_df)}개 중 {len(candidate_codes)}개 종목 선정")
        return candidate_codes

    def select_universe(self, config_name: str, universe_date: date) -> set:
        """
        주어진 설정에 따라 특정 날짜의 유니버스를 동적으로 생성합니다.
        """
        # 1. 설정 로드
        try:
            config = self.UNIVERSE_CONFIGS[config_name]
        except KeyError:
            logger.error(f"'{config_name}'에 해당하는 유니버스 설정을 찾을 수 없습니다.")
            return set()
        
        # 2. 기본 필터링 (우선주, 스팩 등 제외)
        candidate_codes = self.select_universe_initial_filter()
        if not candidate_codes:
            return set()
            
        # 3. 동적 필터링을 위한 데이터 조회
        filter_data_df = self.db_manager.fetch_data_for_universe_filter(
            list(candidate_codes), universe_date
        )
        
        # [수정] 더 명확한 로깅
        logger.info(f"동적 필터링 대상 종목: {len(filter_data_df)}개")
        if filter_data_df.empty:
            logger.warning("동적 필터링에 사용할 데이터가 없습니다.")
            return set()

        # 4. 동적 필터링 적용
        # DataFrame의 boolean indexing을 활용하여 간결하게 처리
        # NULL 값은 비교 연산 시 자동으로 False 처리되어 제외됨 (우리가 원했던 방식)
        if 'min_price' in config:
            filter_data_df = filter_data_df[filter_data_df['price'] >= config['min_price']]
        if 'max_price' in config:
            filter_data_df = filter_data_df[filter_data_df['price'] <= config['max_price']]
        
        if 'min_market_cap' in config:
            filter_data_df = filter_data_df[filter_data_df['market_cap'] >= config['min_market_cap'] ]
        if 'max_market_cap' in config:
            filter_data_df = filter_data_df[filter_data_df['market_cap'] <= config['max_market_cap'] ]
        
        if 'min_avg_trading_value' in config:
            # 단위: 억 원
            filter_data_df = filter_data_df[filter_data_df['trading_value'] >= config['min_avg_trading_value'] ]
        
        # (필요시 다른 지표들도 위와 같은 방식으로 추가)
        
        if filter_data_df.empty:
            logger.warning("동적 필터링(가격, 시총 등) 결과 유니버스 후보 종목이 없습니다.")
            return set() # 빈 set을 반환하고 함수 종료


        # 5. 스코어링
        # 필터링을 통과한 데이터프레임(filter_data_df)을 스코어링 함수에 전달
        scored_df = self._calculate_scores(filter_data_df, config)

        # 6. 최종 선정
        # stock_score가 높은 순으로 정렬
        final_df = scored_df.sort_values(by='stock_score', ascending=False)
        
        # max_universe_per_theme 적용 (향후 구현)
        
        # max_universe 적용
        max_stocks = config.get('max_universe', 200) # 기본값 200
        final_df = final_df.head(max_stocks)
        
        final_codes = set(final_df['stock_code'].tolist())
        logger.info(f"최종 유니버스 선정 완료: {len(final_codes)}개 종목")
        
        return final_codes

    def _calculate_scores(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """[수정] 각 스코어를 명시적으로 처리하고, 최종 가중합산 점수를 10분위 등급으로 변환합니다."""
        if df.empty:
            return df

        scored_df = df.copy()
        scored_df['stock_score'] = 0.0

        # --- 1. 가격 추세 점수 (price_trend_score) ---
        weight = config.get('score_price_trend', 0)
        if weight != 0 and 'price_trend_score' in scored_df.columns:
            median_value = scored_df['price_trend_score'].median()
            scored_df['price_trend_score'].fillna(median_value, inplace=True)
            scored_df['stock_score'] += scored_df['price_trend_score'] * weight
        
        # --- 2. 거래량 점수 (trading_volume_score) ---
        weight = config.get('score_trading_volume', 0)
        if weight != 0 and 'trading_volume_score' in scored_df.columns:
            median_value = scored_df['trading_volume_score'].median()
            scored_df['trading_volume_score'].fillna(median_value, inplace=True)
            scored_df['stock_score'] += scored_df['trading_volume_score'] * weight

        # --- 3. 변동성 점수 (volatility_score) ---
        weight = config.get('score_volatility', 0)
        if weight != 0 and 'volatility_score' in scored_df.columns:
            median_value = scored_df['volatility_score'].median()
            scored_df['volatility_score'].fillna(median_value, inplace=True)
            scored_df['stock_score'] += scored_df['volatility_score'] * weight
            
        # --- 4. 테마 언급 점수 (theme_mention_score) ---
        weight = config.get('score_theme_mention', 0)
        if weight != 0 and 'theme_mention_score' in scored_df.columns:
            median_value = scored_df['theme_mention_score'].median()
            scored_df['theme_mention_score'].fillna(median_value, inplace=True)
            scored_df['stock_score'] += scored_df['theme_mention_score'] * weight

        # --- 5. 수급 점수 (supply_demand_score) ---
        weight = config.get('score_supply_demand', 0)
        if weight != 0 and 'supply_demand_score' in scored_df.columns:
            median_value = scored_df['supply_demand_score'].median()
            scored_df['supply_demand_score'].fillna(median_value, inplace=True)
            scored_df['stock_score'] += scored_df['supply_demand_score'] * weight
        
        # 최종 가중합산 점수를 0~9 사이의 10분위 등급으로 변환
        if not scored_df['stock_score'].dropna().empty and scored_df['stock_score'].nunique() > 1:
            try:
                scored_df['stock_score'] = pd.qcut(scored_df['stock_score'], 10, labels=False, duplicates='drop')
            except (ValueError, IndexError):
                scored_df['stock_score'] = pd.cut(scored_df['stock_score'].rank(method='first'), 10, labels=False)
        else:
            logger.warning("stock_score의 값이 모두 동일하여 10분위 등급을 매길 수 없습니다.")
            scored_df.loc[scored_df['stock_score'].notna(), 'stock_score'] = 5

        return scored_df

    def update_daily_factors_scores(self, target_date: date):
        """
        [수정됨] 하루에 한 번, 모든 개별 팩터 점수를 일괄 계산하여
        daily_factors 테이블에 업데이트합니다.

        점수 계산 로직:
        1. 각 팩터(factor)를 10분위로 나누어 0~9점의 등급(grade)을 부여합니다.
        2. 개별 점수들을 가중 합산하여 최종 stock_score를 계산합니다.
        """
        logger.info(f"[{target_date}] 모든 팩터 스코어 일괄 계산 및 업데이트 시작.")

        # 1. 스코어링에 필요한 모든 원본 데이터 조회
        all_factors_df = self.db_manager.fetch_all_factors_for_scoring(target_date)
        if all_factors_df.empty:
            logger.warning("스코어링에 사용할 팩터 데이터가 없습니다.")
            return

        # 2. 각 점수(Score)를 어떤 팩터(Factor) 기준으로 계산할지 매핑
        #    - 키: 점수를 저장할 컬럼명
        #    - 값: 점수 계산의 기반이 될 원본 팩터 컬럼명
        factor_to_score_map = {
            'price_trend_score': 'dist_from_ma20',          # 가격 추세: 20일 이평선 이격도 (높을수록 좋음)
            'trading_volume_score': 'trading_value',    # 거래량: 체결 강도 (높을수록 좋음)
            'volatility_score': 'historical_volatility_20d',# 변동성: 20일 역사적 변동성 (낮을수록 좋음)
            'theme_mention_score': 'theme_mention_score'    # 테마성: 테마 언급 점수 (높을수록 좋음)
        }
        
        # 점수 계산 결과를 담을 DataFrame 복사
        scored_df = all_factors_df.copy()

        # 3. 각 개별 점수 일괄 계산 (0~9 등급)
        for score_col, factor_col in factor_to_score_map.items():
            if factor_col not in scored_df.columns:
                logger.warning(f"'{factor_col}' 컬럼이 없어 '{score_col}'를 계산할 수 없습니다. 0점을 부여합니다.")
                scored_df[score_col] = 0
                continue

            median_value = scored_df[factor_col].median()
            scored_df[factor_col].fillna(median_value, inplace=True)
            
            # ✨ [핵심 수정] qcut 실행 전, 유효한 데이터가 있는지 확인하는 방어 코드 추가
            if scored_df[factor_col].dropna().empty:
                logger.warning(f"'{factor_col}'에 유효한 데이터가 없어 '{score_col}'를 건너뜁니다. 0점을 부여합니다.")
                scored_df[score_col] = 0
                continue

            try:
                score_grade = pd.qcut(scored_df[factor_col], 10, labels=False, duplicates='drop')
            except ValueError:
                score_grade = pd.cut(scored_df[factor_col].rank(method='first'), 10, labels=False)
            # 작은 값이 점수가 높을 경우 처리
            if score_col == 'volatility_score':
                score_grade = 9 - score_grade

            scored_df[score_col] = score_grade


        # 4. 최종 종합 점수(stock_score) 계산 (가중치 부여)
        weights = {
            'price_trend_score': 0.4,       # 가격 추세 40%
            'trading_volume_score': 0.3,    # 거래량 30%
            'theme_mention_score': 0.2,     # 테마성 20%
            'volatility_score': 0.1         # 변동성 10%
        }
        
        # 가중합 계산 전, 점수 컬럼의 혹시 모를 NaN 값을 0으로 채움
        scored_df['stock_score'] = (
            scored_df['price_trend_score'].fillna(0) * weights['price_trend_score'] +
            scored_df['trading_volume_score'].fillna(0) * weights['trading_volume_score'] +
            scored_df['theme_mention_score'].fillna(0) * weights['theme_mention_score'] +
            scored_df['volatility_score'].fillna(0) * weights['volatility_score']
        )
        
        # 최종 점수도 0~9 사이의 등급으로 변환
        try:
            scored_df['stock_score'] = pd.qcut(scored_df['stock_score'], 10, labels=False, duplicates='drop')
        except ValueError:
            scored_df['stock_score'] = pd.cut(scored_df['stock_score'].rank(method='first'), 10, labels=False)


        # 5. DB에 저장할 최종 데이터 포맷팅
        cols_to_save = ['stock_code', 'stock_score', 'price_trend_score', 'trading_volume_score', 'volatility_score', 'theme_mention_score']
        
        final_df_to_save = scored_df[cols_to_save].copy()
        final_df_to_save['date'] = target_date
        # ✨ [핵심 수정] DB에 전달하기 전, 모든 NaN 값을 Python의 None으로 변환합니다.
        final_df_to_save.replace({np.nan: None}, inplace=True)
        update_list = final_df_to_save.to_dict('records')
            
        # 6. DB에 단 한 번의 호출로 일괄 저장
        success = self.db_manager.update_daily_factors_scores(update_list)
        if success:
            logger.info(f"총 {len(update_list)}개 종목의 모든 팩터 스코어 업데이트 완료.")
        else:
            logger.error("팩터 스코어 업데이트 실패.")