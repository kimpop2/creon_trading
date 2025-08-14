# manager/setup_manager.py

import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
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

logger = logging.getLogger(__name__)

class SetupManager(DataManager):
    """
    초기 데이터 셋업 및 일일 데이터 업데이트를 담당하는 클래스.
    DataManager를 상속받아 데이터 처리의 기본 기능을 활용합니다.
    """
    def __init__(self, api_client: CreonAPIClient, db_manager: DBManager):
        super().__init__(api_client, db_manager)
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