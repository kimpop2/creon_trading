"""
시장 데이터 로딩 및 관리를 위한 클래스
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class DataLoader:
    def __init__(self):
        """데이터 로더 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 테스트용 종목 리스트 (KOSPI 200 종목 중 일부)
        self.stock_list = {
            'A005930': '삼성전자',
            'A000660': 'SK하이닉스',
            'A035420': 'NAVER',
            'A035720': '카카오',
            'A051910': 'LG화학',
            'A006400': '삼성SDI',
            'A105560': 'KB금융',
            'A055550': '신한지주',
            'A086790': '하나금융지주',
            'A114260': 'KODEX 레버리지'  # 안전자산으로 사용
        }
        
        # 데이터 저장소 초기화
        self.data_store = {
            'daily': {},
            'minute': {}
        }
        
    def load_market_data(self, start_date, end_date):
        """
        지정된 기간의 시장 데이터를 로드합니다.
        
        Args:
            start_date (datetime): 시작일
            end_date (datetime): 종료일
            
        Returns:
            dict: 일봉/분봉 데이터가 포함된 딕셔너리
        """
        try:
            # 데이터 디렉토리 확인
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                self.logger.warning(f"데이터 디렉토리가 없어 생성했습니다: {data_dir}")
            
            # 각 종목별 데이터 로드
            for stock_code in self.stock_list.keys():
                # 일봉 데이터 로드
                daily_data = self._load_daily_data(stock_code, start_date, end_date)
                if daily_data is not None:
                    self.data_store['daily'][stock_code] = daily_data
                
                # 분봉 데이터 로드
                minute_data = self._load_minute_data(stock_code, start_date, end_date)
                if minute_data is not None:
                    self.data_store['minute'][stock_code] = minute_data
            
            if not self.data_store['daily'] and not self.data_store['minute']:
                self.logger.error("로드된 데이터가 없습니다.")
                return None
                
            return self.data_store
            
        except Exception as e:
            self.logger.error(f"시장 데이터 로드 중 오류 발생: {str(e)}", exc_info=True)
            return None
    
    def _load_daily_data(self, stock_code, start_date, end_date):
        """일봉 데이터 로드"""
        try:
            # 실제 구현에서는 Creon API를 통해 데이터를 가져옵니다.
            # 테스트를 위해 임의의 데이터 생성
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            data = pd.DataFrame(index=dates)
            
            # 임의의 가격 데이터 생성 (실제로는 API에서 가져온 데이터 사용)
            base_price = 50000 if stock_code != 'A114260' else 10000
            volatility = 0.02 if stock_code != 'A114260' else 0.03
            
            np.random.seed(int(stock_code))  # 재현성을 위한 시드 설정
            returns = np.random.normal(0.0001, volatility, len(dates))
            prices = base_price * (1 + returns).cumprod()
            
            data['open'] = prices * (1 + np.random.normal(0, 0.001, len(dates)))
            data['high'] = data['open'] * (1 + abs(np.random.normal(0, 0.002, len(dates))))
            data['low'] = data['open'] * (1 - abs(np.random.normal(0, 0.002, len(dates))))
            data['close'] = prices
            data['volume'] = np.random.randint(1000, 1000000, len(dates))
            
            return data
            
        except Exception as e:
            self.logger.error(f"{stock_code} 일봉 데이터 로드 중 오류: {str(e)}")
            return None
    
    def _load_minute_data(self, stock_code, start_date, end_date):
        """분봉 데이터 로드"""
        try:
            # 실제 구현에서는 Creon API를 통해 데이터를 가져옵니다.
            # 테스트를 위해 임의의 데이터 생성
            minute_data = {}
            current_date = start_date
            
            while current_date <= end_date:
                if current_date.weekday() < 5:  # 주말 제외
                    # 거래 시간 (9:00-15:30)
                    times = pd.date_range(
                        start=current_date.replace(hour=9, minute=0),
                        end=current_date.replace(hour=15, minute=30),
                        freq='1min'
                    )
                    
                    # 임의의 가격 데이터 생성
                    base_price = 50000 if stock_code != 'A114260' else 10000
                    volatility = 0.001 if stock_code != 'A114260' else 0.002
                    
                    # 유효한 시드 값 생성 (0 ~ 2^32-1 범위 내)
                    seed_value = abs(hash(f"{stock_code}{current_date.strftime('%Y%m%d')}")) % (2**32)
                    np.random.seed(seed_value)
                    returns = np.random.normal(0.0001, volatility, len(times))
                    prices = base_price * (1 + returns).cumprod()
                    
                    data = pd.DataFrame(index=times)
                    data['open'] = prices * (1 + np.random.normal(0, 0.0005, len(times)))
                    data['high'] = data['open'] * (1 + abs(np.random.normal(0, 0.001, len(times))))
                    data['low'] = data['open'] * (1 - abs(np.random.normal(0, 0.001, len(times))))
                    data['close'] = prices
                    data['volume'] = np.random.randint(100, 100000, len(times))
                    
                    minute_data[current_date.date()] = data
                
                current_date += timedelta(days=1)
            
            return minute_data if minute_data else None
            
        except Exception as e:
            self.logger.error(f"{stock_code} 분봉 데이터 로드 중 오류: {str(e)}")
            return None 