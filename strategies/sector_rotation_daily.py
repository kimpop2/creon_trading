"""
섹터 로테이션 일봉 전략
섹터별 순환 매매를 통한 리스크 분산 전략
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from strategies.strategy import DailyStrategy

logger = logging.getLogger(__name__)

class SectorRotationDaily(DailyStrategy):
    """
    섹터 로테이션 전략
    - 섹터별 상대 강도 계산
    - 상위 섹터 선택 및 종목 배분
    - 정기적 리밸런싱
    """
    
    def __init__(self, data_store: Dict, strategy_params: Dict[str, Any], broker):
        super().__init__(data_store, strategy_params, broker)
        self.strategy_name = "SectorRotationDaily"
        
        # 파라미터 검증
        self._validate_parameters()
        
    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'momentum_period', 'rebalance_weekday', 'num_top_sectors', 
            'stocks_per_sector', 'safe_asset_code'
        ]
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"섹터 로테이션 전략 파라미터 검증 완료: "
                   f"모멘텀기간={self.strategy_params['momentum_period']}일, "
                   f"리밸런싱요일={self.strategy_params['rebalance_weekday']}, "
                   f"상위섹터수={self.strategy_params['num_top_sectors']}개")
    
    def run_daily_logic(self, current_date: datetime.date):
        """
        섹터 로테이션 전략 실행
        수정: 전일 데이터까지만 사용하여 장전 판단이 가능하도록 함
        """
        try:
            logger.info(f'[{current_date.isoformat()}] --- 섹터 로테이션 전략 실행 중 (전일 데이터 기준) ---')
            
            # 수정: 전일 영업일 계산
            prev_trading_day = None
            for stock_code in self.data_store['daily']:
                df = self.data_store['daily'][stock_code]
                if not df.empty and current_date in df.index.date:
                    idx = list(df.index.date).index(current_date)
                    if idx > 0:
                        prev_trading_day = df.index.date[idx-1]
                        break

            # 수정: 전일 데이터가 없으면 실행하지 않음
            if prev_trading_day is None:
                logger.warning(f'{current_date}: 전일 데이터를 찾을 수 없어 섹터 로테이션 전략을 건너뜁니다.')
                return
            
            if not self._should_rebalance(prev_trading_day):  # 수정: prev_trading_day 사용
                return
            # 1. 섹터별 모멘텀 계산 (전일 데이터 기준)
            sector_momentums = self._calculate_sector_momentums(prev_trading_day)  # 수정: prev_trading_day 사용
            # 2. 상위 섹터 선정
            top_sectors = self._select_top_sectors(sector_momentums)
            # 3. 섹터별 상위 종목 선정 (전일 데이터 기준)
            sector_stocks = self._select_sector_stocks(top_sectors, prev_trading_day)  # 수정: prev_trading_day 사용
            # 4. 전체 매수 후보 종목 리스트 생성
            buy_candidates = set()
            for stocks in sector_stocks.values():
                buy_candidates.update(stocks)
            # 5. 점수 기준 정렬 (여기서는 모멘텀 점수 없음, 임의로 1.0 부여)
            sorted_stocks = [(code, 1.0) for code in buy_candidates]
            # 6. 신호 생성 및 업데이트 (부모 클래스 메서드 사용) - 전일 데이터 기준
            current_positions = self._generate_signals(prev_trading_day, buy_candidates, sorted_stocks)  # 수정: prev_trading_day 사용
            # 7. 리밸런싱 계획 요약 로깅 (전일 데이터 기준)
            self._log_rebalancing_summary(prev_trading_day, buy_candidates, current_positions)  # 수정: prev_trading_day 사용
        except Exception as e:
            logger.error(f"섹터 로테이션 전략 실행 중 오류: {str(e)}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
    
    def _should_rebalance(self, current_date: datetime.date) -> bool:
        """리밸런싱 여부 확인"""
        weekday = current_date.weekday()
        target_weekday = self.strategy_params['rebalance_weekday']
        
        return weekday == target_weekday
    
    def _calculate_sector_momentums(self, current_date: datetime.date) -> Dict[str, float]:
        """섹터별 모멘텀 계산"""
        sector_momentums = {}
        momentum_period = self.strategy_params['momentum_period']
        
        # 섹터별 종목 정보 가져오기
        sector_stocks = self.broker.stock_selector.sector_stocks_config
        
        for sector_name, stocks in sector_stocks.items():
            sector_scores = []
            
            for stock_name, _ in stocks:
                stock_code = self.broker.api_client.get_stock_code(stock_name)
                if not stock_code or stock_code not in self.data_store['daily']:
                    continue
                
                # 개별 종목 모멘텀 계산
                stock_momentum = self._calculate_stock_momentum(stock_code, current_date, momentum_period)
                if stock_momentum is not None:
                    sector_scores.append(stock_momentum)
            
            # 섹터 모멘텀 = 섹터 내 종목들의 평균 모멘텀
            if sector_scores:
                sector_momentums[sector_name] = np.mean(sector_scores)
                logger.info(f"섹터 '{sector_name}' 모멘텀: {sector_momentums[sector_name]:.4f}")
        
        return sector_momentums
    
    def _calculate_stock_momentum(self, stock_code: str, current_date: datetime.date, 
                                momentum_period: int) -> float:
        """개별 종목 모멘텀 계산"""
        df = self.data_store['daily'][stock_code]
        if df.empty or len(df) < momentum_period:
            return None
        
        # 최근 데이터 추출
        recent_df = df[df.index.date <= current_date].tail(momentum_period + 10)
        if len(recent_df) < momentum_period:
            return None
        
        # 가격 모멘텀 계산
        start_price = recent_df.iloc[0]['close']
        end_price = recent_df.iloc[-1]['close']
        price_momentum = (end_price - start_price) / start_price
        
        # 거래량 모멘텀 계산
        recent_volume = recent_df['volume'].tail(momentum_period//2).mean()
        old_volume = recent_df['volume'].head(momentum_period//2).mean()
        
        if old_volume > 0:
            volume_momentum = (recent_volume - old_volume) / old_volume
        else:
            volume_momentum = 0
        
        # 가격 모멘텀 70%, 거래량 모멘텀 30% 가중치
        total_momentum = price_momentum * 0.7 + volume_momentum * 0.3
        
        return total_momentum
    
    def _select_top_sectors(self, sector_momentums: Dict[str, float]) -> List[str]:
        """상위 섹터 선택"""
        if not sector_momentums:
            return []
        
        # 모멘텀 기준 내림차순 정렬
        sorted_sectors = sorted(sector_momentums.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 N개 섹터 선택
        num_top_sectors = self.strategy_params['num_top_sectors']
        top_sectors = [sector for sector, _ in sorted_sectors[:num_top_sectors]]
        
        # 선택된 섹터들의 모멘텀 로깅
        for i, (sector, momentum) in enumerate(sorted_sectors[:num_top_sectors]):
            logger.info(f"상위 섹터 {i+1}: {sector} (모멘텀: {momentum:.4f})")
        
        return top_sectors
    
    def _select_sector_stocks(self, top_sectors: List[str], 
                            current_date: datetime.date) -> Dict[str, List[str]]:
        """섹터별 상위 종목 선택"""
        sector_stocks = {}
        stocks_per_sector = self.strategy_params['stocks_per_sector']
        
        for sector in top_sectors:
            sector_stock_codes = []
            sector_stock_names = self.broker.stock_selector.sector_stocks_config.get(sector, [])
            
            # 섹터 내 종목들의 모멘텀 계산
            stock_momentums = {}
            for stock_name, _ in sector_stock_names:
                stock_code = self.broker.api_client.get_stock_code(stock_name)
                if not stock_code or stock_code not in self.data_store['daily']:
                    continue
                
                momentum = self._calculate_stock_momentum(
                    stock_code, current_date, self.strategy_params['momentum_period']
                )
                if momentum is not None:
                    stock_momentums[stock_code] = momentum
            
            # 모멘텀 기준 상위 종목 선택
            if stock_momentums:
                sorted_stocks = sorted(stock_momentums.items(), key=lambda x: x[1], reverse=True)
                top_stock_codes = [code for code, _ in sorted_stocks[:stocks_per_sector]]
                sector_stocks[sector] = top_stock_codes
                
                # 선택된 종목들의 정보 로깅
                for i, (stock_code, momentum) in enumerate(sorted_stocks[:stocks_per_sector]):
                    stock_name = self.broker.api_client.get_stock_name(stock_code)
                    logger.info(f"  {sector} 섹터 종목 {i+1}: {stock_code} ({stock_name}) "
                               f"(모멘텀: {momentum:.4f})")
        
        return sector_stocks 