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
    
    def calculate_signals(self, current_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """
        섹터 로테이션 신호 계산
        """
        signals = {}
        
        try:
            # 리밸런싱 날짜 확인
            if not self._should_rebalance(current_date):
                logger.info(f"[{current_date}] 리밸런싱 날짜가 아닙니다.")
                return {}
            
            # 1. 섹터별 모멘텀 계산
            sector_momentums = self._calculate_sector_momentums(current_date)
            
            # 2. 상위 섹터 선택
            top_sectors = self._select_top_sectors(sector_momentums)
            
            # 3. 섹터별 상위 종목 선택
            sector_stocks = self._select_sector_stocks(top_sectors, current_date)
            
            # 4. 매매 신호 생성
            signals = self._generate_trading_signals(sector_stocks, current_date)
            
            logger.info(f"[{current_date}] 섹터 로테이션 신호 생성 완료: {len(signals)}개 종목")
            
        except Exception as e:
            logger.error(f"섹터 로테이션 신호 계산 중 오류: {str(e)}")
        
        return signals
    
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
    
    def _generate_trading_signals(self, sector_stocks: Dict[str, List[str]], 
                                current_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """매매 신호 생성"""
        signals = {}
        
        # 현재 포트폴리오 상태 확인
        current_positions = set(self.broker.positions.keys())
        
        # 목표 종목 목록 생성
        target_stocks = []
        for sector, stocks in sector_stocks.items():
            target_stocks.extend(stocks)
        
        # 매수 신호 생성 (목표 종목 중 포트폴리오에 없는 종목)
        for stock_code in target_stocks:
            if stock_code not in current_positions:
                current_price = self._get_current_price(stock_code, current_date)
                if current_price > 0:
                    # 섹터별 동일 비중 배분
                    target_amount = self.broker.cash / len(target_stocks)
                    target_quantity = int(target_amount / current_price)
                    
                    if target_quantity > 0:
                        signals[stock_code] = {
                            'signal': 'buy',
                            'quantity': target_quantity,
                            'price': current_price,
                            'signal_date': current_date,
                            'strategy': self.strategy_name
                        }
                        logger.info(f"매수 신호 - {stock_code}: 목표수량 {target_quantity}주")
        
        # 매도 신호 생성 (포트폴리오에 있지만 목표 종목이 아닌 경우)
        for stock_code in current_positions:
            if (stock_code not in target_stocks and 
                stock_code != self.strategy_params['safe_asset_code']):
                current_quantity = self.broker.get_position_size(stock_code)
                if current_quantity > 0:
                    signals[stock_code] = {
                        'signal': 'sell',
                        'quantity': current_quantity,
                        'price': self._get_current_price(stock_code, current_date),
                        'signal_date': current_date,
                        'strategy': self.strategy_name
                    }
                    logger.info(f"매도 신호 - {stock_code}: 전체수량 {current_quantity}주")
        
        return signals
    
    def _get_current_price(self, stock_code: str, current_date: datetime.date) -> float:
        """현재가 조회"""
        if stock_code not in self.data_store['daily']:
            return 0.0
        
        df = self.data_store['daily'][stock_code]
        if df.empty:
            return 0.0
        
        # 해당 날짜의 종가 반환
        date_data = df[df.index.date == current_date]
        if not date_data.empty:
            return date_data.iloc[-1]['close']
        
        # 해당 날짜가 없으면 최근 종가 반환
        recent_data = df[df.index.date <= current_date]
        if not recent_data.empty:
            return recent_data.iloc[-1]['close']
        
        return 0.0 