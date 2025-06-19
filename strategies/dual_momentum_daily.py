# strategies/dual_momentum_daily.py

import datetime
import logging
import pandas as pd
import numpy as np 
from datetime import timedelta
from typing import Dict, List, Tuple, Any

from util.strategies_util import * # utils.py에 있는 모든 함수를 임포트한다고 가정 
from strategies.strategy import DailyStrategy 

logger = logging.getLogger(__name__)

class DualMomentumDaily(DailyStrategy): # DailyStrategy 상속 
    def __init__(self, data_store, strategy_params, broker): 
        super().__init__(data_store, strategy_params, broker) # BaseStrategy의 __init__ 호출
        self.signals = {} # {stock_code: {'signal', 'signal_date', 'traded_today', 'target_quantity'}} 
        #self.last_rebalance_date = None 
        self._initialize_signals_for_all_stocks() 
        self.strategy_name = "DualMomentumDaily"
        
        # 파라미터 검증
        self._validate_parameters()

    def _validate_parameters(self):
        """전략 파라미터 검증"""
        required_params = [
            'momentum_period', 'rebalance_weekday', 'num_top_stocks', 'safe_asset_code'
        ]
        
        for param in required_params:
            if param not in self.strategy_params:
                raise ValueError(f"필수 파라미터 누락: {param}")
        
        logger.info(f"듀얼 모멘텀 전략 파라미터 검증 완료: "
                   f"모멘텀기간={self.strategy_params['momentum_period']}일, "
                   f"리밸런싱요일={self.strategy_params['rebalance_weekday']}, "
                   f"선택종목수={self.strategy_params['num_top_stocks']}개")

    def run_daily_logic(self, current_daily_date):
        """주간 듀얼 모멘텀 로직을 실행하고 신호를 생성합니다."""
        if current_daily_date.weekday() != self.strategy_params['rebalance_weekday']:
            return

        logging.info(f'{current_daily_date.isoformat()} - --- 주간 모멘텀 로직 실행 중 ---')

        # 1. 모멘텀 스코어 계산
        momentum_scores = {}
        for stock_code in self.data_store['daily']:
            if stock_code == self.strategy_params['safe_asset_code']:
                continue  # 안전자산은 모멘텀 계산에서 제외

            daily_df = self.data_store['daily'][stock_code]
            if daily_df.empty:
                continue
            
            historical_data = self._get_historical_data_up_to(
                'daily',
                stock_code,
                current_daily_date,
                lookback_period=self.strategy_params['momentum_period'] + 1
            )

            if len(historical_data) < self.strategy_params['momentum_period']:
                logging.debug(f'{stock_code} 종목의 모멘텀 계산을 위한 데이터가 부족합니다.')
                continue

            momentum_score = calculate_momentum(historical_data, self.strategy_params['momentum_period']).iloc[-1]
            momentum_scores[stock_code] = momentum_score
        
        if not momentum_scores:
            logging.warning('계산된 모멘텀 스코어가 없습니다.')
            return

        # 2. 안전자산 모멘텀 계산
        safe_asset_momentum = self._calculate_safe_asset_momentum(current_daily_date)

        # 3. 매수 대상 종목 선정
        buy_candidates, sorted_stocks = self._select_buy_candidates(momentum_scores, safe_asset_momentum)
        buy_candidates = set()
        for rank, (stock_code, _) in enumerate(sorted_stocks, 1):
            if rank <= self.strategy_params['num_top_stocks']:
                buy_candidates.add(stock_code)
        if not buy_candidates:
            return
        # 4. 신호 생성 및 업데이트 (부모 클래스 메서드 사용)
        current_positions = self._generate_signals(current_daily_date, buy_candidates, sorted_stocks)
        # 5. 리밸런싱 계획 요약 로깅
        self._log_rebalancing_summary(current_daily_date, buy_candidates, current_positions)

    def calculate_signals(self, current_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """
        듀얼 모멘텀 신호 계산
        """
        signals = {}
        
        try:
            # 1. 상대 모멘텀 계산 (섹터 내 순위)
            relative_momentum_scores = self._calculate_relative_momentum(current_date)
            
            # 2. 절대 모멘텀 계산 (시장 대비 성과)
            absolute_momentum_scores = self._calculate_absolute_momentum(current_date)
            
            # 3. 듀얼 모멘텀 점수 결합
            dual_momentum_scores = self._combine_momentum_scores(
                relative_momentum_scores, absolute_momentum_scores
            )
            
            # 4. 상위 종목 선택
            top_stocks = self._select_top_stocks(dual_momentum_scores)
            
            # 5. 신호 생성
            signals = self._generate_trading_signals(top_stocks, current_date)
            
            logger.info(f"[{current_date}] 듀얼 모멘텀 신호 생성 완료: {len(signals)}개 종목")
            
        except Exception as e:
            logger.error(f"듀얼 모멘텀 신호 계산 중 오류: {str(e)}")
        
        return signals
    
    def _calculate_relative_momentum(self, current_date: datetime.date) -> Dict[str, float]:
        """상대 모멘텀 계산 (섹터 내 순위)"""
        relative_scores = {}
        
        # 섹터별로 그룹화하여 계산
        sector_stocks = self._get_sector_stocks()
        
        for sector, stocks in sector_stocks.items():
            sector_scores = {}
            
            for stock_name, _ in stocks:
                stock_code = self.broker.api_client.get_stock_code(stock_name)
                if not stock_code or stock_code not in self.data_store['daily']:
                    continue
                
                # 모멘텀 기간 동안의 수익률 계산
                momentum_period = self.strategy_params['momentum_period']
                end_date = current_date
                start_date = end_date - timedelta(days=momentum_period * 2)  # 충분한 데이터 확보
                
                df = self.data_store['daily'][stock_code]
                if df.empty or len(df) < momentum_period:
                    continue
                
                # 최근 데이터만 사용
                recent_df = df[df.index.date <= end_date].tail(momentum_period + 5)
                if len(recent_df) < momentum_period:
                    continue
                
                # 모멘텀 계산 (가격 변화율)
                start_price = recent_df.iloc[0]['close']
                end_price = recent_df.iloc[-1]['close']
                momentum = (end_price - start_price) / start_price
                
                # 거래량 가중치 적용
                avg_volume = recent_df['volume'].mean()
                volume_weight = min(avg_volume / 1000000, 2.0)  # 거래량 가중치 (최대 2배)
                
                sector_scores[stock_code] = momentum * volume_weight
            
            # 섹터 내 순위 점수 (0~1)
            if sector_scores:
                sorted_stocks = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (stock_code, score) in enumerate(sorted_stocks):
                    relative_scores[stock_code] = 1.0 - (rank / len(sorted_stocks))
        
        return relative_scores
    
    def _calculate_absolute_momentum(self, current_date: datetime.date) -> Dict[str, float]:
        """절대 모멘텀 계산 (시장 대비 성과)"""
        absolute_scores = {}
        
        # 시장 지수 (KOSPI) 대비 성과 계산
        market_code = 'A001'  # KOSPI 지수
        if market_code not in self.data_store['daily']:
            logger.warning("시장 지수 데이터가 없어 절대 모멘텀 계산을 건너뜁니다.")
            return {}
        
        market_df = self.data_store['daily'][market_code]
        if market_df.empty:
            return {}
        
        # 시장 모멘텀 계산
        momentum_period = self.strategy_params['momentum_period']
        market_recent = market_df[market_df.index.date <= current_date].tail(momentum_period + 5)
        if len(market_recent) < momentum_period:
            return {}
        
        market_start = market_recent.iloc[0]['close']
        market_end = market_recent.iloc[-1]['close']
        market_momentum = (market_end - market_start) / market_start
        
        # 개별 종목들의 시장 대비 성과 계산
        for stock_code in self.data_store['daily']:
            if stock_code == market_code:
                continue
                
            df = self.data_store['daily'][stock_code]
            if df.empty or len(df) < momentum_period:
                continue
            
            recent_df = df[df.index.date <= current_date].tail(momentum_period + 5)
            if len(recent_df) < momentum_period:
                continue
            
            stock_start = recent_df.iloc[0]['close']
            stock_end = recent_df.iloc[-1]['close']
            stock_momentum = (stock_end - stock_start) / stock_start
            
            # 시장 대비 초과 성과
            excess_return = stock_momentum - market_momentum
            
            # 절대 모멘텀 점수 (0~1)
            absolute_scores[stock_code] = max(0, min(1, (excess_return + 0.1) * 5))  # 정규화
        
        return absolute_scores
    
    def _combine_momentum_scores(self, relative_scores: Dict[str, float], 
                                absolute_scores: Dict[str, float]) -> Dict[str, float]:
        """모멘텀 점수 결합"""
        combined_scores = {}
        
        # 두 점수의 교집합
        common_stocks = set(relative_scores.keys()) & set(absolute_scores.keys())
        
        for stock_code in common_stocks:
            relative_score = relative_scores[stock_code]
            absolute_score = absolute_scores[stock_code]
            
            # 가중 평균 (상대 모멘텀 60%, 절대 모멘텀 40%)
            combined_score = relative_score * 0.6 + absolute_score * 0.4
            combined_scores[stock_code] = combined_score
        
        return combined_scores
    
    def _select_top_stocks(self, momentum_scores: Dict[str, float]) -> List[str]:
        """상위 종목 선택"""
        if not momentum_scores:
            return []
        
        # 점수 기준 내림차순 정렬
        sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 N개 종목 선택
        num_top_stocks = self.strategy_params['num_top_stocks']
        top_stocks = [stock_code for stock_code, _ in sorted_stocks[:num_top_stocks]]
        
        # 선택된 종목들의 점수 로깅
        for i, (stock_code, score) in enumerate(sorted_stocks[:num_top_stocks]):
            stock_name = self.broker.api_client.get_stock_name(stock_code)
            logger.info(f"매수 후보 {i+1}: {stock_code} ({stock_name}) (점수: {score:.2f})")
        
        return top_stocks
    
    def _generate_trading_signals(self, top_stocks: List[str], 
                                current_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """매매 신호 생성"""
        signals = {}
        
        # 현재 포트폴리오 상태 확인
        current_positions = set(self.broker.positions.keys())
        
        # 매수 신호 생성
        for stock_code in top_stocks:
            if stock_code not in current_positions:
                # 매수 수량 계산
                target_amount = self.broker.cash * 0.15  # 포트폴리오의 15%
                current_price = self._get_current_price(stock_code, current_date)
                
                if current_price > 0:
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
        
        # 매도 신호 생성 (포트폴리오에 있지만 상위 종목이 아닌 경우)
        for stock_code in current_positions:
            if stock_code not in top_stocks and stock_code != self.strategy_params['safe_asset_code']:
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
    
    def _get_sector_stocks(self) -> Dict[str, List[Tuple[str, str]]]:
        """섹터별 종목 정보 반환"""
        return self.broker.stock_selector.sector_stocks_config

