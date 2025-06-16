"""
시장 상황 기반 최적화 클래스
시장 상태에 따라 전략과 파라미터를 최적화하는 클래스
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Type, Optional, Callable
from datetime import date, datetime, timedelta
from .base_optimizer import BaseOptimizer
from .strategy_optimizer import StrategyOptimizer
from itertools import product

class MarketOptimizer(BaseOptimizer):
    def __init__(self, backtester, initial_cash: float = 10_000_000):
        """
        시장 상황 기반 최적화 클래스 초기화
        
        Args:
            backtester: 백테스터 인스턴스
            initial_cash: 초기 자본금
        """
        super().__init__(backtester, initial_cash)
        self.logger = logging.getLogger(__name__)
        self.strategy_optimizer = StrategyOptimizer(backtester, initial_cash)
        
        # 시장 상태 정의
        self.market_states = {
            'bull': {
                'description': '상승장',
                'indicators': {
                    'trend': 'up',
                    'volatility': 'low',
                    'momentum': 'high'
                }
            },
            'bear': {
                'description': '하락장',
                'indicators': {
                    'trend': 'down',
                    'volatility': 'high',
                    'momentum': 'low'
                }
            },
            'sideways': {
                'description': '횡보장',
                'indicators': {
                    'trend': 'neutral',
                    'volatility': 'medium',
                    'momentum': 'medium'
                }
            },
            'volatile': {
                'description': '변동성 장',
                'indicators': {
                    'trend': 'mixed',
                    'volatility': 'very_high',
                    'momentum': 'mixed'
                }
            }
        }
    
    def optimize(
            self,
            daily_strategies: List[Any],
            minute_strategies: List[Any],
            param_grids: Dict[str, Dict[str, List[Any]]],
            start_date: datetime,
            end_date: datetime,
            market_state: Optional[str] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None
        ) -> Optional[Dict[str, Any]]:
        """
        전략 파라미터 최적화 실행
        
        Args:
            daily_strategies: 일봉 전략 리스트
            minute_strategies: 분봉 전략 리스트
            param_grids: 파라미터 그리드
            start_date: 시작일
            end_date: 종료일
            market_state: 시장 상태 (선택)
            progress_callback: 진행률 콜백 함수
            
        Returns:
            최적화 결과 딕셔너리 또는 None
        """
        try:
            # 시장 데이터 로드
            market_data = self._get_market_data('U001', start_date, end_date)
            if market_data is None:
                self.logger.error("시장 데이터 로드 실패")
                return None
            
            # 시장 상태 분석
            if market_state is None:
                market_state = self._analyze_market_state(start_date, end_date)
            
            # 파라미터 그리드 조정
            adjusted_grids = self._adjust_param_grids_for_market_state(param_grids, market_state)
            
            # 총 조합 수 계산
            total_combinations = 1
            for strategy_type in ['daily', 'minute']:
                if strategy_type in adjusted_grids:
                    for param_name, param_values in adjusted_grids[strategy_type].items():
                        total_combinations *= len(param_values)
            
            current_combination = 0
            best_performance = float('-inf')
            best_params = None
            best_metrics = None
            
            # 일봉 전략 최적화
            for daily_strategy in daily_strategies:
                daily_params = adjusted_grids.get('daily', {})
                for params in self._generate_param_combinations(daily_params):
                    daily_strategy.set_parameters(params)
                    performance = self._evaluate_strategy(daily_strategy, market_data, 'daily')
                    
                    # 백테스트 실행 결과에서 metrics를 받아서 best_metrics로 저장
                    portfolio_values, metrics = self.backtester.run(
                        start_date=market_data.index[0],
                        end_date=market_data.index[-1]
                    )
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {'daily': params}
                        best_metrics = metrics
                    
                    current_combination += 1
                    if progress_callback:
                        progress_callback(current_combination, total_combinations)
            
            # 분봉 전략 최적화
            for minute_strategy in minute_strategies:
                minute_params = adjusted_grids.get('minute', {})
                for params in self._generate_param_combinations(minute_params):
                    minute_strategy.set_parameters(params)
                    performance = self._evaluate_strategy(minute_strategy, market_data, 'minute')
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {'minute': params}
                        best_metrics = self._calculate_performance_metrics(minute_strategy)
                    
                    current_combination += 1
                    if progress_callback:
                        progress_callback(current_combination, total_combinations)
            
            if best_params is None:
                self.logger.error("최적 파라미터를 찾을 수 없습니다.")
                return None
            
            return {
                'market_state': market_state,
                'best_params': best_params,
                'performance': best_performance,
                'detailed_metrics': best_metrics
            }
            
        except Exception as e:
            self.logger.error(f"최적화 중 오류 발생: {str(e)}", exc_info=True)
            return None
    
    def _analyze_market_state(self,
                            start_date: date,
                            end_date: date) -> Optional[str]:
        """
        시장 상태 분석
        
        Args:
            start_date: 분석 시작일
            end_date: 분석 종료일
            
        Returns:
            Optional[str]: 시장 상태 ('bull', 'bear', 'sideways', 'volatile') 또는 None
        """
        try:
            # 테스트 환경 확인
            is_test_env = not hasattr(self.backtester, 'data_store') or not hasattr(self.backtester.data_store, 'get_daily_data')
            
            # KOSPI 데이터 가져오기
            kospi_data = self._get_market_data('U001', start_date, end_date)
            
            # 테스트 환경이거나 데이터가 없는 경우
            if is_test_env:
                if isinstance(kospi_data, dict):
                    if 'daily' in kospi_data:
                        kospi_data = kospi_data['daily']
                    elif 'minute' in kospi_data:
                        kospi_data = kospi_data['minute']
                    else:
                        self.logger.info("테스트 환경: 기본 시장 상태 'sideways' 반환")
                        return 'sideways'
            
            if kospi_data is None or (isinstance(kospi_data, pd.DataFrame) and kospi_data.empty):
                self.logger.info("데이터 없음: 기본 시장 상태 'sideways' 반환")
                return 'sideways'
            
            # kospi_data가 DataFrame이고 'close' 컬럼이 있는지 확인
            if not isinstance(kospi_data, pd.DataFrame) or 'close' not in kospi_data.columns:
                self.logger.warning("시장 데이터에 'close' 컬럼이 없거나 DataFrame이 아님: 기본 시장 상태 'sideways' 반환")
                return 'sideways'
            
            # 시장 지표 계산
            trend = self._calculate_trend(kospi_data)
            volatility = self._calculate_volatility(kospi_data)
            momentum = self._calculate_momentum(kospi_data)
            
            if not all([trend, volatility, momentum]):
                self.logger.warning("시장 지표 계산 실패: 기본 시장 상태 'sideways' 반환")
                return 'sideways'
            
            # 시장 상태 판단
            market_state = self._determine_market_state(trend, volatility, momentum)
            self.logger.info(f"시장 지표 - 추세: {trend}, 변동성: {volatility}, 모멘텀: {momentum}")
            self.logger.info(f"판단된 시장 상태: {market_state}")
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"시장 상태 분석 중 오류 발생: {str(e)}")
            return 'sideways'  # 오류 발생 시 기본값 반환
    
    def _get_market_data(self, code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            code: 종목 코드
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            pd.DataFrame: 시장 데이터
        """
        try:
            # 테스트 환경에서는 모의 데이터 생성
            if self.backtester.is_test_mode:
                mock_data = self._generate_mock_market_data(start_date, end_date)
                if mock_data is None:
                    self.logger.error("모의 데이터 생성 실패")
                    return pd.DataFrame()
                return mock_data
            
            # 실제 데이터 조회
            data = self.backtester.get_market_data(code, start_date, end_date)
            if data is None:
                self.logger.error("데이터 조회 실패")
                return pd.DataFrame()
            
            # 딕셔너리를 DataFrame으로 변환
            if isinstance(data, dict):
                try:
                    # 테스트용 데이터 구조 처리 ('daily'/'minute' 키가 있는 경우)
                    if 'daily' in data:
                        if isinstance(data['daily'], pd.DataFrame):
                            return data['daily']  # 일봉 데이터 반환
                        elif isinstance(data['daily'], dict):
                            return pd.DataFrame(data['daily'])  # 딕셔너리를 DataFrame으로 변환
                    elif 'minute' in data:
                        if isinstance(data['minute'], pd.DataFrame):
                            return data['minute']  # 분봉 데이터 반환
                        elif isinstance(data['minute'], dict):
                            return pd.DataFrame(data['minute'])  # 딕셔너리를 DataFrame으로 변환
                    
                    # 일반적인 딕셔너리 처리
                    if all(isinstance(k, (str, datetime, date)) for k in data.keys()):
                        data = pd.DataFrame.from_dict(data, orient='index')
                    else:
                        data = pd.DataFrame(data)
                    
                    # 인덱스가 datetime이 아니면 변환 시도
                    if not isinstance(data.index, pd.DatetimeIndex):
                        try:
                            data.index = pd.to_datetime(data.index)
                        except Exception as e:
                            self.logger.error(f"인덱스 변환 실패: {str(e)}")
                            return pd.DataFrame()
                except Exception as e:
                    self.logger.error(f"DataFrame 변환 실패: {str(e)}")
                    return pd.DataFrame()
            
            # DataFrame 검증
            if not isinstance(data, pd.DataFrame):
                self.logger.error("데이터가 DataFrame이 아닙니다")
                return pd.DataFrame()
            
            if data.empty:
                self.logger.warning("데이터가 비어있습니다")
                return pd.DataFrame()
            
            # 필수 컬럼 확인
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"필수 컬럼 누락: {missing_columns}")
                return pd.DataFrame()
            
            # 데이터 타입 변환
            try:
                for col in required_columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                self.logger.error(f"데이터 타입 변환 실패: {str(e)}")
                return pd.DataFrame()
            
            # 결측치 처리
            data = data.dropna(subset=required_columns)
            if data.empty:
                self.logger.warning("유효한 데이터가 없습니다")
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 조회 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def _generate_mock_market_data(self,
                                 start_date: date,
                                 end_date: date) -> pd.DataFrame:
        """
        테스트용 모의 시장 데이터 생성
        
        Args:
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            pd.DataFrame: 모의 시장 데이터
        """
        try:
            # 날짜 범위 생성
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # 기본 가격 설정 (더 현실적인 초기값)
            base_price = 2500  # KOSPI 수준에 맞춤
            np.random.seed(42)  # 재현성을 위한 시드 설정
            
            # 가격 변동 생성 (더 현실적인 변동성 적용)
            daily_returns = np.random.normal(0.0003, 0.012, len(dates))  # 변동성 조정
            prices = base_price * (1 + daily_returns).cumprod()
            
            # OHLCV 데이터 생성 (더 현실적인 가격 범위 적용)
            daily_volatility = np.abs(np.random.normal(0, 0.008, len(dates)))  # 변동성 조정
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + daily_volatility),
                'low': prices * (1 - daily_volatility),
                'close': prices,
                'volume': np.random.lognormal(13, 0.8, len(dates)) * 1000  # 거래량 조정
            }, index=dates)
            
            # 데이터 검증 및 보정
            data['high'] = data[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, 0.002, len(dates))) * data['close']
            data['low'] = data[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, 0.002, len(dates))) * data['close']
            
            # 데이터 검증
            if data.isnull().any().any():
                self.logger.error("생성된 모의 데이터에 결측치가 있습니다.")
                return None
                
            if (data['high'] < data['low']).any():
                self.logger.error("생성된 모의 데이터에 잘못된 OHLC 값이 있습니다.")
                return None
            
            self.logger.info(f"모의 시장 데이터 생성 완료: {len(data)}일")
            return data
            
        except Exception as e:
            self.logger.error(f"모의 데이터 생성 실패: {str(e)}")
            return None
    
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """
        추세 계산
        
        Args:
            data: 일봉 데이터
            
        Returns:
            str: 추세 ('up', 'down', 'neutral', 'mixed')
        """
        # 20일 이동평균선 계산
        ma20 = data['close'].rolling(window=20).mean()
        
        # 현재가와 이동평균선 비교
        current_price = data['close'].iloc[-1]
        ma20_current = ma20.iloc[-1]
        ma20_prev = ma20.iloc[-2]
        
        # 추세 판단
        if current_price > ma20_current and ma20_current > ma20_prev:
            return 'up'
        elif current_price < ma20_current and ma20_current < ma20_prev:
            return 'down'
        elif abs(current_price - ma20_current) / ma20_current < 0.02:
            return 'neutral'
        else:
            return 'mixed'
    
    def _calculate_volatility(self, data: pd.DataFrame) -> str:
        """
        변동성 계산
        
        Args:
            data: 일봉 데이터
            
        Returns:
            str: 변동성 ('low', 'medium', 'high', 'very_high')
        """
        # 20일 변동성 계산
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # 연간화
        
        current_volatility = volatility.iloc[-1]
        
        # 변동성 판단
        if current_volatility < 0.15:
            return 'low'
        elif current_volatility < 0.25:
            return 'medium'
        elif current_volatility < 0.35:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_momentum(self, data: pd.DataFrame) -> str:
        """
        모멘텀 계산
        
        Args:
            data: 일봉 데이터
            
        Returns:
            str: 모멘텀 ('low', 'medium', 'high', 'mixed')
        """
        # 20일 모멘텀 계산
        momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
        
        # 모멘텀 판단
        if momentum > 5:
            return 'high'
        elif momentum < -5:
            return 'low'
        elif abs(momentum) < 2:
            return 'medium'
        else:
            return 'mixed'
    
    def _determine_market_state(self,
                              trend: str,
                              volatility: str,
                              momentum: str) -> str:
        """
        시장 상태 판단
        
        Args:
            trend: 추세
            volatility: 변동성
            momentum: 모멘텀
            
        Returns:
            str: 시장 상태
        """
        # 변동성이 매우 높은 경우
        if volatility == 'very_high':
            return 'volatile'
            
        # 추세와 모멘텀이 일치하는 경우
        if trend == 'up' and momentum == 'high':
            return 'bull'
        elif trend == 'down' and momentum == 'low':
            return 'bear'
            
        # 추세가 중립적이고 변동성이 중간인 경우
        if trend == 'neutral' and volatility == 'medium':
            return 'sideways'
            
        # 그 외의 경우는 변동성에 따라 판단
        if volatility in ['high', 'very_high']:
            return 'volatile'
        elif trend == 'up':
            return 'bull'
        elif trend == 'down':
            return 'bear'
        else:
            return 'sideways'
    
    def _adjust_param_grids_for_market_state(self,
                                           param_grids: Dict[str, Dict[str, List[Any]]],
                                           market_state: str) -> Dict[str, Dict[str, List[Any]]]:
        """
        시장 상태에 따른 파라미터 그리드 조정
        
        Args:
            param_grids: 원본 파라미터 그리드
            market_state: 시장 상태
            
        Returns:
            Dict[str, Dict[str, List[Any]]]: 조정된 파라미터 그리드
        """
        adjusted_grids = param_grids.copy()
        
        # 시장 상태별 파라미터 조정 규칙
        market_state_rules = {
            'bull': {
                'DualMomentumDaily': {
                    'momentum_period': [10, 12, 15],  # 단기 모멘텀 선호
                    'num_top_stocks': [5, 7, 10]      # 더 많은 종목 선택
                },
                'RSIMinute': {
                    'minute_rsi_period': [30, 45],    # 단기 RSI 선호
                    'minute_rsi_oversold': [25, 30],  # 더 공격적인 매수
                    'minute_rsi_overbought': [70, 75] # 더 공격적인 매도
                },
                'MockDailyStrategy': {
                    'window': [5, 10],                # 단기 이동평균 선호
                    'threshold': [0.01, 0.015],       # 더 공격적인 진입
                    'momentum_period': [10, 15],      # 단기 모멘텀 선호
                    'volume_ma': [5, 10]             # 단기 거래량 선호
                },
                'MockMinuteStrategy': {
                    'window': [10, 20],               # 단기 이동평균 선호
                    'threshold': [0.005, 0.01],       # 더 공격적인 진입
                    'rsi_period': [14, 21],           # 단기 RSI 선호
                    'volume_ma': [5, 10]             # 단기 거래량 선호
                }
            },
            'bear': {
                'DualMomentumDaily': {
                    'momentum_period': [20, 25, 30],  # 장기 모멘텀 선호
                    'num_top_stocks': [3, 5, 7]       # 더 적은 종목 선택
                },
                'RSIMinute': {
                    'minute_rsi_period': [45, 60],    # 장기 RSI 선호
                    'minute_rsi_oversold': [20, 25],  # 더 보수적인 매수
                    'minute_rsi_overbought': [75, 80] # 더 보수적인 매도
                },
                'MockDailyStrategy': {
                    'window': [20, 30],               # 장기 이동평균 선호
                    'threshold': [0.02, 0.025],       # 더 보수적인 진입
                    'momentum_period': [25, 30],      # 장기 모멘텀 선호
                    'volume_ma': [15, 20]            # 장기 거래량 선호
                },
                'MockMinuteStrategy': {
                    'window': [30, 40],               # 장기 이동평균 선호
                    'threshold': [0.015, 0.02],       # 더 보수적인 진입
                    'rsi_period': [28, 42],           # 장기 RSI 선호
                    'volume_ma': [15, 20]            # 장기 거래량 선호
                }
            },
            'sideways': {
                'DualMomentumDaily': {
                    'momentum_period': [15, 20, 25],  # 중기 모멘텀 선호
                    'num_top_stocks': [5, 7, 10]      # 중립적인 종목 수
                },
                'RSIMinute': {
                    'minute_rsi_period': [45, 60],    # 중기 RSI 선호
                    'minute_rsi_oversold': [30, 35],  # 중립적인 매수
                    'minute_rsi_overbought': [65, 70] # 중립적인 매도
                },
                'MockDailyStrategy': {
                    'window': [10, 20],               # 중기 이동평균 선호
                    'threshold': [0.015, 0.02],       # 중립적인 진입
                    'momentum_period': [15, 25],      # 중기 모멘텀 선호
                    'volume_ma': [10, 15]            # 중기 거래량 선호
                },
                'MockMinuteStrategy': {
                    'window': [20, 30],               # 중기 이동평균 선호
                    'threshold': [0.01, 0.015],       # 중립적인 진입
                    'rsi_period': [21, 28],           # 중기 RSI 선호
                    'volume_ma': [10, 15]            # 중기 거래량 선호
                }
            },
            'volatile': {
                'DualMomentumDaily': {
                    'momentum_period': [10, 15, 20],  # 단기 모멘텀 선호
                    'num_top_stocks': [3, 5, 7]       # 적은 종목 선택
                },
                'RSIMinute': {
                    'minute_rsi_period': [30, 45],    # 단기 RSI 선호
                    'minute_rsi_oversold': [20, 25],  # 보수적인 매수
                    'minute_rsi_overbought': [75, 80] # 보수적인 매도
                },
                'MockDailyStrategy': {
                    'window': [5, 10],                # 단기 이동평균 선호
                    'threshold': [0.02, 0.025],       # 보수적인 진입
                    'momentum_period': [10, 15],      # 단기 모멘텀 선호
                    'volume_ma': [5, 10]             # 단기 거래량 선호
                },
                'MockMinuteStrategy': {
                    'window': [10, 20],               # 단기 이동평균 선호
                    'threshold': [0.015, 0.02],       # 보수적인 진입
                    'rsi_period': [14, 21],           # 단기 RSI 선호
                    'volume_ma': [5, 10]             # 단기 거래량 선호
                }
            }
        }
        
        # 시장 상태에 따른 파라미터 조정
        if market_state in market_state_rules:
            for strategy, params in market_state_rules[market_state].items():
                if strategy in adjusted_grids:
                    # 기존 파라미터와 새로운 파라미터 병합
                    adjusted_grids[strategy] = {
                        **adjusted_grids[strategy],
                        **params
                    }
        
        return adjusted_grids 

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        파라미터 그리드에서 가능한 모든 조합을 생성
        """
        if not param_grid:
            return [{}]
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        return [dict(zip(keys, v)) for v in product(*values)] 

    def _evaluate_strategy(self, strategy, market_data, data_type):
        """
        전략의 성능을 평가합니다.
        
        Args:
            strategy: 평가할 전략 인스턴스
            market_data: 시장 데이터
            data_type: 데이터 타입 ('daily' 또는 'minute')
            
        Returns:
            float: 평가 점수 (예: 수익률)
        """
        try:
            # 백테스터에 전략 설정
            if data_type == 'daily':
                self.backtester.set_strategies(daily_strategy=strategy)
            else:
                self.backtester.set_strategies(minute_strategy=strategy)
            
            # 백테스트 실행
            portfolio_values, metrics = self.backtester.run(
                start_date=market_data.index[0],
                end_date=market_data.index[-1]
            )
            
            # 수익률을 평가 점수로 사용
            return metrics['total_return']
            
        except Exception as e:
            self.logger.error(f"전략 평가 중 오류 발생: {str(e)}")
            return float("-inf") 