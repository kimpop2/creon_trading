"""
전략 조합 최적화 클래스
일봉/분봉 전략의 최적 조합을 찾고 평가하는 클래스
"""

import logging
from typing import Dict, Any, List, Tuple, Type, Optional
from datetime import date
from .base_optimizer import BaseOptimizer
import numpy as np

class StrategyOptimizer(BaseOptimizer):
    def __init__(self, backtester, initial_cash: float = 10_000_000):
        """
        전략 조합 최적화 클래스 초기화
        
        Args:
            backtester: 백테스터 인스턴스
            initial_cash: 초기 자본금
        """
        super().__init__(backtester, initial_cash)
        self.logger = logging.getLogger(__name__)
        self.available_strategies = {}
    
    def optimize(self,
                daily_strategies: Dict[str, Type],
                minute_strategies: Dict[str, Type],
                param_grids: Dict[str, Dict[str, List[Any]]],
                start_date: date,
                end_date: date) -> Optional[Dict[str, Any]]:
        """
        전략 조합 최적화 실행
        
        Args:
            daily_strategies: 일봉 전략 클래스 딕셔너리
            minute_strategies: 분봉 전략 클래스 딕셔너리
            param_grids: 전략별 파라미터 그리드
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            
        Returns:
            Optional[Dict[str, Any]]: 최적 전략 조합 및 성과 지표
        """
        self.logger.info("전략 조합 최적화를 시작합니다.")
        
        try:
            # 전략 클래스 등록
            self._inject_strategy_classes(daily_strategies, minute_strategies)
            
            # 전략 조합 생성
            strategy_combinations = self._generate_strategy_combinations(
                daily_strategies,
                minute_strategies
            )
            
            if not strategy_combinations:
                self.logger.error("유효한 전략 조합이 없습니다.")
                return None
            
            self.logger.info(f"총 {len(strategy_combinations)}개의 전략 조합을 테스트합니다.")
            
            # 최적화 결과 저장
            best_metrics = float('-inf')
            best_result = None
            total_tests = 0
            successful_tests = 0
            failed_tests = 0
            
            # 파라미터 조합 생성
            daily_param_combinations = self._generate_parameter_combinations(param_grids['daily'])
            minute_param_combinations = self._generate_parameter_combinations(param_grids['minute'])
            
            total_tests = len(strategy_combinations) * len(daily_param_combinations) * len(minute_param_combinations)
            self.logger.info(f"총 {total_tests}개의 파라미터 조합을 테스트합니다.")
            
            # 각 전략 조합과 파라미터 조합 테스트
            for daily_name, minute_name in strategy_combinations:
                for daily_params in daily_param_combinations:
                    for minute_params in minute_param_combinations:
                        self.logger.info(f"\n테스트 {successful_tests + failed_tests + 1}/{total_tests}:")
                        self.logger.info(f"전략: {daily_name} + {minute_name}")
                        self.logger.info(f"일봉 파라미터: {daily_params}")
                        self.logger.info(f"분봉 파라미터: {minute_params}")
                        
                        try:
                            # 전략 인스턴스 생성 (실제 파라미터 적용)
                            daily_strategy = self._create_strategy_instance(
                                daily_strategies[daily_name],
                                daily_params
                            )
                            minute_strategy = self._create_strategy_instance(
                                minute_strategies[minute_name],
                                minute_params
                            )
                            
                            if daily_strategy is None or minute_strategy is None:
                                self.logger.warning(f"전략 인스턴스 생성 실패: {daily_name} + {minute_name}")
                                failed_tests += 1
                                continue
                            
                            # 전략 설정
                            self.backtester.set_strategies(
                                daily_strategy=daily_strategy,
                                minute_strategy=minute_strategy
                            )
                            
                            # 백테스트 실행
                            portfolio_value, metrics = self.backtester.run(
                                start_date=start_date,
                                end_date=end_date
                            )
                            
                            if metrics is None:
                                self.logger.warning(f"백테스트 실패: {daily_name} + {minute_name}")
                                failed_tests += 1
                                continue
                            
                            # 성과 지표 확인
                            sharpe_ratio = metrics.get('sharpe_ratio', 0)
                            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                                self.logger.warning(f"유효하지 않은 성과 지표: {sharpe_ratio}")
                                failed_tests += 1
                                continue
                            
                            self.logger.info(f"성과 지표: {sharpe_ratio:.4f}")
                            
                            # 최고 성과 업데이트
                            if sharpe_ratio > best_metrics:
                                best_metrics = sharpe_ratio
                                best_result = {
                                    'best_combination': {
                                        'daily': daily_name,
                                        'minute': minute_name
                                    },
                                    'best_params': {
                                        'daily_params': daily_params,
                                        'minute_params': minute_params
                                    },
                                    'best_metrics': metrics
                                }
                            
                            successful_tests += 1
                            
                        except Exception as e:
                            self.logger.error(f"파라미터 평가 중 오류 발생: {str(e)}")
                            failed_tests += 1
                            continue
            
            # 최적화 결과 요약
            self.logger.info(f"최적화 완료: 총 {total_tests}개 테스트 중 {successful_tests}개 성공, {failed_tests}개 실패")
            
            if best_result is None:
                self.logger.error("최적화 실패: 유효한 결과가 없습니다.")
                return {
                    'best_combination': None,
                    'best_params': None,
                    'best_metrics': None,
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests
                }
            
            # 최적화 결과 로깅
            self.logger.info("최적 전략 조합을 찾았습니다:")
            self.logger.info(f"일봉 전략: {best_result['best_combination']['daily']}")
            self.logger.info(f"분봉 전략: {best_result['best_combination']['minute']}")
            self.logger.info(f"최고 성과 지표: {best_metrics:.4f}")
            
            # 최종 결과 반환
            best_result.update({
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests
            })
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"최적화 중 오류 발생: {str(e)}")
            return None
    
    def _inject_strategy_classes(self,
                               daily_strategies: Dict[str, Type],
                               minute_strategies: Dict[str, Type]) -> None:
        """
        전략 클래스 등록
        
        Args:
            daily_strategies: 일봉 전략 클래스 딕셔너리
            minute_strategies: 분봉 전략 클래스 딕셔너리
        """
        self.available_strategies.clear()
        self.available_strategies.update({
            'daily': daily_strategies,
            'minute': minute_strategies
        })
    
    def _generate_strategy_combinations(self,
                                     daily_strategies: Dict[str, Type],
                                     minute_strategies: Dict[str, Type]) -> List[Tuple[str, str]]:
        """
        전략 조합 생성
        
        Args:
            daily_strategies: 일봉 전략 클래스 딕셔너리
            minute_strategies: 분봉 전략 클래스 딕셔너리
            
        Returns:
            List[Tuple[str, str]]: 전략 조합 리스트
        """
        combinations = []
        for daily_name in daily_strategies:
            for minute_name in minute_strategies:
                combinations.append((daily_name, minute_name))
        return combinations
    
    def _create_strategy_instance(self,
                                strategy_class: Type,
                                param_grid: Dict[str, Any]) -> Optional[Any]:
        """
        전략 인스턴스 생성
        
        Args:
            strategy_class: 전략 클래스
            param_grid: 파라미터 그리드 또는 파라미터 딕셔너리
            
        Returns:
            Optional[Any]: 전략 인스턴스
        """
        try:
            # 전략 인스턴스 생성 시 strategy_params로 파라미터 전달
            strategy = strategy_class(
                data_store=self.backtester.data_store,
                strategy_params=param_grid,
                broker=self.backtester.broker
            )
            return strategy
            
        except Exception as e:
            self.logger.error(f"전략 인스턴스 생성 중 오류 발생: {str(e)}")
            return None
    
    def _generate_parameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        파라미터 그리드에서 가능한 모든 조합을 생성합니다.
        
        Args:
            param_grid: 파라미터 그리드 (파라미터 이름: 가능한 값들의 리스트)
            
        Returns:
            List[Dict[str, Any]]: 파라미터 조합 리스트
        """
        import itertools
        
        # 파라미터 이름과 값 리스트 분리
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # 모든 가능한 조합 생성
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations 