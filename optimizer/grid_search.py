"""
그리드 서치 최적화 클래스
파라미터 그리드를 기반으로 최적의 파라미터 조합을 찾는 클래스
"""

import itertools
import logging
from typing import Dict, Any, List, Tuple, Type, Optional
from datetime import date
from .base_optimizer import BaseOptimizer
import numpy as np

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, backtester, initial_cash: float = 10_000_000):
        """
        그리드 서치 최적화 클래스 초기화
        
        Args:
            backtester: 백테스터 인스턴스
            initial_cash: 초기 자본금
        """
        super().__init__(backtester, initial_cash)
        self.logger = logging.getLogger(__name__)
        
    def optimize(self,
                daily_strategy: Type,
                minute_strategy: Type,
                param_grid: Dict[str, Dict[str, List[Any]]],
                start_date: date,
                end_date: date) -> Optional[Dict[str, Any]]:
        """
        그리드 서치 최적화 실행
        
        Args:
            daily_strategy: 일봉 전략 클래스
            minute_strategy: 분봉 전략 클래스
            param_grid: 파라미터 그리드
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            
        Returns:
            Optional[Dict[str, Any]]: 최적화 결과
            
        Raises:
            ValueError: 파라미터 그리드가 유효하지 않은 경우
            TypeError: 전략 클래스가 유효하지 않은 경우
        """
        self.logger.info("그리드 서치 최적화를 시작합니다.")
        
        try:
            # 입력값 검증
            if not isinstance(param_grid, dict):
                raise ValueError("파라미터 그리드는 딕셔너리여야 합니다.")
            
            if 'daily_params' not in param_grid or 'minute_params' not in param_grid:
                raise ValueError("파라미터 그리드에 'daily_params'와 'minute_params'가 필요합니다.")
            
            if not isinstance(daily_strategy, type) or not isinstance(minute_strategy, type):
                raise TypeError("전략 클래스가 유효하지 않습니다.")
            
            # 필수 메서드 확인
            required_methods = ['run_daily_logic', 'run_minute_logic', 'set_parameters']
            for strategy_class in [daily_strategy, minute_strategy]:
                for method in required_methods:
                    if not hasattr(strategy_class, method):
                        raise TypeError(f"전략 클래스에 필수 메서드 '{method}'가 없습니다.")
            
            # 파라미터 조합 생성
            daily_param_combinations = self._generate_param_combinations(param_grid['daily_params'])
            minute_param_combinations = self._generate_param_combinations(param_grid['minute_params'])
            
            if not daily_param_combinations or not minute_param_combinations:
                raise ValueError("유효한 파라미터 조합이 없습니다.")
            
            self.logger.info(f"총 {len(daily_param_combinations) * len(minute_param_combinations)}개의 파라미터 조합을 테스트합니다.")
            
            # 최적화 결과 저장
            best_metrics = float('-inf')
            best_result = None
            total_tests = len(daily_param_combinations) * len(minute_param_combinations)
            successful_tests = 0
            failed_tests = 0
            
            # 각 파라미터 조합 테스트
            for i, daily_params in enumerate(daily_param_combinations, 1):
                for j, minute_params in enumerate(minute_param_combinations, 1):
                    test_num = (i - 1) * len(minute_param_combinations) + j
                    self.logger.info(f"테스트 {test_num}/{total_tests}")
                    
                    try:
                        # 파라미터 검증
                        if not self._validate_params(daily_params, minute_params):
                            self.logger.warning("파라미터 조합이 유효하지 않습니다.")
                            failed_tests += 1
                            continue
                        
                        # 전략 인스턴스 생성
                        daily_strategy_instance = daily_strategy(params=daily_params)
                        minute_strategy_instance = minute_strategy(params=minute_params)
                        
                        # 백테스트 실행
                        portfolio_value, metrics = self.backtester.run(
                            daily_strategy=daily_strategy_instance,
                            minute_strategy=minute_strategy_instance,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if metrics is None:
                            self.logger.warning("백테스트 실패")
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
                raise ValueError("최적화 실패: 유효한 결과가 없습니다.")
            
            # 최적화 결과 로깅
            self.logger.info("최적 파라미터 조합을 찾았습니다:")
            self.logger.info(f"일봉 전략 파라미터: {best_result['best_params']['daily_params']}")
            self.logger.info(f"분봉 전략 파라미터: {best_result['best_params']['minute_params']}")
            self.logger.info(f"최고 성과 지표: {best_metrics:.4f}")
            
            # 최종 결과 반환
            best_result.update({
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests
            })
            
            return best_result
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"최적화 실패 (입력값 오류): {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"최적화 중 오류 발생: {str(e)}")
            return None
    
    def _validate_params(self,
                        daily_params: Dict[str, Any],
                        minute_params: Dict[str, Any]) -> bool:
        """
        파라미터 유효성 검증
        
        Args:
            daily_params: 일봉 전략 파라미터
            minute_params: 분봉 전략 파라미터
            
        Returns:
            bool: 파라미터가 유효한지 여부
        """
        try:
            # 파라미터가 딕셔너리인지 확인
            if not isinstance(daily_params, dict) or not isinstance(minute_params, dict):
                self.logger.warning("파라미터는 딕셔너리여야 합니다.")
                return False
            
            # 필수 파라미터 확인 (전략 클래스에 따라 다를 수 있음)
            required_params = {
                'daily': ['window'],  # threshold는 선택적
                'minute': ['window']  # threshold는 선택적
            }
            
            # 일봉 전략 파라미터 검증
            for param in required_params['daily']:
                if param not in daily_params:
                    self.logger.warning(f"일봉 전략에 필수 파라미터 '{param}'가 없습니다.")
                    return False
                if daily_params[param] is None:
                    self.logger.warning(f"일봉 전략 파라미터 '{param}'의 값이 None입니다.")
                    return False
            
            # 분봉 전략 파라미터 검증
            for param in required_params['minute']:
                if param not in minute_params:
                    self.logger.warning(f"분봉 전략에 필수 파라미터 '{param}'가 없습니다.")
                    return False
                if minute_params[param] is None:
                    self.logger.warning(f"분봉 전략 파라미터 '{param}'의 값이 None입니다.")
                    return False
            
            # 파라미터 값 검증 (숫자형 파라미터만)
            for params in [daily_params, minute_params]:
                for param, value in params.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            self.logger.warning(f"파라미터 '{param}'의 값이 유효하지 않습니다.")
                            return False
                        if value <= 0 and param in ['window', 'threshold']:
                            self.logger.warning(f"파라미터 '{param}'는 양수여야 합니다.")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"파라미터 검증 중 오류 발생: {str(e)}")
            return False
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        파라미터 조합 생성
        
        Args:
            param_grid: 파라미터 그리드
            
        Returns:
            List[Dict[str, Any]]: 파라미터 조합 리스트
        """
        if not param_grid:
            return [{}]
            
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []
        
        for value_tuple in self._cartesian_product(*values):
            combinations.append(dict(zip(keys, value_tuple)))
            
        return combinations
    
    def _cartesian_product(self, *args) -> List[Tuple]:
        """
        카르테시안 곱 계산
        
        Args:
            *args: 리스트들
            
        Returns:
            List[Tuple]: 조합 리스트
        """
        if not args:
            return [()]
            
        result = []
        for x in args[0]:
            for y in self._cartesian_product(*args[1:]):
                result.append((x,) + y)
        return result 