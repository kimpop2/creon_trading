"""
기본 최적화 클래스
모든 최적화 클래스의 기본이 되는 추상 클래스
"""

import abc
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, date

class BaseOptimizer(abc.ABC):
    def __init__(self, backtester, initial_cash: float = 10_000_000):
        """
        기본 최적화 클래스 초기화
        
        Args:
            backtester: 백테스터 인스턴스
            initial_cash: 초기 자본금
        """
        self.backtester = backtester
        self.initial_cash = initial_cash
        self.logger = logging.getLogger(__name__)
        
        # 최적화 결과 저장
        self._results = []
        self.best_result = None
        self.target_metric = 'sharpe_ratio'  # 기본값 설정
        
    @property
    def results(self) -> List[Dict[str, Any]]:
        """결과 리스트 반환"""
        return self._results
    
    @results.setter
    def results(self, value: List[Dict[str, Any]]):
        """결과 리스트 설정 및 best_result 업데이트"""
        self._results = value
        if value:
            # target_metric 기준으로 최고 성과 결과 찾기
            self.best_result = max(
                value,
                key=lambda x: x['metrics'][self.target_metric]
            )
        else:
            self.best_result = None
        
    @abc.abstractmethod
    def optimize(self, **kwargs) -> Dict[str, Any]:
        """
        최적화 실행 (추상 메서드)
        자식 클래스에서 구현해야 함
        """
        pass
    
    def evaluate_params(self, 
                       daily_strategy: Any,
                       minute_strategy: Any,
                       params: Dict[str, Any],
                       start_date: date,
                       end_date: date,
                       target_metric: str = None) -> Dict[str, Any]:
        """
        주어진 파라미터로 백테스트 실행 및 결과 평가
        
        Args:
            daily_strategy: 일봉 전략 인스턴스
            minute_strategy: 분봉 전략 인스턴스
            params: 전략 파라미터
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            target_metric: 최적화 대상 지표 (기본값: None, 이 경우 self.target_metric 사용)
            
        Returns:
            Dict[str, Any]: 백테스트 결과 및 평가 지표
        """
        try:
            # target_metric 설정
            if target_metric is not None:
                self.target_metric = target_metric
            
            # 전략 파라미터 설정
            daily_strategy.set_parameters(params.get('daily_params', {}))
            minute_strategy.set_parameters(params.get('minute_params', {}))
            
            # 백테스트 실행
            self.backtester.set_strategies(daily_strategy, minute_strategy)
            portfolio_values, metrics = self.backtester.run(start_date, end_date)
            
            # 결과 저장
            result = {
                'params': params,
                'metrics': metrics,
                'portfolio_values': portfolio_values,
                'timestamp': datetime.now()
            }
            
            # 결과 리스트에 추가
            self.results.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"파라미터 평가 중 오류 발생: {str(e)}")
            return None
    
    def save_results(self, filename: str = None) -> None:
        """
        최적화 결과를 CSV 파일로 저장
        
        Args:
            filename: 저장할 파일명 (기본값: optimizer_results_{timestamp}.csv)
        """
        if not self.results:
            self.logger.warning("저장할 결과가 없습니다.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimizer_results_{timestamp}.csv"
            
        try:
            # 결과를 DataFrame으로 변환
            results_df = pd.DataFrame([
                {
                    **result['params'],
                    **result['metrics']
                }
                for result in self.results
            ])
            
            # CSV 파일로 저장
            results_df.to_csv(filename, index=False)
            self.logger.info(f"최적화 결과가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {str(e)}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        최고 성과 파라미터 반환
        
        Returns:
            Dict[str, Any]: 최고 성과 파라미터
        """
        if self.best_result is None:
            return None
        return self.best_result['params']
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        최고 성과 지표 반환
        
        Returns:
            Dict[str, float]: 최고 성과 지표
        """
        if self.best_result is None:
            return None
        return self.best_result['metrics'] 