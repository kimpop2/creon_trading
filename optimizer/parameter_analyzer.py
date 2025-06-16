"""
파라미터 분석 클래스
최적화 결과를 분석하고 파라미터의 영향도를 시각화하는 클래스
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Type
import os
from datetime import datetime, date
from .base_optimizer import BaseOptimizer

class ParameterAnalyzer(BaseOptimizer):
    def __init__(self, backtester, initial_cash: float = 10_000_000):
        """
        파라미터 분석 클래스 초기화
        
        Args:
            backtester: 백테스터 인스턴스
            initial_cash: 초기 자본금
        """
        super().__init__(backtester, initial_cash)
        self.logger = logging.getLogger(__name__)
        
    def optimize(self,
                daily_strategies: Dict[str, Type],
                minute_strategies: Dict[str, Type],
                param_grids: Dict[str, Dict[str, List[Any]]],
                start_date: date,
                end_date: date) -> Optional[Dict[str, Any]]:
        """
        파라미터 최적화 실행 (BaseOptimizer 추상 메서드 구현)
        
        Args:
            daily_strategies: 일봉 전략 클래스 딕셔너리
            minute_strategies: 분봉 전략 클래스 딕셔너리
            param_grids: 전략별 파라미터 그리드
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            
        Returns:
            Optional[Dict[str, Any]]: 최적화 결과
        """
        self.logger.info("파라미터 분석을 시작합니다.")
        
        try:
            # 전략 인스턴스 생성
            daily_strategy = self._create_strategy_instance(
                list(daily_strategies.values())[0],
                param_grids.get(list(daily_strategies.keys())[0], {})
            )
            minute_strategy = self._create_strategy_instance(
                list(minute_strategies.values())[0],
                param_grids.get(list(minute_strategies.keys())[0], {})
            )
            
            if daily_strategy is None or minute_strategy is None:
                self.logger.error("전략 인스턴스 생성 실패")
                return None
            
            # 백테스트 실행
            portfolio_value, metrics = self.backtester.run(
                daily_strategy=daily_strategy,
                minute_strategy=minute_strategy,
                start_date=start_date,
                end_date=end_date
            )
            
            if metrics is None:
                return None
            
            # 결과 반환
            return {
                'best_strategy_combination': {
                    'daily': list(daily_strategies.keys())[0],
                    'minute': list(minute_strategies.keys())[0]
                },
                'best_params': {
                    'daily_params': {
                        name: getattr(daily_strategy, name)
                        for name in param_grids.get(list(daily_strategies.keys())[0], {}).keys()
                    },
                    'minute_params': {
                        name: getattr(minute_strategy, name)
                        for name in param_grids.get(list(minute_strategies.keys())[0], {}).keys()
                    }
                },
                'best_metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"최적화 중 오류 발생: {str(e)}")
            return None
    
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
            # 기본 파라미터 추출
            default_params = {}
            for param_name, value in param_grid.items():
                if isinstance(value, list) and value:
                    default_params[param_name] = value[0]
                else:
                    default_params[param_name] = value
            
            # 전략 인스턴스 생성
            strategy = strategy_class(
                data_store=self.backtester.data_store,
                strategy_params=default_params,
                broker=self.backtester.broker
            )
            return strategy
            
        except Exception as e:
            self.logger.error(f"전략 인스턴스 생성 실패: {str(e)}")
            return None
    
    def analyze_parameter_impact(self,
                               results: List[Dict[str, Any]],
                               target_metric: str = 'sharpe_ratio') -> Optional[Dict[str, Dict[str, float]]]:
        """
        파라미터 영향도 분석
        
        Args:
            results: 최적화 결과 리스트
            target_metric: 분석 대상 지표
            
        Returns:
            Optional[Dict[str, Dict[str, float]]]: 파라미터별 영향도
        """
        try:
            if not results:
                self.logger.error("분석할 결과가 없습니다.")
                return None
            
            # 결과 데이터프레임 생성
            analysis_data = []
            for result in results:
                if result is None or 'best_params' not in result:
                    continue
                    
                params = result.get('best_params', {})
                metrics = result.get('best_metrics', {})
                
                if not params or not metrics:
                    continue
                
                # 일봉/분봉 파라미터 분리
                daily_params = params.get('daily_params', {})
                minute_params = params.get('minute_params', {})
                
                # 일봉 전략 파라미터 분석
                for param_name, param_value in daily_params.items():
                    analysis_data.append({
                        'strategy_type': 'daily',
                        'param_name': param_name,
                        'param_value': param_value,
                        'metric_value': metrics.get(target_metric, 0)
                    })
                
                # 분봉 전략 파라미터 분석
                for param_name, param_value in minute_params.items():
                    analysis_data.append({
                        'strategy_type': 'minute',
                        'param_name': param_name,
                        'param_value': param_value,
                        'metric_value': metrics.get(target_metric, 0)
                    })
            
            if not analysis_data:
                self.logger.error("분석할 데이터가 없습니다.")
                return None
            
            # 데이터프레임 변환
            df = pd.DataFrame(analysis_data)
            
            # 파라미터별 영향도 계산
            impact_analysis = {}
            
            # 일봉/분봉 전략별 분석
            for strategy_type in ['daily', 'minute']:
                strategy_df = df[df['strategy_type'] == strategy_type]
                if strategy_df.empty:
                    continue
                
                param_impacts = {}
                for param_name in strategy_df['param_name'].unique():
                    param_df = strategy_df[strategy_df['param_name'] == param_name]
                    
                    # 상관관계 계산
                    if param_df['param_value'].dtype in [np.int64, np.float64]:
                        # 데이터 유효성 검사
                        if len(param_df) < 2:  # 최소 2개의 데이터 포인트 필요
                            param_impacts[param_name] = 0.0
                            continue
                            
                        # 결측치 제거
                        valid_data = param_df.dropna(subset=['param_value', 'metric_value'])
                        if len(valid_data) < 2:
                            param_impacts[param_name] = 0.0
                            continue
                            
                        # 표준편차가 0인 경우 처리
                        if valid_data['param_value'].std() == 0 or valid_data['metric_value'].std() == 0:
                            param_impacts[param_name] = 0.0
                            continue
                            
                        try:
                            correlation = valid_data['param_value'].corr(valid_data['metric_value'])
                            param_impacts[param_name] = correlation if not np.isnan(correlation) else 0.0
                        except Exception as e:
                            self.logger.warning(f"상관관계 계산 중 오류 발생: {str(e)}")
                            param_impacts[param_name] = 0.0
                    else:
                        # 범주형 파라미터의 경우 그룹별 평균 성과 차이 계산
                        group_means = param_df.groupby('param_value')['metric_value'].mean()
                        if len(group_means) > 1:
                            mean_value = group_means.mean()
                            if mean_value != 0:  # 0으로 나누기 방지
                                impact = (group_means.max() - group_means.min()) / mean_value
                                param_impacts[param_name] = impact if not np.isnan(impact) else 0.0
                            else:
                                param_impacts[param_name] = 0.0
                        else:
                            param_impacts[param_name] = 0.0
                
                impact_analysis[strategy_type] = param_impacts
            
            # 분석 결과 시각화
            self._visualize_parameter_impact(impact_analysis, target_metric)
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"파라미터 분석 중 오류 발생: {str(e)}")
            return None
    
    def _visualize_parameter_impact(self,
                                  impact_analysis: Dict[str, Dict[str, float]],
                                  target_metric: str) -> None:
        """
        파라미터 영향도 시각화
        
        Args:
            impact_analysis: 파라미터별 영향도
            target_metric: 분석 대상 지표
        """
        try:
            # 결과 저장 디렉토리 생성
            output_dir = 'optimization_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # 시각화 데이터 준비
            plot_data = []
            for strategy_type, param_impacts in impact_analysis.items():
                for param_name, impact in param_impacts.items():
                    plot_data.append({
                        'strategy_type': strategy_type,
                        'param_name': param_name,
                        'impact': abs(impact)  # 절대값 사용
                    })
            
            if not plot_data:
                return
            
            # 데이터프레임 변환
            df = pd.DataFrame(plot_data)
            
            # 시각화
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='param_name', y='impact', hue='strategy_type')
            plt.title(f'Parameter Impact on {target_metric}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(output_dir, f'parameter_impact_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {str(e)}")
    
    def generate_analysis_report(self,
                               results: List[Dict[str, Any]],
                               save_dir: str = 'analysis_results') -> Optional[str]:
        """
        분석 보고서 생성
        
        Args:
            results: 최적화 결과 리스트
            save_dir: 저장 디렉토리
            
        Returns:
            Optional[str]: 보고서 파일 경로
        """
        if not results:
            self.logger.warning("분석할 결과가 없습니다.")
            return None
        
        try:
            # 저장 디렉토리 생성
            os.makedirs(save_dir, exist_ok=True)
            
            # 데이터프레임 생성
            df = self._prepare_analysis_dataframe(results)
            if df is None or df.empty:
                return None
            
            # 파라미터 영향도 분석
            impact_analysis = self.analyze_parameter_impact(
                results,
                target_metric='sharpe_ratio',
                save_dir=save_dir
            )
            
            if impact_analysis is None:
                return None
            
            # 파라미터 분포 시각화
            self._plot_parameter_distribution(df, save_dir)
            
            # 성과 지표 분포 시각화
            self._plot_metric_distribution(df, save_dir)
            
            # 보고서 파일 경로 반환
            report_path = os.path.join(save_dir, 'analysis_report.txt')
            
            # 보고서 작성
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== 파라미터 최적화 분석 보고서 ===\n\n")
                f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 기본 통계
                f.write("1. 기본 통계\n")
                f.write(f"총 테스트 수: {len(results)}\n")
                f.write(f"성공한 테스트 수: {len(df)}\n\n")
                
                # 파라미터 영향도
                f.write("2. 파라미터 영향도\n")
                for param_type, params in impact_analysis.items():
                    f.write(f"\n{param_type}:\n")
                    for param_name, impact in params.items():
                        f.write(f"- {param_name}: {impact:.4f}\n")
                
                # 성과 지표 통계
                f.write("\n3. 성과 지표 통계\n")
                metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
                for metric in metrics:
                    if metric in df.columns:
                        f.write(f"\n{metric}:\n")
                        f.write(f"- 평균: {df[metric].mean():.4f}\n")
                        f.write(f"- 최대: {df[metric].max():.4f}\n")
                        f.write(f"- 최소: {df[metric].min():.4f}\n")
                        f.write(f"- 표준편차: {df[metric].std():.4f}\n")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"분석 보고서 생성 중 오류 발생: {str(e)}")
            return None
    
    def _plot_parameter_distribution(self, df: pd.DataFrame, save_dir: str) -> None:
        """
        파라미터 분포 시각화
        
        Args:
            df: 분석 데이터프레임
            save_dir: 저장 디렉토리
        """
        try:
            # 파라미터 컬럼 선택
            param_columns = [col for col in df.columns if col.endswith(('period', 'stocks', 'oversold'))]
            
            if not param_columns:
                return
            
            # 서브플롯 생성
            n_cols = min(3, len(param_columns))
            n_rows = (len(param_columns) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(param_columns, 1):
                plt.subplot(n_rows, n_cols, i)
                
                # 히스토그램 생성
                sns.histplot(data=df, x=col, bins=20)
                plt.title(f'{col} 분포')
                plt.xlabel(col)
                plt.ylabel('빈도')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'parameter_distribution.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"파라미터 분포 시각화 중 오류 발생: {str(e)}")
    
    def _plot_metric_distribution(self, df: pd.DataFrame, save_dir: str) -> None:
        """
        성과 지표 분포 시각화
        
        Args:
            df: 분석 데이터프레임
            save_dir: 저장 디렉토리
        """
        try:
            # 성과 지표 컬럼 선택
            metric_columns = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
            metric_columns = [col for col in metric_columns if col in df.columns]
            
            if not metric_columns:
                return
            
            # 서브플롯 생성
            n_cols = min(2, len(metric_columns))
            n_rows = (len(metric_columns) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(12, 5 * n_rows))
            
            for i, col in enumerate(metric_columns, 1):
                plt.subplot(n_rows, n_cols, i)
                
                # 박스플롯 생성
                sns.boxplot(data=df, y=col)
                plt.title(f'{col} 분포')
                plt.ylabel(col)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'metric_distribution.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"성과 지표 분포 시각화 중 오류 발생: {str(e)}") 