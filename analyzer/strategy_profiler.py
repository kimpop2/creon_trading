# analyzer/strategy_profiler.py (전체 파일 교체)

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StrategyProfiler:
    """
    백테스트의 일별 성과와 국면 데이터를 분석하여 전략의 국면별 성과 프로파일을 생성합니다.
    """
    def generate_profiles(self,
                          performance_df: pd.DataFrame,
                          regime_data_df: pd.DataFrame,
                          run_info_df: pd.DataFrame,
                          model_id: int) -> List[Dict[str, Any]]:
        """
        Args:
            performance_df (pd.DataFrame): 'date', 'daily_return', 'run_id' 컬럼 포함
            regime_data_df (pd.DataFrame): 'date', 'regime_id' 컬럼 포함
            run_info_df (pd.DataFrame): 'run_id', 'strategy_daily' 등 전략 이름 정보 포함
            model_id (int): 프로파일의 기준이 되는 HMM 모델 ID
        
        Returns:
            List[Dict[str, Any]]: DB에 저장할 프로파일 데이터 딕셔너리의 리스트
        """
        if performance_df.empty or regime_data_df.empty or run_info_df.empty:
            logger.warning("성과, 국면, 또는 실행 정보 데이터가 비어 있어 프로파일링을 건너뜁니다.")
            return []
            
        performance_df['date'] = pd.to_datetime(performance_df['date']).dt.date
        regime_data_df['date'] = pd.to_datetime(regime_data_df['date']).dt.date

        merged_df = pd.merge(performance_df, regime_data_df, on='date', how='inner')
        merged_df = pd.merge(merged_df, run_info_df[['run_id', 'strategy_daily']], on='run_id', how='left')
        
        merged_df.dropna(subset=['regime_id', 'strategy_daily', 'daily_return'], inplace=True)
        if merged_df.empty:
            logger.warning("병합 후 유효한 데이터가 없어 프로파일링을 중단합니다.")
            return []
            
        merged_df['regime_id'] = merged_df['regime_id'].astype(int)

        grouped = merged_df.groupby(['strategy_daily', 'regime_id'])
        
        profiles = []
        for (strategy, regime), group in grouped:
            returns = group['daily_return']
            
            if np.std(returns) == 0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

            total_return = returns.sum()
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
            num_days = len(returns)

            profile = {
                'strategy_name': strategy,
                'model_id': model_id,
                'regime_id': int(regime),
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'win_rate': win_rate,
                'num_trades': num_days,
                'profiling_start_date': group['date'].min().strftime('%Y-%m-%d'),
                'profiling_end_date': group['date'].max().strftime('%Y-%m-%d'),
            }
            profiles.append(profile)
            logger.info(f"프로파일 생성: 전략 '{strategy}', 국면 '{regime}', 샤프 지수: {sharpe_ratio:.4f}, 거래일 수: {num_days}")

        return profiles