# analyzer/strategy_profiler.py (전체 파일 교체)

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import json # [추가] 파라미터를 JSON 문자열로 변환하기 위해 임포트

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
        [수정됨] mdd와 params_json을 포함한 완전한 프로파일을 생성합니다.
        """
        if performance_df.empty or regime_data_df.empty or run_info_df.empty:
            logger.warning("성과, 국면, 또는 실행 정보 데이터가 비어 있어 프로파일링을 건너뜁니다.")
            return []
            
        performance_df['date'] = pd.to_datetime(performance_df['date']).dt.date
        regime_data_df['date'] = pd.to_datetime(regime_data_df['date']).dt.date

        # run_info_df에서 params_json_daily를 미리 파싱해 둡니다.
        # db_manager에서 이미 dict로 변환했을 수 있으므로 확인 후 처리
        if 'params_json_daily' in run_info_df.columns:
            def safe_json_loads(s):
                if isinstance(s, str):
                    return json.loads(s)
                return s # 이미 dict이면 그대로 반환
            run_info_df['params_dict'] = run_info_df['params_json_daily'].apply(safe_json_loads)

        merged_df = pd.merge(performance_df, regime_data_df, on='date', how='inner')
        merged_df = pd.merge(merged_df, run_info_df[['run_id', 'strategy_daily', 'params_dict']], on='run_id', how='left')
        
        merged_df.dropna(subset=['regime_id', 'strategy_daily', 'daily_return'], inplace=True)
        if merged_df.empty:
            logger.warning("병합 후 유효한 데이터가 없어 프로파일링을 중단합니다.")
            return []
            
        merged_df['regime_id'] = merged_df['regime_id'].astype(int)

        grouped = merged_df.groupby(['strategy_daily', 'regime_id'])
        
        profiles = []
        for (strategy, regime), group in grouped:
            returns = group['daily_return']
            
            # --- [추가] MDD 계산 로직 ---
            if not returns.empty:
                cumulative_returns = (1 + returns).cumprod()
                peak = cumulative_returns.expanding(min_periods=1).max()
                drawdown = (cumulative_returns - peak) / peak
                mdd = drawdown.min()
            else:
                mdd = 0.0
            # --- MDD 계산 끝 ---
            
            if np.std(returns) == 0:
                sharpe_ratio = 0.0
            else:
                # 연환산 샤프 지수
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

            total_return = (1 + returns).prod() - 1 # 복리 수익률로 계산
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
            num_days = len(returns)

            # --- [추가] 파라미터 정보 추출 로직 ---
            params_dict = group['params_dict'].iloc[0] if not group['params_dict'].empty else {}
            params_json_str = json.dumps(params_dict)
            # --- 파라미터 추출 끝 ---

            profile = {
                'strategy_name': strategy,
                'model_id': model_id,
                'regime_id': int(regime),
                'sharpe_ratio': sharpe_ratio,
                'mdd': mdd, # [추가]
                'total_return': total_return,
                'win_rate': win_rate,
                'num_trades': num_days, # 여기서는 거래일 수를 거래 수로 간주
                'params_json': params_json_str, # [추가]
                'profiling_start_date': group['date'].min().strftime('%Y-%m-%d'),
                'profiling_end_date': group['date'].max().strftime('%Y-%m-%d'),
            }
            profiles.append(profile)
            logger.info(f"프로파일 생성: 전략 '{strategy}', 국면 '{regime}', 샤프 지수: {sharpe_ratio:.4f}, MDD: {mdd:.4f}")

        return profiles