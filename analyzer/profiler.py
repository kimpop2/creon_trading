# analyzer/profiler.py

import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StrategyProfiler:
    """
    거래 기록과 장세 데이터를 분석하여 전략의 장세별 성과 프로파일을 생성합니다.
    """
    def generate_profiles(self,
                          trade_logs_df: pd.DataFrame,
                          regime_data_df: pd.DataFrame,
                          model_id: int) -> List[Dict[str, Any]]:
        """
        Args:
            trade_logs_df (pd.DataFrame): 'trade_datetime', 'strategy_name', 'realized_pnl' 등 컬럼 포함
            regime_data_df (pd.DataFrame): 'date', 'regime_id' 컬럼 포함
            model_id (int): 프로파일의 기준이 되는 HMM 모델 ID
        
        Returns:
            List[Dict[str, Any]]: DB에 저장할 프로파일 데이터 딕셔너리의 리스트
        """
        if trade_logs_df.empty or regime_data_df.empty:
            logger.warning("거래 로그 또는 장세 데이터가 비어 있어 프로파일링을 건너뜁니다.")
            return []
            
        # --- 데이터 전처리 ---
        # 날짜 타입 통일 (시간 정보 제거)
        trade_logs_df['date'] = pd.to_datetime(trade_logs_df['trade_datetime']).dt.date
        regime_data_df['date'] = pd.to_datetime(regime_data_df['date']).dt.date

        # 거래 기록에 해당 날짜의 장세 ID 병합
        merged_df = pd.merge(trade_logs_df, regime_data_df, on='date', how='left')
        merged_df.dropna(subset=['regime_id'], inplace=True) # 장세 ID가 없는 거래는 제외
        merged_df['regime_id'] = merged_df['regime_id'].astype(int)

        # --- 장세별/전략별 그룹화 및 성과 지표 계산 ---
        grouped = merged_df.groupby(['strategy_name', 'regime_id'])
        
        profiles = []
        for (strategy, regime), group in grouped:
            # (향후 여기에 샤프지수, MDD 등 복잡한 지표 계산 로직 추가 필요)
            # 현재는 계획서의 기본 지표를 계산합니다.
            realized_pnl = group['realized_pnl'].sum() # 컬럼명을 'realized_pnl'로 가정
            num_trades = len(group)
            win_trades = len(group[group['realized_pnl'] > 0])
            win_rate = win_trades / num_trades if num_trades > 0 else 0.0

            profile = {
                'strategy_name': strategy,
                'model_id': model_id,
                'regime_id': int(regime),
                'sharpe_ratio': 0.0,  # 임시 값, 실제 계산 필요
                'mdd': 0.0,           # 임시 값, 실제 계산 필요
                'total_return': realized_pnl, # 'total_return'은 누적수익률이지만 여기선 실현손익 합계로 대체
                'win_rate': win_rate,
                'num_trades': num_trades,
                'profiling_start_date': merged_df['date'].min().strftime('%Y-%m-%d'),
                'profiling_end_date': merged_df['date'].max().strftime('%Y-%m-%d'),
            }
            profiles.append(profile)
            logger.info(f"프로파일 생성 완료: 전략 '{strategy}', 장세 '{regime}', 거래 수: {num_trades}")

        return profiles