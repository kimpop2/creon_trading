# analyzer/analyze_hmm_results.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# --- 프로젝트 경로 설정 및 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from manager.db_manager import DBManager
from config.settings import LIVE_HMM_MODEL_NAME
from pykrx import stock

# --- ▼ [핵심 수정] 한글 폰트 설정 ---
# 운영체제에 따라 다른 한글 폰트를 설정합니다.
if sys.platform == "win32":
    plt.rc('font', family='Malgun Gothic')
elif sys.platform == "darwin": # Mac OS
    plt.rc('font', family='AppleGothic')
else: # Linux
    # Linux에서는 나눔고딕 등 별도 폰트 설치가 필요할 수 있습니다.
    # plt.rc('font', family='NanumGothic')
    pass

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
# --- ▲ [핵심 수정] ---


def analyze_and_visualize_hmm():
    """ HMM 모델의 학습 결과를 분석하고 시각화합니다. """
    print("HMM 모델 결과 분석을 시작합니다...")
    
    db_manager = DBManager()
    LIVE_HMM_MODEL_NAME = 'wf_model_202508'
    model_info = db_manager.fetch_hmm_model_by_name(LIVE_HMM_MODEL_NAME)
    if not model_info:
        print(f"'{LIVE_HMM_MODEL_NAME}' 모델을 DB에서 찾을 수 없습니다.")
        return
    model_id = model_info['model_id']
    
    regime_df = db_manager.fetch_daily_regimes(model_id)
    if regime_df.empty:
        print(f"모델 ID {model_id}에 대한 국면 데이터가 없습니다.")
        return
        
    start_date = regime_df['date'].min()
    end_date = regime_df['date'].max()

    kospi_df = stock.get_index_ohlcv(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), "1001")
    kospi_df.index = pd.to_datetime(kospi_df.index)
    
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    analysis_df = pd.merge(kospi_df, regime_df, left_index=True, right_on='date', how='inner')
    
    print("\n--- 국면별 통계 분석 ---")
    regime_labels = {
        0: '국면 0 (횡보)',
        1: '국면 1 (급등)',
        2: '국면 2 (상승)',
        3: '국면 3 (하락)'
    }
    # ... (통계 분석 코드는 이전과 동일) ...
    for regime_id in sorted(analysis_df['regime_id'].unique()):
        regime_data = analysis_df[analysis_df['regime_id'] == regime_id]
        daily_return = regime_data['종가'].pct_change()
        
        print(f"\n[국면 {regime_labels[regime_id]}]")
        print(f"  - 기간: {regime_data['date'].min().date()} ~ {regime_data['date'].max().date()}")
        print(f"  - 총 거래일 수: {len(regime_data)}일")
        print(f"  - 평균 일일 수익률: {daily_return.mean():.2%}")
        print(f"  - 일일 수익률 표준편차: {daily_return.std():.2%}")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(analysis_df['date'], analysis_df['종가'], label='KOSPI 종가', color='black', linewidth=0.8)
    
    # --- ▼ [핵심 수정] 색상 및 범례 텍스트 변경 ---
    colors = ['lightgreen', 'salmon', 'orange','skyblue']

    # --- ▲ [핵심 수정] ---

    for regime_id in sorted(analysis_df['regime_id'].unique()):
        color = colors[regime_id % len(colors)]
        ax.fill_between(analysis_df['date'], 0, analysis_df['종가'].max()*1.1, 
                        where=analysis_df['regime_id'] == regime_id, 
                        facecolor=color, alpha=0.5, label=regime_labels.get(regime_id, f'Regime {regime_id}'))

    ax.set_title(f'HMM 시장 국면 분석 vs KOSPI ({start_date} ~ {end_date})')
    ax.set_ylabel('KOSPI 지수')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    output_filename = "hmm_regime_analysis.png"
    plt.savefig(output_filename)
    print(f"\n분석 결과 차트를 '{output_filename}' 파일로 저장했습니다.")
    
    db_manager.close()

if __name__ == "__main__":
    analyze_and_visualize_hmm()