# analyzer/hmm_results.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import logging # 로깅 모듈 임포트

# --- 1. 프로젝트 경로 설정 및 로거 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 한글 폰트 및 차트 설정 ---
def setup_korean_font():
    """운영체제에 맞는 한글 폰트를 설정합니다."""
    if sys.platform == "win32":
        plt.rc('font', family='Malgun Gothic')
    elif sys.platform == "darwin":
        plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
    
# --- 3. 국면 라벨 및 색상 생성 함수 (이전과 동일) ---
def generate_regime_details(analysis_data: pd.DataFrame) -> dict:
    """4-분면 정의에 따라 각 국면의 '라벨'과 시각화를 위한 '색상'을 함께 생성합니다."""
    # ... (이전 답변의 최종 코드를 그대로 사용) ...
    df = analysis_data.copy()
    if 'daily_return' not in df.columns:
        df['daily_return'] = df['종가'].pct_change()
    df.dropna(inplace=True)
    if df.empty: return {}

    regime_stats = df.groupby('regime_id')['daily_return'].agg(['mean', 'std']).reset_index()
    vol_median = regime_stats['std'].median()
    
    temp_labels = {}
    for _, row in regime_stats.iterrows():
        regime_num = int(row['regime_id'])
        vol_label = "고변동성" if row['std'] > vol_median else "저변동성"
        trend_label = "강세장" if row['mean'] > 0 else "약세장"
        temp_labels[regime_num] = f"{vol_label} {trend_label}"

    final_results = {}
    label_counts = pd.Series(temp_labels).value_counts()
    processed_counts = {label: 0 for label in label_counts[label_counts > 1].index}

    for regime_num in sorted(temp_labels.keys()):
        base_label = temp_labels[regime_num]
        type_num = 0
        final_label = f"국면 {regime_num} ({base_label})"

        if base_label in processed_counts:
            processed_counts[base_label] += 1
            type_num = processed_counts[base_label]
            final_label = f"국면 {regime_num} ({base_label} Type {type_num})"
        
        color = 'lightgray'
        if base_label == "고변동성 강세장":
            color = ['orange', 'darkorange'][type_num -1] if type_num > 0 else 'orange'
        elif base_label == "저변동성 강세장":
            color = ['salmon', 'lightcoral'][type_num -1] if type_num > 0 else 'salmon'
        elif base_label == "고변동성 약세장":
            color = ['skyblue', 'steelblue'][type_num -1] if type_num > 0 else 'skyblue'
        elif base_label == "저변동성 약세장":
            color = ['lightcyan', 'paleturquoise'][type_num -1] if type_num > 0 else 'lightcyan'
            
        final_results[regime_num] = {'label': final_label, 'color': color}
            
    return final_results



# --- 5. 메인 분석 및 시각화 함수 ---
def analyze_and_visualize_hmm(model_name: str):
    """지정된 HMM 모델의 학습 결과를 분석하고 시각화합니다."""
    logger.info(f"'{model_name}' 모델 결과 분석을 시작합니다...")
    db_manager = None
    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        analysis_df = backtest_manager.prepare_analysis_data(model_name)
        if analysis_df.empty:
            return

        analysis_df['daily_return'] = analysis_df['종가'].pct_change()
        
        regime_details = generate_regime_details(analysis_df)

        logger.info("\n--- 자동으로 생성된 국면 라벨 및 색상 ---")
        for k, v in sorted(regime_details.items()):
            logger.info(f"  {k}: {v['label']} (Color: {v['color']})")

        logger.info("\n--- 국면별 통계 분석 ---")
        for regime_id in sorted(analysis_df['regime_id'].unique()):
            regime_data = analysis_df[analysis_df['regime_id'] == regime_id]
            label_text = regime_details.get(regime_id, {}).get('label', f'국면 {regime_id}')
            
            log_message = (
                f"\n[{label_text}]\n"
                f"  - 기간: {regime_data['date'].min().date()} ~ {regime_data['date'].max().date()}\n"
                f"  - 총 거래일 수: {len(regime_data)}일\n"
                f"  - 평균 일일 수익률: {regime_data['daily_return'].mean():.4%}\n"
                f"  - 일일 수익률 표준편차: {regime_data['daily_return'].std():.4%}"
            )
            logger.info(log_message)

        # ... (시각화 로직은 이전과 동일) ...
        setup_korean_font()
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(analysis_df['date'], analysis_df['종가'], label='KOSPI 종가', color='black', linewidth=1.0, zorder=10)
        
        handles, labels = ax.get_legend_handles_labels()
        
        for regime_id, details in sorted(regime_details.items()):
            label_text, color = details['label'], details['color']
            ax.fill_between(analysis_df['date'], 0, analysis_df['종가'].max()*1.2, 
                            where=analysis_df['regime_id'] == regime_id, 
                            facecolor=color, alpha=0.4)
            if label_text not in labels:
                handles.append(plt.Rectangle((0,0),1,1, color=color, alpha=0.5))
                labels.append(label_text)

        start_date = analysis_df['date'].min().date()
        end_date = analysis_df['date'].max().date()
        title = f"HMM 시장 국면 분석 vs KOSPI ({start_date} ~ {end_date})"
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('KOSPI 지수')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(handles, labels, loc='upper left')
        ax.set_ylim(bottom=analysis_df['종가'].min() * 0.95)
        fig.autofmt_xdate()
        
        output_filename = f"{model_name}_analysis.png"
        plt.savefig(output_filename)
        logger.info(f"\n분석 결과 차트를 '{output_filename}' 파일로 저장했습니다.")

    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()

# --- 6. 스크립트 실행 ---
if __name__ == "__main__":
    MODEL_NAME_TO_ANALYZE = 'EKLMNO_4s_2208-2508'
    analyze_and_visualize_hmm(MODEL_NAME_TO_ANALYZE)