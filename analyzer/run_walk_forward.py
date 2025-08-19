# analyzer/run_walk_forward.py

import logging
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
import numpy as np
import sys
import os

# --- 프로젝트 경로 설정 및 필요 모듈 임포트 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from optimizer.walk_forward_optimizer import WalkForwardOptimizer
from analyzer.train_and_save_hmm import generate_and_save_regimes_for_period
from trading.hmm_backtest import HMMBacktest
from strategies.pass_minute import PassMinute
from config.settings import INITIAL_CASH, STRATEGY_CONFIGS, COMMON_PARAMS

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ▼ [신규 추가] hmm_results.py의 핵심 로직 이식 ---
def setup_korean_font():
    """운영체제에 맞는 한글 폰트를 설정합니다."""
    try:
        if sys.platform == "win32":
            font_path = "c:/Windows/Fonts/malgun.ttf"
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
        elif sys.platform == "darwin":
            rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        logger.warning(f"한글 폰트 설정 중 오류 발생: {e}")

def generate_regime_details(analysis_data: pd.DataFrame) -> dict:
    """4-분면 정의에 따라 각 국면의 '라벨'과 시각화를 위한 '색상'을 동적으로 생성합니다."""
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
        final_label = f"국면 {regime_num} ({base_label})"
        if base_label in processed_counts:
            processed_counts[base_label] += 1
            final_label = f"국면 {regime_num} ({base_label} Type {processed_counts[base_label]})"
        
        color = '#E0E0E0' # 기본 회색
        if base_label == "고변동성 강세장": color = '#FFCDD2' # 연홍색
        elif base_label == "저변동성 강세장": color = '#FFF9C4' # 연노랑
        elif base_label == "고변동성 약세장": color = '#BBDEFB' # 연파랑
        elif base_label == "저변동성 약세장": color = '#C8E6C9' # 연녹색
        final_results[regime_num] = {'label': final_label, 'color': color}
            
    return final_results
# --- ▲ 이식 완료 ---

# --- 헬퍼 함수: 성능 지표 계산 및 로깅 (이전과 동일) ---
def calculate_metrics(series: pd.Series) -> dict:
    if series.empty or len(series) < 2: return {}
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = ((1 + total_return) ** (252 / len(series))) - 1 if len(series) > 0 else 0
    returns = series.pct_change().dropna()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    mdd = drawdown.min()
    win_rate = (returns > 0).mean()
    num_trades = len(returns[returns != 0])
    return {"최종 수익률": total_return, "연평균 수익률": cagr, "샤프 지수": sharpe_ratio, "MDD": mdd, "승률": win_rate, "매매 횟수": num_trades}

def log_performance_report(results: dict):
    logger.info("\n\n" + "="*80)
    logger.info("### Walk-Forward Analysis 최종 성과 보고서 ###")
    logger.info("="*80)
    for period, data in results.items():
        logger.info(f"\n--- {period} ---")
        logger.info(f"  - 기간: {data['시작일']} ~ {data['종료일']}")
        logger.info(f"  - 연초 포트폴리오 가치: {data['연초 가치']:,.0f}원")
        logger.info(f"  - 연말 포트폴리오 가치: {data['연말 가치']:,.0f}원")
        for key, value in data['지표'].items():
            if isinstance(value, float) and ('률' in key or 'MDD' in key): logger.info(f"  - {key}: {value:.2%}")
            elif isinstance(value, float): logger.info(f"  - {key}: {value:.2f}")
            else: logger.info(f"  - {key}: {value}")
        logger.info(f"  - 최초 투자금 대비 성장: {(data['연말 가치'] / INITIAL_CASH):.2f}배")
    logger.info("\n" + "="*80)

# --- ▼ [핵심 수정] 차트 생성 함수를 최종 버전으로 교체 ---
def create_performance_chart_for_year(year_series: pd.Series, strategy_results_year: dict, regime_data_year: dict, file_path: str, backtest_manager: BacktestManager):
    year = year_series.index.year.unique()[0]
    logger.info(f"{year}년도 성과 비교 차트 생성을 시작합니다...")
    
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(20, 10))

    # 1. 국면 배경색상 표시 (동적 라벨링 적용)
    regime_details = regime_data_year.get('details', {})
    regime_legends = {}
    if 'periods' in regime_data_year:
        for _, row in regime_data_year['periods'].iterrows():
            regime_id = row['regime_id']
            color = regime_details.get(regime_id, {}).get('color', '#E0E0E0')
            ax.axvspan(row['start_date'], row['end_date'], color=color, alpha=0.5, zorder=1)
            if regime_id not in regime_legends:
                 label = regime_details.get(regime_id, {}).get('label', f'국면 {regime_id}')
                 regime_legends[label] = plt.Rectangle((0,0),1,1, color=color, alpha=0.5)

    # 2. 모든 데이터를 정규화(시작점 100)하여 비교
    kospi_series = backtest_manager.cache_daily_ohlcv(
        COMMON_PARAMS['market_index_code'], year_series.index.min(), year_series.index.max()
    )['close']
    
    comparison_df = pd.DataFrame({'WFO 포트폴리오': year_series, 'KOSPI': kospi_series})
    for name, series in strategy_results_year.items():
        comparison_df[f'전략: {name}'] = series

    normalized_df = comparison_df.interpolate(method='time').fillna(method='ffill')
    normalized_df = (normalized_df / normalized_df.iloc[0]) * 100

    # 3. 그래프 플롯 (단일 Y축 사용)
    ax.plot(normalized_df.index, normalized_df['WFO 포트폴리오'], color='black', linewidth=2.5, zorder=10, label='WFO 포트폴리오')
    ax.plot(normalized_df.index, normalized_df['KOSPI'], color='dimgrey', linestyle='--', linewidth=1.5, zorder=5, label='KOSPI')
    
    strategy_cols = [col for col in normalized_df.columns if '전략' in col]
    palette = plt.get_cmap('viridis')
    for i, col in enumerate(strategy_cols):
        ax.plot(normalized_df.index, normalized_df[col], linestyle='-', alpha=0.8, zorder=4, label=col, color=palette(i/len(strategy_cols) if len(strategy_cols) > 0 else 0))

    # 4. 차트 종합 스타일링
    ax.set_title(f'{year}년 Walk-Forward Analysis 성과 비교', fontsize=18)
    ax.set_ylabel('정규화 가치 (시작일 = 100)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    sorted_regime_legends = sorted(regime_legends.items(), key=lambda item: item[0])
    handles.extend([item[1] for item in sorted_regime_legends])
    labels.extend([item[0] for item in sorted_regime_legends])
    
    ax.legend(handles, labels, fontsize=10, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(file_path)
    logger.info(f"성과 차트가 '{file_path}'에 저장되었습니다.")
    plt.close(fig)


# =================================================================
# 메인 실행 로직
# =================================================================
if __name__ == "__main__":
    
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client=api_client, db_manager=db_manager)
    optimizer = WalkForwardOptimizer(backtest_manager)

    today = date.today()
    current_year = today.year
    periods = []
    for i in range(3):
        validation_year = current_year - i
        test_start = date(validation_year, 1, 1)
        test_end = today if i == 0 else date(validation_year, 12, 31)
        train_end = test_start - timedelta(days=1)
        train_start = date(train_end.year - 2 + 1, 1, 1)
        periods.insert(0, {'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end})
    
    wfa_first_day = periods[0]['train_start']
    wfa_last_day = periods[-1]['test_end']
    logger.info(f"WFA 전체 기간({wfa_first_day} ~ {wfa_last_day})에 대한 마스터 데이터 사전 로딩을 시작합니다...")
    backtest_manager.prepare_pykrx_data_for_period(wfa_first_day, wfa_last_day)
    logger.info("마스터 데이터 사전 로딩 완료.")

    all_oos_results = []
    compounding_capital = INITIAL_CASH
    model_prefix = "EKLMNO_4s"

    for i, p in enumerate(periods):
        stage = i + 1
        logger.info(f"\n\n{'='*30} 전진분석 Stage {stage} 시작 {'='*30}")
        
        model_name, model_id = optimizer.run_hmm_training_step(p['train_start'], p['train_end'], model_prefix)
        if not model_id: continue
        
        generate_and_save_regimes_for_period(model_name=model_name, start_date=p['test_start'], end_date=p['test_end'], backtest_manager=backtest_manager)
        profiling_success = optimizer.run_in_sample_profiling_step(p['train_start'], p['train_end'], model_id, model_name)
        if not profiling_success: continue

        oos_series, run_id = optimizer.run_out_of_sample_validation_step(
            p['test_start'], p['test_end'], model_name, model_id, compounding_capital
        )
        if oos_series.empty: continue
        
        all_oos_results.append(oos_series)
        compounding_capital = oos_series.iloc[-1]
        
        # --- ▼ [핵심 수정] 매 Stage가 끝날 때마다 해당 연도의 차트 생성 ---
        logger.info(f"--- Stage {stage}: 개별 전략 성과 기록 및 연간 차트 생성 ---")
        
        strategy_results_year = {}
        active_strategies = [name for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")]
        for strategy_name in active_strategies:
            tester = HMMBacktest(backtest_manager, INITIAL_CASH, p['test_start'], p['test_end'], save_to_db=False)
            strategy_class = globals().get(strategy_name)
            if strategy_class:
                strategy_instance = strategy_class(broker=tester.broker, data_store=tester.data_store)
                series, _, _, _ = tester.reset_and_rerun([strategy_instance], PassMinute(broker=tester.broker, data_store=tester.data_store), mode='strategy')
                if not series.empty:
                    strategy_results_year[strategy_name] = series
        
        analysis_df = backtest_manager.prepare_analysis_data(model_name)
        regime_data_year = {}
        if not analysis_df.empty:
            regime_details = generate_regime_details(analysis_df)
            regime_df = backtest_manager.fetch_daily_regimes(model_id)
            test_regimes = regime_df[(regime_df['date'] >= p['test_start']) & (regime_df['date'] <= p['test_end'])].copy()
            if not test_regimes.empty:
                test_regimes['date'] = pd.to_datetime(test_regimes['date'])
                regime_groups = test_regimes.groupby((test_regimes['regime_id'] != test_regimes['regime_id'].shift()).cumsum())
                period_data = regime_groups.agg(start_date=('date', 'min'), end_date=('date', 'max'), regime_id=('regime_id', 'first')).reset_index(drop=True)
                regime_data_year = {'details': regime_details, 'periods': period_data}

        year = p['test_start'].year
        chart_path = os.path.join(project_root, 'reports', f'wfo_report_{year}.png')
        create_performance_chart_for_year(oos_series, strategy_results_year, regime_data_year, chart_path, backtest_manager)

    # --- 4. 최종 결과 분석 및 리포팅 (통합 보고서) ---
    if all_oos_results:
        final_wfo_series = pd.concat(all_oos_results)
        report_data = {}
        for i, series in enumerate(all_oos_results):
            period_name = f"{series.index.min().year}년 전진분석"
            report_data[period_name] = {'시작일': series.index.min().date(), '종료일': series.index.max().date(), '연초 가치': series.iloc[0], '연말 가치': series.iloc[-1], '지표': calculate_metrics(series)}
        report_data["전체 기간 통합"] = {'시작일': final_wfo_series.index.min().date(), '종료일': final_wfo_series.index.max().date(), '연초 가치': final_wfo_series.iloc[0], '연말 가치': final_wfo_series.iloc[-1], '지표': calculate_metrics(final_wfo_series)}
        log_performance_report(report_data)

    else:
        logger.warning("Walk-Forward 분석 결과가 없어 리포트를 생성할 수 없습니다.")

    # --- 5. 시스템 종료 ---
    db_manager.close()