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

# --- 로거 및 한글 폰트 설정 (이전과 동일) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
except FileNotFoundError:
    logger.warning("Malgun Gothic 폰트를 찾을 수 없어 한글이 깨질 수 있습니다.")

# --- 헬퍼 함수: 성능 지표 계산 및 로깅, 차트 생성 (이전과 동일) ---
def calculate_metrics(series: pd.Series) -> dict:
    # ... (내용 변경 없음)
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
    return {"최종 수익률": total_return, "연평균 수익률": cagr, "샤프 지수": sharpe_ratio, "MDD": mdd, "승률": win_rate}

def log_performance_report(results: dict):
    # ... (내용 변경 없음)
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

# --- ▼ [핵심 수정] 차트 생성 함수를 새롭고 안정적인 로직으로 교체 ---
def create_performance_chart(wfo_series: pd.Series, strategy_results: dict, regime_data: dict, file_path: str, backtest_manager: BacktestManager):
    """
    hmm_results.py와 run_hmm_backtest.py의 장점을 결합한 최종 성과 분석 차트를 생성합니다.
    """
    logger.info("최종 성과 비교 차트 생성을 시작합니다...")
    
    # 1. 한글 폰트 설정
    try:
        font_path = "c:/Windows/Fonts/malgun.ttf"
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)
    except FileNotFoundError:
        logger.warning("Malgun Gothic 폰트를 찾을 수 없어 차트의 한글이 깨질 수 있습니다.")
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(18, 9))

    # 2. 국면 배경색상 표시 (hmm_results.py 로직)
    colors = ['#E6F4EA', '#FFF3E0', '#FFEBEE', '#E3F2FD'] # 안정, 주의, 위험, 기회 (국면 0, 1, 2, 3)
    regime_legends = {}
    for model_id, df in regime_data.items():
        for _, row in df.iterrows():
            ax.axvspan(row['start_date'], row['end_date'], color=colors[row['regime_id']], alpha=0.4, zorder=1)
            if row['regime_id'] not in regime_legends:
                 regime_legends[row['regime_id']] = plt.Rectangle((0,0),1,1, color=colors[row['regime_id']], alpha=0.4)

    # 3. KOSPI 지수 및 포트폴리오 데이터 준비 (run_hmm_backtest.py 로직)
    kospi_series = backtest_manager.cache_daily_ohlcv(
        COMMON_PARAMS['market_index_code'], wfo_series.index.min(), wfo_series.index.max()
    )['close']
    
    # 데이터프레임으로 통합 후 시작점을 100으로 정규화
    comparison_df = pd.DataFrame({'Portfolio': wfo_series, 'KOSPI': kospi_series})
    
    # 개별 전략 시리즈 추가 (비어있지 않은 경우에만)
    for name, series_list in strategy_results.items():
        if series_list and all(not s.empty for s in series_list):
            # 비어있지 않은 시리즈만 concat
            valid_series = [s for s in series_list if not s.empty]
            if valid_series:
                comparison_df[f'Strategy: {name}'] = pd.concat(valid_series)

    normalized_df = (comparison_df.interpolate(method='time').fillna(method='ffill') / comparison_df.iloc[0]) * 100

    # 4. 그래프 플롯
    ax.plot(normalized_df.index, normalized_df['Portfolio'], label='Walk-Forward Portfolio', color='black', linewidth=2.5, zorder=5)
    ax.plot(normalized_df.index, normalized_df['KOSPI'], label='KOSPI', color='dimgrey', linestyle='--', linewidth=1.5, zorder=4)
    
    strategy_cols = [col for col in normalized_df.columns if 'Strategy' in col]
    palette = plt.get_cmap('viridis')
    for i, col in enumerate(strategy_cols):
        ax.plot(normalized_df.index, normalized_df[col], label=col, linestyle='-', alpha=0.8, zorder=3, color=palette(i/len(strategy_cols)))

    # 5. 차트 스타일링
    ax.set_title('Walk-Forward Analysis Performance vs Benchmarks', fontsize=18)
    ax.set_ylabel('Normalized Value (Start = 100)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=normalized_df.min().min() * 0.95)
    
    # 범례(Legend) 핸들링
    handles, labels = ax.get_legend_handles_labels()
    regime_labels = [f'국면 {k}' for k in sorted(regime_legends.keys())]
    handles.extend(regime_legends.values())
    labels.extend(regime_labels)
    ax.legend(handles, labels, fontsize=10, loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_path)
    logger.info(f"성과 차트가 '{file_path}'에 저장되었습니다.")
    plt.close(fig)


# =================================================================
# 메인 실행 로직
# =================================================================
if __name__ == "__main__":
    
    # --- 1. 시스템 컴포넌트 초기화 ---
    api_client = CreonAPIClient()
    db_manager = DBManager()
    backtest_manager = BacktestManager(api_client=api_client, db_manager=db_manager) # API 클라이언트는 필요 시점에 생성
    optimizer = WalkForwardOptimizer(backtest_manager)

    # --- 2. 3단계 전진분석 기간 자동 생성 ---
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
    
    for i, p in enumerate(periods):
        logger.info(f"  - Stage {i+1}: Train({p['train_start']} ~ {p['train_end']}), Test({p['test_start']} ~ {p['test_end']})")

    # --- ▼ [핵심 수정] 전체 기간에 대한 마스터 데이터 사전 로딩 ---
    wfa_first_day = periods[0]['train_start']
    wfa_last_day = periods[-1]['test_end']
    logger.info(f"WFA 전체 기간({wfa_first_day} ~ {wfa_last_day})에 대한 마스터 데이터 사전 로딩을 시작합니다...")
    backtest_manager.prepare_pykrx_data_for_period(wfa_first_day, wfa_last_day)
    logger.info("마스터 데이터 사전 로딩 완료.")
    # --- ▲ 수정 완료 ---

    all_oos_results = []
    all_strategy_results = {name: [] for name, config in STRATEGY_CONFIGS.items() if config.get("strategy_status")}
    all_regime_data = {}
    
    compounding_capital = INITIAL_CASH
    model_prefix = "EKLMNO_4s"

    # --- 3. WFA 루프 실행 ---
    for i, p in enumerate(periods):
        stage = i + 1
        logger.info(f"\n\n{'='*30} 전진분석 Stage {stage} 시작 {'='*30}")
        
        # 학습: HMM 모델 생성
        model_name, model_id = optimizer.run_hmm_training_step(p['train_start'], p['train_end'], model_prefix)
        if not model_id: continue
        
        # 학습: 검증 기간 국면 데이터 미리 생성
        generate_and_save_regimes_for_period(
            model_name=model_name, start_date=p['test_start'], end_date=p['test_end'], backtest_manager=backtest_manager
        )

        # 학습: 전략 프로파일링
        profiling_success = optimizer.run_in_sample_profiling_step(p['train_start'], p['train_end'], model_id, model_name)
        if not profiling_success: continue

        # 검증: 최종 백테스트 (연결된 자금으로 실행)
        oos_series, run_id = optimizer.run_out_of_sample_validation_step(
            p['test_start'], p['test_end'], model_name, model_id, compounding_capital
        )
        if oos_series.empty: continue
        
        all_oos_results.append(oos_series)
        compounding_capital = oos_series.iloc[-1]

        # 차트용 데이터 수집
        regime_df = backtest_manager.fetch_daily_regimes(model_id)
        if not regime_df.empty:
            test_regimes = regime_df[(regime_df['date'] >= p['test_start']) & (regime_df['date'] <= p['test_end'])].copy()
            if not test_regimes.empty:
                test_regimes['date'] = pd.to_datetime(test_regimes['date'])
                test_regimes['start_date'] = test_regimes['date']
                regime_groups = test_regimes.groupby((test_regimes['regime_id'] != test_regimes['regime_id'].shift()).cumsum())
                all_regime_data[model_id] = regime_groups.agg(start_date=('date', 'min'), end_date=('date', 'max'), regime_id=('regime_id', 'first')).reset_index(drop=True)

        logger.info(f"--- Stage {stage}: 개별 전략 성과 기록 중 ---")
        for strategy_name in all_strategy_results.keys():
            tester = HMMBacktest(backtest_manager, INITIAL_CASH, p['test_start'], p['test_end'], save_to_db=False)
            strategy_class = globals().get(strategy_name)
            if strategy_class:
                strategy_instance = strategy_class(broker=tester.broker, data_store=tester.data_store)
                series, _, _, _ = tester.reset_and_rerun([strategy_instance], PassMinute(broker=tester.broker, data_store=tester.data_store), mode='strategy')
                all_strategy_results[strategy_name].append(series)

    # --- 4. 최종 결과 분석 및 리포팅 ---
    if all_oos_results:
        final_wfo_series = pd.concat(all_oos_results)
        report_data = {}
        for i, series in enumerate(all_oos_results):
            period_name = f"{series.index.min().year}년 전진분석"
            report_data[period_name] = {'시작일': series.index.min().date(), '종료일': series.index.max().date(), '연초 가치': series.iloc[0], '연말 가치': series.iloc[-1], '지표': calculate_metrics(series)}
        report_data["전체 기간 통합"] = {'시작일': final_wfo_series.index.min().date(), '종료일': final_wfo_series.index.max().date(), '연초 가치': final_wfo_series.iloc[0], '연말 가치': final_wfo_series.iloc[-1], '지표': calculate_metrics(final_wfo_series)}
        log_performance_report(report_data)

        # 비어있는 시리즈 리스트를 가진 전략을 필터링
        valid_strategy_results = {name: series_list for name, series_list in all_strategy_results.items() if series_list}
        
        chart_path = os.path.join(project_root, 'reports', f'wfo_report_{today.strftime("%Y%m%d")}.png')
        create_performance_chart(final_wfo_series, valid_strategy_results, all_regime_data, chart_path, backtest_manager)
    else:
        logger.warning("Walk-Forward 분석 결과가 없어 리포트를 생성할 수 없습니다.")

    # --- 5. 시스템 종료 ---
    db_manager.close()