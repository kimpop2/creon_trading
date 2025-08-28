import logging
import argparse
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 필요 모듈 임포트 ---
from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager
from trading.hmm_backtest import HMMBacktest
from config.settings import STRATEGY_CONFIGS, INITIAL_CASH, COMMON_PARAMS

# --- 동적 전략 로딩을 위해 모든 전략 클래스 임포트 ---
from strategies.sma_daily import SMADaily
from strategies.dual_momentum_daily import DualMomentumDaily
from strategies.breakout_daily import BreakoutDaily
from strategies.closing_bet_daily import ClosingBetDaily
from strategies.pass_minute import PassMinute

# --- 로거 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """ 
    사용법 reports 폴더에 backtest_report_ 차트 이미지 파일 생성 == 최고 성능 측정
    python analyzer/run_hmm_backtest.py --model-name EKLMNO_4s_2301-2412 --start 2025-06-01 --end 2025-08-10
    """
    parser = argparse.ArgumentParser(description="전체 기간에 대한 포트폴리오 백테스트를 실행합니다.")
    parser.add_argument('--model-name', type=str, required=True, help="백테스트의 기준이 될 HMM 모델 이름")
    parser.add_argument('--start', type=str, required=True, help="백테스트 기간 시작일 (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, required=True, help="백테스트 기간 종료일 (YYYY-MM-DD)")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    logger.info("="*50)
    logger.info("전체 기간 포트폴리오 백테스트를 시작합니다.")
    logger.info(f"대상 모델: {args.model_name}, 기간: {args.start} ~ {args.end}")
    logger.info("="*50)
    
    backtest_system = None
    try:
        # 1. 시스템 컴포넌트 초기화
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        
        # 2. 백테스트 시스템 인스턴스 생성
        backtest_system = HMMBacktest(
            manager=backtest_manager,
            initial_cash=INITIAL_CASH,
            start_date=start_date,
            end_date=end_date,
            save_to_db=False 
        )

        # 3. 활성화된 전략 동적 로딩
        daily_strategies_list = []
        active_strategy_names = []
        for name, config in STRATEGY_CONFIGS.items():
            if config.get('strategy_status'):
                strategy_class = globals().get(name)
                if strategy_class:
                    instance = strategy_class(
                        broker=backtest_system.broker,
                        data_store=backtest_system.data_store
                    )
                    daily_strategies_list.append(instance)
                    active_strategy_names.append(name)
        
        if not daily_strategies_list:
            logger.error("실행할 활성 전략이 없습니다. settings.py를 확인하세요.")
            return

        logger.info(f"백테스트에 포함된 활성 전략: {active_strategy_names}")

        minute_strategy = PassMinute(
            broker=backtest_system.broker,
            data_store=backtest_system.data_store
        )
        backtest_system.set_strategies(
            daily_strategies=daily_strategies_list,
            minute_strategy=minute_strategy
        )

        # 4. 데이터 준비 및 백테스트 실행
        logger.info("백테스트에 필요한 데이터 사전 로딩을 시작합니다...")
        backtest_manager.prepare_pykrx_data_for_period(start_date, end_date)
        backtest_system.prepare_for_system()
        
        logger.info("전체 기간 백테스트 실행...")
        portfolio_series, metrics, _, _ = backtest_system.reset_and_rerun(
            daily_strategies=daily_strategies_list,
            minute_strategy=minute_strategy,
            mode='hmm',
            model_name=args.model_name
        )

        # 5. 최종 결과 출력 및 시각화
        if not portfolio_series.empty:
            logger.info("\n" + "="*50)
            logger.info("### 최종 백테스트 결과 요약 ###")
            logger.info("="*50)
            
            final_value = portfolio_series.iloc[-1]
            logger.info(f"  - 최종 포트폴리오 가치: {final_value:,.0f}원")
            
            for key, value in metrics.items():
                if "return" in key or "mdd" in key:
                    logger.info(f"  - {key}: {value:.2%}")
                elif "ratio" in key:
                    logger.info(f"  - {key}: {value:.2f}")
                else:
                    logger.info(f"  - {key}: {value}")
            
            # --- ▼ [핵심 수정] KOSPI 데이터 로드 및 정규화, 그래프 비교 로직 추가 ---
            logger.info("\nKOSPI 지수와 성과 비교 그래프를 생성합니다...")
            kospi_code = COMMON_PARAMS.get('market_index_code', 'U001')
            kospi_df = backtest_manager.cache_daily_ohlcv(kospi_code, start_date, end_date)

            if not kospi_df.empty:
                # 포트폴리오와 코스피 데이터의 날짜를 맞춤
                comparison_df = pd.DataFrame({'Portfolio': portfolio_series})
                comparison_df = pd.merge(comparison_df, kospi_df[['close']].rename(columns={'close': 'KOSPI'}), 
                                         left_index=True, right_index=True, how='inner')

                # 시작점을 1로 정규화하여 수익률 추이 비교
                normalized_df = comparison_df / comparison_df.iloc[0]

                # 수익 곡선 그래프 저장
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.plot(normalized_df.index, normalized_df['Portfolio'], label='Portfolio (Normalized)')
                ax.plot(normalized_df.index, normalized_df['KOSPI'], label='KOSPI (Normalized)', color='red', linestyle='--')
                
                ax.set_title(f"Portfolio vs KOSPI ({args.start} ~ {args.end})", fontsize=16)
                ax.set_xlabel("Date")
                ax.set_ylabel("Normalized Value")
                ax.legend()
            else:
                logger.warning("KOSPI 데이터를 찾을 수 없어 포트폴리오 그래프만 생성합니다.")
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value')
                ax.set_title(f"Portfolio Backtest Result ({args.start} ~ {args.end})", fontsize=16)
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio Value (KRW)")
                ax.legend()
            
            # reports 폴더가 없으면 생성
            reports_dir = os.path.join(project_root, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            report_filename = f"backtest_report_{args.model_name}_{args.start}-{args.end}.png"
            report_path = os.path.join(reports_dir, report_filename)
            plt.savefig(report_path)
            logger.info(f"\n수익 곡선 그래프가 '{report_path}'에 저장되었습니다.")
            # --- ▲ 수정 종료 ---

        else:
            logger.warning("백테스트 결과가 비어있어 요약을 생성할 수 없습니다.")

    except Exception as e:
        logger.critical(f"백테스트 실행 중 심각한 오류 발생: {e}", exc_info=True)
    finally:
        if backtest_system and backtest_system.manager:
            backtest_system.manager.close()
        logger.info("="*50)
        logger.info("백테스트를 종료합니다.")
        logger.info("="*50)

if __name__ == "__main__":
    main()