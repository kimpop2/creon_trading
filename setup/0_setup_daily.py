#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매일 정해진 시간에 실행되는 데이터 셋업 스크립트
SetupManager를 사용하여 종목 정보, 거래일 캘린더 등을 업데이트합니다.
create_stock_tables.sql에 정의된 테이블 중 핵심 테이블만 대상으로 셋업합니다:
1. stock_info - 종목 기본 정보 (매매의 기본, 모든 종목 셋업)
2. market_calendar - 주식시장 캘린더

일봉/분봉 데이터는 백테스팅 과정에서 자동으로 처리되므로 셋업에서 제외합니다.
"""

import os
import sys
import logging
from datetime import datetime, date
import schedule
import time

# 프로젝트 루트 경로를 sys.path에 추가 (data 디렉토리에서 실행 가능하도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.creon_api import CreonAPIClient
from manager.db_manager import DBManager
from manager.backtest_manager import BacktestManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_daily.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def daily_setup_job():
    """
    매일 실행되는 메인 셋업 작업
    핵심 테이블만 대상으로 셋업합니다.
    """
    logger.info("=== 일일 데이터 셋업 작업 시작 ===")
    
    try:
        api_client = CreonAPIClient()
        db_manager = DBManager()
        backtest_manager = BacktestManager(api_client, db_manager)
        
        # 1. 모든 종목 기본 정보 업데이트 (stock_info 테이블)
        logger.info("1. 종목 기본 정보 업데이트 시작 (stock_info 테이블)")
        if backtest_manager.update_all_stock_info():
            logger.info("✓ 종목 기본 정보 업데이트 완료")
        else:
            logger.error("✗ 종목 기본 정보 업데이트 실패")
        
        # 2. 주식시장 캘린더 업데이트 (market_calendar 테이블)
        logger.info("2. 주식시장 캘린더 업데이트 시작 (market_calendar 테이블)")
        if backtest_manager.update_market_calendar():
            logger.info("✓ 주식시장 캘린더 업데이트 완료")
        else:
            logger.error("✗ 주식시장 캘린더 업데이트 실패")
        
        logger.info("=== 일일 데이터 셋업 작업 완료 ===")
        
    except Exception as e:
        logger.error(f"일일 셋업 작업 중 오류 발생: {e}", exc_info=True)
    finally:
        backtest_manager.close()

def run_once():
    """
    한 번만 실행하는 함수 (테스트용)
    """
    logger.info("=== 일회성 데이터 셋업 실행 ===")
    daily_setup_job()

def run_scheduled():
    """
    스케줄에 따라 실행하는 함수
    """
    logger.info("=== 스케줄된 데이터 셋업 서비스 시작 ===")
    
    # 매일 오전 08:30 에 실행 (장 시작 전 - 장전시간외 08:50 고려 )
    schedule.every().day.at("08:30").do(daily_setup_job)
    
    # 매일 오후 3:30 에 실행 (장 마감 후 - 시간외 단일가 16:00 고려)
    schedule.every().day.at("15:30").do(daily_setup_job)
    
    logger.info("스케줄 설정 완료: 매일 09:00, 18:00에 실행")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 스케줄 확인

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 셋업 스크립트')
    parser.add_argument('--once', action='store_true', help='한 번만 실행')
    parser.add_argument('--scheduled', action='store_true', help='스케줄에 따라 실행')
    
    args = parser.parse_args()
    
    if args.once:
        run_once()
    elif args.scheduled:
        run_scheduled()
    else:
        # 기본값: 한 번만 실행
        run_once() 