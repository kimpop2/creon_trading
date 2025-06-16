"""
optimizer 모듈 테스트 실행 스크립트
optimizer 모듈의 모든 단위 테스트를 실행하고 결과를 로깅합니다.
"""

import unittest
import sys
import os
import logging
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def setup_logging():
    """로깅 설정"""
    log_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'optimizer_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_tests():
    """optimizer 모듈 테스트 실행"""
    logger = setup_logging()
    logger.info("optimizer 모듈 테스트를 시작합니다.")
    
    try:
        # 테스트 로더 생성
        loader = unittest.TestLoader()
        
        # 현재 디렉토리에서 테스트 스위트 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(current_dir, pattern='test_*.py')
        
        # 테스트 실행기 생성
        runner = unittest.TextTestRunner(verbosity=2)
        
        # 테스트 실행
        result = runner.run(suite)
        
        # 결과 출력
        logger.info("\n=== optimizer 모듈 테스트 결과 요약 ===")
        logger.info(f"실행된 테스트 수: {result.testsRun}")
        logger.info(f"성공한 테스트 수: {result.testsRun - len(result.failures) - len(result.errors)}")
        logger.info(f"실패한 테스트 수: {len(result.failures)}")
        logger.info(f"오류 발생 테스트 수: {len(result.errors)}")
        
        # 실패한 테스트 출력
        if result.failures:
            logger.info("\n=== 실패한 테스트 ===")
            for failure in result.failures:
                logger.error(f"\n{failure[0]}\n{failure[1]}")
        
        # 오류 발생 테스트 출력
        if result.errors:
            logger.info("\n=== 오류 발생 테스트 ===")
            for error in result.errors:
                logger.error(f"\n{error[0]}\n{error[1]}")
        
        # 종료 코드 설정
        sys.exit(len(result.failures) + len(result.errors))
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("optimizer 모듈 테스트가 완료되었습니다.")

if __name__ == '__main__':
    run_tests() 