# manager/capital_manager.py

import logging
from typing import Dict, Any, List
import sys
import os
# trading.abstract_broker에서 AbstractBroker를 임포트하여 타입 힌팅에 사용합니다.
# 실제 런타임 의존성을 만들지 않기 위해 TYPE_CHECKING 블록을 사용할 수 있습니다.
from typing import TYPE_CHECKING
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.insert(0, project_root)
if TYPE_CHECKING:
    from trading.abstract_broker import AbstractBroker
from config.settings import STRATEGY_CONFIGS, ACTIVE_STRATEGIES_FOR_HMM
from trading.abstract_broker import AbstractBroker
logger = logging.getLogger(__name__)


class CapitalManager:
    """
    정적 자금 배분을 담당하며, settings.py에서 직접 포트폴리오 설정을 로드합니다.
    """
    def __init__(self, broker: AbstractBroker):
        # [핵심 2] 생성자에서 portfolio_configs 인자 제거
        if not hasattr(broker, 'get_current_cash_balance'):
            raise TypeError("제공된 broker 객체가 AbstractBroker의 인터페이스를 따르지 않습니다.")
        
        self.broker = broker
        
        # [핵심 3] settings.py 정보를 기반으로 내부 설정 자동 구성
        self.strategy_configs = {}
        for strategy_name in ACTIVE_STRATEGIES_FOR_HMM:
            if strategy_name in STRATEGY_CONFIGS:
                config = STRATEGY_CONFIGS[strategy_name]
                # 'name'과 'portfolio_params' 안의 'weight'를 추출
                self.strategy_configs[strategy_name] = {
                    "name": strategy_name,
                    "weight": config.get("portfolio_params", {}).get("weight", 0)
                }
        
        logger.info(f"CapitalManager 초기화 완료. {len(self.strategy_configs)}개 활성 전략의 가중치 로드.")


    def get_account_equity(self, current_prices: Dict[str, float]) -> float:
        """
        1단계: 총자산 (Account Equity)을 계산합니다.
        (총자산 = 현재 보유 현금 + 보유 주식의 현재가치 총합)

        :param current_prices: 보유 주식의 현재가를 담은 딕셔너리.
        :return: 총자산 평가 금액.
        """
        cash_balance = self.broker.get_current_cash_balance()
        portfolio_value = self.broker.get_portfolio_value(current_prices)
        account_equity = cash_balance + portfolio_value
        logger.debug(f"총자산(Account Equity) 계산: {account_equity:,.0f}원 (현금: {cash_balance:,.0f} + 주식: {portfolio_value:,.0f})")
        return account_equity

    def get_total_principal(self, account_equity: float, principal_ratio: float) -> float:
        """
        2단계: 총 투자원금 (Total Principal)을 계산합니다.
        (총 투자원금 = 총자산 * 총 투자원금 비율)
        :param account_equity: 총자산.
        :param principal_ratio: 총 투자원금 비율 (0.0 ~ 1.0).
        :return: 총 투자원금.
        """
        if not 0.0 <= principal_ratio <= 1.0:
            raise ValueError("총 투자원금 비율(principal_ratio)은 0과 1 사이의 값이어야 합니다.")
        
        total_principal = account_equity * principal_ratio
        logger.debug(f"총 투자원금(Total Principal) 계산: {total_principal:,.0f}원 (총자산의 {principal_ratio:.0%})")
        return total_principal, None

    def get_strategy_capital(self, strategy_name: str, total_principal: float) -> float:
        """
        3단계: 전략별 투자금 (Strategy Capital)을 계산합니다.
        (전략별 투자금 = 총 투자원금 * 해당 전략의 가중치)
        :param strategy_name: 자금을 계산할 전략의 이름.
        :param total_principal: 총 투자원금.
        :return: 해당 전략에 할당된 투자금.
        """
        strategy_config = self.strategy_configs.get(strategy_name)
        if not strategy_config:
            raise ValueError(f"'{strategy_name}'에 대한 전략 설정을 찾을 수 없습니다.")
        
        weight = strategy_config.get('weight', 0.0)
        strategy_capital = total_principal * weight
        logger.debug(f"[{strategy_name}] 전략별 투자금(Strategy Capital) 계산: {strategy_capital:,.0f}원 (총 투자원금의 {weight:.0%})")
        return strategy_capital
