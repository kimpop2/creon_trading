# manager/capital_manager.py

import logging
from typing import Dict, Any, List

# trading.abstract_broker에서 AbstractBroker를 임포트하여 타입 힌팅에 사용합니다.
# 실제 런타임 의존성을 만들지 않기 위해 TYPE_CHECKING 블록을 사용할 수 있습니다.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading.abstract_broker import AbstractBroker

logger = logging.getLogger(__name__)

class CapitalManager:
    """
    5단계 계층적 자금 관리 프레임워크를 기반으로,
    시스템의 모든 자금 관련 계산 및 의사결정을 전담하는 클래스.
    """
    def __init__(self, broker: 'AbstractBroker', portfolio_configs: List[Dict[str, Any]]):
        """
        CapitalManager를 초기화합니다.

        :param broker: AbstractBroker를 구현한 브로커 객체.
        :param portfolio_configs: 각 전략의 설정을 담은 딕셔너리 리스트.
               예: [{'name': 'SMADaily', 'weight': 0.6, 'max_position_count': 10}, ...]
        """
        if not hasattr(broker, 'get_current_cash_balance') or not hasattr(broker, 'get_portfolio_value'):
            raise TypeError("제공된 broker 객체가 AbstractBroker의 인터페이스를 따르지 않습니다.")
        
        self.broker = broker
        # 전략 설정을 이름으로 쉽게 조회할 수 있도록 딕셔너리로 변환
        self.strategy_configs = {config['name']: config for config in portfolio_configs}
        logger.info("CapitalManager 초기화 완료.")

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
        return total_principal

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
