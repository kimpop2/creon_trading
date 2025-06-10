# analyzer/backtest_result.py

import pandas as pd
import numpy as np
import logging
import datetime

class BacktestResult:
    """백테스트 성과 분석 클래스"""
    
    def __init__(self, initial_cash: float):
        """
        Args:
            initial_cash: 초기 투자금
        """
        self.initial_cash = initial_cash
        self.portfolio_values = []
        self.dates = []
        self.trade_history = []
        
    def add_portfolio_value(self, date: datetime.datetime, value: float):
        """포트폴리오 가치를 기록합니다."""
        self.portfolio_values.append(value)
        self.dates.append(date)
    
    def add_trade(self, trade_info):
        """매매 기록을 추가합니다."""
        self.trade_history.append(trade_info)
    
    def get_portfolio_value_series(self):
        """포트폴리오 가치 시계열 데이터를 반환합니다."""
        return pd.Series(self.portfolio_values, index=self.dates)
    
    def calculate_returns(self, portfolio_values: pd.Series):
        """일별 수익률을 계산합니다."""
        return portfolio_values.pct_change().dropna()
    
    def calculate_metrics(self, risk_free_rate: float = 0.03):
        """
        포트폴리오 성과 지표를 계산합니다.
        
        Args:
            risk_free_rate: 무위험 수익률 (연율화된 값, 예: 0.03 = 3%)
            
        Returns:
            성과 지표 딕셔너리
        """
        portfolio_values = self.get_portfolio_value_series()
        daily_returns = self.calculate_returns(portfolio_values)
        
        # 누적 수익률 계산
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # MDD (Maximum Drawdown) 계산
        cumulative_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / cumulative_max - 1
        mdd = drawdowns.min()
        
        # 연간 수익률 계산 (연율화)
        total_days = len(daily_returns)
        total_years = total_days / 252  # 거래일 기준
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (1 / total_years) - 1
        
        # 연간 변동성 계산 (연율화)
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # 샤프 지수 계산
        excess_returns = annual_return - risk_free_rate
        sharpe_ratio = excess_returns / annual_volatility if annual_volatility != 0 else 0
        
        # 승률 계산
        win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
        
        # 평균 수익거래 대비 손실거래 비율 (Profit Factor)
        positive_returns = daily_returns[daily_returns > 0].mean()
        negative_returns = abs(daily_returns[daily_returns < 0].mean())
        profit_factor = positive_returns / negative_returns if negative_returns != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'mdd': mdd,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def get_trade_summary(self):
        """매매 기록을 분석하여 요약 정보를 반환합니다."""
        if not self.trade_history:
            return {}
            
        total_trades = len(self.trade_history)
        buy_trades = len([t for t in self.trade_history if t['order_type'] == 'buy'])
        sell_trades = len([t for t in self.trade_history if t['order_type'] == 'sell'])
        
        total_commission = sum(t['commission'] for t in self.trade_history)
        total_volume = sum(t['price'] * t['size'] for t in self.trade_history)
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_commission': total_commission,
            'total_volume': total_volume,
            'turnover_ratio': total_volume / self.initial_cash
        }
    
    def print_summary(self, start_date: datetime.date, end_date: datetime.date):
        """백테스트 결과 요약을 출력합니다."""
        metrics = self.calculate_metrics()
        trade_summary = self.get_trade_summary()
        
        logging.info("\n=== 백테스트 결과 ===")
        logging.info(f"시작일: {start_date.isoformat()}")
        logging.info(f"종료일: {end_date.isoformat()}")
        logging.info(f"초기자금: {self.initial_cash:,.0f}원")
        logging.info(f"최종 포트폴리오 가치: {self.portfolio_values[-1]:,.0f}원")
        
        # 수익률 및 위험 지표
        logging.info("\n--- 수익률 및 위험 지표 ---")
        logging.info(f"총 수익률: {metrics['total_return']*100:.2f}%")
        logging.info(f"연간 수익률: {metrics['annual_return']*100:.2f}%")
        logging.info(f"연간 변동성: {metrics['annual_volatility']*100:.2f}%")
        logging.info(f"샤프 지수: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
        logging.info(f"승률: {metrics['win_rate']*100:.2f}%")
        logging.info(f"수익비 (Profit Factor): {metrics['profit_factor']:.2f}")
        
        # 매매 정보
        if trade_summary:
            logging.info("\n--- 매매 정보 ---")
            logging.info(f"총 거래 횟수: {trade_summary['total_trades']}회")
            logging.info(f"매수 거래: {trade_summary['buy_trades']}회")
            logging.info(f"매도 거래: {trade_summary['sell_trades']}회")
            logging.info(f"총 거래대금: {trade_summary['total_volume']:,.0f}원")
            logging.info(f"총 수수료: {trade_summary['total_commission']:,.0f}원")
            logging.info(f"회전율: {trade_summary['turnover_ratio']:.2f}")