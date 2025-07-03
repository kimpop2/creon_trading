# main_live_trading_demo.py

import logging
import time
from trade_creon_api_mock import CreonAPIMock
from trade_live_broker_core import LiveBroker

# Configure a root logger to see all output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_demo():
    logger = logging.getLogger('DemoMain')
    logger.info("Starting live trading core demo...")

    # 1. Initialize the mock Creon API client
    mock_api = CreonAPIMock()
    mock_api.connect()

    # 2. Initialize the LiveBroker, passing the mock API client
    broker = LiveBroker(api_client=mock_api)

    logger.info("\n--- Demo Scenario: Placing Orders ---")

    # --- Scenario 1: Buy Order ---
    logger.info("\n--- SCENARIO 1: BUY ORDER (A005930) ---")
    buy_order_id = broker.execute_order(stock_code="A005930", order_type="buy", price=75000.0, quantity=10)
    if buy_order_id:
        logger.info(f"Buy order {buy_order_id} submitted to broker. Waiting for callbacks...")
        # In a real system, you'd now wait for the Creon API's event loop to
        # trigger broker.on_order_reply and broker.on_conclusion
        # For this mock, the callbacks are fired immediately within send_order for simplicity.
    time.sleep(1) # Give some time for logs to appear

    logger.info(f"\n--- Current Broker State after Buy Attempt ---")
    logger.info(f"Cash: {broker.get_current_cash():,.0f} KRW")
    logger.info(f"Positions: {broker.get_current_positions()}")
    logger.info(f"Pending Orders: {broker.pending_orders}")
    logger.info(f"Transaction Log Length: {len(broker.get_transaction_log())}")

    # --- Scenario 2: Sell Order ---
    logger.info("\n--- SCENARIO 2: SELL ORDER (A005930) ---")
    # Assuming the previous buy was successful and we now have A005930
    sell_order_id = broker.execute_order(stock_code="A005930", order_type="sell", price=75500.0, quantity=5)
    if sell_order_id:
        logger.info(f"Sell order {sell_order_id} submitted to broker. Waiting for callbacks...")
    time.sleep(1) # Give some time for logs to appear

    logger.info(f"\n--- Current Broker State after Sell Attempt ---")
    logger.info(f"Cash: {broker.get_current_cash():,.0f} KRW")
    logger.info(f"Positions: {broker.get_current_positions()}")
    logger.info(f"Pending Orders: {broker.pending_orders}")
    logger.info(f"Transaction Log Length: {len(broker.get_transaction_log())}")

    # --- Scenario 3: Attempt to buy with insufficient cash ---
    logger.info("\n--- SCENARIO 3: BUY ORDER (A000660) - Insufficient Cash ---")
    broker.execute_order(stock_code="A000660", order_type="buy", price=100000.0, quantity=1000) # Should fail
    time.sleep(0.5)

    logger.info(f"\n--- Final Broker State ---")
    logger.info(f"Cash: {broker.get_current_cash():,.0f} KRW")
    logger.info(f"Positions: {broker.get_current_positions()}")
    logger.info(f"Pending Orders: {broker.pending_orders}")
    logger.info(f"Transaction Log:")
    for log_entry in broker.get_transaction_log():
        logger.info(f"  - {log_entry}")

    mock_api.cleanup()
    logger.info("Demo finished.")

if __name__ == "__main__":
    run_demo()