from collections import deque

import example_utils

from hyperliquid.utils import constants


# XXX: Load model here

# XXX: Helper methods for handling data
MINUTE_BARS = deque(maxlen=100)
def _handle_minute_bars(ws_msg):
    print(f"ws_msg: {ws_msg}")


def _handle_bbo(ws_msg):
    print(f"ws_msg: {ws_msg}")


def _handle_order_updates(ws_msg):
    print(f"ws_msg: {ws_msg}")










def main():
    address, info, _ = example_utils.setup(constants.TESTNET_API_URL)
    # info.subscribe({"type": "candle", "coin": "BTC", "interval": "1m"}, _handle_minute_bars)
    # info.subscribe({"type": "orderUpdates", "user": address}, _handle_order_updates)
    info.subscribe({"type": "bbo", "coin": "BTC"}, _handle_bbo)
    



    # Some subscriptions do not return snapshots, so you will not receive a message until something happens
    # info.subscribe({"type": "allMids"}, print)
    # info.subscribe({"type": "l2Book", "coin": "BTC"}, print)
    # info.subscribe({"type": "trades", "coin": "BTC/USDC"}, print)
    # info.subscribe({"type": "userEvents", "user": address}, print)
    # info.subscribe({"type": "userFills", "user": address}, print) 
    # info.subscribe({"type": "userFundings", "user": address}, print)
    # info.subscribe({"type": "userNonFundingLedgerUpdates", "user": address}, print)
    # info.subscribe({"type": "webData2", "user": address}, print)

if __name__ == "__main__":
    main()
