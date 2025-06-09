import json

import example_utils

from hyperliquid.utils import constants


address, info, exchange = example_utils.setup(base_url=constants.TESTNET_API_URL, skip_ws=True)
RESTING_ORDERS = {}
POSITIONS = {}


def cancel_all_orders(address):
    open_orders = info.open_orders(address)
    cancelled_orders = []
    for open_order in open_orders:
        print(f"cancelling order {open_order}")
        result = exchange.cancel(open_order["coin"], open_order["oid"])
        cancelled_orders.append(result)

    return cancelled_orders

def buy_lmt_order(coin="HYPE/USDC", qty=0.07, lmt_price=150.00):
    order_result = exchange.order(coin, True, qty, lmt_price, {"limit": {"tif": "Gtc"}})
    print(order_result)
    status = order_result["response"]["data"]["statuses"][0]
    RESTING_ORDERS[coin] = [status["resting"]["oid"], qty, lmt_price, {"limit": {"tif": "Gtc"}}, "buy"]
    
    return order_result


def sell_lmt_order(coin="HYPE/USDC", qty=0.07, lmt_price=150.00):
    order_result = exchange.order(coin, False, qty, lmt_price, {"limit": {"tif": "Gtc"}})
    print(order_result)
    status = order_result["response"]["data"]["statuses"][0]
    RESTING_ORDERS[coin] = [status["resting"]["oid"], qty, lmt_price, {"limit": {"tif": "Gtc"}}, "sell"]
    
    return order_result

def mkt_buy_order(coin="HYPE/USDC", qty=0.07, lmt_price=150.00):
    order_result = exchange.market_open(coin, True, qty, None, 0.01)
    return order_result

def mkt_sell_order(coin="HYPE/USDC", qty=0.07, lmt_price=150.00):
    order_result = exchange.market_open(coin, False, qty, None, 0.01)
    return order_result

def mkt_close(coin):
    order_result = exchange.market_close(coin)
    return order_result

def close():
    pass



def main():
    address, info, exchange = example_utils.setup(base_url=constants.TESTNET_API_URL, skip_ws=True)

    # Get the user state and print out position information
    print(f"address: {address}")
    user_state = info.user_state(address)
    positions = []
    for position in user_state["assetPositions"]:
        positions.append(position["position"])
    if len(positions) > 0:
        print("positions:")
        for position in positions:
            print(json.dumps(position, indent=2))
    else:
        print("no open positions")

    # Place an order that should rest by setting the price very low
    order_result = exchange.order("HYPE/USDC", True, .07, 150, {"limit": {"tif": "Gtc"}})
    print(order_result)

    # Query the order status by oid
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            order_status = info.query_order_by_oid(address, status["resting"]["oid"])
            print("Order status by oid:", order_status)

    # Cancel the order
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            cancel_result = exchange.cancel("FARTCOIN", status["resting"]["oid"])
            print(cancel_result)


if __name__ == "__main__":
    main()
