import os
from hyperliquid.info import Info
from hyperliquid.utils import constants


address = os.getenv("ETH_COMP_PUBLIC_KEY")


def get_hyperliquid_order_history(include_subaccounts=False):
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    fills = info.user_fills(address)

    trades = []

    for f in fills:
        dir = f['dir']
        if dir == 'Open Short' or dir == 'Close Long':
            side = 'sell'
        elif dir == 'Open Long' or dir == 'Close Short':
            side = 'buy'
        trade = {
            "exchange": "hyperliquid",
            "symbol": f['coin'],
            "trade_id": f['tid'],
            "side": side,
            "type": 'unknown',
            "price": float(f['px']),
            "filled_quantity": float(f['sz']),
            "fee": float(f['fee']),
            "timestamp": f['time'],
            "user_address": address.lower()
        }

        if trade["filled_quantity"] > 0:
            trades.append(trade)

    if include_subaccounts:
        subaccounts = info.query_sub_accounts(address)
        for sub in subaccounts:
            sub_fills = info.user_fills(sub['subAccountUser'])
            for f in sub_fills:
                dir = f['dir']
                if dir == 'Open Short' or dir == 'Close Long':
                    side = 'sell'
                elif dir == 'Open Long' or dir == 'Close Short':
                    side = 'buy'
                trade = {
                    "exchange": "hyperliquid",
                    "symbol": f['coin'],
                    "trade_id": f['hash'],
                    "side": side,
                    "type": 'unknown',
                    "price": float(f['px']),
                    "filled_quantity": float(f['sz']),
                    "fee": float(f['fee']),
                    "timestamp": f['time'],
                    "user_address": sub['subAccountUser'].lower()
                }

                if trade["filled_quantity"] > 0:
                    trades.append(trade)

    return trades
    # user_fills_by_time
    # url = f"{BASE_URL}/api/v1/user/orders/history"
    # response = requests.get(url, headers=headers)
    #
    # if response.status_code != 200:
    #     print(f"Error {response.status_code}: {response.text}")
    #     return {}
    #
    # data = response.json().get("data", [])

    # trades = []
    # for d in data:
        # Convert timestamp to human-readable format
        #
        # trade = {
        #     "exchange": "extended",
        #     "symbol": d.get("market", "").split("-")[0] if "-" in d.get("market", "") else d.get("market", ""),
        #     "trade_id": d['id'],
        #     "side": d.get("side", "").lower(),
        #     "type": d.get("type", "").lower(),
        #     "price": float(d.get("averagePrice", 0.0)),
        #     "filled_quantity": float(d.get("filledQty", 0.0)),
        #     "fee": float(d.get("payedFee", 0.0)),
        #     "timestamp": d.get("createdTime"),
        #     "user_address": WALLET_ADDRESS.lower()
        # }
        #
        # if trade["filled_quantity"] > 0:
        #     trades.append(trade)

    # return trades


def get_hyperliquid_balance(include_subaccounts=False):
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(address)

    total_equity = float(user_state['marginSummary']['accountValue'])

    if include_subaccounts:
        subaccounts = info.query_sub_accounts(address)
        print(subaccounts)
        for sub in subaccounts:
            total_equity = total_equity + float(info.user_state(sub['subAccountUser'])['marginSummary']['accountValue'])
            user_state = info.user_state(sub['subAccountUser'])
            print(user_state)

    balance = {
        "exchange": "hyperliquid",
        "equity": total_equity,
    }

    return balance


def get_hyperliquid_positions(include_subaccounts=False):
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(address)

    positions_dict = []

    for pos in user_state['assetPositions']:
        size = float(pos['position']['szi'])
        position_dict = {
            "exchange": "hyperliquid",
            "symbol": pos['position']['coin'],
            "side": "long" if size > 0 else "short",
            "base_amount": abs(size),
        }

        positions_dict.append(position_dict)

    if include_subaccounts:
        subaccounts = info.query_sub_accounts(address)
        for sub in subaccounts:
            user_state = info.user_state(sub['subAccountUser'])

            for pos in user_state['assetPositions']:
                size = float(pos['position']['szi'])
                position_dict = {
                    "exchange": "hyperliquid",
                    "symbol": pos['position']['coin'],
                    "side": "long" if size > 0 else "short",
                    "base_amount": abs(size),
                }

                positions_dict.append(position_dict)

    return positions_dict


def get_hyperliquid_collected_funding(include_subaccounts=False):
    # user_funding_history
    start_time = 1755522896959  # Before competition started
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    funding_history = info.user_funding_history(address, startTime=start_time)

    funding_records = []
    for f in funding_history:
        funding_record = {
            "exchange": "hyperliquid",
            "symbol": f['delta']['coin'],
            "timestamp": f['time'],
            "funding": float(f['delta']['usdc']),
            "user_address": address,
        }
        funding_records.append(funding_record)

    if include_subaccounts:
        subaccounts = info.query_sub_accounts(address)
        for sub in subaccounts:
            funding_history = info.user_funding_history(sub['subAccountUser'], startTime=start_time)

            for f in funding_history:
                funding_record = {
                    "exchange": "hyperliquid",
                    "symbol": f['delta']['coin'],
                    "timestamp": f['time'],
                    "funding": float(f['delta']['usdc']),
                    "user_address": address,
                }
                funding_records.append(funding_record)

    return funding_records
    # funding_history = info.user_funding_history(address, startTime=start_time)
    # print(funding_history)
    # for f in data:
    #     funding_record = {
    #         "exchange": "extended",
    #         "symbol": f['market'].split("-")[0] if "-" in f['market'] else f['market'],
    #         "timestamp": f['paidTime'],
    #         "funding": float(f.get("fundingFee", 0.0)),
    #         "user_address": WALLET_ADDRESS.lower(),
    #     }
    #     funding_records.append(funding_record)
    pass
    # """Loop over positions and add collected_funding from funding history."""
    # url = f"{BASE_URL}/api/v1/user/funding/history"
    #
    # params = {
    #     "fromTime": 1755522896959  # Before competition started
    # }
    #
    # response = requests.get(url, headers=headers, params=params)
    # if response.status_code != 200:
    #     print(f"Funding Error {response.status_code}: {response.text}")
    #     return 0.0
    #
    # data = response.json().get("data", [])
    #
    # funding_records = []

    #
    # return funding_records


if __name__ == '__main__':

    print(get_hyperliquid_order_history())
    print(get_hyperliquid_positions(include_subaccounts=False))
    print(get_hyperliquid_balance(include_subaccounts=False))
    print(get_hyperliquid_collected_funding(include_subaccounts=False))