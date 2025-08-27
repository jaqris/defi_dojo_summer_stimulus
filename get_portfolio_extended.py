import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

# API details
API_KEY = os.getenv("EXTENDED_API_KEY")
BASE_URL = "https://api.starknet.extended.exchange"

WALLET_ADDRESS = os.getenv("ETH_COMP_PUBLIC_KEY")
# Headers (API key + required User-Agent)
headers = {
    "X-Api-Key": API_KEY,
    "User-Agent": "MyPythonClient/1.0"
}


def get_extended_order_history():
    url = f"{BASE_URL}/api/v1/user/orders/history"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return {}

    data = response.json().get("data", [])

    trades = []
    for d in data:
        # Convert timestamp to human-readable format

        trade = {
            "exchange": "extended",
            "symbol": d.get("market", "").split("-")[0] if "-" in d.get("market", "") else d.get("market", ""),
            "trade_id": d['id'],
            "side": d.get("side", "").lower(),
            "type": d.get("type", "").lower(),
            "price": float(d.get("averagePrice", 0.0)),
            "filled_quantity": float(d.get("filledQty", 0.0)),
            "fee": float(d.get("payedFee", 0.0)),
            "timestamp": d.get("createdTime"),
            "user_address": WALLET_ADDRESS.lower()
        }

        if trade["filled_quantity"] > 0:
            trades.append(trade)

    return trades


def get_extended_balance():
    url = f"{BASE_URL}/api/v1/user/balance"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return {}

    data = response.json().get("data", [])

    balance = {
        "exchange": "extended",
        # "balance": float(data.get("balance", 0.0)),
        "equity": float(data.get("equity", 0.0)),
        # "unrealized_pnl": float(data.get("unrealisedPnl", 0.0)),
    }

    return balance


def get_extended_positions():
    """Fetch positions for a given market + side, transform to positions_dict format."""
    url = f"{BASE_URL}/api/v1/user/positions"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return []

    data = response.json().get("data", [])
    # print(data)
    positions_dict = []

    for pos in data:
        # Extract symbol (market like "PUMP-USD" -> "PUMP")
        symbol = pos["market"].split("-")[0] if "-" in pos["market"] else pos["market"]

        mapped = {
            "exchange": "extended",
            "symbol": symbol,
            "side": pos["side"].lower(),
            "base_amount": float(pos["size"]),
        }
        positions_dict.append(mapped)

    return positions_dict


def get_extended_collected_funding():
    """Loop over positions and add collected_funding from funding history."""
    url = f"{BASE_URL}/api/v1/user/funding/history"

    params = {
        "fromTime": 1755522896959  # Before competition started
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Funding Error {response.status_code}: {response.text}")
        return 0.0

    data = response.json().get("data", [])

    funding_records = []
    for f in data:
        funding_record = {
            "exchange": "extended",
            "symbol": f['market'].split("-")[0] if "-" in f['market'] else f['market'],
            "timestamp": f['paidTime'],
            "funding": float(f.get("fundingFee", 0.0)),
            "user_address": WALLET_ADDRESS.lower(),
        }
        funding_records.append(funding_record)

    return funding_records


if __name__ == "__main__":
    funding = get_extended_collected_funding()
    print(funding)
