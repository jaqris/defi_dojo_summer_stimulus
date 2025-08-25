import os

import requests
import time

# API details
API_KEY = os.getenv("EXTENDED_API_KEY")
BASE_URL = "https://api.starknet.extended.exchange"

# Headers (API key + required User-Agent)
headers = {
    "X-Api-Key": API_KEY,
    "User-Agent": "MyPythonClient/1.0"
}


def get_extended_balance():
    url = f"{BASE_URL}/api/v1/user/balance"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return {}

    data = response.json().get("data", [])

    balance = {
        "exchange": "extended",
        "balance": float(data.get("balance", 0.0)),
        "equity": float(data.get("equity", 0.0)),
        "unrealized_pnl": float(data.get("unrealisedPnl", 0.0)),
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
            # "market_id": pos["id"],  # Position ID
            "symbol": symbol,
            "side": pos["side"].lower(),  # "LONG"/"SHORT" -> "long"/"short"
            "base_amount": float(pos["size"]),
            # "avg_entry_price": float(pos["openPrice"]),
            # "liquidation_price": float(pos["liquidationPrice"]),
            # "mark_price": float(pos["markPrice"]),
            # "collected_funding": 0.0,  # will be filled later
            # "creation_timestamp": pos["createdAt"]
        }
        positions_dict.append(mapped)

    return positions_dict


def get_extended_collected_funding(positions):
    """Loop over positions and add collected_funding from funding history."""
    url = f"{BASE_URL}/api/v1/user/funding/history"

    for pos in positions:
        params = {
            "market": pos["symbol"] + "-USD",  # assuming all are against USD
            "side": pos["side"],
            "fromTime": pos["creation_timestamp"]
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Funding Error {response.status_code}: {response.text}")
            return 0.0

        funding_data = response.json().get("data", [])
        total_funding = sum(float(f["fundingFee"]) for f in funding_data if "fundingFee" in f)
        pos["collected_funding"] = total_funding

        # Small sleep to avoid rate limiting (adjust as needed)
        time.sleep(0.2)


def get_extended_positions_dict():

    balances = get_extended_balance()
    positions = get_extended_positions()
    get_extended_collected_funding(positions)

    return positions


if __name__ == "__main__":
    positions = get_extended_positions_dict()
    print(positions)
