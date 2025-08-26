import os
import lighter
import asyncio
import time
import requests
from lighter.api_client import ApiClient
from lighter.models.position_fundings import PositionFunding
from lighter.rest import ApiException
from pprint import pprint
from typing import Dict


# Defining the host is optional and defaults to https://mainnet.zklighter.elliot.ai
# See configuration.py for a list of all supported configuration parameters.
BASE_URL = "https://mainnet.zklighter.elliot.ai"
PRIVATE_KEY = os.getenv("LIGHTER_SECRET_KEY")
ACCOUNT_INDEX = int(os.getenv("LIGHTER_ACCOUNT_INDEX"))
API_KEY_INDEX = int(os.getenv("LIGHTER_API_KEY_INDEX"))
WALLET_ADDRESS = os.getenv("ETH_COMP_PUBLIC_KEY")


def fetch_market_index_to_symbol_map():
    order_book_url = f"{BASE_URL}/api/v1/orderBookDetails"
    resp = requests.get(order_book_url)
    resp.raise_for_status()
    order_book = resp.json()

    mapping: Dict[int, str] = {}

    for item in order_book['order_book_details']:
        # The fields might be named differently; adjust as needed
        market_id = item.get("market_id")
        symbol = item.get("symbol")

        if market_id is None or symbol is None:
            continue

        try:
            market_id = int(market_id)
        except (ValueError, TypeError):
            continue

        mapping[market_id] = symbol

    return mapping



async def get_lighter_balance():
    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    account_instance = lighter.AccountApi(api_client)

    result = await account_instance.account(by="l1_address", value=WALLET_ADDRESS)

    accounts = result.accounts

    balance = 0.0
    equity = 0.0
    unrealized_pnl = 0.0
    for account in accounts:

        balance = balance + float(account.available_balance if account.available_balance else 0.0)
        equity = equity + float(account.total_asset_value if account.total_asset_value else 0.0)

        positions = account.positions
        for pos in positions:
            unrlzd = float(pos.unrealized_pnl) if pos.unrealized_pnl else 0.0
            unrealized_pnl += unrlzd

    balance = {
        "exchange": "lighter",
        # "balance": balance,
        "equity": equity,
        # "unrealized_pnl": unrealized_pnl
    }

    return balance


async def get_lighter_positions():
    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    account_instance = lighter.AccountApi(api_client)

    result = await account_instance.account(by="l1_address", value=WALLET_ADDRESS)
    position_dicts = []
    accounts = result.accounts

    for account in accounts:
        positions = account.positions
        for pos in positions:
            pos_dict = {
                "exchange": "lighter",
                # "market_id": pos.market_id,
                "symbol": pos.symbol,
                "side": "long" if pos.sign == 1 else "short",
                "base_amount": float(pos.position),
                # "avg_entry_price": float(pos.avg_entry_price),
                # "liquidation_price": float(pos.additional_properties['liquidation_price']),
            }
            position_dicts.append(pos_dict)

    return position_dicts


async def get_lighter_mark_prices_for_positions(candlestick_instance, positions_dict):
    end_timestamp = round(time.time())
    for pos in positions_dict:
        result = await candlestick_instance.candlesticks(
            market_id=pos['market_id'],
            resolution='1m',
            start_timestamp=end_timestamp - 60,
            end_timestamp=end_timestamp,
            count_back=1
        )
        pos['mark_price'] = result.candlesticks[0].close if result.candlesticks else None\


async def get_lighter_collected_funding():
    market_mappings = fetch_market_index_to_symbol_map()

    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    client = lighter.SignerClient(
        url=BASE_URL,
        private_key=PRIVATE_KEY,
        account_index=ACCOUNT_INDEX,
        api_key_index=API_KEY_INDEX
    )
    auth_token, _ = client.create_auth_token_with_expiry()

    # Create instances of the API classes
    account_instance = lighter.AccountApi(api_client)

    funding_records = []
    next_cursor = None

    while True:
        try:
            result = await account_instance.position_funding(
                ACCOUNT_INDEX,
                authorization=auth_token,
                auth=auth_token,
                limit=100,
                cursor=next_cursor
            )
            for f in result.position_fundings:
                funding_record = {
                    "exchange": "lighter",
                    "symbol": market_mappings[f.market_id],
                    "timestamp": f.timestamp * 1000,
                    "funding": float(f.change if f.change else 0.0),
                    "user_address": WALLET_ADDRESS.lower()
                }
                funding_records.append(funding_record)

            next_cursor = result.next_cursor if result.next_cursor else None

            if not next_cursor:
                break
        except ApiException as e:
            print("Exception when calling AccountApi->position_funding: %s\n" % e)
            break

    return funding_records


async def get_lighter_order_history():
    market_mappings = fetch_market_index_to_symbol_map()

    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    client = lighter.SignerClient(
        url=BASE_URL,
        private_key=PRIVATE_KEY,
        account_index=ACCOUNT_INDEX,
        api_key_index=API_KEY_INDEX
    )
    auth_token, _ = client.create_auth_token_with_expiry()
    order_instance = lighter.OrderApi(api_client)

    result = await order_instance.trades(
        sort_by='timestamp',
        limit=50,
        authorization=auth_token,
        auth=auth_token,
        account_index=ACCOUNT_INDEX,
    )

    trades = []

    for r in result.trades:
        if r.bid_account_id == ACCOUNT_INDEX:
            side = "buy"  # bid
            if r.is_maker_ask:
                type = "market"  # taker
                fee = float(r.taker_fee or 0.0)
            else:
                type = "limit"  # maker
                fee = float(r.maker_fee or 0.0)

        else:
            side = "sell"  # ask
            if r.is_maker_ask:
                type = "limit"  # maker
                fee = float(r.maker_fee or 0.0)
            else:
                type = "market"  # taker
                fee = float(r.taker_fee or 0.0)

        # print(r)
        trade = {
            "exchange": "lighter",
            "symbol": market_mappings[r.market_id],
            "trade_id": r.trade_id,
            "side": side,
            "type": type,
            "price": float(r.price if r.price else 0.0),
            "filled_quantity": float(r.size if r.size else 0.0),
            "fee": fee,
            "timestamp": r.timestamp,
            "user_address": WALLET_ADDRESS.lower()
        }
        trades.append(trade)
    return trades



async def get_lighter_positions_dict():
    """
        Returns a list of dicts with the following fields:
        [
          {
            "exchange": "lighter",
            "market_id": 45,
            "symbol": "PUMP",
            "side": "short",
            "base_amount": 4000000.0,
            "avg_entry_price": 0.003163,
            "liquidation_price": 0.0041078863711627904,
            "mark_price": 0.002966,
            "collected_funding": 11.911396,
            "creation_timestamp": 1755722896959
          }
        ]

    """
    # Initialize the API client
    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    client = lighter.SignerClient(
        url=BASE_URL,
        private_key=PRIVATE_KEY,
        account_index=ACCOUNT_INDEX,
        api_key_index=API_KEY_INDEX
    )
    auth_token, _ = client.create_auth_token_with_expiry()

    # Create instances of the API classes
    account_instance = lighter.AccountApi(api_client)
    order_instance = lighter.OrderApi(api_client)
    candlestick_instance = lighter.CandlestickApi(api_client)

    # Get most of position info
    position_dicts = await get_lighter_positions(account_instance)

    # Add mark prices, collected funding, and creation timestamps to pos dicts
    await get_lighter_mark_prices_for_positions(candlestick_instance, position_dicts)
    await get_lighter_collected_funding(account_instance, auth_token=auth_token, positions=position_dicts)
    await get_lighter_order_history(order_instance, auth_token=auth_token, positions=position_dicts)

    await api_client.close()
    await client.close()

    return position_dicts


if __name__ == "__main__":
    trades = asyncio.run(get_lighter_order_history())
    print(trades)