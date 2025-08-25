import os
import lighter
import asyncio
import time
from lighter.api_client import ApiClient
from lighter.models.position_fundings import PositionFunding
from lighter.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://mainnet.zklighter.elliot.ai
# See configuration.py for a list of all supported configuration parameters.
BASE_URL = "https://mainnet.zklighter.elliot.ai"
PRIVATE_KEY = os.getenv("LIGHTER_SECRET_KEY")
ACCOUNT_INDEX = int(os.getenv("LIGHTER_ACCOUNT_INDEX"))
API_KEY_INDEX = int(os.getenv("LIGHTER_API_KEY_INDEX"))


async def get_lighter_balance():
    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    account_instance = lighter.AccountApi(api_client)

    result = await account_instance.account(by="l1_address", value="0x84EAec4953E02A07E9Ab79DB98C4dA1287Ed8FfB")

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
        "balance": balance,
        "equity": equity,
        "unrealized_pnl": unrealized_pnl
    }

    return balance


async def get_lighter_positions():
    api_client = ApiClient(configuration=lighter.Configuration(host=BASE_URL))
    account_instance = lighter.AccountApi(api_client)

    result = await account_instance.account(by="l1_address", value="0x84EAec4953E02A07E9Ab79DB98C4dA1287Ed8FfB")
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


async def get_lighter_collected_funding(account_instance, auth_token, positions):
    for pos in positions:
        market_id = pos['market_id']
        result = await account_instance.position_funding(
            ACCOUNT_INDEX,
            authorization=auth_token,
            auth=auth_token,
            limit=100,
            market_id=market_id
        )

        data = [pf.to_dict() for pf in result.position_fundings]
        collected_funding = 0
        for item in data:
            collected_funding = collected_funding + float(item['change'])
        pos['collected_funding'] = collected_funding


async def get_lighter_transactions(order_instance, auth_token, positions):
    for pos in positions:
        result = await order_instance.trades(
            sort_by='timestamp',
            limit=5,
            authorization=auth_token,
            auth=auth_token,
            market_id=45,
            account_index=ACCOUNT_INDEX,
        )
        pos['creation_timestamp'] = result.trades[0].timestamp if result.trades else None


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
    await get_lighter_transactions(order_instance, auth_token=auth_token, positions=position_dicts)

    await api_client.close()
    await client.close()

    return position_dicts


if __name__ == "__main__":
    positions = asyncio.run(get_lighter_positions_dict())
    print(positions)