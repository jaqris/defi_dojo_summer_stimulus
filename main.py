import os
import asyncio
import streamlit as st
import plotly.express as px
import pandas as pd
import requests
import numpy as np
from sqlalchemy import create_engine, text
from get_portfolio_extended import get_extended_balance, get_extended_positions, get_extended_order_history, get_extended_collected_funding
from get_portfolio_lighter import get_lighter_balance, get_lighter_positions, get_lighter_order_history, get_lighter_collected_funding
from get_portfolio_drift import get_drift_balance, get_drift_positions, get_drift_order_history, get_drift_collected_funding
from get_portfolio_hyperliquid import get_hyperliquid_balance, get_hyperliquid_positions, get_hyperliquid_order_history, get_hyperliquid_collected_funding
from concurrent.futures import ThreadPoolExecutor
# from exchange_adapters import ExtendedAdapter, HyperliquidAdapter
from exchange_adapters.drift import DriftAdapter
from exchange_adapters.lighter import LighterAdapter

FROM_DATABASE = True


# Pick host depending on environment
if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
    db_host = os.environ.get("RENDER_PSQL_ZDY_DB_HOST_INTERNAL")
else:
    db_host = os.environ.get("RENDER_PSQL_ZDY_DB_HOST_EXTERNAL")

# Database connection URI
db_uri = (
    f"postgresql+pg8000://{os.environ.get('RENDER_PSQL_ZDY_DB_USER')}:"
    f"{os.environ.get('RENDER_PSQL_ZDY_DB_PASSWORD')}@{db_host}:"
    f"{os.environ.get('RENDER_PSQL_ZDY_DB_PORT')}/"
    f"{os.environ.get('RENDER_PSQL_ZDY_DB_NAME')}"
)

# Create engine
engine = create_engine(db_uri, echo=False, future=True)

lighter = LighterAdapter()
drift = DriftAdapter()
# extended = ExtendedAdapter()
# hyperliquid = HyperliquidAdapter()


def aggregate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    # ensure numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["filled_quantity"] = pd.to_numeric(df["filled_quantity"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)

    # cash flow = price * quantity
    df["value"] = df["price"] * df["filled_quantity"]

    agg = []
    for sym, g in df.groupby("symbol"):
        buys = g[g["side"].str.lower() == "buy"]
        sells = g[g["side"].str.lower() == "sell"]

        total_buy_qty = buys["filled_quantity"].sum()
        total_buy_value = buys["value"].sum()

        total_sell_qty = sells["filled_quantity"].sum()
        total_sell_value = sells["value"].sum()

        total_fees = g["fee"].sum()

        avg_buy_price = total_buy_value / total_buy_qty if total_buy_qty > 0 else None
        avg_sell_price = total_sell_value / total_sell_qty if total_sell_qty > 0 else None

        pnl_no_fee = total_sell_value - total_buy_value
        pnl_with_fee = pnl_no_fee - total_fees

        agg.append({
            "symbol": sym,
            "total_sell_qty": total_sell_qty,
            "avg_sell_price": avg_sell_price,
            "total_buy_qty": total_buy_qty,
            "avg_buy_price": avg_buy_price,
            "PnL_no_fee": pnl_no_fee,
            "total_fees": total_fees,
            "PnL_with_fee": pnl_with_fee
        })

    result = pd.DataFrame(agg)

    total_pnl_without_fees = result["PnL_no_fee"].sum()
    total_fees = result["total_fees"].sum()
    total_pnl_with_fees = result["PnL_with_fee"].sum()

    # add totals row
    totals = {
        "symbol": "Total",
        "total_sell_qty": None,
        "avg_sell_price": None,
        "total_buy_qty": None,
        "avg_buy_price": None,
        "PnL_no_fee": result["PnL_no_fee"].sum(),
        "total_fees": result["total_fees"].sum(),
        "PnL_with_fee": result["PnL_with_fee"].sum()
    }
    result = pd.concat([result, pd.DataFrame([totals])], ignore_index=True)

    # ---- HOURLY SUMMARY ----
    df["hour"] = pd.to_datetime(df["timestamp"], unit="ms").dt.floor("h")

    hourly_summary = (
        df.assign(
            PnL_no_fee=np.where(
                df["side"].str.lower() == "sell",
                df["value"],
                -df["value"]
            )
        )
            .groupby("hour", as_index=False)
            .agg(
            PnL_no_fee=("PnL_no_fee", "sum"),
            total_fees=("fee", "sum")
        )
    )
    hourly_summary["PnL_with_fee"] = hourly_summary["PnL_no_fee"] - hourly_summary["total_fees"]

    return result, total_pnl_without_fees, total_fees, total_pnl_with_fees, hourly_summary


def aggregate_funding(df: pd.DataFrame) -> pd.DataFrame:
    # pivot: symbols down rows, exchanges across columns, sum of funding in cells
    agg = pd.pivot_table(
        df,
        values="funding",
        index="symbol",
        columns="exchange",
        aggfunc="sum",
        fill_value=0,
        margins=True,          # adds total row and total column
        margins_name="Total"   # name for totals
    ).reset_index()

    # overall total funding
    total_funding = df["funding"].sum()

    # 12 most recent funding records (convert ms -> datetime)
    most_recent = (
        df.sort_values(by="timestamp", ascending=False)
          .head(12)
          .copy()
    )
    most_recent["timestamp"] = pd.to_datetime(most_recent["timestamp"], unit="ms")

    # hourly aggregation of total funding
    df["hour"] = pd.to_datetime(df["timestamp"], unit="ms").dt.floor("h")
    hourly_funding = (
        df.groupby("hour", as_index=False)["funding"]
          .sum()
          .rename(columns={"funding": "total_funding"})
    )

    return agg, total_funding, most_recent, hourly_funding


def plot_cumulative_funding(df: pd.DataFrame):
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Cumulative funding by symbol
    df_sorted = df.sort_values("datetime")
    cum_df = (
        df_sorted.groupby("symbol")[["datetime", "funding"]]
            .apply(lambda g: g.assign(cumulative=g["funding"].cumsum()))
            .reset_index(drop=True)
    )

    # Add total
    total_df = (
        df_sorted[["datetime", "funding"]]
            .assign(cumulative=df_sorted["funding"].cumsum())
    )
    total_df["symbol"] = "TOTAL"

    cum_df = pd.concat([cum_df, total_df], ignore_index=True)

    return px.line(
        cum_df,
        x="datetime",
        y="cumulative",
        color="symbol",
        title="Cumulative Funding",
        markers=False
    )


def daily_contribution_chart(df_funding: pd.DataFrame):
    """
    Returns a stacked bar chart of funding contributions per symbol, aggregated weekly.
    """
    df = df_funding.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['week'] = df['timestamp'].dt.to_period('D').apply(lambda r: r.start_time)

    # aggregate funding per week and symbol
    weekly = (
        df.groupby(['week', 'symbol'], as_index=False)
        .agg({'funding': 'sum'})
    )

    fig = px.bar(
        weekly,
        x='week',
        y='funding',
        color='symbol',
        title='Daily Contribution per Symbol',
        barmode='stack'
    )
    fig.update_layout(xaxis_title="Week", yaxis_title="Funding")
    return fig


def rolling_avg_chart(df_funding: pd.DataFrame, window: int = 7):
    """
    Returns a line chart of rolling average funding per symbol.
    Default window = 7 days.
    """
    df = df_funding.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp').sort_index()

    # aggregate daily funding first
    daily = (
        df.groupby([pd.Grouper(freq='1D'), 'symbol'])
        .agg({'funding': 'sum'})
        .reset_index()
    )

    # calculate rolling mean
    daily['rolling_avg'] = (
        daily.groupby('symbol')['funding']
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

    fig = px.line(
        daily,
        x='timestamp',
        y='rolling_avg',
        color='symbol',
        title=f'{window}D Rolling Average Funding per Symbol'
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Funding (Rolling Avg)")
    return fig


# async def get_all_balances():
#     extended_task = asyncio.to_thread(get_extended_balance)  # wrap sync function
#     hyperliquid_task = asyncio.to_thread(get_hyperliquid_balance)
#     lighter_task = get_lighter_balance()  # already async
#     drift_task = get_drift_balance()      # already async
#
#     results = await asyncio.gather(
#         extended_task, lighter_task, drift_task, hyperliquid_task
#     )
#     return pd.DataFrame(results)



def get_balances():
    # df = asyncio.run(get_all_balances())

    with ThreadPoolExecutor() as executor:
        futures = [
            # executor.submit(get_extended_balance),
            executor.submit(lambda: asyncio.run(get_lighter_balance())),
            executor.submit(lambda: asyncio.run(get_drift_balance())),
            # executor.submit(get_hyperliquid_balance),
        ]
        results = [f.result() for f in futures]
    df = pd.DataFrame(results)

    total_equity = df['equity'].sum()
    # Add total row
    total = pd.DataFrame({
        'exchange': ['Total'],
        'equity': [df['equity'].sum()],
        'maintenance_margin': None,
        'notional_exposure': None,
        'leverage': None,
        'health_ratio': None
    })

    df_total = pd.concat([df, total], ignore_index=True)
    df_total['equity'] = df_total['equity'].round(2)

    return df_total[['exchange', 'equity', 'maintenance_margin',
                     'notional_exposure', 'leverage', 'health_ratio']], total_equity


def get_prices_from_lighter():
    url = "https://mainnet.zklighter.elliot.ai/api/v1/orderBookDetails"
    resp = requests.get(url)
    resp.raise_for_status()
    order_book = resp.json()['order_book_details']

    return {m['symbol']: m['last_trade_price'] for m in order_book}


def get_prices_from_extended():
    url = "https://api.extended.exchange/api/v1/info/markets"
    resp = requests.get(url)
    resp.raise_for_status()
    markets = resp.json()['data']

    return {m['assetName']: float(m['marketStats']['markPrice']) for m in markets}


def get_prices_for_open_positions(df_positions):
    lighter_prices = get_prices_from_lighter()

    # Fill from lighter first
    df_positions["price"] = df_positions["symbol"].map(lighter_prices)

    # 2. Check if we still have missing values
    missing_mask = df_positions["price"].isna()
    if missing_mask.any():
        # Only pull extended if needed
        extended_prices = get_prices_from_extended()
        df_positions.loc[missing_mask, "price"] = (
            df_positions.loc[missing_mask, "symbol"].map(extended_prices)
        )

    return df_positions


def get_latest_funding_for_exchange_coin(exchange, coin):
    query = text(f"""
            WITH ranked AS (
                SELECT
                    exchange,
                    symbol,
                    fundingrate,
                    interval,
                    datetime,
                    ROW_NUMBER() OVER (PARTITION BY exchange, symbol ORDER BY datetime DESC) AS rn
                FROM perp_market_data
                WHERE 
                    exchange = '{exchange}' AND
                    symbol ILIKE '{coin}/%'
                    AND datetime >= NOW() - INTERVAL '25 hours'
            )
            SELECT
                exchange,
                split_part(symbol,'/',1) as symbol,
                MAX(CASE WHEN rn = 1 THEN fundingrate/ interval END) * (24 ) * 365 * 100 AS apy_1hr,
                AVG(CASE WHEN rn <= 8 THEN fundingrate/ interval END) * (24) * 365 * 100 AS apy_8hr,
                AVG(CASE WHEN rn <= 24 THEN fundingrate/ interval END) * (24) * 365 * 100 AS apy_24hr
            FROM ranked
            GROUP BY exchange, symbol
        """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


async def get_all_balances():
    # extended_balance = await extended.get_balances()
    # hyperliquid_balance = await hyperliquid.get_balances()
    lighter_balance = await lighter.history.get_balances()
    drift_balance = await drift.history.get_balances()

    df = pd.DataFrame([
        #extended_balance,
        lighter_balance,
        drift_balance,
        #hyperliquid_balance
    ])
    total_equity = df['equity'].sum()
    # Add total row
    total = pd.DataFrame({
        'exchange': ['Total'],
        'equity': [df['equity'].sum()],
        'maintenance_margin': None,
        'notional_exposure': None,
        'leverage': None,
        'health_ratio': None
    })

    df_total = pd.concat([df, total], ignore_index=True)
    df_total['equity'] = df_total['equity'].round(2)

    return df_total[['exchange', 'equity', 'maintenance_margin',
                     'notional_exposure', 'leverage', 'health_ratio']], total_equity


async def get_all_positions():
    extended_positions = await extended.get_positions()
    hyperliquid_positions = await hyperliquid.get_positions()  # asyncio.to_thread(get_hyperliquid_positions)
    lighter_positions = await lighter.get_positions()  # async
    drift_positions = await drift.history.get_positions()      # async

    result = pd.DataFrame(lighter_positions + drift_positions + extended_positions + hyperliquid_positions)
    result = result.loc[result['base_amount'] != 0]
    return result


def get_positions():
    if FROM_DATABASE:
        query = text("""
                WITH subquery AS (
                    SELECT
                        exchange,
                        symbol,
                        SUM(
                            CASE
                                WHEN side = 'sell' THEN -1 * filled_quantity
                                ELSE filled_quantity
                            END
                        ) AS base_amount,
                        AVG(price) AS price
                    FROM order_history
                    GROUP BY exchange, symbol
                )
                SELECT
                    exchange,
                    symbol,
                    CASE WHEN base_amount < 0 THEN 'short' ELSE 'long' END AS side,
                    ROUND(ABS(base_amount) * 100) / 100 AS base_amount
                FROM subquery
                WHERE ABS(base_amount) * price > 0.01
                ORDER BY symbol, side;
            """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                # executor.submit(get_extended_positions),
                executor.submit(lambda: asyncio.run(lighter.get_positions())),
                executor.submit(lambda: asyncio.run(get_drift_positions())),
                # executor.submit(get_hyperliquid_positions),
            ]
            results = [f.result() for f in futures]

        # results is a list of lists of dicts (one per exchange)
        df = pd.DataFrame(sum(results, []))  # flatten list of lists

        return df.sort_values(by=["symbol", "side"])


async def get_all_orders():
    extended_orders = await extended.get_order_history()
    lighter_orders = await lighter.get_order_history()
    drift_orders = await drift.history.get_order_history()
    hyperliquid_orders = await hyperliquid.get_order_history()
    df = pd.DataFrame(extended_orders + lighter_orders + drift_orders + hyperliquid_orders)
    df = df.sort_values(by=['timestamp']).reset_index(drop=True)
    return df[['exchange', 'symbol', 'side', 'price', 'filled_quantity', 'fee', 'timestamp']]

def get_orders():
    if FROM_DATABASE:
        with engine.connect() as conn:
            query = text("""
                SELECT exchange, symbol, side, price, filled_quantity, fee, timestamp
                FROM order_history
                ORDER BY timestamp
            """)
            df = pd.read_sql(query, conn)
        return df.reset_index(drop=True)
    else:
        extended_orders = get_extended_order_history()
        lighter_orders = asyncio.run(get_lighter_order_history())
        drift_orders = get_drift_order_history()
        hyperliquid_orders = get_hyperliquid_order_history()

        df = pd.DataFrame(extended_orders + lighter_orders + drift_orders + hyperliquid_orders)
        df = df.sort_values(by=['timestamp']).reset_index(drop=True)

        return df[['exchange', 'symbol', 'side', 'price', 'filled_quantity', 'fee', 'timestamp']]


async def get_all_collected_funding():
    extended_funding = await extended.get_collected_funding()
    hyperliquid_funding = await hyperliquid.get_collected_funding()
    lighter_funding = await lighter.get_collected_funding()
    drift_funding = await drift.history.get_collected_funding()

    df = pd.DataFrame(extended_funding + lighter_funding + drift_funding + hyperliquid_funding)
    # df = pd.DataFrame(lighter_funding)
    return df.sort_values(by=['timestamp']).reset_index(drop=True)


def get_collected_funding():
    if FROM_DATABASE:
        with engine.connect() as conn:
            query = text("""
                SELECT exchange, symbol, funding, timestamp
                FROM collected_funding
                ORDER BY timestamp
            """)
            df = pd.read_sql(query, conn)
        return df.reset_index(drop=True)
    else:
        extended_funding = get_extended_collected_funding()
        lighter_funding = asyncio.run(get_lighter_collected_funding())
        drift_funding = get_drift_collected_funding()
        hyperliquid_funding = get_hyperliquid_collected_funding()

        df = pd.DataFrame(extended_funding + lighter_funding + drift_funding)
        # df = pd.DataFrame(lighter_funding)
        return df.sort_values(by=['timestamp']).reset_index(drop=True)


def calculate_apy(
        df_hourly_funding: pd.DataFrame,
        df_hourly_pnl: pd.DataFrame,
        hours: int = 24,
        capital: float = 10_000):
    """
    Calculate APY over the last 24 hours based on cumulative funding + pnl.

    Args:
        df_hourly_funding: DataFrame with columns ["hour", "total_funding"]
        df_hourly_pnl: DataFrame with columns ["hour", "PnL_with_fee"]
        capital: starting balance (default 10,000)

    Returns:
        apy (float): annualized percentage yield
        balance_curve (pd.Series): balance over time
    """
    # merge funding + pnl by hour
    df = pd.merge(df_hourly_funding, df_hourly_pnl, on="hour", how="outer").fillna(0)

    # sort by hour
    df = df.sort_values("hour")

    # net cash flow each hour
    df["net"] = df["total_funding"] + df["PnL_with_fee"]

    # cumulative balance
    df["balance"] = capital + df["net"].cumsum()

    # --- cap hours ---
    available_hours = len(df)
    hours = min(hours, available_hours)

    # select last N hours
    last_nh = df.tail(hours)

    start_balance = last_nh["balance"].iloc[0]
    end_balance = last_nh["balance"].iloc[-1]

    # growth over this period
    period_return_pct = (end_balance - start_balance) / start_balance

    # scale to daily return (normalize for hours span)
    daily_return_pct = period_return_pct * (24 / hours)

    # annualize without compounding (APR)
    apr = period_return_pct * (365 * 24 / hours)

    return apr, daily_return_pct, start_balance, end_balance


def add_liquidation_prices(df_positions: pd.DataFrame, df_balances: pd.DataFrame) -> pd.DataFrame:
    """
    Adds liquidation price to df_positions using df_balances.
    Formula: P_liq = P_m + (E - MR) / Q

    df_positions: DataFrame with columns [exchange, symbol, base_amount, price]
    df_balances: DataFrame with columns [exchange, equity, margin_required, notional_exposure]

    Returns: df_positions with 'liq_price' column added (if not already present).
    """

    # Merge balances into positions
    merged = df_positions.merge(
        df_balances[["exchange", "equity", "maintenance_margin"]],
        on="exchange",
        how="left"
    )

    def calc_liq(row):
        Q = row["base_amount"]
        Pm = row["price"]
        E = row["equity"]
        MR = row["maintenance_margin"]

        # Avoid division by zero (no position)
        if Q == 0:
            return None
        
        if row['side'] == 'short':
            liq_price = Pm + (E - 1.20 * MR) / abs(Q)
        else:
            liq_price = Pm - (E - 1.20 * MR) / Q

        return liq_price if liq_price > 0 else None

    if "liquidation_price" not in merged.columns:
        merged["liquidation_price"] = None

    merged["liquidation_price"] = merged.apply(calc_liq, axis=1)
    merged["liq_distance_pct"] = ((merged["liquidation_price"] - merged["price"]).abs() / merged["price"]) * 100
    # Return df_positions with new column
    return merged[df_positions.columns.tolist() + ["liquidation_price", "liq_distance_pct"]]


def enrich_positions(df_positions: pd.DataFrame, df_balances: pd.DataFrame):
    results = []

    for _, row in df_positions.iterrows():
        exchange = row["exchange"]
        coin = row["symbol"]

        exchange_coin_funding = get_latest_funding_for_exchange_coin(exchange, coin)
        results.append(exchange_coin_funding)

    if results:
        df_funding = pd.concat(results, ignore_index=True)
    else:
        df_funding = pd.DataFrame(columns=["exchange", "symbol", "latest_funding", "avg_last_8", "avg_last_24"])

    # Merge on exchange + coin (base coin match: symbol LIKE coin/%)
    df_positions = df_positions.merge(
        df_funding,
        on=["exchange", "symbol"],
        how="left"
    )

    df_positions['apy_1hr'] = df_positions.apply(
        lambda row: -row['apy_1hr'] if row['side'] == 'long' else row['apy_1hr'],
        axis=1
    )

    df_positions['apy_8hr'] = df_positions.apply(
        lambda row: -row['apy_8hr'] if row['side'] == 'long' else row['apy_8hr'],
        axis=1
    )

    df_positions['apy_24hr'] = df_positions.apply(
        lambda row: -row['apy_24hr'] if row['side'] == 'long' else row['apy_24hr'],
        axis=1
    )
    df_positions = get_prices_for_open_positions(df_positions)
    df_positions = add_liquidation_prices(df_positions, df_balances)

    return df_positions.loc[abs(df_positions['base_amount']) > 0]


async def main():
    st.set_page_config(page_title="Summer Stimulus", layout="wide")
    st.title("Jaqris' Portfolio Overview")
    st.markdown(f"Currently only trading on Drift and Lighter. \n"
                f"[Drift Portfolio](https://app.drift.trade/overview/history?authority=ENo6PA4ypDxy9rfDZiusnZw4ZfKpFzbqkd3HEFyRTDhT) \n"
                f"[Lighter Portfolio](https://lightlens.vercel.app/traders/0x84EAec4953E02A07E9Ab79DB98C4dA1287Ed8FfB)")

    # Get data
    result = await get_all_balances()
    df_balances, total_equity = result
    df_positions = get_positions()
    df_funding = get_collected_funding()
    df_orders = get_orders()
    
    # max timestamp
    st.text(f"Last updated: {pd.to_datetime(df_funding['timestamp'].max(), unit='ms').floor('s')} UTC")

    df_funding_summary, total_funding, recent_funding, df_hourly_funding = aggregate_funding(df_funding)
    df_order_summary, total_pnl_without_fees, total_fees, total_pnl_with_fees, df_hourly_pnl = aggregate_pnl(df_orders)

    # APY overall (use all hours in data)
    apy_overall, _, _, _ = calculate_apy(df_hourly_funding, df_hourly_pnl, hours=2400)
    apy_7d, _, _, _ = calculate_apy(df_hourly_funding, df_hourly_pnl, hours=7 * 24)
    apy_24h, _, _, _ = calculate_apy(df_hourly_funding, df_hourly_pnl, hours=24)
    apy_8h, _, _, _ = calculate_apy(df_hourly_funding, df_hourly_pnl, hours=8)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total equity", f"${total_equity:.2f}")
    col2.metric("Collected funding", f"${total_funding:.2f}")
    col3.metric("PnL (fees+entry/exit arb)", f"${total_pnl_with_fees:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("APR (overall)", f"{apy_overall*100:.2f}%")
    col2.metric("APR (last 7d)", f"{apy_7d*100:.2f}%")
    col3.metric("APR (last 24h)", f"{apy_24h*100:.2f}%")
    col4.metric("APR (last 8h)", f"{apy_8h * 100:.2f}%")

    # get latest funding for position data
    df_positions = enrich_positions(df_positions, df_balances)

    # Display data
    col1, col2 = st.columns([4, 6])
    with col1:
        st.header("Current Balances")
        st.table(df_balances[['exchange', 'equity', 'leverage', 'health_ratio']])
    with col2:
        st.header("Current Positions")
        st.dataframe(df_positions.style.format({
            "base_amount": "{:.0f}",
            "apy_1hr": "{:.2f}%",
            "apy_8hr": "{:.2f}%",
            "apy_24hr": "{:.2f}%",
            "price": "${:.4f}",
            "liquidation_price": "${:.4f}",
            "liq_distance_pct": "{:.0f}%",

        }))
        st.markdown("** Liquidation price is an estimate. Verify on exchange directly")

    col1, col2, col3 = st.columns(3)

    # Cumulative funding
    with col1:
        st.plotly_chart(plot_cumulative_funding(df_funding), use_container_width=True)
    with col2:
        st.plotly_chart(daily_contribution_chart(df_funding), use_container_width=True)
    with col3:
        st.plotly_chart(rolling_avg_chart(df_funding), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Collected Funding")
        st.table(df_funding_summary)
    with col2:
        st.header("Most Recent Fundings")
        st.table(recent_funding)

    st.header("PNL Summary")
    st.table(df_order_summary)

    # st.header("Order History")
    # st.table(df_orders)

    await lighter.close()
    await drift.close()
    # await extended.close()
    # await hyperliquid.close()


if __name__ == "__main__":
    asyncio.run(main())