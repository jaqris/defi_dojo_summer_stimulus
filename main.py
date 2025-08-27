import os
import asyncio
import streamlit as st
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, text
from get_portfolio_extended import get_extended_balance, get_extended_positions, get_extended_order_history, get_extended_collected_funding
from get_portfolio_lighter import get_lighter_balance, get_lighter_positions, get_lighter_order_history, get_lighter_collected_funding
from get_portfolio_drift import get_drift_balance, get_drift_positions, get_drift_order_history, get_drift_collected_funding
from get_portfolio_hyperliquid import get_hyperliquid_balance, get_hyperliquid_positions, get_hyperliquid_order_history, get_hyperliquid_collected_funding

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

        net_position = total_buy_qty - total_sell_qty

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

    return result, total_pnl_without_fees, total_fees, total_pnl_with_fees


def aggregate_funding(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate per exchange + symbol
    agg = (
        df.groupby(["exchange", "symbol"], as_index=False)["funding"]
        .sum()
        .rename(columns={"funding": "total_funding"})
    )

    total_funding = agg["total_funding"].sum()
    # totals row
    totals = {
        "exchange": "Total",
        "symbol": "",
        "total_funding": agg["total_funding"].sum()
    }
    agg = pd.concat([agg, pd.DataFrame([totals])], ignore_index=True)

    # 12 most recent funding records:
    most_recent = df.sort_values(by="timestamp", ascending=False).head(12)

    # convert timestamp to date time
    most_recent["timestamp"] = pd.to_datetime(most_recent["timestamp"], unit="ms")

    return agg, total_funding, most_recent


def aggregate_funding_by_4hr(df: pd.DataFrame, freq: str = "4H") -> pd.DataFrame:
    # Convert timestamp (ms) to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Aggregate per symbol
    symbol_agg = (
        df.set_index("datetime")
            .groupby("symbol")["funding"]
            .resample(freq)
            .sum()
            .reset_index()
    )

    # Aggregate total (all symbols combined)
    total_agg = (
        df.set_index("datetime")
            .resample(freq)["funding"]
            .sum()
            .reset_index()
    )
    total_agg["symbol"] = "TOTAL"

    # Combine
    agg_df = pd.concat([symbol_agg, total_agg], ignore_index=True)

    return agg_df


def plot_funding(agg_df: pd.DataFrame, freq: str):
    return px.line(
        agg_df,
        x="datetime",
        y="funding",
        color="symbol",
        title=f"Funding ({freq})",
        markers=True
    )


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


def get_balances():
    extended_balance = get_extended_balance()
    lighter_balance = asyncio.run(get_lighter_balance())
    drift_balance = asyncio.run(get_drift_balance())
    hyperliquid_balance = get_hyperliquid_balance()

    # Create DataFrame from list of dicts
    df = pd.DataFrame([extended_balance, lighter_balance, drift_balance, hyperliquid_balance])

    total_equity = df['equity'].sum()
    # Add total row
    total = pd.DataFrame({
        'exchange': ['Total'],
        # 'balance': [df['balance'].sum()],
        'equity': [df['equity'].sum()],
        # 'unrealized_pnl': [df['unrealized_pnl'].sum()]
    })

    df_total = pd.concat([df, total], ignore_index=True)


    return df_total[['exchange', 'equity']], total_equity


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
                        ) AS base_amount
                    FROM order_history
                    GROUP BY exchange, symbol
                )
                SELECT 
                    exchange,
                    symbol,
                    CASE WHEN base_amount < 0 THEN 'short' ELSE 'long' END AS side,
                    ABS(base_amount) AS base_amount
                FROM subquery
                WHERE base_amount != 0
                ORDER BY symbol, side;
            """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df
    else:
        extended_positions = get_extended_positions()
        lighter_positions = asyncio.run(get_lighter_positions())
        drift_positions = asyncio.run(get_drift_positions())
        hyperliquid_positions = get_hyperliquid_positions()

        df = pd.DataFrame(extended_positions + lighter_positions + drift_positions + hyperliquid_positions)

        return df.sort_values(by=['symbol', 'side'])


def get_orders():
    if FROM_DATABASE:
        with engine.connect() as conn:
            query = text("""
                SELECT exchange, symbol, side, type, price, filled_quantity, fee, timestamp
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

def main():
    st.set_page_config(page_title="Summer Stimulus", layout="wide")
    st.title("Jaqris' Portfolio Overview")

    # Get data
    df_balances, total_equity = get_balances()
    df_positions = get_positions()
    df_funding = get_collected_funding()
    df_orders = get_orders()

    # max timestamp
    st.text(f"Last updated: {pd.to_datetime(df_funding['timestamp'].max(), unit='ms').floor('s')} UTC")

    df_funding_summary, total_funding, recent_funding = aggregate_funding(df_funding)
    df_order_summary, total_pnl_without_fees, total_fees, _ = aggregate_pnl(df_orders)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total equity", f"${total_equity:.2f}")
    col2.metric("Collected funding", f"${total_funding:.2f}")
    col3.metric("PnL (excl. fees)", f"${total_pnl_without_fees:.2f}")
    col4.metric("Fees paid", f"${total_fees:.2f}")

    # col4.metric("PnL (disregarded from balance)", f"${total_pnl_without_fees:.2f}")

    # Display data
    col1, col2 = st.columns(2)
    with col1:
        st.header("Current Balances")
        st.table(df_balances)
        # st.header("Current Positions")
        # st.table(df_positions)
    with col2:
        st.header("Current Positions")
        st.table(df_positions)

    st.header("PNL Summary")
    st.table(df_order_summary)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Collected Funding")
        st.table(df_funding_summary)
    with col2:
        st.header("Most Recent Fundings")
        st.table(recent_funding)

    col1, col2, col3 = st.columns(3)

    # Cumulative funding
    with col1:
        st.plotly_chart(plot_cumulative_funding(df_funding), use_container_width=True)

    # 1D funding
    agg_1d = aggregate_funding_by_4hr(df_funding, "1D")
    with col2:
        st.plotly_chart(plot_funding(agg_1d, "1D"), use_container_width=True)

    # 8H funding
    agg_8h = aggregate_funding_by_4hr(df_funding, "8H")
    with col3:
        st.plotly_chart(plot_funding(agg_8h, "8H"), use_container_width=True)

    st.header("Order History")
    st.table(df_orders)


if __name__ == "__main__":
    main()