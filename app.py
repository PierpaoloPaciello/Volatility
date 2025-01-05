import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Streamlit App Title
st.set_page_config(page_title="Volatility Backtest", layout="wide")
st.title("Volatility Strategy Backtest")

# Sidebar for User Input
st.sidebar.header("Strategy Parameters")

INITIAL_INVESTMENT = st.sidebar.number_input("Initial Investment", value=50000, step=1000)
INVESTMENT_STEPS = [
    50000,
    75000,
    100000,
    125000,
    150000,
    175000
]
PROFIT_THRESHOLD = st.sidebar.slider("Profit Threshold (x Average Cost)", 1.0, 2.0, 1.5, step=0.1)
LOSS_THRESHOLD = st.sidebar.slider("Loss Threshold (as % of Max Price)", 0.5, 1.0, 0.65, step=0.01)
CREDIT_LIMIT = st.sidebar.number_input("Bank Credit Limit", value=1000000, step=10000)
CREDIT_RATE = st.sidebar.number_input("Credit Rate (Annual)", value=0.05, step=0.01)

st.sidebar.markdown("---")

# Load Data
st.header("Market Data")

with st.spinner("Downloading data from Yahoo Finance..."):
    data = yf.download("^SHORTVOL", start="2006-01-01")
data = data["Close"].reset_index()
data.columns = ["Date", "Price"]
data.dropna(inplace=True)

# Display data
st.write("### Price Data")
st.dataframe(data.tail())

# Backtest Logic
st.header("Backtest Results")

# Initialize variables
position_size = INITIAL_INVESTMENT / data["Price"].iloc[0]  # Initial position in shares
average_cost = data["Price"].iloc[0]
current_step = 0
max_price = data["Price"].iloc[0]  # Start tracking max price

used_credit = INITIAL_INVESTMENT  # Initial credit used
profit_account = 0  # Separate profit account
credit_interest_cost = 0  # Accumulated interest cost

portfolio_log = []  # To store daily portfolio details
annual_profits = {}  # To store annual profits
annual_interest_costs = {}  # To store annual interest costs
total_net_profit = {}  # Annual net profit (profit - interest)

tracking_max_price = True  # Enable tracking max price after take profit

# Backtesting Loop
for i, row in data.iterrows():
    price = row["Price"]
    date = row["Date"]

    # Track the maximum price after take profit
    if tracking_max_price:
        max_price = max(max_price, price)

    # Determine current position value
    position_value = position_size * price

    # Calculate daily credit interest
    daily_interest = (used_credit * CREDIT_RATE) / 252
    credit_interest_cost += daily_interest

    # Log annual interest costs
    year = date.year
    if year not in annual_interest_costs:
        annual_interest_costs[year] = 0
    annual_interest_costs[year] += daily_interest

    # Log portfolio details for the day
    portfolio_log.append({
        'Date': date,
        'Price': price,
        'Position Size': position_size,
        'Position Value': position_value,
        'Average Cost': average_cost,
        'Last Max Price': max_price,
        'Profit Account': profit_account,
        'Used Credit': used_credit,
        'Interest Cost': credit_interest_cost,
        'Action': None
    })

    # Action 1: Take profit if position value reaches PROFIT_THRESHOLD gain
    if position_value >= PROFIT_THRESHOLD * (position_size * average_cost):
        profit = position_value - INITIAL_INVESTMENT
        profit_account += profit

        # Log annual profits
        if year not in annual_profits:
            annual_profits[year] = 0
        annual_profits[year] += profit

        # Reset to initial position
        position_size = INITIAL_INVESTMENT / price
        average_cost = price
        max_price = price  # Reset max price tracking
        current_step = 0  # Reset the step
        used_credit = INITIAL_INVESTMENT  # Reset used credit to 50k
        tracking_max_price = True  # Enable max price tracking
        portfolio_log[-1]['Action'] = 'Take Profit, Reset to 50k'
        continue

    # Action 2: Resize to 50k if current price equals or exceeds average cost (only after scaling)
    if current_step > 0 and price >= average_cost:
        # Calculate excess profit from reducing position size
        total_cost = position_size * average_cost
        sell_value = position_size * price
        excess_profit = sell_value - total_cost  # Realized profit from scaled positions

        # Add realized profit to profit account
        profit_account += excess_profit

        # Log annual profits
        if year not in annual_profits:
            annual_profits[year] = 0
        annual_profits[year] += excess_profit

        # Reset position to 50k
        position_size = INITIAL_INVESTMENT / price
        max_price = max_price  # Maintain last layer's max price
        current_step = 0  # Reset the step
        used_credit = INITIAL_INVESTMENT  # Reset used credit to 50k
        tracking_max_price = False  # Disable max price tracking
        portfolio_log[-1]['Action'] = f'Resize to 50k, Realized Profit: €{excess_profit:.2f}'
        continue

    # Action 3: Scale up if price drops LOSS_THRESHOLD below the last max price
    if price <= max_price * LOSS_THRESHOLD:
        if current_step < len(INVESTMENT_STEPS) - 1:
            current_step += 1  # Move to the next step
            new_investment = INVESTMENT_STEPS[current_step]
            additional_investment = new_investment - position_value
            additional_shares = additional_investment / price
            position_size += additional_shares

            # Update average cost
            total_cost = average_cost * (position_size - additional_shares) + price * additional_shares
            average_cost = total_cost / position_size

            # Update max price to the layer value
            max_price *= LOSS_THRESHOLD  # Update to the next layer

            # Update used credit
            used_credit += additional_investment
            tracking_max_price = False  # Disable max price tracking during scaling
            portfolio_log[-1]['Action'] = f'Scale to {new_investment}€, Used Credit Updated'
            continue

# Convert log to DataFrame
portfolio_log_df = pd.DataFrame(portfolio_log)

# Display logs
st.subheader("Portfolio Logs")
st.dataframe(portfolio_log_df)

# Plot Portfolio Details
st.subheader("Portfolio Performance")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_log_df['Date'], y=portfolio_log_df['Position Value'], mode='lines', name='Position Value'))
fig.add_trace(go.Scatter(x=portfolio_log_df['Date'], y=portfolio_log_df['Price'], mode='lines', name='Price', yaxis='y2'))
fig.update_layout(
    title='Position Value and Price Over Time',
    xaxis_title='Date',
    yaxis_title='Position Value',
    yaxis2=dict(title='Price', overlaying='y', side='right')
)
st.plotly_chart(fig, use_container_width=True)

# Annual Profits and Costs
annual_profits_df = pd.DataFrame(list(annual_profits.items()), columns=['Year', 'Profit'])
annual_interest_df = pd.DataFrame(list(annual_interest_costs.items()), columns=['Year', 'Interest Cost'])

# Annual Profits Plot
st.subheader("Annual Profits")
fig_profits = go.Figure()
fig_profits.add_trace(go.Bar(x=annual_profits_df['Year'], y=annual_profits_df['Profit'], name='Annual Profit'))
fig_profits.update_layout(
    title='Annual Profits',
    xaxis_title='Year',
    yaxis_title='Profit (€)'
)
st.plotly_chart(fig_profits, use_container_width=True)

# Annual Interest Costs Plot
st.subheader("Annual Interest Costs")
fig_annual_interest = go.Figure()
fig_annual_interest.add_trace(go.Bar(x=annual_interest_df['Year'], y=annual_interest_df['Interest Cost'], name='Annual Interest Cost'))
fig_annual_interest.update_layout(
    title='Annual Interest Costs',
    xaxis_title='Year',
    yaxis_title='Interest Cost (€)'
)
st.plotly_chart(fig_annual_interest, use_container_width=True)
