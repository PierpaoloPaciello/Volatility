import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Streamlit App Title
st.set_page_config(page_title="Volatility Backtest", layout="wide")
st.title("Short Volatility Martingale")

# Sidebar for User Input
st.sidebar.header("Strategy Parameters")

INITIAL_INVESTMENT = st.sidebar.number_input("Initial Investment", value=50000, step=1000)
INVESTMENT_STEPS = [
    INITIAL_INVESTMENT,
    INITIAL_INVESTMENT * 1.5,
    INITIAL_INVESTMENT * 2,
    INITIAL_INVESTMENT * 2.5,
    INITIAL_INVESTMENT * 3,
    INITIAL_INVESTMENT * 3.5,
]
PROFIT_THRESHOLD = st.sidebar.slider("Profit Threshold (x Average Cost)", 1.0, 2.0, 1.5, step=0.1)
LOSS_THRESHOLD = st.sidebar.slider("Loss Threshold (as % of Max Price)", 0.5, 1.0, 0.65, step=0.01)
CREDIT_LIMIT = st.sidebar.number_input("Bank Credit Limit", value=1000000, step=10000)
CREDIT_RATE = st.sidebar.number_input("Credit Rate (Annual)", value=0.05, step=0.01)

st.sidebar.markdown("---")

# Load Data
#st.header("Market Data")

with st.spinner("Downloading data from Yahoo Finance..."):
    data = yf.download("^SHORTVOL", start="2006-01-01")
data = data["Close"].reset_index()
data.columns = ["Date", "Price"]
data.dropna(inplace=True)

# Display data
st.write("### Price Data")
st.dataframe(data.tail())

# Backtest Logic
#st.header("Backtest Results")

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

# ---------------------------
# Layers Table Preparation
# ---------------------------
#st.header("Layer Overview")

# Retrieve the last max price from the portfolio logs
last_max_price = portfolio_log_df["Last Max Price"].iloc[-1]

# Calculate the current layer and the 4 layers below
layers = [last_max_price * (LOSS_THRESHOLD ** i) for i in range(5)]

# Create the Layers Table
layer_table = pd.DataFrame({
    "Layer": [f"Layer {i + 1}" for i in range(len(layers))],
    "Max Price": layers
})

# Highlight the first row (current layer) and second row (next layer)
def highlight_layers(row):
    if row.name == 0:  # Current layer
        return ['background-color: steelblue'] * len(row)
    elif row.name == 1:  # Next layer
        return ['background-color: gray'] * len(row)
    else:
        return [''] * len(row)

# Display the Layers Table
st.subheader("Layers Table")
st.dataframe(layer_table.style.apply(highlight_layers, axis=1))


# ---------------------------
# Position Values Table Preparation
# ---------------------------
st.subheader("Position Values Table")

# Retrieve the last 5 position values and average costs from the portfolio logs
last_5_entries = portfolio_log_df.tail(5)

# Calculate the take profit prices and % to take profit for each day
position_table_data = []
for idx, row in last_5_entries.iterrows():
    current_price = row["Price"]
    average_cost = row["Average Cost"]
    position_value = row["Position Value"]
    
    # Calculate take profit price based on current difference
    take_profit_price = average_cost + (average_cost * PROFIT_THRESHOLD - average_cost)
    pct_to_profit = ((take_profit_price - current_price) / current_price) * 100

    position_table_data.append({
        "Position Value": position_value,
        "Current Price": current_price,
        "Average Cost": average_cost,
        "Take Profit Price": take_profit_price,
        "% to Take Profit": f"{pct_to_profit:.2f}%"
    })

# Convert to DataFrame
position_table = pd.DataFrame(position_table_data)

# Display the Position Values Table
st.dataframe(position_table)


# Display logs
st.subheader("Portfolio Logs")
sorted_portfolio_log_df = portfolio_log_df.iloc[::-1]
# Display the sorted DataFrame
st.dataframe(sorted_portfolio_log_df.style.background_gradient(cmap='viridis'))

#st.dataframe(portfolio_log_df)

# Plot Portfolio Details
st.subheader("")
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
st.subheader("")
fig_profits = go.Figure()
fig_profits.add_trace(go.Bar(x=annual_profits_df['Year'], y=annual_profits_df['Profit'], name='Annual Profit'))
fig_profits.update_layout(
    title='Annual Profits',
    xaxis_title='Year',
    yaxis_title='Profit (€)'
)
st.plotly_chart(fig_profits, use_container_width=True)

# Annual Interest Costs Plot
st.subheader("")
fig_annual_interest = go.Figure()
fig_annual_interest.add_trace(go.Bar(x=annual_interest_df['Year'], y=annual_interest_df['Interest Cost'], name='Annual Interest Cost'))
fig_annual_interest.update_layout(
    title='Annual Interest Costs',
    xaxis_title='Year',
    yaxis_title='Interest Cost (€)'
)
st.plotly_chart(fig_annual_interest, use_container_width=True)

