
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas_ta as ta

def RSI_graph(df2):
    window_length = 20
    df2['diff'] = df2['Close'].diff(1)

    # Calculate Avg. Gains/Losses
    df2['gain'] = df2['diff'].clip(lower=0).round(2)
    df2['loss'] = df2['diff'].clip(upper=0).abs().round(2)

    # Get initial Averages
    df2['avg_gain'] = df2['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
    df2['avg_loss'] = df2['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]

    # Get WMS averages
    # Average Gains
    for i, row in enumerate(df2['avg_gain'].iloc[window_length+1:]):
        df2['avg_gain'].iloc[i + window_length + 1] =            (df2['avg_gain'].iloc[i + window_length] *
             (window_length - 1) +
             df2['gain'].iloc[i + window_length + 1])\
            / window_length
    # Average Losses
    for i, row in enumerate(df2['avg_loss'].iloc[window_length+1:]):
        df2['avg_loss'].iloc[i + window_length + 1] =            (df2['avg_loss'].iloc[i + window_length] *
             (window_length - 1) +
             df2['loss'].iloc[i + window_length + 1])\
            / window_length
        
    df2['rsi'] = 100 - (100 / (1.0 + df2['avg_gain'] / df2['avg_loss']))

    df2.drop(['gain','loss','avg_gain', 'avg_loss', 'diff'], axis = 1, inplace = True)
    # Create Figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.35, 0.65])
    
    # Create Candlestick chart for price data
    fig.add_trace(go.Candlestick(
        x=df2.index,
        open=df2['Open'],
        high=df2['High'],
        low=df2['Low'],
        close=df2['Close'],
        increasing_line_color='mediumspringgreen',
        decreasing_line_color='midnightblue',
        showlegend=False), row=1, col=1)

    # Make RSI Plot
    fig.add_trace(go.Scatter(
        x=df2.index,
        y=df2['rsi'],
        line=dict(color='mediumspringgreen', width=4),
        showlegend=False,
    ), row=2, col=1
    )

    # Add upper/lower bounds
    fig.update_yaxes(range=[-10, 110], row=2, col=1)
    fig.add_hline(y=0, col=1, row=2, line_color="midnightblue", line_width=2)
    fig.add_hline(y=100, col=1, row=2, line_color="midnightblue", line_width=2)

    # Add overbought/oversold
    fig.add_hline(y=30, col=1, row=2, line_color='midnightblue', line_width=4, line_dash='dash')
    fig.add_hline(y=70, col=1, row=2, line_color='midnightblue', line_width=4, line_dash='dash')

    # Customize font, colors, hide range slider
    layout = go.Layout(
        plot_bgcolor='#FDFDFD',
        height=700,
        width=900,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # update and display
    fig.update_layout(layout)
    return fig


def MACD_graph(df2):
     # Get the 26-day EMA of the closing price
    k = df2['Close'].ewm(span=12, adjust=False, min_periods=12).mean()

    # Get the 12-day EMA of the closing price
    d = df2['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d

    # Get the 9-Day EMA of the MACD for the Trigger line
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()

    # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
    macd_h = macd - macd_s

    # Add all of our new values for the MACD to the dataframe
    df2['macd'] = df2.index.map(macd)
    df2['macd_h'] = df2.index.map(macd_h)
    df2['macd_s'] = df2.index.map(macd_s)

    # View our data
    pd.set_option("display.max_columns", None)
    df2.drop(['macd_h', 'macd_s'], axis = 1, inplace = True)
    
    
    # calculate MACD values
    df2.ta.macd(close='Close', fast=12, slow=26, append=True)
    # Force lowercase (optional)
    df2.columns = [x.lower() for x in df2.columns]
    # Construct a 2 x 1 Plotly figure
    fig = make_subplots(rows=2, cols=1, row_width=[0.35, 0.65]) #shared_xaxes=True,
    # price Line
    fig.append_trace(
        go.Scatter(
            x=df2.index,
            y=df2['open'],
            line=dict(color='midnightblue', width=2),
            name='open',
            showlegend=False,
            #legendgroup='1',
        ), row=1, col=1
    )
    # Candlestick chart for pricing
    fig.append_trace(
        go.Candlestick(
            x=df2.index,
            open=df2['open'],
            high=df2['high'],
            low=df2['low'],
            close=df2['close'],
            increasing_line_color='midnightblue',
            decreasing_line_color='#CC0000',
            showlegend=False
        ), row=1, col=1
    )
    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=df2.index,
            y=df2['macd_12_26_9'],
            line=dict(color='midnightblue', width=4),
            name='macd',
            #showlegend=False,
            legendgroup='2',
        ), row=2, col=1
    )
    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=df2.index,
            y=df2['macds_12_26_9'],
            line=dict(color='firebrick', width=4),
            #showlegend=False,
            legendgroup='2',
            name='signal'
        ), row=2, col=1
    )
    # Colorize the histogram values
    colors = np.where(df2['macdh_12_26_9'] < 0, '#CC0000', 'midnightblue')
    # Plot the histogram
    fig.append_trace(
        go.Bar(
            x=df2.index,
            y=df2['macdh_12_26_9'],
            name='histogram',
            marker_color=colors,
        ), row=2, col=1
    )
    layout = go.Layout(
        plot_bgcolor='#FDFDFD',
        height=700,
        width=900,
        # # Font Families
        # font_family='Monospace',
        # font_color='#000000',
        # font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # Update options and show plot
    fig.update_layout(layout)
    return fig


def OSC_graph(df2):
    plt.style.use('fivethirtyeight')
    Coin_so = df2.tail(365).copy()

    Coin_so['14-high'] = Coin_so['high'].rolling(14).max()
    Coin_so['14-low'] = Coin_so['low'].rolling(14).min()
    Coin_so['%K'] = (Coin_so['close'] - Coin_so['14-low'])*100/(Coin_so['14-high'] - Coin_so['14-low'])
    Coin_so['%D'] = Coin_so['%K'].rolling(3).mean()
    ax = Coin_so[['%K', '%D']].plot()
    Coin_so['close'].plot(ax=ax, secondary_y=True)
    ax.axhline(20, linestyle='--', color="r")
    ax.axhline(80, linestyle="--", color="r")
    plt.rcParams["figure.figsize"] = (30,7)
    
    return plt

