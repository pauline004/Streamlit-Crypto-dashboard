#Description: Create a Cryptocurrencies Dashboard

#import Libraries
from tweepy import *
import tweepy 
import datetime as dt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import wordcloud
import yfinance as yf
from st_btn_select import st_btn_select
import numpy as np
import Indicators
import matplotlib.pyplot as plt
import arima_pred
import twitter_sent_analysis
from quantiphy import Quantity
import mplfinance as mpl
import yahoo_fin.stock_info as si
from plotly.subplots import make_subplots
import nltk

nltk.download('vader_lexicon')

st.set_page_config(layout="wide")

image = Image.open("4.png")
st.image(image, use_column_width=True)

today = dt.datetime.today()
prev_year = dt.datetime.today() - dt.timedelta(days=365)

st.sidebar.header("User Input")

def get_input():
    asset_type = st.sidebar.radio("Choose the type of your financial asset",('Cryptocurrency', 'Company'))
    start_date = st.sidebar.text_input("Start Date", prev_year.strftime("%Y-%m-%d")) 
    end_date = st.sidebar.text_input("End Date", today.strftime("%Y-%m-%d"))
    crypto_money = st.sidebar.text_input("Please enter the Asset Symbol", "BTC-USD")
    return start_date, end_date, crypto_money, asset_type

def get_asset_name(tickerSymbol):
    tickerSymbol = tickerSymbol.upper()
    return tickerSymbol

def get_data(tickerSymbol, start, end):
    tickerSymbol = tickerSymbol.upper()
    df = yf.download(tickerSymbol,
                      start=start,
                      end=end,
                      progress=False,
    )
    return df #.loc[start:end]


def get_page_name():
    page = st_btn_select(('Company Overview','Price Prediction', 'Technical Indicators', 'Sentiment Analysis'),format_func=lambda name: name.capitalize(),)
    return page

start, end, tickerSymbol,asset = get_input()
df = get_data(tickerSymbol, start, end)
asset_name = get_asset_name(tickerSymbol)
page = get_page_name()
ticker = yf.Ticker(tickerSymbol)


price = ticker.info['regularMarketPrice']
previousdayprice= ticker.info['regularMarketPreviousClose']
percent_change = round(((price - previousdayprice) / previousdayprice)* 100.0, 2)
marketcap= ticker.info['marketCap']


def comp_overview():

    col1, col2, col3 = st.columns(3)
    col1.metric("Current " + asset_name + " Price in USD", price, previousdayprice)  # Live Price
    col2.metric("Change in price from previous day", "{}%".format(percent_change))
    col3.metric("Market Cap", str(Quantity(marketcap)))

    #Part 1: Brief Summary
    with st.expander(asset + ' Brief Summary'):
        st.subheader(asset + ' Full name')
        st.text(ticker.info['shortName'])
        st.subheader('Sector')
        st.text(ticker.info['sector'])
        st.subheader('Size of the '+ asset)
        st.text(str(Quantity(ticker.info['fullTimeEmployees'])))

    #Part 2: Fig 1 - candlestisk Chart with with MA
    st.title(asset_name + " CandleStick with MA and Volume Traded")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        show_nontrading_days = st.checkbox('Show non-trading days', False)
    with c2:
        date_from = st.date_input('Show data from', prev_year.date())
    with c3:
        mav1 = st.number_input('Moving Average', min_value=3, max_value=30, value=20, step=1)

    df1 = get_data(tickerSymbol, str(date_from), end)

    fig1, ax = mpl.plot(
        df1,
        #title=f'{symbol}, {date_from}',
        type="candle",
        show_nontrading=show_nontrading_days,
        mav=(int(mav1)),
        volume=True,
        style="starsandstripes",
        figsize=(15, 10),
        returnfig=True
    )
    st.pyplot(fig1)

    #Part 3: COmpany Financial Indicators
    multipliers = {'K':1000, 'M':1000000, 'B':1000000000, 'T':1000000000000, 'P':1000000000000000}
    def string_to_int(string): #Function to get string into int
        if string[-1].isdigit(): # check if no suffix
            return int(string)
        mult = multipliers[string[-1]] # look up suffix to get multiplier
         # convert number to float, multiply by multiplier, then make int
        return int(float(string[:-1]) * mult)

    overview_df = si.get_stats(tickerSymbol)
    overview_df = overview_df.set_index('Attribute')
    overview_dict = si.get_quote_table(tickerSymbol)
    income_statement = si.get_income_statement(tickerSymbol)
    year_end = overview_df.loc['Fiscal Year Ends'][0]
    market_cap = string_to_int(overview_dict['Market Cap'])
    market_cap_cs = str(Quantity(market_cap))
    sales = income_statement.loc['totalRevenue'][0]
    gross_profit = income_statement.loc['grossProfit'][0]
    ebit = income_statement.loc['ebit'][0]
    net_profit = income_statement.loc['netIncome'][0]
    ev = ticker.info['enterpriseValue']
    ev_cs = str(Quantity(ev))

    gross_margin = gross_profit/sales
    operating_margin = ebit/sales
    net_margin = net_profit/sales
    price_earnings_ratio = round(market_cap/net_profit, 3)
    ev_sales_ratio = round(ev/sales, 3)

    st.title(asset_name + 'Financial Indicators')
    overview_dict = [ev_cs, market_cap_cs, str(ev_sales_ratio), str(price_earnings_ratio)]
    overview_index = ['Enterprise value', 'Market cap', 'EV/sales ratio', 'P/E ratio']
    fig2 = pd.DataFrame(overview_dict, overview_index)
    st.dataframe(fig2)

    with st.expander('Profit margins (as of Fiscal Year Ends {})'.format(year_end)):
        profit_margin_dict = {'Values': [gross_margin, operating_margin, net_margin]}
        profit_margin_index = ['Gross margin', 'Operating margin', 'Net margin']
        profit_margin_df = pd.DataFrame(profit_margin_dict, index=profit_margin_index)
        st.table(profit_margin_df)
        st.bar_chart(profit_margin_df)

    with st.expander('Analysts recommendation'):
        fig5 = ticker.recommendations.head()
        st.table(fig5)

    # Part4: Company Stakeholders
    st.title(asset_name + ' Stakeholders')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Major Holders")
        fig3 = ticker.major_holders
        fig3.columns = ['Percent', 'Name']
        st.table(fig3)

    with col2:
        # Fig 4
        st.subheader("Instutional Holders")
        instu_holders = ticker.institutional_holders
        fig4 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
        fig4.add_trace(go.Pie(labels=instu_holders['Holder'], values=instu_holders['% Out'], name="PRICE"),
                       row=1, col=1)
        fig4.update_traces(hole=.5, hoverinfo="label+value+name")
        fig4.update_layout(
            title="Institutional Holders",
            title_x=0.3,
            paper_bgcolor="#FDFDFD",
            legend=dict(
                x=0.5,
                y=1,
                traceorder="reversed",
                title_font_family="Times New Roman",
                font=dict(
                    family="Courier",
                    size=12,
                    color="black")
            ))
        st.plotly_chart(fig4)



def crypto_overview():

    col1, col2, col3 = st.columns(3)
    col1.metric("Current " + asset_name + " Price in USD", str(Quantity(price)), str(Quantity(previousdayprice)))  # Live Price
    col2.metric("Change in price from previous day", "{}%".format(percent_change))
    col3.metric("Market Cap", str(Quantity(marketcap)))

    # Part 1: Brief Summary
    with st.expander(asset + ' Brief Summary'):
        st.subheader(' Full name')
        st.text(ticker.info['name'])
        st.subheader('Quote type')
        st.text(ticker.info['quoteType'])
        st.subheader('Description')
        st.text(ticker.info['description'])

    # Part 2: Fig 1 - candlestisk Chart with with MA
    st.title(asset_name + " CandleStick with MA and Volume Traded")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        show_nontrading_days = st.checkbox('Show non-trading days', False)
    with c2:
        date_from = st.date_input('Show data from', prev_year.date())
    with c3:
        mav1 = st.number_input('Moving Average', min_value=3, max_value=30, value=20, step=1)

    df1 = get_data(tickerSymbol, str(date_from), end)

    fig1, ax = mpl.plot(
        df1,
        # title=f'{symbol}, {date_from}',
        type="candle",
        show_nontrading=show_nontrading_days,
        mav=(int(mav1)),
        volume=True,
        style="starsandstripes",
        figsize=(15, 10),
        returnfig=True
    )
    st.pyplot(fig1)

    # Part 3: Key Crypto Numbers
    st.title(asset_name + ' Key Numbers')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historical Data")
        fig3 = ticker.major_holders
        fig3.columns = ['Name', 'Values']
        st.table(fig3)

    with col2:
        st.subheader("Summary")
        fig4 = ticker.institutional_holders
        fig4.columns = ['Name', 'Values']
        st.table(fig4)


def get_rsi(data):
    return Indicators.RSI_graph(data)


def get_macd(data):
    return Indicators.MACD_graph(data)

def get_oscillator(data):
    return Indicators.OSC_graph(data)

def get_prediction(data):
    predictions, originals , plot, next_value  = arima_pred.predict_arima(data)

    testdf = pd.DataFrame()
    testdf['original'] = originals
    testdf['pred'] = predictions
    return plot, testdf,next_value


rsi_graph = go.Figure(get_rsi(df))
macd_graph = go.Figure(get_macd(df))
osc_graph = get_oscillator(df)
pred_fig, pred_df, prediction= get_prediction(df)

def indicators(asset_name):
    st.title(asset_name + " Technical Indicators")
    date_from = st.date_input('Show data from', prev_year.date())
    df2 = get_data(tickerSymbol, str(date_from), end)

    rsi_graph = go.Figure(get_rsi(df2))
    macd_graph = go.Figure(get_macd(df2))
    osc_graph = get_oscillator(df2)

    st.subheader("Relative Strength Index (RSI)")
    st.plotly_chart(rsi_graph)

    st.subheader("Moving Average Convergence Divergence (MACD)")
    st.plotly_chart(macd_graph)

    st.subheader("Stochastic Oscillator")
    st.pyplot(osc_graph)
        
def predicted_value():
    st.header("Predicted price")
    st.subheader(str(round(prediction[0], 3)))
    st.header("Prediction versus Original Price Table")
    st.write(pred_df)
    st.header("Predicted price Line Chart")
    st.pyplot(pred_fig)


def get_sentiment_chart(tickerSymbol,n_tweets):
    tw_list = twitter_sent_analysis.sent_analysis(tickerSymbol,n_tweets)
    by_day_sentiment = tw_list.groupby([pd.Grouper(key='date', freq='min'), 'sentiment']) \
    .size().unstack('sentiment')
    by_day_sentiment.reset_index(inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=by_day_sentiment['date'], y=by_day_sentiment['positive'],
                        mode='lines',
                        name='positive'))
                    
    fig.add_trace(go.Scatter(x=by_day_sentiment['date'], y=by_day_sentiment['neutral'],
                        mode='lines',
                        name='neutral'))
    fig.add_trace(go.Scatter(x=by_day_sentiment['date'], y=by_day_sentiment['negative'],
                        mode='lines', name='negative'))

    return fig

def get_sentiment_cloud(tickerSymbol,n_tweets):
    tw_list = twitter_sent_analysis.sent_analysis(tickerSymbol,n_tweets)
    return twitter_sent_analysis.get_wordcloud(tw_list)

def get_pie(tickerSymbol,n_tweets):
    tw_list = twitter_sent_analysis.sent_analysis(tickerSymbol,n_tweets)
    return twitter_sent_analysis.get_pie_chart(tw_list)    


def sentiment_display(tickerSymbol):
    n_tweets = st.number_input("Select Number of tweets to analyse",min_value=100, max_value=2500)

    sent_graph = get_sentiment_chart(tickerSymbol,n_tweets)
    wdcloud = get_sentiment_cloud(tickerSymbol,n_tweets)
    pc = get_pie(tickerSymbol,n_tweets)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Sentiment from looking at latest Tweets")
        st.plotly_chart(sent_graph)

    with col2:
        st.header("Sentiment distribution")
        st.plotly_chart(pc)

    st.header("Wordcloud")
    st.caption("Displays the most frequently words in the recent Tweets")
    st.image(wdcloud)


if asset == 'Cryptocurrency':
    if "-USD" in asset_name:
        if page == 'Company Overview':
            crypto_overview()
        elif page == 'Technical Indicators':
            indicators(asset_name)
        elif page == 'Price Prediction':
            predicted_value()
        elif page == 'Sentiment Analysis':
            sentiment_display(tickerSymbol)
            
    else:
        st.write("Not a cryptocurrency, please select 'Company' !!")

if asset == 'Company':
    if "-USD" not in asset_name:
        if page == 'Company Overview':
            comp_overview()
        elif page == 'Technical Indicators':
            indicators(asset_name)
        elif page == 'Price Prediction':
            predicted_value()
        elif page == 'Sentiment Analysis':
            sentiment_display(tickerSymbol)
    else:
        st.write("Not a company, please select 'Cryptocurrency' !!")


