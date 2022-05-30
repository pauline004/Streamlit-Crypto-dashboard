from tweepy import *
import tweepy 
import pandas as pd
import numpy as np
import re 
import string
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime as dt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


today = dt.datetime.today()

consumer_key = 'VhYqm8YCEvFPgcFLpaG5SUgXM'
consumer_secret = 'D6h5k1Ti5qE38Jh2UrJRRjihPPCreCzy8f4hhucp15WIvrud1J'
access_key= '1408046358-Fi79jDCTMW01iLeYxhLX12vEhlJX29rqkG0PGnC'
access_secret = 'BrzRZSvnsXKS04tF666HpPHX8p5LHQjUdyYrGf2Jc38gx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)




##### Sentiment Analysis
def percentage(part,whole):
    return 100 * float(part)/float(whole)


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('rt', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


def create_wordcloud(text):
    mask = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    mask = mask,
    max_words=3000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(str(text))
    wc.to_file("wc.png")
    path="wc.png"
    image =Image.open(path)
    return image 



def sent_analysis(ticker, tweet_amount):
    keyword =   '#'+ticker+ ' -filter:retweets'
    noOfTweet = tweet_amount
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(noOfTweet)
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []
    date_list  = []

    for tweet in tweets:
        tweet_list.append(tweet.text)
        date_list.append(tweet.created_at)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity

        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1

    positive = percentage(positive, noOfTweet)
    negative = percentage(negative, noOfTweet)
    neutral = percentage(neutral, noOfTweet)
    polarity = percentage(polarity, noOfTweet)

    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')

    tweet_list = pd.DataFrame(tweet_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)

    percentile_list = pd.DataFrame(np.column_stack([date_list, tweet_list]), 
                               columns=['date', 'text'])
    percentile_list.date = percentile_list.date.dt.tz_convert('Europe/Lisbon')          



    tweet_list.drop_duplicates(inplace = True)

    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    tw_list['text'] = tw_list['text'].apply(lambda x:clean_text(x))
    tw_list['date'] = percentile_list['date']
    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tw_list['text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = "positive"
        else:
            tw_list.loc[index, 'sentiment'] = "neutral"
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp

    return tw_list



def create_graph(tw_list):
    sentiment_to_graph = tw_list.groupby([pd.Grouper(key='date', freq='5min'), 'sentiment']) \
    .size().unstack('sentiment')
    sentiment_to_graph.reset_index(inplace=True) 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sentiment_to_graph['date'], y=sentiment_to_graph['positive'],
                        mode='lines',
                        name='positive'))
    fig.add_trace(go.Scatter(x=sentiment_to_graph['date'], y=sentiment_to_graph['neutral'],
                        mode='lines',
                        name='neutral'))
    fig.add_trace(go.Scatter(x=sentiment_to_graph['date'], y=sentiment_to_graph['negative'],
                        mode='lines', name='negative'))      


    # fig.update_xaxes(
    #     rangeselector=dict(
    #         buttons=list([
    #             dict(count=5, label="5minutes", step="minute", stepmode="todate"),
    #             dict(count=15, label="15minutes", step="minute", stepmode="todate"),
    #             dict(count=30, label="30minutes", step="minute", stepmode="todate"),
    #             dict(count=1, label="Daily", step="day", stepmode="todate"),
    #         ])
    #     )
    # )
    return fig, tw_list

def get_wordcloud(tw_list):
    return create_wordcloud(tw_list["text"].values)


def get_pie_chart(tw_list):
    
    tw_list['Count'] = 1 #create a new column for the pie chart

    fig6 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig6.add_trace(go.Pie(labels=tw_list['sentiment'], values=tw_list['Count'], name="Sentiment"),
                   row=1, col=1)
    fig6.update_traces(hole=.5, hoverinfo="label+value+name")
    fig6.update_layout(
        title="Pie chart",
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
        )
    ) 
    return fig6       