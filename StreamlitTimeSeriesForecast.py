#pip install ipython 
#pip install nbformat

import streamlit as st  # pip install streamlit

from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import re

import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt

import json
import requests  # pip install requests
import io

from datetime import datetime
from datetime import timedelta
from zoneinfo import ZoneInfo
import time

from prophet import Prophet # pip install prophet
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation 
from prophet.diagnostics import performance_metrics

import matplotlib.pyplot as plt

import plotly.io as pio
pio.renderers.default = "notebook_connected+iframe"
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

#from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)

#st.write(pio.renderers)

st.set_page_config(
    page_title="Time Series Forecast Project",
    layout='centered', #wide
    initial_sidebar_state= "expanded", #"expanded", #collapsed
    menu_items={
        'Report a bug': "https://github.com/efrenmo"
    }
)

# ---------- Load Animations ---------- #
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_btc = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_Rfya22/Bitcoin.json")
#lottie_btc = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zlrpnoxz.json")




# ---------- Table of Contents ---------- #
with st.sidebar:
    class Toc:

        def __init__(self):
            self._items = []
            self._placeholder = None
        
        def title(self, text):
            self._markdown(text, "h1")

        def header(self, text):
            self._markdown(text, "h2", " " * 2)

        def subheader(self, text):
            self._markdown(text, "h3", " " * 4)

        def placeholder(self, sidebar=False):
            self._placeholder = st.sidebar.empty() if sidebar else st.empty()

        def generate(self):
            if self._placeholder:
                self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
        
        def _markdown(self, text, level, space=""):            
            key = re.sub('[^0-9a-zA-Z_-]+', '-', text).lower()
            #key = "".join(filter(str.isalnum, text)).lower()

            st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
            self._items.append(f"{space}* <a href='#{key}'>{text}</a>")

    toc = Toc()
   
      
    
    #st_lottie(
    #    lottie_btc,
    #    speed=1,
    #    reverse=False,
    #    loop=True,
    #    #quality= "medium" #"low", # medium ; high
    #    #renderer= "canvas"   #"svg", # canvas
    #    height=300,
    #    width=300,
    #    key=None,
    #)

    #st.markdown("<h1 style='text-align: center;'>Table of contents</h1>", unsafe_allow_html=True) #; color: black
    st.title("Table of contents")  


    toc.placeholder()


# ---------- REPORT STARTS ---------- #

st.title('Bitcoin Time Series Forecast')

# ---------- SECTION 1: Intro, Methodology, and Data Source ---------- #

with st.container():      
    tab1, tab2, tab3 = st.tabs(["Introduction", "Data Source", "Methodology"])
    
    with tab1:
        #st.write('*Click on the desired tab')
        st.markdown("""
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-hand-index-thumb" viewBox="0 0 16 16">
        <path d="M6.75 1a.75.75 0 0 1 .75.75V8a.5.5 0 0 0 1 0V5.467l.086-.004c.317-.012.637-.008.816.027.134.027.294.096.448.182.077.042.15.147.15.314V8a.5.5 0 0 0 1 0V6.435l.106-.01c.316-.024.584-.01.708.04.118.046.3.207.486.43.081.096.15.19.2.259V8.5a.5.5 0 1 0 1 0v-1h.342a1 1 0 0 1 .995 1.1l-.271 2.715a2.5 2.5 0 0 1-.317.991l-1.395 2.442a.5.5 0 0 1-.434.252H6.118a.5.5 0 0 1-.447-.276l-1.232-2.465-2.512-4.185a.517.517 0 0 1 .809-.631l2.41 2.41A.5.5 0 0 0 6 9.5V1.75A.75.75 0 0 1 6.75 1zM8.5 4.466V1.75a1.75 1.75 0 1 0-3.5 0v6.543L3.443 6.736A1.517 1.517 0 0 0 1.07 8.588l2.491 4.153 1.215 2.43A1.5 1.5 0 0 0 6.118 16h6.302a1.5 1.5 0 0 0 1.302-.756l1.395-2.441a3.5 3.5 0 0 0 .444-1.389l.271-2.715a2 2 0 0 0-1.99-2.199h-.581a5.114 5.114 0 0 0-.195-.248c-.191-.229-.51-.568-.88-.716-.364-.146-.846-.132-1.158-.108l-.132.012a1.26 1.26 0 0 0-.56-.642 2.632 2.632 0 0 0-.738-.288c-.31-.062-.739-.058-1.05-.046l-.048.002zm2.094 2.025z"/>
        </svg> Click on the desired tab 
        """, unsafe_allow_html=True)
        
        
        toc.header("Introduction")
        
        st.write("""
        In this project I'll use [**Prophet's**](https://facebook.github.io/prophet/) time series forecasting 
        algorithm to attempt to forecast Bitcoin's price for the next thrirty days. 

        [**Prophet**](https://facebook.github.io/prophet/)  is an open source forecasting package that was
        developed by Facebook’s data science research team. The procedure for forecasting time series data 
        is based on an univariate additive model where non-linear trends are fit with **`yearly, weekly, 
        and daily seasonality, plus holiday effects`**. It works best with time series that have **`strong
        seasonal effects`** and **`several seasons`** of historical data. Prophet is **`robust`** to missing
        data and shifts in the trend, and typically handles outliers well.


        [**Bitcoin** **(BTC)**](https://www.coingecko.com/en/coins/bitcoin) is a scarse decentralized digital
        currency introduced into the world in 2009 by a person or group of people using the pseudonym **`Satoshi Nakamoto`**.
        It can be transferred on the peer-to-peer Bitcoin network. Bitcoin transactions are verified by network nodes
        through cryptography and recorded in a public distributed ledger called a blockchain.

        As of August 31st 2022 

        *   Circulating supply of BTC: 18,925,000
        *   Market Capitalization: $412 Billion USD
        *   Market Cap Rank: #1 *(among all cryptocurrencies)*

        *   Trading Volume: $27,851,916,129 USD

        *   Current Price: $20,000 USD per unit

        """)

    with tab2:
        st.header("Data Source")
        
        st.write("""
        For this project I'll be using data from `May 11th, 2020 to September 16th, 2022`. The profile of this asset has change
        dramatically in recent years, that is why I'm choosing to use relatively newer data; and is around this time when a pivot
        point is marked in the life trajectory of this asset. 

        The period used is characterized by the increased presence of institutional interest, which many cite as a contributing
        factor to the latest bull run that began in late 2020 and ended in late 2021. As for the exact date, "May 11th, 2020",
        it marks the start of the third cycle of a very critical event engraved in the monetary policy of the cryptocurrency,
        called the Bitcoin Halving. This cyclical event happens after every 210,000 blocks have been mined on the blockchain,
        a process that takes roughly four years to complete.
        
        At a bitcoin halving event, the rate at which new Bitcoin units are released into circulation is reduced by 50%. It's
        important to note that 21 million units is the max cap for this asset, and 90% of the supply is already in circulation.
        That is what makes this event so important, as it creates disinflationary pressure on the digital currency driving Bitcoin's
        medium-term and long-term price development.  
        
        Initially, I wanted to use data from the day it was first listed on an exchange back in 2010. Also, I wanted to add a custom
        seasonality component to Prophet by using the Bitcoin halving dates to mark the start and end of a season. However, after
        comparing cross-validation results among models using different date ranges, I found out I get more accurate results by
        using data from the last Bitcoin halving and onwards. And, since I need at least two season cycles (8 years of data) to
        be able to add a custom seasonality component to Prophet, I could not add the custom component into the algorithm. Prophet
        will however detect automatically any perceived yearly, monthly, or weekly seasonality and forecast accordingly. 

        To import data from cryptocurrency exchanges I used the `CCXT library`. The CCXT library is normally used to connect and trade
        with cryptocurrency exchanges and payment processing services worldwide. It provides quick access to market data for storage,
        analysis, visualization, indicator development, algorithmic trading, strategy back-testing, bot programming, and related software
        engineering.

        """)

    with tab3:  
        st.header("Methodology")      
        #st.header('Methodology')
        st.write("""

        In principle, you don't need to specify any hyperparameters. Prophet can automatically detect and set a good set of hyperparameters
        for you. However, my experience with this particular dataset made me decide to manually set some hyperparameters when training the model.
        The hyperparameters that I manually set are: 

        *   `seasonality_mode`: Options are "additive", "multiplicative". Default is 'additive', but many business time series will have multiplicative seasonality. This is best identified just from looking at the time series and seeing if the magnitude of seasonal fluctuations grows with the magnitude of the time series. In our case, it is multiplicative.
        *   `changepoint_range`: This is the proportion of the history in which the trend is allowed to change. This defaults to 0.8, 80% of the history, meaning the model will not fit any trend changes in the last 20% of the time series. This is fairly conservative, to avoid overfitting to trend changes at the very end of the time series where there isn’t enough runway left to fit it well.
        *   `n_changepoints`: This is the number of automatically placed changepoints. The default of 25 should be plenty to capture the trend changes in a typical time series.

        
        I found that manipulating the above parameters would produce more accurate results, measured by cross validation techniques, than just running the
        Prophet algorithm with default values.

        Nevertheless, there were 2 very important hyperparameters that I left out from the initial model training, `changepoint_prior_scale` and
        `seasonality_prior_scale`.

        Prophet suggests these are the most impactful when tuned right. So, I later retrained the model with the output of a hyperparameter tuning
        script that optimized the model for `changepoint_prior_scale` and `seasonality_prior_scale` with the purpose of comparing this second version of the
        model with the first version of the model.  |

        The hyperparameter tuning script performed a grid search on a 4x4 grid for every possible combination, and output the best set of values that would
        minimize the error and increased accuracy.   

        *   `changepoint_prior_scale`: This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. If it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.
        *   `seasonality_prior_scale`: This parameter controls the flexibility of the seasonality. Similarly, a large value allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality. The default is 10., which applies basically no regularization. That is because we very rarely see overfitting here (there’s inherent regularization with the fact that it is being modeled with a truncated Fourier series, so it’s essentially low-pass filtered). A reasonable range for tuning it would probably be [0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is forced to be very small. This likely also makes sense on a log scale, since it is effectively an L2 penalty like in ridge regression.


        """)

# ---------- SECTION 2: Importing and Exploring the Dataset ---------- #

toc.header("Importing and Exploring the Dataset")

st.write("""
Connecting to Binance Exchange and importing data using [**CCXT library**](https://github.com/ccxt/ccxt) 
""")

st.code("""
# Connect to cryptocurrency Binance exchange 
binance = ccxt.binance()

# Fetching Ticker
btc_ticker = binance.fetch_tickers(['BTC/USDT'])

# Fetching Open, High, Low, Volume (OHLCV)
binance_BTCUSDT_ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe= '1d', limit=864)

# Creating dataframe with fetched data
BTC_Data = pd.DataFrame(binance_BTCUSDT_ohlcv,columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Changing the original data format from UTC timestamp in milliseconds to datetime
BTC_Data.Date = pd.to_datetime(BTC_Data.Date, unit='ms')

""")


@st.cache(suppress_st_warning=True)
def data_upload(url):
    #st.write("Cache miss")
    download = requests.get(url).content

    # Reading the downloaded content and turning it into a pandas dataframe
    BTC_Data = pd.read_csv(io.StringIO(download.decode('utf-8')))

    BTC_Data.drop(columns=BTC_Data.columns[0], axis=1, inplace=True)

    return(BTC_Data)


url = "https://raw.githubusercontent.com/efrenmo/Forecasting_BTC_with_Prophet/main/BTC_Data_ccxt.csv" # Make sure the url is the raw version of the file on GitHub
BTC_Data = data_upload(url)


#st.dataframe(BTC_Data.style.format({'Volume':'{:.2f}'}))
st.dataframe(BTC_Data.style.format({'Close':'{:.2f}','Volume':'{:.2f}','Low':'{:.2f}','High':'{:.2f}','Open':'{:.2f}'}))


#st.write(BTC_Data.dtypes.astype(str))

st.subheader('Data Statistics')
st.write(BTC_Data.describe()) 

  
# ---------- SECTION 3: Visualizing the Data Set ---------- #    

toc.header("Visualizing the Data Set")

tab1, tab2 = st.tabs(["📊Chart", "🔣Code"])


with tab2:        
    st.subheader('Plotly Chart Code')    
    with st.echo():        
        chart = go.Figure()

        chart.add_trace(go.Scatter(
            x= BTC_Data['Date'], 
            y=BTC_Data['Close'].round(decimals = 2), 
            name = "Price",
            yaxis="y2",            
            showlegend = False
            )
        )

        chart.add_trace(go.Bar(
            x= BTC_Data['Date'], 
            y=BTC_Data['Volume'], 
            name = "Volume",
            yaxis="y",
            marker = {'color' : '#19D3F3', 'line_width':0.15}, #19D3F3 00CC99
            showlegend = False,

            )
        )

        chart.update_layout(
            xaxis=dict(
                autorange=True,
                range=["2020-05-11", "2022-09-16"],
                rangeslider=dict(
                    autorange=True,
                    range=["2020-05-11", "2022-09-16"],
                ),
                type="date",
                title_text= "Date",
                showgrid=True
            ),
            yaxis=dict(
                anchor="x",
                autorange=True,
                domain=[0, 0.3],
                #linecolor="#607d8b",
                mirror=True,
                range=[2500, 435000],
                showline=True,
                side="left",
                #tickfont={"color": "#607d8b"},
                tickmode="auto",
                ticks="",
                title="Volume",
                #titlefont={"color": "#607d8b"},
                type="linear",
                zeroline=False,        
            ),
            yaxis2=dict(
                anchor="x",
                autorange=True,
                domain=[0.3, 1],
                #linecolor="#6600FF",
                mirror=True,
                range=[8000, 70000],
                showline=True,
                side="left",
                tickfont={"color": "#6600FF"},
                tickmode="auto",
                ticks="",
                title="Price",
                titlefont={"color": "#6600FF"},
                type="linear",
                zeroline=False,
                fixedrange=False      
            )
        )

        # Add range buttons
        chart.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
    
        chart.update_layout(            
            title={
                'text':"Bitcoin Daily Price & Volume",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font_size':18
            },
            dragmode="zoom",
            hovermode="x",
            legend=dict(traceorder="reversed"),
            height=600,
            width=800,
            #template="plotly_white",
            margin=dict(
                t=100,
                b=50,
                l=1,
                r=40
            ),
        )
    
        chart.show()
    

with tab1:    
    st.markdown("""
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-hand-index-thumb" viewBox="0 0 16 16">
        <path d="M6.75 1a.75.75 0 0 1 .75.75V8a.5.5 0 0 0 1 0V5.467l.086-.004c.317-.012.637-.008.816.027.134.027.294.096.448.182.077.042.15.147.15.314V8a.5.5 0 0 0 1 0V6.435l.106-.01c.316-.024.584-.01.708.04.118.046.3.207.486.43.081.096.15.19.2.259V8.5a.5.5 0 1 0 1 0v-1h.342a1 1 0 0 1 .995 1.1l-.271 2.715a2.5 2.5 0 0 1-.317.991l-1.395 2.442a.5.5 0 0 1-.434.252H6.118a.5.5 0 0 1-.447-.276l-1.232-2.465-2.512-4.185a.517.517 0 0 1 .809-.631l2.41 2.41A.5.5 0 0 0 6 9.5V1.75A.75.75 0 0 1 6.75 1zM8.5 4.466V1.75a1.75 1.75 0 1 0-3.5 0v6.543L3.443 6.736A1.517 1.517 0 0 0 1.07 8.588l2.491 4.153 1.215 2.43A1.5 1.5 0 0 0 6.118 16h6.302a1.5 1.5 0 0 0 1.302-.756l1.395-2.441a3.5 3.5 0 0 0 .444-1.389l.271-2.715a2 2 0 0 0-1.99-2.199h-.581a5.114 5.114 0 0 0-.195-.248c-.191-.229-.51-.568-.88-.716-.364-.146-.846-.132-1.158-.108l-.132.012a1.26 1.26 0 0 0-.56-.642 2.632 2.632 0 0 0-.738-.288c-.31-.062-.739-.058-1.05-.046l-.048.002zm2.094 2.025z"/>
        </svg> Click on the "Code" tab to see the code for this chart
        """, unsafe_allow_html=True)
    st.plotly_chart(chart)

# ---------- SECTION 4: Data  Preparation  ---------- #

toc.header("Data  Preparation")
#st.header('Data  Preparation')

toc.subheader("Preparing Data for the Algorithm")
#st.subheader("Preparing Data for the Algorithm")

st.write("""
Prophet in its most simple form requires only two columns, **`'datesatamp' and 'price'`**

""")

with st.echo():      
    # Isolating Date and Daily Closing Price
    BTC_2020_2022_Date_Price = BTC_Data[['Date', 'Close']]
    # Renaming of the columns is needed for the FP algorithm 
    BTC_2020_2022_Date_Price = BTC_2020_2022_Date_Price.rename({'Date':'ds', 'Close':'y'}, axis=1)
    # Converting the data type from string to datetime for the "ds" column
    BTC_2020_2022_Date_Price["ds"] = pd.to_datetime(BTC_2020_2022_Date_Price["ds"])
    # Sorting dataframe by date
    BTC_2020_2022_Date_Price.sort_values(by = 'ds', inplace = True)

#st.dataframe(BTC_2020_2022_Date_Price)


# ---------- SECTION 4B: Data  Preparation: Train Test Split  ---------- #

toc.subheader("Train Test Split")

with st.echo(): 
    # Train test split
    df_train = BTC_2020_2022_Date_Price[BTC_2020_2022_Date_Price['ds']<='2022-08-16']
    df_test = BTC_2020_2022_Date_Price[BTC_2020_2022_Date_Price['ds']>'2022-08-16']

    # Print the number of records and date range for training and testing dataset.
    print('The training dataset has', len(df_train), 'records, ranging from', df_train['ds'].min(), 'to', df_train['ds'].max())
    print('The testing dataset has', len(df_test), 'records, ranging from', df_test['ds'].min(), 'to', df_test['ds'].max())

col1, col2= st.columns(2)
with col1:
    st.subheader('Training Data')
    st.write('The training dataset has', len(df_train), 'records, ranging from', df_train['ds'].min(), 'to', df_train['ds'].max())
    st.dataframe(df_train)
with col2:
    st.subheader('Testing Data')
    st.write('The testing dataset has', len(df_test), 'records, ranging from', df_test['ds'].min(), 'to', df_test['ds'].max())
    st.dataframe(df_test)


# ---------- PART 1 of Forecasting with Prophet ---------- #
# ---------- PART 1 - SECTION 1: Forecasting with Prophet  ---------- #

toc.header("Forecasting with Prophet")


toc.header("Part 1")


# ---------- PART 1 - SECTION 2: Model Training  ---------- #

toc.subheader("Model Training")
st.write("""
To train a model in Prophet, first we create an instance of the 
model class and then we call the fit method.
""")


st.code("""
Prophet_Model = Prophet(seasonality_mode='multiplicative', 
                        changepoint_range=0.75, 
                        n_changepoints=30, 
                        yearly_seasonality= 4, 
                        interval_width=0.95
                        )

Prophet_Model.fit(df_train)
""")


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def my_func():
    #st.write("Cache miss")
    Prophet_Model = Prophet(
        seasonality_mode='multiplicative',
        changepoint_range=0.75,
        n_changepoints=30,
        yearly_seasonality= 4,
        interval_width=0.95
    )   
    Prophet_Model.fit(df_train)
    return(Prophet_Model)

Prophet_Model = my_func()


# ---------- PART 1 - SECTION 3: Model Forecasting  ---------- #

toc.subheader('Model Forecasting')

st.write("""
Here we use the trained prophet Model to make the prediction for the next 30 days.
""")

with st.echo(): 
    future = Prophet_Model.make_future_dataframe(periods= 30)
    forecast_model= Prophet_Model.predict(future)



tab1, tab2, tab3 = st.tabs(["📊Chart", "🔣Code", "📃Data"])

with tab2:
    st.subheader('Plotly Chart Code')
    with st.echo():
        # Set layout with background color you want (rgba values)
        # This one is for white background
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')

        figure = go.Figure(layout = layout)


        # Lower Band
        figure.add_trace(go.Scatter(x=forecast_model['ds'],
                                    y=forecast_model['yhat_lower'],
                                    hoverinfo = 'none',
                                    showlegend = False,                    
                                    marker = {'color': "rgba(0, 0, 0,0)"}))

        # Upper Band
        figure.add_trace(go.Scatter(name = 'Confidence Interval',
                                    x=forecast_model['ds'], 
                                    y=forecast_model['yhat_upper'],                                                        
                                    fill = "tonexty", 
                                    #fillcolor = '#EAEAEA', 
                                    fillcolor = '#E4E8F0', 
                                    #fillcolor = "rgba(231, 234, 241,.75)",
                                    hoverinfo = 'none',
                                    mode="none"
                                    ))

        # yhat
        figure.add_trace(go.Scatter(name = 'Model Line of Best Fit',
                                    x=forecast_model['ds'], 
                                    y=forecast_model['yhat'],
                                    mode="lines",
                                    line = {'width' : 4},
                                    marker = {'color' : '#19D3F3'}))

        # Actual Price
        figure.add_trace(go.Scatter(x=df_train['ds'],
                                    y=df_train['y'], 
                                    name = 'Train Data', 
                                    mode="markers", 
                                    fill = None,  
                                    marker = {
                                        'color': "#fffaef",
                                        #'color': "#FFF5D5",                                        
                                        'size':5,
                                        'line': {
                                            'color':'#000000', 
                                            'width': .75}}))

        figure.add_trace(go.Scatter(x=df_test['ds'],
                                    y=df_test['y'], 
                                    name = 'Test Data', 
                                    mode="markers", 
                                    fill = None,  
                                    marker = {
                                        #'color': "#FAB8C6",
                                        'color': "#FF5D7C",
                                        #'color': "#FC8EA0",                                        
                                        'size':5,
                                        'line': {
                                            'color':'#000000', 
                                            'width': .75}}))

        figure.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='lightgray')  
        figure.update_yaxes(title_text="Price", showgrid=True, gridwidth=1, gridcolor='lightgray')  

        # update layout by changing the plot size, hiding legends & rangeslider
        figure.update_layout(
            height=600,
            width=950,
            legend= {'borderwidth' : 2,'bordercolor' : "lightgrey"},
            #title={
            #    'text':"BTC Forecast Model",
            #    'y':0.9,
            #    'x':0.5,
            #    'xanchor':'center',
            #    'yanchor':'top',
            #    'font_size':18
            #},             
            margin=dict(
                l=1,
                r=40,
                t=50,
                b=50                
            )
        )

        figure.show()

with tab1:
    st.write("*Click on the desired tab")
    toc.subheader('BTC Forecast Model Visualization')        
    st.plotly_chart(figure)

with tab3:
    st.subheader("**Forecast Model Output**")
    st.write("Includes upper and lower bounds for each day.")
    st.code("""
    forecast_model[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    """)
    
    forecast_model_inst = forecast_model[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.dataframe(forecast_model_inst.style.format({'yhat':'{:.2f}','yhat_lower':'{:.2f}','yhat_upper':'{:.2f}'}))

# ---------- PART 1 - SECTION 4: Model Forecast Components  ---------- #

toc.subheader('Forecast Components') 
#st.subheader('Forecast Components Visualization') 
st.write("""
`Trend`: We see a sharp increase at the end of 2020. It reaches the top around April 2021. It then gradually decreases till the November 2021. To continue falling at an accelarated pace

`Weekly Seasonality`: It shows Bitcoin's price starts decreasing Monday to reach the lowest point fo the week on Thursday. Then the prices starts to climb to its highest point on Saturday.

`Yearly Seasonality`: The year starts with a gradual uptrend till April, where it reverses sharply, reaching its lowest point in July. Here, it does a relatively sharp reversal once again to a positive slope, to reah its highest point on November. 

""")

with st.echo(): 
    fig2 = Prophet_Model.plot_components(forecast_model)
st.pyplot(fig2)


# ---------- PART 1 - SECTION 5: Cross validation  ---------- #

toc.subheader("Cross validation")

st.write("""
Prophet includes functionality for time series cross validation to measure
forecast error using historical data. This is done by selecting cutoff points
in the history, and for each of them fitting the model using data only up to that
cutoff point. We can then compare the forecasted values to the actual values.



We invoke the `cross_validation` function, and we specify the following:



`Prophet_Model`: The trained model.

`initial = 605 days`: The initial model will be trained on the first 605 days of data.

`period = 15 days`: 15 days will be added to the training dataset for each additional model.

`horizon = 30 days`: The model forecasts the next 30 days. When only horizon
is given, Prophet defaults initial to be triple the horizon, and period to be half of the horizon.

`parallel = processes`: Enables parallel processing for cross-validation. When the
parallel cross-validation can be done on a single machine, "processes" provide the highest performance. For larger problems, "dask" can be used to do cross-validation on multiple machines.

""")

st.code("""

forecast_cv = cross_validation(
    Prophet_Model, 
    initial = '605 days', 
    period = '15 days', 
    horizon = '30 days'

""")


@st.cache(suppress_st_warning=True)
def cross_val():
    #st.write("Cache miss")
    forecast_cv = cross_validation(
        Prophet_Model, 
        initial = '605 days', 
        period = '15 days', 
        horizon = '30 days')
    return(forecast_cv)

forecast_cv = cross_val()


# ---------- PART 1 - SECTION 6: Performance Matrix  ---------- #

toc.subheader("Performance Matrix")
st.write("""
`pm` is the performance matrix dataframe showing several model fit statistics:

*   **MSE:** mean squared error
*   **RMSE:** root mean squared error
*   **MAE:** mean absolute error
*   **MAPE:** mean absolute percent error


""")


with st.echo():    
    pm = performance_metrics(forecast_cv, rolling_window=0.05)

st.write("""
The dataframe below show us the aggregated value for each of the metrics calculated in the cross validation.
""")

with st.echo():
    # For charting purposes we need to change the datatype for column "horizon (days)" from timedelta to float
    pm['horizon'] = pm['horizon'].astype('timedelta64[D]')
    pm.rename(columns={"horizon": "horizon (days)"}, inplace=True)
    pm[['horizon (days)', 'mape', 'mae', 'mse', 'rmse', 'coverage']]


st.code("pm[['horizon (days)', 'mape', 'mae', 'mse', 'rmse', 'coverage']].describe()")
st.write(pm[['horizon (days)', 'mape', 'mae', 'mse', 'rmse', 'coverage']].describe())
    

# ---------- PART 1 - SECTION 7: Performance Matric Visualization  ---------- #

tab1, tab2, tab3 = st.tabs(["📊Chart", "🔣Code", "📃Additional Data"])

with tab3:  
    st.write("""
    The below dataframe show us the performance metrics for each of the folds in the cross validation. 
    There were 13 cut-offs. Therefore, on the performace metric visualization below you will see 13 dots per horizon day.
    """)   
    with st.echo():
        pm_all = performance_metrics(forecast_cv, rolling_window=-1)

    with st.echo():
        # For charting purposes we need to change the datatype for column "horizon (days)" from timedelta to float
        pm_all['horizon'] = pm_all['horizon'].astype('timedelta64[D]')
        pm_all.rename(columns={"horizon": "horizon (days)"}, inplace=True)
        st.dataframe(pm_all)

with tab2:
    st.subheader('Plotly Chart Code')   
    with st.echo():
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        figure2 = go.Figure(layout = layout)
        
        # Model 1 - Folds
        figure2.add_trace(go.Scatter(name = 'Model 1 Folds',
                                    x=pm_all['horizon (days)'], 
                                    y=pm_all['mape'],
                                    mode="markers",
                                    line = {'width' : 4},
                                    #marker = {'color' : '#19D3F3'}))
                                    marker = {'color' : '#B8BFE6'}))
        
        # Model 1 - Aggregated    
        figure2.add_trace(go.Scatter(name = 'M1 Aggregate',
                                    x=pm['horizon (days)'], 
                                    y=pm['mape'],
                                    mode="lines",
                                    line = {'width' : 4},
                                    #marker = {'color' : '#19D3F3'}))
                                    marker = {'color' : '#3948A5'}))
        
        figure2.update_xaxes(title_text="Horizon (Day)", showgrid=True, gridwidth=1, gridcolor='lightgray')  
        figure2.update_yaxes(title_text="MAPE", showgrid=True, gridwidth=1, gridcolor='lightgray')  

        figure2.update_layout( 
                            legend= {'borderwidth' : 2,'bordercolor' : "lightgrey"}, 
                            #title_text= "Performance Metric Chart", 
                            title_font_size=20,
                            width=900,
                            height=500,
                            title_x=0.5,
                            margin=dict(
                                l=1,
                                r=40,
                                t=40,
                                b=50
                            )             
                )

with tab1:
    st.markdown("""
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-hand-index-thumb" viewBox="0 0 16 16">
        <path d="M6.75 1a.75.75 0 0 1 .75.75V8a.5.5 0 0 0 1 0V5.467l.086-.004c.317-.012.637-.008.816.027.134.027.294.096.448.182.077.042.15.147.15.314V8a.5.5 0 0 0 1 0V6.435l.106-.01c.316-.024.584-.01.708.04.118.046.3.207.486.43.081.096.15.19.2.259V8.5a.5.5 0 1 0 1 0v-1h.342a1 1 0 0 1 .995 1.1l-.271 2.715a2.5 2.5 0 0 1-.317.991l-1.395 2.442a.5.5 0 0 1-.434.252H6.118a.5.5 0 0 1-.447-.276l-1.232-2.465-2.512-4.185a.517.517 0 0 1 .809-.631l2.41 2.41A.5.5 0 0 0 6 9.5V1.75A.75.75 0 0 1 6.75 1zM8.5 4.466V1.75a1.75 1.75 0 1 0-3.5 0v6.543L3.443 6.736A1.517 1.517 0 0 0 1.07 8.588l2.491 4.153 1.215 2.43A1.5 1.5 0 0 0 6.118 16h6.302a1.5 1.5 0 0 0 1.302-.756l1.395-2.441a3.5 3.5 0 0 0 .444-1.389l.271-2.715a2 2 0 0 0-1.99-2.199h-.581a5.114 5.114 0 0 0-.195-.248c-.191-.229-.51-.568-.88-.716-.364-.146-.846-.132-1.158-.108l-.132.012a1.26 1.26 0 0 0-.56-.642 2.632 2.632 0 0 0-.738-.288c-.31-.062-.739-.058-1.05-.046l-.048.002zm2.094 2.025z"/>
        </svg> Click on the desired
        """, unsafe_allow_html=True)   
    
    toc.subheader('Performance Metric Visualization') 
    st.plotly_chart(figure2)


st.write(""" 
The **MAPE** values for this prediction ranges between 7.32% and 15.78%.

On the graph below, the blue line shows the mean absolute percentage error (MAPE),
where the mean is taken over a 5% rolling window of the grey dots.

The x-axis is the horizon. The horizon was set to be 30 days into the future.
The y-axis is the metric we are interested in. We use MAPE in this visualization.
On each day, we can see 13 dots. This is because there are 13 models 
(with cutoffs between 2022-01-18 and 2022-07-17) in the cross-validation,
and each dot represents the MAPE for each model. The line is the aggregated performance
across all the models.

We can see that MAPE value increases with days, which is expected because time
series tend to make better predictions for the near future than the far future.

""")
    
# Visualize the performance metrics
#fc_cv_fig = plot_cross_validation_metric(forecast_cv, metric='mape', rolling_window=0.05)
#st.pyplot(fc_cv_fig)
#plt.show()

st.subheader('Mean Absolute Percentage Error By Sklearn')
st.write('**calculating the MAPE between expected and predicted values**')
with st.echo():
    df_y_yhat = pd.merge(BTC_2020_2022_Date_Price, forecast_model[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],on='ds') 

    # calculate MAPE between expected and predicted values
    y_true = df_y_yhat['y'].values
    y_pred = df_y_yhat['yhat'].values
    mape_1 = mean_absolute_percentage_error(y_true, y_pred)
    # % as a placeholder, which is replaced by mape_2 in this case. 
    # The f then refers to "Floating point decimal format". 
    # The .3 indicates to round to 3 places after the decimal point.
    '**MAPE: %.4f**' % mape_1

st.write(" ")


# ---------- PART 2  ---------- #
# ---------- PART 2 - SECTION 1: Fine Tuning Hyperparameters  ---------- #

toc.header("Part 2")

toc.subheader("Fine Tuning Hyperparameters")

st.write("""

While the model in part 1 was no “basic model”, meaning some hyperparameters and values used were other than the default ones, it was tuned manually. 

Here in part 2 we are going to build upon our first model by exposing it to 2 new hyperparameters, and optimize for these two hyperparameters
automatically using cross-validation and a tuning script. 


Prophet documentation suggests, our model can benefit the most by tuning the following hyperparameters:  

*   `changepoint_prior_scale`: This is one of the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.

*   `seasonality_prior_scale`: This parameter controls the flexibility of the seasonality. Similarly, a large value allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality. The default is 10., which applies basically no regularization. That is because we very rarely see overfitting here (there’s inherent regularization with the fact that it is being modeled with a truncated Fourier series, so it’s essentially low-pass filtered). A reasonable range for tuning it would probably be [0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is forced to be very small. This likely also makes sense on a log scale, since it is effectively an L2 penalty like in ridge regression.

The script will perform a 4x4 grid search to explore each possible combination of the provided range of values for `changepoint_prior_scale` and  `seasonality_prior_scale`,
and output the combination that best optimizes the model. 

""")

st.code(""" 
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],   
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here
maes = []  # Store the MAE for each params here
mapes = [] # Store the MAPE for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params, yearly_seasonality= 4, seasonality_mode='multiplicative', changepoint_range=0.70, n_changepoints=30, interval_width=0.95)   
    m.fit(df_train)  # Fit model with given params
    df_cv = cross_validation(m, initial='605 days',
        period='15 days', 
        horizon = '30 days', 
        #parallel="processes"    
        )
    df_p = performance_metrics(df_cv, rolling_window=0.05)
    
    maes.append(df_p['mae'].values[0])
    mapes.append(df_p['mape'].values[0])
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mapes
tuning_results['mae'] = maes
tuning_results['rmse'] = rmses

""")
    


@st.cache(suppress_st_warning=True)
def data_upload2(url2):
    #st.write("Cache miss")
    download = requests.get(url-2).content

    # Reading the downloaded content and turning it into a pandas dataframe
    tuning_results = pd.read_csv(io.StringIO(download.decode('utf-8')))

    tuning_results.drop(columns=tuning_results.columns[0], axis=1, inplace=True)

    return(tuning_results)


url2 = "https://raw.githubusercontent.com/efrenmo/Forecasting_BTC_with_Prophet/main/tuning_results_df.csv" # Make sure the url is the raw version of the file on GitHub
tuning_results = data_upload(url2)


#st.code('tuning_results.sort_values(['mape','mae'])')
st.code("tuning_results.sort_values(['mape','mae'])")
st.dataframe(tuning_results.sort_values(['mape','mae']))


st.write('The fine tuning script returned the following optimized combination of parameters:')

with st.echo():
    '**changepoint_prior_scale:**', tuning_results['changepoint_prior_scale'].loc[tuning_results['mape'].idxmin()]

    '**seasonality_prior_scale:**', tuning_results['seasonality_prior_scale'].loc[tuning_results['mape'].idxmin()]


# ---------- PART2 - SECTION 2: Re-training the Model ---------- #

toc.subheader("Re-training the Model")
st.write("""
Re-train the model with the new hyperparameters,  `changepoint_range` and `changepoint_prior_scale` and their optimized values.
""")


st.code("""
Prophet_Model = Prophet(seasonality_mode='multiplicative', 
                        changepoint_range=0.70, 
                        n_changepoints=30, 
                        yearly_seasonality= 4,
                        changepoint_prior_scale = 0.1,
                        seasonality_prior_scale = 10, 
                        interval_width=0.95
                        )

Prophet_Model.fit(df_train)
""")


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def my_func2():
    st.write("Cache miss")
    Tuned_Prophet_Model = Prophet(
        seasonality_mode='multiplicative',
        changepoint_range=0.75,
        n_changepoints=30,
        yearly_seasonality= 4,
        changepoint_prior_scale = 0.1,
        seasonality_prior_scale = 10,
        interval_width=0.95
        )   
    Tuned_Prophet_Model.fit(df_train)
    return(Tuned_Prophet_Model)

Tuned_Prophet_Model = my_func2()


with st.echo():    
    T_future = Tuned_Prophet_Model.make_future_dataframe(periods= 30)
    T_forecast_model= Tuned_Prophet_Model.predict(T_future)
    T_forecast_model[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(5)

st.write(" ")


# ---------- PART2 - SECTION 3: Cross Validation ---------- #
toc.subheader('P2 Cross Validation')

st.write('To calculate the new performance matrix we need to run the cross-validation using the newly optimized model created in the previous section.')

st.code("""

Tuned_forecast_cv = cross_validation(
    Tuned_Prophet_Model, 
    initial = '605 days', 
    period = '15 days', 
    horizon = '30 days'

""")


@st.cache(suppress_st_warning=True)
def cross_val2():
    #st.write("Cache miss")
    Tuned_forecast_cv = cross_validation(
        Tuned_Prophet_Model, 
        initial = '605 days', 
        period = '15 days', 
        horizon = '30 days')
    return(Tuned_forecast_cv)

Tuned_forecast_cv = cross_val2()


# ---------- PART2 - SECTION 4: Performance Matrix ---------- #

toc.subheader("P2 Performance Matrix")
st.write("""

`pm` is the performance matrix dataframe showing several model fit statistics:

*   **MSE:** mean squared error
*   **RMSE:** root mean squared error
*   **MAE:** mean absolute error
*   **MAPE:** mean absolute percent error


`rolling_window`: Proportion of data to use in each rolling window for computing the metrics.
                  
""")

# Model performance metrics

with st.echo():  
    T_pm = performance_metrics(Tuned_forecast_cv, rolling_window=0.05)

with st.echo():
    # For charting purposes we need to change the datatype for column "horizon (days)" from timedelta to float
    T_pm['horizon'] = T_pm['horizon'].astype('timedelta64[D]')
    T_pm.rename(columns={"horizon": "horizon (days)"}, inplace=True)
    T_pm[['horizon (days)', 'mape', 'mae', 'mse', 'rmse', 'coverage']]


with st.echo():
    st.write(T_pm[['horizon (days)', 'mape', 'mae', 'mse', 'rmse', 'coverage']].describe())


# ---------- PART 2 - SECTION 5: Performance Metric Visualization  ---------- #

tab1, tab2, tab3 = st.tabs(["📊Chart", "🔣Code", "📃Additional Data"])

with tab3:  
    st.write("""
    The below dataframe show us the performance metrics for each of the folds in the cross validation. 
    There were 13 cut-offs. Therefore, on the performace metric visualization below you will see 13 dots per horizon day.
    """)   
    with st.echo():
        T_pm_all = performance_metrics(Tuned_forecast_cv, rolling_window=-1)

    with st.echo():
        # For charting purposes we need to change the datatype for column "horizon (days)" from timedelta to float
        T_pm_all['horizon'] = T_pm_all['horizon'].astype('timedelta64[D]')
        T_pm_all.rename(columns={"horizon": "horizon (days)"}, inplace=True)
        st.dataframe(T_pm_all)

with tab2:
    st.subheader('Plotly Chart Code')   
    with st.echo():
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        figure3 = go.Figure(layout = layout)
        
        # Model 2 - Folds
        figure3.add_trace(go.Scatter(name = 'Model 2 Folds',
                                    x=T_pm_all['horizon (days)'], 
                                    y=T_pm_all['mape'],
                                    mode="markers",
                                    line = {'width' : 4},                                    
                                    marker = {'color' : '#F5D3E6'}))
        
        # Model 2 - Aggregated    
        figure3.add_trace(go.Scatter(name = 'M2 Aggregate',
                                    x=T_pm['horizon (days)'], 
                                    y=T_pm['mape'],
                                    mode="lines",
                                    line = {'width' : 4},                                    
                                    marker = {'color' : '#B826B1'}))
        
        figure3.update_xaxes(title_text="Horizon (Day)", showgrid=True, gridwidth=1, gridcolor='lightgray')  
        figure3.update_yaxes(title_text="MAPE", showgrid=True, gridwidth=1, gridcolor='lightgray')  

        figure3.update_layout( 
                            legend= {'borderwidth' : 2,'bordercolor' : "lightgrey"}, 
                            #title_text= "Performance Metric Chart", 
                            title_font_size=20,
                            width=900,
                            height=500,
                            title_x=0.5,
                            margin=dict(
                                l=1,
                                r=40,
                                t=40,
                                b=50
                            )             
                )

with tab1:   
    st.write('*Click on the "Code" tab to see the code for this chart')
    toc.subheader('P2 Performance Metric Visualization') 
    st.plotly_chart(figure3)


st.subheader('Mean Absolute Percentage Error By Sklearn')
st.write('**calculating the MAPE between expected and predicted values**')

with st.echo():
    df_y_yhat2 = pd.merge(BTC_2020_2022_Date_Price, T_forecast_model[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],on='ds') 

    # calculate MAPE between expected and predicted values
    y_true = df_y_yhat2['y'].values
    y_pred = df_y_yhat2['yhat'].values
    mape_2 = mean_absolute_percentage_error(y_true, y_pred)
    # % as a placeholder, which is replaced by mape_2 in this case. 
    # The f then refers to "Floating point decimal format". 
    # The .3 indicates to round to 3 places after the decimal point.
    '**MAPE: %.4f**' % mape_2

# ---------- PART 3 - Results and Conclusion  ---------- #

toc.header('Results')

st.write('Before and After Hyperparameter Tuning Performance Metric Comparison')

tab1, tab2 = st.tabs(["📊Chart", "🔣Code"])

with tab2:
    st.subheader('Plotly Chart Code')   
    with st.echo():
        layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
        figure4 = go.Figure(layout = layout)
        
        # Model 1 - Folds
        figure4.add_trace(go.Scatter(name = 'Model 1 Folds',
                                    x=pm_all['horizon (days)'], 
                                    y=pm_all['mape'],
                                    mode="markers",
                                    line = {'width' : 4},
                                    #marker = {'color' : '#19D3F3'}))
                                    marker = {'color' : '#B8BFE6'}))
        
        # Model 1 - Aggregated    
        figure4.add_trace(go.Scatter(name = 'Before Fine Tuning',
                                    x=pm['horizon (days)'], 
                                    y=pm['mape'],
                                    mode="lines",
                                    line = {'width' : 4},
                                    #marker = {'color' : '#19D3F3'}))
                                    marker = {'color' : '#3948A5'}))     
               
        # Model 2 - Folds
        figure4.add_trace(go.Scatter(name = 'Model 2 Folds',
                                    x=T_pm_all['horizon (days)'], 
                                    y=T_pm_all['mape'],
                                    mode="markers",
                                    line = {'width' : 4},                                    
                                    marker = {'color' : '#F5D3E6'}))
        
        # Model 2 - Aggregated    
        figure4.add_trace(go.Scatter(name = 'After Fine Tuning',
                                    x=T_pm['horizon (days)'], 
                                    y=T_pm['mape'],
                                    mode="lines",
                                    line = {'width' : 4},                                    
                                    marker = {'color' : '#B826B1'}))
        
        figure4.update_xaxes(title_text="Horizon (Day)", showgrid=True, gridwidth=1, gridcolor='lightgray')  
        figure4.update_yaxes(title_text="MAPE", showgrid=True, gridwidth=1, gridcolor='lightgray')  

        figure4.update_layout( 
                            legend= {'borderwidth' : 2,'bordercolor' : "lightgrey"}, 
                            #title_text= "Performance Metric Chart", 
                            title_font_size=20,
                            width=900,
                            height=500,
                            title_x=0.5,
                            margin=dict(
                                l=1,
                                r=40,
                                t=40,
                                b=50
                            )             
                )
    with tab1:   
        st.markdown("""
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-hand-index-thumb" viewBox="0 0 16 16">
        <path d="M6.75 1a.75.75 0 0 1 .75.75V8a.5.5 0 0 0 1 0V5.467l.086-.004c.317-.012.637-.008.816.027.134.027.294.096.448.182.077.042.15.147.15.314V8a.5.5 0 0 0 1 0V6.435l.106-.01c.316-.024.584-.01.708.04.118.046.3.207.486.43.081.096.15.19.2.259V8.5a.5.5 0 1 0 1 0v-1h.342a1 1 0 0 1 .995 1.1l-.271 2.715a2.5 2.5 0 0 1-.317.991l-1.395 2.442a.5.5 0 0 1-.434.252H6.118a.5.5 0 0 1-.447-.276l-1.232-2.465-2.512-4.185a.517.517 0 0 1 .809-.631l2.41 2.41A.5.5 0 0 0 6 9.5V1.75A.75.75 0 0 1 6.75 1zM8.5 4.466V1.75a1.75 1.75 0 1 0-3.5 0v6.543L3.443 6.736A1.517 1.517 0 0 0 1.07 8.588l2.491 4.153 1.215 2.43A1.5 1.5 0 0 0 6.118 16h6.302a1.5 1.5 0 0 0 1.302-.756l1.395-2.441a3.5 3.5 0 0 0 .444-1.389l.271-2.715a2 2 0 0 0-1.99-2.199h-.581a5.114 5.114 0 0 0-.195-.248c-.191-.229-.51-.568-.88-.716-.364-.146-.846-.132-1.158-.108l-.132.012a1.26 1.26 0 0 0-.56-.642 2.632 2.632 0 0 0-.738-.288c-.31-.062-.739-.058-1.05-.046l-.048.002zm2.094 2.025z"/>
        </svg> Click on the "Code" tab to see the code for this chart
        """, unsafe_allow_html=True)
        
        #st.subheader('P2 Performance Metric Visualization') 
        st.plotly_chart(figure4)


st.write("""

While the first model (model before fine tuning) had already some manually tweaked hyperparameters 
(meaning the model used hyperparameters and values other than default ones), the second model was introduced to 2 new hyperparameters that the first model had no exposure to, `changepoint_prior_scale`, and `changepoint_range` .

Since prophet documentation suggested these two were the most impactful if tuned right, I left these parameters to be tuned by an automatic tuning script.
The tuning script performed a 4x4 grid search trying out every possible combination of predetermined values in the grid.

In the figure above, we can clearly see a noticeable reduction in the mean absolute percentage error between model 1 and model 2. Model 1 MAPE values for the 30 day horizon ranged between 7.17% to 15.8%. 
While Model 2 MAPE values for the same horizon ranged between 6.34% to 13.85%

We can see that MAPE values increases as horizon days increases, which is expected because time series tend to make better predictions for the near future than far in the future.

""")

toc.header('Conclusion')

st.write("""
While Prophet is presented as very user friendly product, ready to produce outstanding results with little tweaking,
I find that there is a lot of room for customization that can improve your results in a meaningful way.
But, to make the right tweaking of the parameters, one should understand their data and its domain well, and be knowledgeable about time series forecasting.


Moreover, best results were obtained by using relatively newer data (last 2 years), as opposed to using the full price history of the asset.
My explanation to that is that the the profile of the asset has change drastically in the last 2 years, due to an increase in institutional interest. 
So that older data, prior institutional involvement, renders less weight on future predictions.
It would be interesting to revisit and retrain the model in the upcoming years; as more data accrues, the asset keeps maturing, and a more prominent seasonality emerges.

Regarding future direction, I’ll be looking for a model that, unlike Prophet, will allow me to add regressors for which future values are unknow.
As for which regressors, I’m particularly thinking about the dollar index, the NASDAQ, gold prices, 10 or 20 years US Treasuries, and the yield curve.
""")


st.sidebar.write("""

Created by: Efren Andres Mora

""")
st.sidebar.write("""


""")

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.markdown("<p style='text-align: right;'>Created by: Efren Andres Mora</p>", unsafe_allow_html=True)

toc.generate()


