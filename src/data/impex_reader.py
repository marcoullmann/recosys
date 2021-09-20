import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
from pathlib import Path

load_dotenv(find_dotenv())

dtypes = {'order': 'string', 'product': 'string', 'quantity': 'int64'}
df_small_cols = ['order', 'creationtime', 'product', 'quantity', 'totalPrice', 'user']


def read_raw_vx():
    orderentry = pd.read_csv(os.getenv("RAW_DATA_PATH") + 'vx/orderentry/OrderEntry.csv', dtype=dtypes, thousands=',',
                             delimiter=";", parse_dates=['creationtime'])
    orderentry['order'] = orderentry['order'].str.replace('\W', '')
    order = pd.read_csv(os.getenv("RAW_DATA_PATH") + 'vx/order/Order.csv', thousands=',', delimiter=";",
                        low_memory=False, parse_dates=['creationtime'])
    order_user = order[['code', 'user']]
    merged = pd.merge(orderentry, order_user, 'inner', left_on='order', right_on='code')
    return merged[df_small_cols]


def read_raw_koch():
    df = pd.read_csv(os.getenv("RAW_DATA_PATH") + 'koch/OrderEntry.csv', dtype=dtypes, thousands=',', delimiter=",",
                     parse_dates=['creationtime'])
    df[['order', 'user']] = df.order.str.split(":", expand=True)
    return df[df_small_cols]

def process_youchoose():
    dtypes_youchoose = {'user': 'int64', 'time': 'int64', 'item': 'int64', 'price': 'string', 'qty': 'int64'}
    df = pd.read_csv(os.getenv("RAW_DATA_PATH") + 'yoochoose-buys.csv', dtype=dtypes_youchoose, delimiter=",", parse_dates=['time'])
    df['time'] = (df['time'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
    df = df.head(100000)
    df[["user", "item", "time"]].to_csv("../data/curated/youchoose.csv", header=False ,index=False)

def process_retail_rocket():
    dtypes_retailrocket = {'timestamp': 'int64', 'visitorid': 'string', 'event': 'string', 'itemid': 'int64', 'transactionid': 'string'}
    df = pd.read_csv(os.getenv("RAW_DATA_PATH") + 'retailrocket.csv', dtype=dtypes_retailrocket, delimiter=",")
    df = df[df['event'] == 'transaction']
    df.rename(columns={"timestamp": "time", "visitorid": "user", "itemid": "item"}, inplace=True)
    df['time'] = (pd.to_datetime(df['time'], unit='ms') - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df[["user", "item", "time"]].to_csv("../data/curated/retailrocket.csv", header=False ,index=False)

def read_raw_online_retail():
    df = pd.read_excel(os.getenv("RAW_DATA_PATH") + 'online_retail/OnlineRetail.xlsx')
    cleaned_retail = df.loc[pd.isnull(df.CustomerID) == False]
    item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
    item_lookup['StockCode'] = item_lookup.StockCode.astype(str)

    cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
    cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
    grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
    grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
    # indicate purchased
    grouped_purchased = grouped_cleaned.query('Quantity > 0')

def write_to_curated(df: pd.DataFrame, folder: str):
    output_file = 'orderentry.pickle'
    output_dir = Path(os.getenv("CURATED_DATA_PATH") + folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_dir / output_file, protocol=4)
