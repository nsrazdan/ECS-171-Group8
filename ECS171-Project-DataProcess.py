# ## Data Processing - ECS171 Project Group 8
# **Description**: Converts true/false entries to 1/0 and normalizes numerical data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

def data_processing():
    #import data
    path = 'data/videos oct27-nov1.csv'
    df = pd.read_csv(path)

    #drop redundant columns
    drop_names = ['Unnamed: 0', 'Unnamed: 0.1','Channel_title','Channel_description']
    df = df.drop(drop_names,axis=1)

    #create column 'trending', 1 if yes, 0 if no
    df = process_trends(df)

    #normalize (StandardScaler)
    names_int = ['categoryId','view_count','likes','dislikes','comment_count','Channel_viewCount','Channel_subscriberCount','Channel_videoCount']
    df[names_int] = StandardScaler().fit_transform(df[names_int])

    #one hot encode
    names_category = ['comments_disabled','ratings_disabled','Channel_hiddenSubscriberCount']
    df = ohe(df, names_category)

    #plot data
    names_num = ['categoryId','view_count','likes','dislikes','comment_count','Channel_viewCount','Channel_subscriberCount','Channel_videoCount','comments_disabled','ratings_disabled','Channel_hiddenSubscriberCount','trending']
    df_num = df[names_num] #dataframe with only numerical columns
    plt.subplots(figsize=(20,15))
    sns.heatmap(df_num.corr())
    return df_num

#adds target attribute 'trending' with 1 if trend, 0 if not
def process_trends(df):
    trend_list = []
    for i in range(0,len(df)):
        if math.isnan(df['trending_date'][i]):
            trend_list.append(0)
        else:
            trend_list.append(1)
    df = df.drop('trending_date',axis=1)
    df['trending'] = trend_list
    return df

#ohe columns in 'names', returns new df
def ohe(df, names):
    new_df = df.copy()
    for name in names:
        tmp_df = pd.get_dummies(new_df[name],prefix=name)
        loc = new_df.columns.get_loc(name)
        new_df = new_df.drop([name],axis=1)
        for item in tmp_df.columns:
            new_df.insert(loc,column=item,value=tmp_df[item])
            loc += 1
        new_df = new_df.drop([name+'_False'],axis=1)
        new_df = new_df.rename(columns={name+'_True':name})
    return new_df

df_num = data_processing()