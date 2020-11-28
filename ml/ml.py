# newfile

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import dateutil.parser as parser
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

nltk.download('vader_lexicon')

# read the dataset into pandas dataframe
df = pd.read_csv('./../training_data.csv', delim_whitespace=False).dropna()


class DataProcessor:
    def __init__(self, data):
        self.df = data
    def process_train_data(self):
        df = self.df
        retrieval_time = df['time_retrieved']
        publish_time = df['publishedAt']
        channel_publish_time = df['Channel_publishedAt']
        retrieval_time_11_19_14 = df['11_19_14_update_timestamp']
        columns_to_drop = ['definition', 'publishedAt', 'time_retrieved', 'Channel_title', '11_19_14_update_timestamp', 'Channel_publishedAt', 'video_id', 'channelId', 'thumbnail_link', 'Channel_country']
        df = df.drop(columns_to_drop, axis = 1)

        # switch strings for booleans
        for i in df.index:
            if df['ratings_disabled'][i] == 'True':
                df['ratings_disabled'][i] = True
            elif df['ratings_disabled'][i] == 'False':
                df['ratings_disabled'][i] = False

            if df['Channel_hiddenSubscriberCount'][i] == 'True':
                df['Channel_hiddenSubscriberCount'][i] = True
            elif df['Channel_hiddenSubscriberCount'][i] == 'False':
                df['Channel_hiddenSubscriberCount'][i] = False


            if df['trended_later'][i] == 'True':
                df['trended_later'][i] = True
            elif df['trended_later'][i] == 'False':
                df['trended_later'][i] = False

        age = []
        age_update = []
        channel_age = []
        for i in df.index:
            channel_publish_time[i] = channel_publish_time[i].replace("\"", "")
            age.append(parser.isoparse(retrieval_time[i]) - parser.isoparse(publish_time[i]))
            age_update.append(parser.isoparse(retrieval_time_11_19_14[i]) - parser.isoparse(publish_time[i]))
            channel_age.append(parser.isoparse(channel_publish_time[i]) - parser.isoparse(publish_time[i]))
     
    
        titles = df['title']
        channel_title = df['channelTitle']
        description = df['description']
        channel_description = df['Channel_description']

        title_sentiment_vals = []
        channel_title_sentiment_vals = []
        description_sentiment_vals = []
        channel_description_sentiment_vals = []

        sid = SentimentIntensityAnalyzer()
        for sentence in titles:
            ss = sid.polarity_scores(str(sentence))
            title_sentiment_vals.append(ss['pos']-ss['neg'])

        for sentence in channel_title:
            ss = sid.polarity_scores(str(sentence))
            channel_title_sentiment_vals.append(ss['pos']-ss['neg'])

        for sentence in description:
            ss = sid.polarity_scores(str(sentence))
            description_sentiment_vals.append(ss['pos']-ss['neg'])

        for sentence in channel_description:
            ss = sid.polarity_scores(str(sentence))
            channel_description_sentiment_vals.append(ss['pos']-ss['neg'])

        df['title'] = title_sentiment_vals
        df['channelTitle'] = channel_title_sentiment_vals
        df['description'] = description_sentiment_vals
        df['Channel_description'] = channel_description_sentiment_vals

        le = preprocessing.LabelEncoder()
        df['trending?'] = le.fit_transform(df['trending?'])
        df['ratings_disabled'] = le.fit_transform(df['ratings_disabled'])
        df['Channel_hiddenSubscriberCount'] = le.fit_transform(df['Channel_hiddenSubscriberCount'])
        pd.set_option('display.max_columns', None)
        
        self.process_data = df
          

class MLModelMaker:
    def __init__(self, data, ratio = 0.7):
        self.df = data
        self.ratio = ratio
        self.train, self.test = train_test_split(self.df, train_size=self.ratio, random_state=42)
        self.train.reset_index(inplace=True, drop=True)
        self.test.reset_index(inplace=True, drop=True)
        self.train_X = self.train.loc[:,self.train.columns != 'trending?']
        self.train_Y = self.train['trending?']
        self.test_X = self.test.loc[:,self.test.columns != 'trending?']
        self.test_Y = self.test['trending?']
        
    def trainLogisticRegression(self):
        self.LR_model = LogisticRegression(multi_class='ovr')
        self.LR_model.fit(self.train_X, self.train_Y)
    
    def trainNeuralNetwork(self, num_hidden_layers, num_hidden_layer_nodes, train_ratio, hidden_layer_activations, optimizer, learning_rate, loss, metrics, metrics_names, epochs, batch_size):
        self.ANN_model = keras.Sequential()

        # add hidden layers
        for i in range(num_hidden_layers):
            if i == 0:
                self.ANN_model.add(Dense(num_hidden_layer_nodes[i], input_dim = self.train_X.shape[1], activation = hidden_layer_activations[i]))
            self.ANN_model.add(Dense(num_hidden_layer_nodes[i], activation=hidden_layer_activations[i]))

        # add output layers
        self.ANN_model.add(Dense(1, activation=hidden_layer_activations[num_hidden_layers]))
        self.ANN_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)       
        self.ANN_model.fit(self.train_X, self.train_Y, epochs=epochs, batch_size=batch_size)
        
        
        
### Process Everything:

num_hidden_layers = 3
num_hidden_layer_nodes = [20, 10, 5]
train_ratio = .7
hidden_layer_activations = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
optimizer = 'adam'
learning_rate = .005
loss = 'mean_squared_error'
metrics = [tf.keras.metrics.Accuracy(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
metrics_names = ["accuracy","recall","precision"]
epochs = 150
batch_size = 200
    
processor = DataProcessor(df)
processor.process_train_data()
model_maker = MLModelMaker(processor.process_data)
model_maker.trainLogisticRegression()
model_maker.trainNeuralNetwork(num_hidden_layers, num_hidden_layer_nodes, train_ratio, hidden_layer_activations, optimizer, learning_rate, loss, metrics, metrics_names, epochs, batch_size)

## LR Confusion Matrix
pred_Y = model_maker.LR_model.predict(model_maker.test_X)
print("LR Model:")
print(model_maker.LR_model.score(model_maker.test_X, model_maker.test_Y))
print(confusion_matrix(model_maker.test_Y, pred_Y))

## NN Confusion Matrix
pred_Y = model_maker.ANN_model.predict_classes(model_maker.test_X)
print("ANN Model:")
evaluated_metrics = model_maker.ANN_model.evaluate(model_maker.test_X, model_maker.test_Y)
for i in range(len(metrics)):
    print(metrics_names[i] + ": %.2f" % evaluated_metrics[i])
    
print(confusion_matrix(model_maker.test_Y, pred_Y))

    