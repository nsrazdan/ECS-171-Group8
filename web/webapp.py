import streamlit as st

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

import pickle

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Fetches and Prepares the Data
def prepareData():
    # Import Data Into Pandas Dataframe
    df = pd.read_csv('df_to_import.csv');

    # drop first column (junk values added on while importing)
    df = df.drop(df.columns[0], axis=1)

    # Split DataFrame into training and testing set: 70:30 ratio
    ratio = 0.7
    train, test = train_test_split(df, train_size = ratio, random_state = 42)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    # Separate Data into X and Y
    train_X = train.loc[:,train.columns != 'trending?']
    train_Y = train['trending?']
    test_X = test.loc[:,test.columns != 'trending?']
    test_Y = test['trending?']

    return [train_X, train_Y, test_X, test_Y]


# Load Our Pre-Built Neural Network Model
def loadNeuralNetworkModel():
    ANN_json_file = open('./prebuilt_models/ANN_model.json', 'r')
    ANN_loaded_json = ANN_json_file.read()
    ANN_json_file.close()
    ANN_model = model_from_json(ANN_loaded_json)
    ANN_model.load_weights('./prebuilt_models/ANN_model_weights.h5')
    # st.success("Loaded Neural Network!")
    return ANN_model

# Load Our Pre-Built Logistic Regression Model
def loadLogisticModel():
    LR_model = pickle.load(open("./prebuilt_models/LR_model.sav", 'rb'))
    # st.success("Loaded Logistic Regression Model!")
    return LR_model


# Building A User-Defined Neural Network Model
def buildCustomNeuralNetworkModel(is_model_built):
    num_hidden_layers = st.slider("Number of Hidden Layers: ", 1, 3)

    num_nodes_1 = 0
    num_nodes_2 = 0
    num_nodes_3 = 0
    num_hidden_layer_nodes = []

    if num_hidden_layers == 1:
        num_nodes_1 = st.slider("Number of Nodes in First Layer: ", 1, 30)
    elif num_hidden_layers == 2:
        num_nodes_1 = st.slider("Number of Nodes in First Layer: ", 1, 30)
        num_nodes_2 = st.slider("Number of Nodes in Second Layer: ", 1, 30)
    elif num_hidden_layers == 3:
        num_nodes_1 = st.slider("Number of Nodes in First Layer: ", 1, 30)
        num_nodes_2 = st.slider("Number of Nodes in Second Layer: ", 1, 30)
        num_nodes_3 = st.slider("Number of Nodes in Third Layer: ", 1, 30)

    for i in range(num_hidden_layers):
        if i == 0:
            num_hidden_layer_nodes.append(num_nodes_1)
        elif i == 1:
            num_hidden_layer_nodes.append(num_nodes_2)
        elif i == 2:
            num_hidden_layer_nodes.append(num_nodes_3)
        # st.success(num_hidden_layer_nodes[i]) # For Debugging
    
    learning_rate = st.number_input("Learning Rate: ")

    # NOTE: Adjust the range later
    epochs = st.slider("Number of Epochs: ", 100,500)
    batch_size = st.slider("Batch Size: ", 100,500)
    
    optimizer = st.selectbox("Please Select Which Optimizer You Would Like To Use", ("adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl", "RMSprop", "SGD"))
    loss = st.selectbox("Please Select Which Loss Metric You Would Like To Use", ("mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "cosine_similarity", "logcosh"))

    metrics = [tf.keras.metrics.Accuracy(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
    metrics_names = ["accuracy","recall","precision"]

    # NOTE: Add Customizable Later?
    hidden_layer_activations = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']

    # Import Data Needed For Model Creation
    dataList = prepareData() # Returns: [train_X, train_Y, test_X, test_Y]
    train_X = dataList[0]
    train_Y = dataList[1]
    test_X = dataList[2]
    test_Y = dataList[3]

    if st.button("Build Model"): # User presses button when they are done inputting data
        ANN_model = keras.Sequential()
        # add hidden layers
        for i in range(num_hidden_layers):
            if i == 0:
                ANN_model.add(Dense(num_hidden_layer_nodes[i], input_dim = train_X.shape[1], activation=hidden_layer_activations[i]))
            ANN_model.add(Dense(num_hidden_layer_nodes[i], activation=hidden_layer_activations[i]))
         
        # add output layer
        ANN_model.add(Dense(1, activation=hidden_layer_activations[num_hidden_layers]))
        
        # fit model with metrics
        ANN_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # st.success("Finished Building Model")
        
        # NOTE: If something goes wrong, this is it haha
        train_X = np.asarray(train_X).astype('float32')
        train_Y = np.asarray(train_Y).astype('float32')
        test_X = np.asarray(test_X).astype('float32')
        test_Y = np.asarray(test_Y).astype('float32')

        epoch_cb_num = 0 # set progress bar at 0
        training_progress = st.progress(epoch_cb_num)

        # Callback (Updates Progress Bar)
        class UpdateProgressBar(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs = {}):
                # Update progress bar
                training_progress.progress( (epoch+1)/ epochs)

        # Once model is built, we must train it
        ANN_model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=epochs, batch_size=batch_size, callbacks= [UpdateProgressBar()])
        st.success("Finished Training Model")      
        return ANN_model


# Display Statistics For Logistic Regression Model
def displayLogisticStats(LR_model):
    # Import Data Needed For Model Evaluation
    dataList = prepareData() # Returns: [train_X, train_Y, test_X, test_Y]
    train_X = dataList[0]
    train_Y = dataList[1]
    test_X = dataList[2]
    test_Y = dataList[3]

    
    accuracy = LR_model.score(test_X, test_Y)
    y_scores = LR_model.predict(test_X)

    # Confusion Matrix
    disp_con_mat = plt.figure()
    con_mat = metrics.confusion_matrix(test_Y, y_scores)
    sns.heatmap(con_mat/np.sum(con_mat), annot=True, fmt=".2%", linewidths=1, linecolor = "black", square = True, cmap = 'BuPu');
    plt.ylabel('True Label')
    plt.xlabel("Predicted Label")
    con_mat_title = 'Accuracy Score: {:.2%}'.format(accuracy)
    plt.title(con_mat_title, size = 15)
    st.pyplot(disp_con_mat)


    # ROC Curve
    # --------------------------------------------------
    untrained_prob = [0 for _ in range(len(test_Y))]
    LR_prob = LR_model.predict_proba(test_X)
    LR_prob = LR_prob[:,1]

    untrained_AUC = metrics.roc_auc_score(test_Y, untrained_prob)
    LR_prob_AUC = metrics.roc_auc_score(test_Y, LR_prob)

    # Calculating the true positive rate (tpr) and false positive rate (fpr)
    untrained_fpr, untrained_tpr, _ = metrics.roc_curve(test_Y, untrained_prob)
    LR_fpr, LR_tpr, _ = metrics.roc_curve(test_Y, LR_prob)

    disp_ROC_curve = pyplot.figure()
    pyplot.plot(untrained_fpr, untrained_tpr, linestyle = '--', label = 'Untrained')
    pyplot.plot(LR_fpr, LR_tpr, marker='.', label = 'Logistic Regression')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    st.pyplot(disp_ROC_curve)
    # ---------------------------------------------------------

    return


# TODO: Implement Displaying Model Statistics
# Display Statistics for Neural Network Model
def displayNeuralNetworkStats(ANN_model):
    # Import Data Needed For Model Evaluation
    dataList = prepareData() # Returns: [train_X, train_Y, test_X, test_Y]
    train_X = dataList[0]
    train_Y = dataList[1]
    test_X = dataList[2]
    test_Y = dataList[3]

    st.success("Impelement This")

    return


# Sidebar Customization
# ==================================================================================

st.sidebar.header("Youtube Trending Videos");
st.sidebar.subheader("ECS 171 | Group 8 | Fall 2020");
st.sidebar.text("Nikhil Razdan\tCameron Yuen\nJoshua McGinnis\tKeith Chuong\nOwen Gao\tPhalgun Krishna\nPrajwal Singh\tRohail Asad\nSeth Damany\tTheresa Nowack\nThu Vo\t\tTrevor Carpenter\nTed Kahl");

st.sidebar.text("");
st.sidebar.text("");
st.sidebar.text("\n\n\n*Insert Description/Introduction Here*");

# ===================================================================================




# Choosing Models
# ==================================================================================
st.header("Machine Learning Model Setup")
modelType = st.radio("What type of model would you like to use?", ("Use Pre-Built Model", "Build Custom Neural Network Model"))


# Make the selection box that contains what pre-built models we have appear
if modelType == "Use Pre-Built Model":
    prebuilt_model_selection = st.radio("Please Select Which Pre-Built Model You Would Like To Use", ("Logistic Regression", "Neural Network"))
    LR_model = LogisticRegression()
    if prebuilt_model_selection == "Logistic Regression" and st.button("Load & Evaluate Model"):
        LR_model = loadLogisticModel()
        displayLogisticStats(LR_model)
             



    elif prebuilt_model_selection == "Neural Network" and st.button("Load & Evaluate Model"):
        ANN_model = loadNeuralNetworkModel()
        displayNeuralNetworkStats(ANN_model)
        
        
        

# ----------------Building Custom Neural Network Model--------------------
if modelType == "Build Custom Neural Network Model":
    is_model_built = False # Ensures you can only display stats after building model
    ANN_model = buildCustomNeuralNetworkModel(is_model_built)

    if st.button("Evaluate Neural Network"):
        displayNeuralNetworkStats(ANN_model)
