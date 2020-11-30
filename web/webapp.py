import streamlit as st
from getonevideo import get_video

st.sidebar.header("Youtube Trending Videos");
st.sidebar.subheader("ECS 171 | Group 8 | Fall 2020");
st.sidebar.text("Nikhil Razdan\tCameron Yuen\nJoshua McGinnis\tKeith Chuong\nOwen Gao\tPhalgun Krishna\nPrajwal Singh\tRohail Asad\nSeth Damany\tTheresa Nowack\nThu Vo\t\tTrevor Carpenter\nTed Kahl");

st.sidebar.text("");
st.sidebar.text("");
st.sidebar.text("\n\n\n*Insert Description/Introduction Here*");


# Choosing Models
# --------------------------------------------------------------------------
st.header("Choose Model (Custom vs. Our Best Models)")

modelType = st.radio("What type of model would you like to use?", ("Use Pre-Built Models", "Build Custom Model"));
modelTrait = st.radio("Would you like to use Logistic Regression or a Neural Network?", ("Logistic Regression", "Neural Network"));

# Make the selection box that contains what pre-built models we have appear
if modelType == "Use Pre-Built Models":
	modelType = st.selectbox("Please Select Which Pre-Built Model You Would Like To Use", ("Accuracy", "Precision", "Add more here"));

if modelType == "Build Custom Model":
    if modelTrait == "Logistic Regression":
        learningRate = st.number_input("Learning Rate: ");
        actFunction = st.number_input("Activation Function: ");
        regularization = st.number_input("Regularization: ");
        penalty = st.number_input("Penalty: ");
    else:
        learningRate = st.number_input("Learning Rate: ");
        batchSize = st.number_input("Batch Size: ");
        epochs = st.number_input("Number of Epochs: ");
        numLayers = st.number_input("Number of Layers: ");
        neuronPerLayer = st.number_input("Neurons Per Layer: ");
        actFunction = st.number_input("Activation Function: ");
        momentum = st.number_input("Momentum: ");


# Build/Train Model If User Selected "Build Custom Model"
# --------------------------------------------------------------------------
# Load in dataset, train model
# Maybe add in a loading bar? (Set it to increase a certain amount in between iterations?)


# Load Model
# --------------------------------------------------------------------------
# if modelType == "Accuracy"...
# elif modelType == "Precision"...




# Display Info About The Model
# --------------------------------------------------------------------------
# Pass in the test data and retrieve the model metrics (Accuracy, Loss...)
# Confusion Matrix?
# Display Weights?




# Obtain Video Data From URL
# --------------------------------------------------------------------------
st.header("Video Trendability Prediction");
videoURL = st.text_input("Enter Video URL");
if videoURL != "":
    videoIDIndex = videoURL.find("=");
    videoID = videoURL[videoIDIndex+1 : ];
    videoData = get_video(videoID);
    st.table(videoData); # for debugging



# Run Video Data Through Model
# --------------------------------------------------------------------------
result = None
calculating = False # until the model is activated


# Output the results from the model
# --------------------------------------------------------------------------
calculating = True # when the model is calculating
if result != None:
	st.subheader("Calculating...")
else:
	st.success("Your video has a {result} chance of being trendy on Youtube!")