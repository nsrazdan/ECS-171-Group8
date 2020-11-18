import streamlit as st 

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

# Make the selection box that contains what pre-built models we have appear
if modelType == "Use Pre-Built Models":
    modelType = st.selectbox("Please Select Which Pre-Built Model You Would Like To Use", ("Accuracy", "Precision", "Add more here"));

if modelType == "Build Custom Model":
    modelType = st.radio("What Sort of Model do you Want to Build?", ("Logistic Regression", "Neural Network"))
    if modelType == "Logistic Regression":
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


# Display Info About The Model
# --------------------------------------------------------------------------
