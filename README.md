# CNN-LSTM-EEG

# Overview
A hybrid CNN-LSTM model is trained to localise anomalies in each channel of EEG record. Proposed architecture is divided into two steps. First, Deep CNN is trained for detecting abnormal channels. Furthermore, to detect anomaly time from abnormal channels Long Short-Term Memory (LSTM) network is trained.

# Tools and Technologies

This application was programmed in Python 3.5

# Run

Modify config.py, especially correct data folders for your path..
Run with python ./auto_diagnosis.py
auto_diagnosis.py defines and train CNN models
diagnosis.py gives features from trained deep CNN
hybrid_lstm.py trains an LSTM model to classify sequence of features
