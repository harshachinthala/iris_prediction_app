---
title: Iris Prediction App
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
---
# Iris Flower Prediction App

This app predicts the species of Iris flowers based on sepal and petal measurements using a K-Nearest Neighbors (KNN) classifier. It features an interactive Streamlit interface for real-time inference and dataset visualization.

## 🧠 Tech Stack & Skills

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-blueviolet)

## 🚀 Features

- **Real-time Inference:** Instantly predicts the Iris species as you adjust the input parameters.
- **Interactive Sliders:** Simple and intuitive sidebar sliders for modifying Sepal and Petal dimensions.
- **Probability Distribution:** Displays the prediction probability for each of the three species (Setosa, Versicolor, Virginica).
- **Data Visualization:** Visualizes the entire Iris dataset using a pairplot to show the relationship between features.

## 🔄 Modeling Flow

The application follows a standard machine learning pipeline:

1.  **Data Loading:** The standard Iris dataset is loaded, consisting of 150 samples with 4 features each.
2.  **User Input:** The user selects the Sepal Length, Sepal Width, Petal Length, and Petal Width via the sidebar.
3.  **Model Training:** A K-Nearest Neighbors (KNN) classifier with `k=3` is trained on the entire dataset.
4.  **Prediction:** The trained model predicts the species for the user's input and calculates the class probabilities.
5.  **Visualization:** The result is displayed, and a specific visualization (pairplot) shows the data distribution.

## 📊 Dataset Visualization

The app includes a pairplot visualization of the Iris dataset using Seaborn to help users understand how the different species are distributed across the four feature dimensions.
