# Iris Flower & Classification

## Overview

- This project focuses on Exploratory Data Analysis (EDA), data visualization, feature engineering, and machine learning classification using the classic Iris Flower dataset. The goal is to analyze there lationships between petal and sepal dimensions and classify Iris species using a Support VectorMachine (SVM) model.

## Objective

- The objective of this project is to analyze the Iris Flower dataset using EDA and visualization techniques and build a machine learning model to classify iris species with high accuracy.

## Key Skills Demonstrated

- Exploratory Data Analysis (EDA) using Pandas & NumPy
- Data visualization with Matplotlib & Seaborn
- Feature engineering with ratio-based features
- Statistical analysis and distribution interpretation
- Machine learning classification using Support Vector Machine (SVM)
- Data preprocessing and scaling with StandardScaler
- Model evaluation and performance analysis
- Writing clean, structured, and reproducible Python code


## Dataset

- Source: kaggle
- Name: Iris Flower Dataset
- Rows: 150
- Columns: 5
- Link: https://www.kaggle.com/datasets/arshid/iris-flower-dataset

### Features

- sepal_length – Sepal length (cm)
- sepal_width – Sepal width (cm)
- petal_length – Petal length (cm)
- petal_width – Petal width (cm)
- species – Iris species (categorical)

## Feature Engineering

Two ratio-based features were created to capture proportional relationships:
- PL_SL: Petal Length / Sepal Length
- PW_SW: Petal Width / Sepal Width

These features help capture proportional relationships between petals and sepals.

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Visualizations

The following plots are generated to analyze the data:

- Histogram (distribution by species)
- Scatter Plot (relationships between features)
- Heatmap (features correlations)
- Box Plot (outlier detection)
- Pie Chart (species distribution)
- Pair Plot (pairwise feature relationships)
- Violin Plot & Swarm Plot (distribution by species)
- Min-Max Chart (size comparison between species)

## Machine Learning

- Model: Support Vector Classifier (SVC)
- Kernel: RBF
- Data Scaling: StandardScaler
- Train/Test Split: applied before training

## Model Evaluation

- Model: Support Vector Classifier (RBF kernel)
- Accuracy: High accuracy on the test set
- Dataset split: Train/Test 

## Results

- Setosa species shows the strongest correlations.
- Sepals of Setosa are generally larger.
- Petals of Versicolor and Virginica are larger and similar in size.
- Petals-based features show stronger correlation than sepal-based features.
- Species distribution is perfectly balanced.
- The SVM model achieved 100% accuracy on the test data.

## Conclusion

This project demonstrates a complete data science workflow:

- Data exploration
- Visualization
- Feature engineering
- Machine learning modeling

It provides clear insights into Iris species characteristics and achieves excellent classification performance.