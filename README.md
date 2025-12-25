# Iris Flower & Classification

## Overview
- This project focuses on Exploratory Data Analysis (EDA), data visualization, feature engineering, and machine learning classification using the classic Iris Flower dataset. The goal is to analyze there lationships between petal and sepal dimensions and classify Iris species using a Support VectorMachine (SVM) model.

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
- *Model:* Support Vector Classifier (SVC)
- Kernel: RBF
- Data Scaling: StandardScaler
- Train/Test Split: applied before training

## Steps
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Modeling

## Project Structure
- Opening data and printing basic information. (head(), value_counts(), info(), describe())
- Building new columns. (PL_SL, PW_SW)
- Data reconstruction based on species. (setosa, versicolor, virginica)
- Draw a correlation chart for each column based on species. (Histogram chart)
- Draw a correlation diagram for petals and sepals based on length and width. (Scatter chart)
- Column correlation chart. (Heatmap chart)
- Draw a box diagram. (Box Chart)
- Percentage of each sample in the data. (Pie chart)
- Drawing a diagram of the pairwise relationships of the characteristics of the Iris Flower. (Pair chart)
- Drawing violin and swarm diagrams by species. (Violin chart & Swarm chart)
- Chart of minimum and maximum sizes of any species. (Min-Max chart)
- Data standardization.
- Data partitioning with the train-test-split.
- Building and testing the model.

## Results
- The correlation of the setosa species is greater.
- The sepals of the setosa species are larger in size.
- The sepals and petals of the species versicolor virginica are approximately equal in size.
- The petals of the species versicolor virginica are larger in size.
- The correlation of the columns (petal_length, petal_width, PL_SL, PW_SW) is higher than the columns (sepal_length, sepal_width)
- The number of species in the data is equal.
- We use the SVC model.
- The model's accuracy in identifying Iris species based on the size of petals and sepals is 100%.