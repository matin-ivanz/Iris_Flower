import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

os.chdir('D:\python\Iris_Flower')
data = pd.read_csv('Iris_data.csv')

print(data.head())

for col in data.columns:
    print('\n', data[col].value_counts())

print(data.info())

print(data.describe())

# PL_SL = Petal Length to Sepal Length Ratio
data['PL_SL'] = round(data['petal_length'] / data['sepal_length'] * 100).astype('int64')

# PL_SL = Petal Width to Sepal Width Ratio
data['PW_SW'] = round(data['petal_width'] / data['sepal_width'] * 100).astype('int64')

data_setosa = data[data['species'] == 'Iris-setosa']
data_versicolor = data[data['species'] == 'Iris-versicolor']
data_virginica = data[data['species'] == 'Iris-virginica']

# choose a color for each category
flower_colors = {'data_setosa': 'blue', 'data_versicolor': 'red', 'data_virginica': 'green'}

a1 = 1
plt.figure(figsize=(10, 32))

for col in data.select_dtypes('float64').columns:
    plt.subplot(4, 1, a1)
    plt.title(f"{col} data distribution chart")
    plt.xlabel(f"{col} (cm)")
    plt.ylabel("probability density (1/cm)")

    sns.histplot(data=data, x=col, hue='species', palette=flower_colors.values(), bins=30,
                 element='poly', kde=True, stat='density')
    a1 += 1

plt.show()

a1 = 1
plt.figure(figsize=(10, 16))

for col in data.select_dtypes('int64').columns:
    plt.subplot(2, 1, a1)
    plt.title(f"{col} data distribution chart")
    plt.xlabel(f"{col} (ratio)")
    plt.ylabel("probability density")

    sns.histplot(data=data, x=col, hue='species', palette=flower_colors.values(), bins=30,
                 element='poly', kde=True, stat='density')
    a1 += 1

plt.show()

a1 = 1
quantity = [['sepal_length', 'sepal_width'], ['petal_length', 'petal_width']]
plt.figure(figsize=(10, 14))
mean_data = pd.concat([data_setosa.describe(), data_versicolor.describe(),
                       data_virginica.describe()], axis=1).T['mean']

for i, j in quantity:
    plt.subplot(2, 1, a1)
    plt.title(f"Correlation of {i} to {j}")
    plt.xlabel(i)
    plt.ylabel(j)

    sns.scatterplot(data=data, x=i, y=j, hue='species', palette=flower_colors.values())

    # triangles indicate average sizes.
    plt.scatter(x=mean_data[i], y=mean_data[j], marker='^', color=flower_colors.values(), s=100,
                edgecolors='black')
    plt.grid()
    a1 += 1
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(data.drop('species', axis=1).corr(), cmap='coolwarm', annot=True,
            linewidths=0.5, linecolor='black', square=True, vmin=-1, vmax=1, center=0)
plt.title("Correlation Heatmap of Iris Features")
plt.show()

a1 = 1
plt.figure(figsize=(10, 32))

for col in data.select_dtypes('float64').columns:
    plt.subplot(4, 1, a1)
    plt.title(f"Box Plot of {col} by Species")
    plt.ylabel(f"{col} (cm)")

    sns.boxplot(data=data, x='species', y=col, palette=flower_colors.values(),
                width=0.4)
    plt.grid(axis='y')
    a1 += 1

plt.show()

plt.figure(figsize=(7, 7))
plt.pie(data['species'].value_counts(), labels=flower_colors.keys(), autopct='%1.2f%%', colors=flower_colors.values(),
        startangle=90)
plt.title("Percentage chart of the number of Iris")
plt.show()

sns.pairplot(data=data, hue='species', kind='scatter', palette=flower_colors.values())
plt.show()

a1 = 1
plt.figure(figsize=(10, 32))

for col in data.select_dtypes('float64').columns:
    plt.subplot(4, 1, a1)
    plt.title(f"Violin Plot of {col} by Species")
    plt.ylabel(f"{col} (cm)")

    sns.violinplot(data=data, x='species', y=col, palette=flower_colors.values(),
                   inner='box', bw_adjust=1.5, )
    plt.grid(axis='y')
    a1 += 1

plt.show()

a1 = 1
plt.figure(figsize=(10, 32))

for col in data.select_dtypes('float64').columns:
    plt.subplot(4, 1, a1)
    plt.title(f"Swarm Plot of {col} by Species")
    plt.ylabel(f"{col} (cm)")

    sns.swarmplot(data=data, x='species', y=col, palette=flower_colors.values(),
                  size=10)
    plt.grid(axis='y')
    a1 += 1

plt.show()


def min_max_chart(quantity):
    # data classification
    min_data = quantity.select_dtypes('float64').describe().T['min']
    max_data = quantity.select_dtypes('float64').describe().T['max']

    #definition of variables
    thickness = 0.3
    number_columns = np.arange(len(min_data.index))
    gride_min_max_chart = ["Min", "Max"]

    #draw a diagram
    plt.figure(figsize=(12, 9))
    plt.bar(number_columns - (thickness/2), height=min_data, width=thickness, color='skyblue')
    plt.bar(number_columns + (thickness/2), height=max_data, width=thickness, color='green')

    for i, j in enumerate(min_data):
        plt.text(i - (thickness/2), j, str(f"{j}cm"), ha='center', va='bottom', fontsize=10, color='black')
    for i, j in enumerate(max_data):
        plt.text(i + (thickness/2), j, str(f"{j}cm"), ha='center', va='bottom', fontsize=10, color='black')

    plt.title(f"Chart of {quantity['species'].values[0]}")
    plt.xlabel("Flower components")
    plt.ylabel("ُSize (cm)")
    plt.xticks(number_columns, min_data.index)
    plt.legend(gride_min_max_chart)
    plt.grid(axis='y')

    plt.show()


min_max_chart(data_setosa)
min_max_chart(data_versicolor)
min_max_chart(data_virginica)

x = data.drop('species', axis=1)
y = data['species']

# data conversion
y = y.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

scaler = StandardScaler()
scaled = scaler.fit_transform(x)
scaled_x = pd.DataFrame(scaled, columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2,
                                                    random_state=42)

model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model_score = accuracy_score(y_test, y_pred)

print(f"Accuracy of the model: {int(model_score * 100)}%")
