import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cell_1 = nbf.v4.new_markdown_cell("# Step 1: Load Data")
code_cell_1 = nbf.v4.new_code_cell("""import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows
print(data.head())

# Statistic Summary
data.describe()""")

text_cell_2 = nbf.v4.new_markdown_cell("# Step 2: Visualize the Data")
code_cell_2 = nbf.v4.new_code_cell("""import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize the relationships
sns.pairplot(data, hue='species')
plt.show()""")

text_cell_3 = nbf.v4.new_markdown_cell("# Step 3: Build a Machine Learning Model")
code_cell_3 = nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2, random_state=42)""")

code_cell_4 = nbf.v4.new_code_cell("""# k-NN model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Step 2: Train the model on the training data
knn.fit(X_train, y_train)

# Step 3: Use the trained model to make predictions on the test data
y_pred = knn.predict(X_test)

# Step 4: Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Step 5: Print the accuracy as a percentage
print(f'Accuracy: {accuracy * 100:.2f}%')""")

text_cell_4 = nbf.v4.new_markdown_cell("# Step 4: Create an Interactive App")
code_cell_5 = nbf.v4.new_code_cell("""%%writefile app.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()

# Train the k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris.data, iris.target)

# Streamlit app
st.title('Iris Flower Prediction')
st.write('Enter the measurements of the iris flower to predict its species.')

# User inputs
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button('Predict'):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(input_data)
    species = iris.target_names[prediction][0]
    st.write(f'The predicted species is: {species}')""")

nb['cells'] = [text_cell_1, code_cell_1, text_cell_2, code_cell_2, text_cell_3, code_cell_3, code_cell_4, text_cell_4, code_cell_5]

with open('iris_workflow.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
