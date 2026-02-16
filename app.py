import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #4a4a4a;
    }
</style>
""", unsafe_allow_html=True)

# Load the iris dataset
@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    data['species_name'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return iris, data

iris, data = load_data()

# Sidebar for inputs
st.sidebar.title("🌸 Input Parameters")
st.sidebar.markdown("Adjust the values below to predict the iris species.")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', float(data['sepal length (cm)'].min()), float(data['sepal length (cm)'].max()), float(data['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal Width (cm)', float(data['sepal width (cm)'].min()), float(data['sepal width (cm)'].max()), float(data['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal Length (cm)', float(data['petal length (cm)'].min()), float(data['petal length (cm)'].max()), float(data['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal Width (cm)', float(data['petal width (cm)'].min()), float(data['petal width (cm)'].max()), float(data['petal width (cm)'].mean()))
    return [sepal_length, sepal_width, petal_length, petal_width]

input_features = user_input_features()

# Main Page
st.title('🌸 Iris Flower Prediction App')
st.markdown("This app predicts the **Iris flower species** based on sepal and petal measurements. It also visualizes the dataset distribution.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Input")
    input_df = pd.DataFrame([input_features], columns=iris.feature_names)
    st.write(input_df)

    # Train model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(iris.data, iris.target)
    
    # Prediction
    prediction = knn.predict([input_features])
    prediction_proba = knn.predict_proba([input_features])
    species = iris.target_names[prediction][0]

    st.subheader("Prediction")
    st.success(f"**{species.capitalize()}**")
    
    st.subheader("Prediction Probability")
    st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

with col2:
    st.subheader("Dataset Visualization")
    st.markdown("Pairplot of the Iris dataset. (Note: Your input is not plotted here for simplicity, but represents a point in this space.)")
    
    # Visualization
    fig = sns.pairplot(data, hue='species_name', palette='Set2')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Scikit-learn, and Seaborn.")
