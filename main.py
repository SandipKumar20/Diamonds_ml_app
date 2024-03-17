import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Machine learning App")
st.image("st_logo.png", width=500)

st.title("Case study on Diamonds Dataset")

data = sns.load_dataset("diamonds")
st.write("Dimensions of the dataset:", data.shape)
menu = st.sidebar.radio("Menu", ["Home", "Prediction Price"])

match menu:
    case "Home":
        st.image("diamond.png", width=500)
        st.header("Tabular Data of a diamond")
        if st.checkbox("Tabular Data"):
            st.table(data.head(50))
        st.header("Statistical summary of the data")
        if st.checkbox("Statistics"):
            st.table(data.describe())
        st.header("Correlation graph of the data")
        if st.checkbox("Correlation graph"):
            fig, ax = plt.subplots(figsize=(5, 2.5))
            sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="crest")
            st.pyplot(fig)
        st.title("Graphs")
        graph = st.selectbox("Types of graphs", ["Scatter Plot", "Bar Graph", "Histogram"])
        if graph == "Scatter Plot":
            st.header("Scatter Plot carat vs price")
            value = st.slider("Filter data using carat", 0, 6)
            data = data.loc[data["carat"] >= value]
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=data,x="carat",y="price", hue="cut")
            st.pyplot(fig)
        if graph == "Bar Graph":
            st.header("Bar Graph of cut")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="cut", y=data.cut.index, data=data)
            st.pyplot(fig)
        if graph == "Histogram":
            st.header("Bar Graph of price")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.distplot(data.price, kde=True)
            st.pyplot(fig)
    case "Prediction Price":
        st.title("Prediction price of the diamond")
        st.header("Input values for a diamond")
        carat = st.slider("carat", min_value=0.2, max_value=5.01, step=0.01)
        cut = st.slider("cut", min_value=0, max_value=4, step=1)
        color = st.slider("color", min_value=0, max_value=6, step=1)
        clarity = st.slider("clarity", min_value=0, max_value=7, step=1)
        depth = st.slider("depth", min_value=43, max_value=79, step=1)
        table = st.slider("table", min_value=43, max_value=95, step=1)
        x = st.slider("x", min_value=0.0, max_value=10.74, step=0.01)
        y = st.slider("y", min_value=0.0, max_value=58.9, step=0.01)
        z = st.slider("z", min_value=0.0, max_value=31.8, step=0.01)

        inputs = [[carat, cut, color, clarity, depth, table, x, y, z]]
        ml_algorithm = st.selectbox("ML algorithms", ["Linear Regression", "Random Forest", "MLP"])
        predict = st.button("Predict Price($)")
        if ml_algorithm == "Linear Regression" and predict:
            with open('model_lr.pkl', 'rb') as f:
                lr_model = pickle.load(f)
            prediction = lr_model.predict(inputs)
            prediction_ = np.round(prediction, 2)
            st.write(f"Price of the diamond: {prediction_[0]}")
        if ml_algorithm == "Random Forest" and predict:
            with open("model_rf.pkl", "rb") as f:
                rf_model = pickle.load(f)
            prediction = rf_model.predict(inputs)
            prediction_ = np.round(prediction, 2)
            st.write(f"Price of the diamond: {prediction_[0]}")
        if ml_algorithm == "MLP" and predict:
            with open("model_mlp.pkl", "rb") as f:
                mlp_model = pickle.load(f)
            prediction = mlp_model.predict(inputs)
            prediction_ = np.round(prediction, 2)
            st.write(f"Price of the diamond: {prediction_[0]}")


