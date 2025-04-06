import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare the dataset
df = pd.read_csv("train.csv")
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df = df[df['GrLivArea'] < 4500]
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

# Split and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 🌟 STREAMLIT APP STARTS
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 Advanced House Price Predictor")
st.markdown("##### Predict the sale price of a house using real housing data. Enter the details in the sidebar!")

# 🔧 Sidebar Inputs
st.sidebar.header("🔧 House Features Input")

gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 4000, step=50, value=1800)
garage_cars = st.sidebar.slider("Garage Size (number of cars)", 0, 5, 2)
total_bsmt = st.sidebar.number_input("Basement Area (sq ft)", 0, 3000, step=50, value=900)
full_bath = st.sidebar.slider("Number of Full Bathrooms", 0, 4, 2)
year_built = st.sidebar.slider("Year Built", 1900, 2023, 2005)
overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 7)

# Prepare single row for prediction
input_data = X.iloc[0:1].copy()
input_data['GrLivArea'] = gr_liv_area
input_data['GarageCars'] = garage_cars
input_data['TotalBsmtSF'] = total_bsmt
input_data['FullBath'] = full_bath
input_data['YearBuilt'] = year_built
input_data['OverallQual'] = overall_qual

# Prediction
predicted_price = model.predict(input_data)[0]

st.subheader("💸 Predicted Sale Price")
st.success(f"🏠 Estimated Price: **${predicted_price:,.2f}**")

# Show model RMSE
st.info(f"📉 Model RMSE (Root Mean Squared Error): **${rmse:,.2f}**")

# 📊 Optional Comparison Chart
st.subheader("📊 Price Comparison with Dataset Average")

avg_price = y.mean()
comparison_df = pd.DataFrame({
    'Type': ['Predicted Price', 'Average Price'],
    'Price': [predicted_price, avg_price]
})

fig, ax = plt.subplots()
sns.barplot(x='Type', y='Price', data=comparison_df, palette='viridis')
ax.set_ylabel("Price in USD")
st.pyplot(fig)
