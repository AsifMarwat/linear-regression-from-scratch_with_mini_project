# 📈 Simple Linear Regression From Scratch — No Scikit-Learn

This repo is a lightweight machine learning project where everything is built from scratch using only NumPy — no `scikit-learn` or other ML libraries were used.

It includes:

- A custom linear regression class with training and prediction logic.
- A custom `train_test_split()` function.
- A mini project (Ice Cream Sales Predictor) with a FastAPI web interface to interact with the model in real time.

---

## 🗂️ Project Structure

├── my_linear_reg.py # Custom Linear Regression class (fit, predict, metrics)
├── data_split.py # Custom train-test split function
├── ice_cream_sales/
│ ├── app.py # FastAPI app for prediction demo
│ ├── ice_cream_sales_analysis.ypynb jupyter nb for this mini porject
│ ├── icecream_model.pkl # Saved trained model
│ ├── temperature_icecream_data.csv # Dataset for demo
│ └── templates/
│ └── form.html # Jinja2 template for user input
├── requirements.txt # Dependencies (no scikit-learn)
└── README.md # You are here

---

## 🔧 Custom ML Implementation

### `my_linear_reg.py`

A handcrafted linear regression class `myLinearReg` implemented **from scratch using NumPy**.

#### Methods:

- **`.fit(X_train, y_train)`**  
  Trains the model using the **Ordinary Least Squares (OLS)** method to compute the slope (`m`) and intercept (`b`).

- **`.predict(X_test)`**  
  Returns predicted values using the learned line equation `y = mx + b`.

- **`.metrics(y_test, y_pred)`**  
  Calculates and returns key regression evaluation metrics:
  - **MSE** – Mean Squared Error
  - **MAE** – Mean Absolute Error
  - **R² Score** – Coefficient of Determination

### `data_split.py`

Custom implementation of `train_test_split()` from scratch using NumPy.

#### Features:

- Inputs: `X`, `y`, `test_size` (e.g. 0.2), and optional `random_state`
- Validates input shapes and `test_size` range
- Uses NumPy to:
  - Generate and shuffle indices
  - Split data into training and testing sets based on the test size

Returns:

- `X_train`, `X_test`, `y_train`, `y_test` – as NumPy arrays

## 🍦 Ice Cream Sales Predictor (FastAPI Demo)

The `ice_cream_sales/` folder contains a minimal FastAPI web app that:

- Loads your trained model (`icecream_model.pkl`)
- Accepts user input for temperature via a web form
- Predicts the number of ice creams likely to be sold

### 🔧 How It Works

- Uses `Jinja2Templates` for rendering HTML
- Accepts input from a form (`temp` as float)
- Passes the input to the `myLinearReg` model for prediction
- Displays the prediction on the same page

### 🌐 Run Locally

````bash
cd ice_cream_sales
uvicorn app:app --reload


---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/asifmarwat/linear-regression-from-scratch_with_mini_project.git
cd  linear-regression-from-scratch_with_mini_project

## 2. Install dependencies
For normal usage:
pip install -r requirements.txt

For development (includes Jupyter, dotenv, etc.):
pip install -r dev-requirements.txt

## 3. Run the FastAPI web app
cd ice_cream_sales
uvicorn app:app --reload

📌 Requirements
fastapi
uvicorn[standard]
jinja2
numpy

📚 Dataset
temperature_icecream_data.csv: Contains temperature vs. ice cream sales data

💡 Motivation
The goal of this project is to demystify how linear regression works under the hood by:

Avoiding ML libraries (SkLearn)

Implementing math directly in Python/NumPy

Visualizing and deploying a real-world use case

🧠 Author
Built  by Asif Marwat

````
