# 📊 Sales Prediction using Machine Learning

## Overview

This project demonstrates a complete Machine Learning workflow using the **Sample Superstore Dataset**. It includes data preprocessing, handling missing values, target encoding, feature selection, model training, and prediction using **Linear Regression**.

The project is developed using Python and Scikit-learn and is suitable for beginners learning data preprocessing and regression analysis.

---

## Features

* Load and analyze the Superstore dataset
* Handle missing values using Median Imputation
* Encode categorical features using Target Encoding
* Select the most important features using Mutual Information
* Split the dataset into training and testing sets
* Train a Linear Regression model
* Predict sales values
* Evaluate model performance using MAE and R² Score

---

## Dataset

**Dataset Name:** Sample Superstore Dataset

### Dataset Columns

* Ship Mode
* Segment
* Country
* City
* State
* Postal Code
* Region
* Category
* Sub-Category
* Sales
* Quantity
* Discount
* Profit

---

## Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Category Encoders

---

## Machine Learning Workflow

1. Load the dataset
2. Handle missing values using Median Imputation
3. Apply Target Encoding to categorical features
4. Select the best features using Mutual Information
5. Split the dataset into training and testing sets
6. Train a Linear Regression model
7. Predict sales values
8. Evaluate the model

---

## Project Structure

```
Sales-Prediction-ML/
│
├── SampleSuperstore.csv
├── ml_project2.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Sales-Prediction-ML.git
```

Move into the project directory

```bash
cd Sales-Prediction-ML
```

Install the required libraries

```bash
pip install pandas numpy scikit-learn category_encoders
```

---

## Run the Project

```bash
python ml_project2.py
```

---

## Sample Output

```
Loading Dataset...

Dataset Loaded Successfully
Rows : 9994
Columns : 13

Handling Missing Values...
Missing values after imputation : 0

Applying Target Encoding...

Selecting Best Features...
Selected Features : ['Profit', 'Quantity']

Training Model...

Prediction Results

Predicted Sales : 245.83
Actual Sales    : 261.96
Difference      : 16.13

Model Performance
Mean Absolute Error : 145.37
R² Score            : 0.62
```

---

## Future Improvements

* Implement Random Forest Regression
* Compare multiple regression algorithms
* Hyperparameter tuning
* Data visualization using Matplotlib and Seaborn
* Build a Streamlit web application
* Deploy the model using Flask or FastAPI

---

## Learning Outcomes

Through this project, the following machine learning concepts are demonstrated:

* Data preprocessing
* Missing value handling
* Target encoding
* Feature selection
* Train-Test Split
* Linear Regression
* Model evaluation

---

## Author

**Shivaprasad**

If you found this project useful, feel free to ⭐ star the repository and contribute with improvements.
