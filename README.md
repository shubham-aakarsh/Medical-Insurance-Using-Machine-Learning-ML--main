
# **Medical Insurance Price Prediction Using Machine Learning**

![Insurance Prediction](https://img.shields.io/badge/Machine%20Learning-XGBoost-blue.svg)  
Predicting medical insurance costs based on personal attributes using machine learning.

---

## **Table of Contents**
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Project Pipeline](#project-pipeline)  
4. [Model Used](#model-used)  
5. [Installation and Setup](#installation-and-setup)  
6. [Usage](#usage)  
7. [Results](#results)  
8. [Technologies Used](#technologies-used)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## **Project Overview**
This project aims to predict the medical insurance costs of individuals based on features such as age, BMI, smoker status, and the number of dependents. Using **machine learning**, we build a predictive model to provide insights into how various factors influence insurance premiums. The project demonstrates the full ML pipeline, including data preprocessing, model training, evaluation, and predictions on new data.

---

## **Dataset**
The dataset used in this project is from **Kaggle** and includes the following features:
- **Age**: Age of the individual.
- **BMI**: Body Mass Index (a measure of body fat).
- **Smoker**: Whether the individual is a smoker (yes/no).
- **Children**: Number of dependents.
- **Region**: Region of residence in the US.
- **Charges**: Medical insurance costs (target variable).

No missing values are present in the dataset, and it is well-suited for regression tasks.

---

## **Project Pipeline**
1. **Data Loading and Cleaning**:  
   - Load the dataset using Pandas.
   - Convert categorical variables (e.g., smoker) into numerical values using binary encoding.
   - Drop unnecessary columns (`sex` and `region`).

2. **Exploratory Data Analysis (EDA)**:  
   - Analyze the relationship between features and the target variable.
   - Confirm no missing values are present.

3. **Data Splitting**:  
   - Split the dataset into 80% training and 20% testing sets using `train_test_split()`.

4. **Model Selection and Training**:  
   - Use **XGBoost Regressor** to train the model with hyperparameters like `n_estimators` and `max_depth`.
   - Train the model on the training data.

5. **Model Evaluation**:  
   - Use **R² score** and **cross-validation** to evaluate model performance.

6. **Saving the Model**:  
   - Save the trained model using `pickle` for future use.

7. **Prediction on New Data**:  
   - Create new input data, apply preprocessing, and predict insurance costs using the trained model.

---

## **Model Used**
We selected the **XGBoost Regressor** for this project due to its efficiency and ability to handle non-linear data patterns. It also offers **regularization** to prevent overfitting, making it an ideal choice for structured data.

---

## **Installation and Setup**
Follow these steps to set up the project locally:

1. **Clone the repository:**
   ```bash
   https://github.com/Amit-priyu/Medical-Insurance-Using-Machine-Learning-ML-.git
   ```

2. **Install the required libraries:**
   ```bash
   pip install pandas scikit-learn xgboost pickle-mixin
   ```

3. **Run the notebook:**
   Open the provided Jupyter notebook and follow the code cells step-by-step to reproduce the results.

---

## **Usage**
To use the trained model for predictions:

1. Import the saved model using `pickle`:
   ```python
   import pickle
   model = pickle.load(open('insurancemodelf.pkl', 'rb'))
   ```

2. Create a new data point:
   ```python
   import pandas as pd
   new_data = pd.DataFrame({
       'age': 19,
       'sex': 'male',
       'bmi': 27.9,
       'children': 0,
       'smoker': 'yes',
       'region': 'northeast'
   }, index=[0])

   # Preprocessing
   new_data['smoker'] = new_data['smoker'].map({'yes': 1, 'no': 0})
   new_data = new_data.drop(['sex', 'region'], axis=1)

   # Make a prediction
   prediction = model.predict(new_data)
   print(f'Predicted Insurance Cost: ${prediction[0]:.2f}')
   ```

---

## **Results**
- **Training R² Score:** 0.869  
- **Test R² Score:** 0.901  
- **Cross-Validation Score:** 0.861  

These results indicate that the model performs well, with no signs of overfitting. The prediction for the sample input gave an estimated insurance cost of **$18,035**.

---

## **Technologies Used**
- **Python**: Programming language for data manipulation and model building.
- **Pandas**: Data manipulation and analysis.
- **XGBoost**: Machine learning model for regression.
- **Scikit-learn**: Tools for model evaluation and data splitting.
- **Pickle**: Save and load models for future use.

---

## **Contributing**
Contributions are welcome! If you find a bug or want to improve this project, feel free to submit a **pull request**.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## **License**
This project is licensed under the **Amit**. You are free to use, modify, and distribute this project as per the terms of the license.

---

## **Acknowledgements**
Special thanks to **Kaggle** for the dataset and to the open-source community for providing the libraries used in this project.
