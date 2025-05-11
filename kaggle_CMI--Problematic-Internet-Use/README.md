# Problematic Internet Use Prediction: Kaggle Competition

This project was undertaken as part of a Kaggle competition hosted by the Child Mind Institute, focusing on predicting problematic internet use. The dataset primarily involved tabular data, making it a suitable candidate for advanced tree-based models. Below is a detailed breakdown of the approach and methodologies used during the competition.

## **Objective**
The goal of the competition was to build a predictive model to identify patterns and predict problematic internet usage, leveraging a dataset provided by the Child Mind Institute.

---

## **Approach**

### **1. Exploratory Data Analysis (EDA)**
To gain an initial understanding of the dataset, extensive EDA was performed:
- **Data Visualization**: Charts, histograms, and pair plots were used to identify patterns, relationships, and potential correlations in the data.
- **Statistical Insights**: Basic descriptive statistics were computed, and the distribution of features was analyzed to detect skewness, outliers, and potential missing values.
- **Feature Importance**: Correlation matrices and feature importance scores (based on tree-based algorithms) were analyzed to identify key predictors.

---

### **2. Data Preprocessing**
Various preprocessing techniques were applied to prepare the data for modeling:
- **Missing Value Treatment**: Missing values were handled using imputation techniques such as mean, median, or mode imputation based on feature type.
- **Feature Engineering**: New features were derived where necessary to better capture relationships in the data.
- **Categorical Encoding**: Categorical variables were encoded using methods such as one-hot encoding or label encoding, depending on the nature of the variable and its cardinality.
- **Scaling and Normalization**: While not strictly necessary for tree-based models, feature scaling was applied in select cases to improve interpretability.

---

### **3. Model Selection**
Given the tabular nature of the data, the focus was placed exclusively on tree-based models, as these have demonstrated superior performance for this type of data. The following models were explored:
- **Random Forest**: A baseline model to assess feature importance and benchmark performance.
- **Gradient Boosting Machines (GBM)**: Used for its ability to handle large datasets with minimal preprocessing.
- **XGBoost**: Tuned extensively to improve accuracy and reduce overfitting, leveraging its parallel processing capabilities.
- **LightGBM**: Selected for its speed and efficiency with large datasets and its ability to handle categorical features natively.
- **CatBoost**: Leveraged for its native handling of categorical data and minimal need for extensive preprocessing.
- **TabNet**: A cutting-edge neural network architecture for tabular data. TabNet utilizes attention mechanisms and sequential decision-making to achieve high performance without requiring extensive preprocessing or feature engineering. It was tuned for hyperparameters like the learning rate, number of decision steps, and batch size to maximize accuracy.  
---

### **4. Model Tuning**
Each model was fine-tuned using hyperparameter optimization techniques:
- **Grid Search**: A comprehensive approach to test a wide range of hyperparameter combinations.
- **Random Search**: Applied to quickly identify promising parameter combinations.
- **Bayesian Optimization**: Used in some cases to fine-tune models efficiently based on prior results.
TabNet-specific tuning involved optimizing decision steps, feature dimensions, and sparsity regularization to balance performance and interpretability.  
---

### **5. Evaluation**
The models were evaluated using appropriate metrics to ensure robustness:
- **Metrics Used**: Metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² were used to evaluate model performance.
- **Cross-Validation**: K-Fold cross-validation was employed to ensure model generalizability and avoid overfitting.
- **Feature Importance Analysis**: Post-modeling, feature importance scores were analyzed to draw meaningful insights.

---

### **6. Results**
- Tree-based models, especially **XGBoost** and **LightGBM**, consistently outperformed other models due to their ability to handle the dataset's complexities.
- The final model achieved competitive results on the leaderboard, demonstrating the effectiveness of tree-based approaches for tabular data.

---

## **Conclusion**
This competition provided valuable insights into handling tabular datasets and highlighted the power of tree-based models in predictive tasks. By combining robust preprocessing with advanced modeling techniques, significant progress was made toward understanding and predicting problematic internet use.

---

## **Technologies and Tools**
- **Programming Language**: Python
- **Libraries Used**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM, CatBoost
- **Environment**: Kaggle Notebooks

---
