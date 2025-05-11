## How to Run the Application

1. Navigate to the directory where the file is stored:
   ```bash
   cd "<folder_path>"
   ```

2. Execute the Streamlit application:
   ```bash
   streamlit run mental_health_ui.py
   ```

---

## Flow of the Code

1. **Data Cleaning & Preprocessing**  
   Ensures the datasets are consistent and usable for analysis.

2. **Handling Missing Values**  
   Addresses gaps in the data to enhance the accuracy of the model.

3. **Normalization**  
   Processes the text data to ensure it's ready for better analysis.

4. **Exploratory Data Analysis (EDA)**  
   Identifies key relationships between symptoms and mental health conditions.

5. **Feature Engineering**  
   Encodes symptoms and conditions into input features and labels for the model.

6. **Feature Selection**  
The goal is to identify the most impactful features contributing to the prediction. Since we are using comment embeddings as features, some additional features from the original dataset may not have a significant impact on model performance. However, after experimentation, we found that including all features improved the model's performance. An L1 penalty was applied in Logistic Regression for feature selection, while Random Forest automatically performs feature selection by default.

7. **Model Development (Binary Classification of a Balanced Dataset)**  
   Different models were tested, and the final model was selected based on performance metrics.

8. **Evaluation Metrics**  
   The following evaluation metrics are used to assess the model’s performance:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - Classification Report
   - Confusion Matrix

9. **Model Interpretation**  
   SHAP or LIME is used to interpret the model's predictions.
   #### **Understanding the SHAP Summary Plot**
   This SHAP summary plot provides insights into the impact of various features on the model's predictions.

   #### **1. X-Axis: SHAP Value (Impact on Model Output)**
   - The **SHAP value** represents how much each feature contributes to pushing the model’s prediction higher or lower.
   - **Negative SHAP values** indicate that the feature is lowering the prediction.
   - **Positive SHAP values** indicate that the feature is increasing the prediction.

   #### **2. Y-Axis: Features Sorted by Importance**
   - Features are ranked by their overall impact on the model.
   - The most influential feature (e.g., `work_interfere`) is at the top, while the least impactful feature (e.g., `phys_health_consequence`) is at the bottom.

   #### **3. Color: Feature Value**
   - **Red (High feature value)**: Indicates higher values of the feature.
   - **Blue (Low feature value)**: Indicates lower values of the feature.

   ---

## Model Details

- **Model Used**: Both Random Forest and logistic Regression have Excellent performance across all metrics.
  The Logistic Regression model is used here in streamlit for its less computational cost.

- **File Storage**:  
  The pickle files for the model, label encoder, and ordinal encoder are stored and can be accessed.

- **Model Switching**:  
  To use a different model, simply change the file path to the new pickle file as needed.

## Dataset 
[Kaggle dataset](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey/data)