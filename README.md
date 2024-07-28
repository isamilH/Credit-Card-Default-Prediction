# Credit-Card-Default-Prediction
Project Description: Credit Card Default Prediction Using Machine Learning

Objective:
The primary goal of this project is to develop a machine learning model that predicts the likelihood of a credit card client defaulting on their next month's payment. By accurately predicting defaults, financial institutions can manage risks more effectively, optimize credit allocation, and design targeted interventions to mitigate potential losses.

Dataset:
The dataset used in this project is sourced from the UCI Machine Learning Repository and contains information on credit card clients in Taiwan. The dataset includes features such as demographic information, credit card attributes, and payment history over several months. The target variable is a binary indicator of whether a client defaulted on their next payment.

Key Features:

    LIMIT_BAL: Credit limit of the client.
    SEX: Gender of the client.
    EDUCATION: Education level of the client.
    MARRIAGE: Marital status of the client.
    AGE: Age of the client.
    PAY_0 to PAY_6: Repayment status from April to September.
    BILL_AMT1 to BILL_AMT6: Bill statements from April to September.
    PAY_AMT1 to PAY_AMT6: Payment amounts from April to September.
    default.payment.next.month: Target variable indicating if the client defaulted (1) or not (0).

Exploratory Data Analysis (EDA):

    Distribution plots were created for key features such as credit limit, age, gender, education, and marital status.
    Correlation heatmaps were used to identify relationships between features.
    Box plots highlighted the distribution and outliers for continuous features.

Feature Engineering:

    No significant feature engineering was required as the dataset already contained well-defined features relevant to credit default prediction.

Model Building:

    Various machine learning models were trained and evaluated, including Logistic Regression, Decision Trees, and Random Forests.
    Hyperparameter tuning was performed using grid search to identify the best model configuration.

Model Evaluation:

    The final model was evaluated using classification metrics such as precision, recall, F1-score, and accuracy.
    ROC-AUC score was used to assess the model's ability to distinguish between defaulters and non-defaulters.
    Gain and ROC curves provided visual insights into the model's performance.

Final Model and Results:

    The best model was a Random Forest classifier with hyperparameters: max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300.
    The final model achieved an accuracy of 0.82 and an ROC-AUC score of 0.76.
    The model showed good performance in predicting non-defaults but had lower recall for defaults, indicating room for improvement in identifying defaulters.

SHAP Analysis:

    SHAP values were used to interpret the model's predictions and identify the most influential features.
    Key features influencing the prediction included repayment status (PAY_0), credit limit (LIMIT_BAL), and recent payment amounts (PAY_AMT1).

Conclusion:
The project successfully developed a machine learning model to predict credit card defaults with good overall performance. The model can help financial institutions manage credit risk by identifying high-risk clients. However, further improvements are needed to enhance the model's recall for defaulters. Future work could explore additional features, alternative models, and advanced techniques for handling class imbalance.
