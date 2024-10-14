# Insurance Boost Analysis

This project uses machine learning models, primarily boosting techniques like **XGBoost**, to predict whether a customer is likely to respond positively to an automobile insurance offer. The dataset is imbalanced, and model evaluation is done using the AUC-ROC metric to capture performance effectively.

## Project Overview

The goal is to predict which customers will purchase insurance based on their profile and past behaviors. Boosting models were selected for their ability to handle complex patterns and imbalanced data.

### Key Steps:
1. Data Loading & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Model Building & Hyperparameter Tuning
4. Model Evaluation using AUC-ROC
5. Feature Importance Analysis

## Dataset

The dataset is sourced from Kaggle's **Health Insurance Cross Sell Prediction** competition. The data includes features like age, gender, driving license status, vehicle age, premium amount, and customer response to past offers.

You can access the dataset at the following link:  
[Kaggle Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e7)

## Installation

### Prerequisites

- **Python 3.8+**
- Required Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `seaborn`
  - `matplotlib`
  - `jupyterlab`

### Clone the Repository

```bash
git clone https://github.com/yourusername/insurance-boost-analysis.git
cd insurance-boost-analysis
```
### Install Dependencies
You can install the required packages by running:

```bash
pip install -r requirements.txt
```
### Run the Jupyter Notebook
Launch the Jupyter notebook to view and run the analysis:

```bash
jupyter lab insurance-boost-analysis.ipynb
```

### Project Structure
insurance-boost-analysis.ipynb: Jupyter notebook containing all the analysis and model building steps.
README.md: This file.

### Exploratory Data Analysis (EDA)
During the EDA phase, we analyzed key features, including:

- **Age Distribution:** Customers range from 20 to 85 years old, with an average of 38.38 years.
- **Annual Premium:** Premium amounts vary greatly, ranging from ₹2,630 to ₹540,165.
- **Previously Insured:** Around 46.3% of customers were previously insured, impacting their response to new offers.
- **Response Imbalance:** Only about 12.3% of customers responded positively to the insurance offer, leading to an imbalanced dataset, which is handled carefully in modeling.

### Model Building & Evaluation
We used **XGBoost** as the primary model due to its strong performance with imbalanced data. Other techniques like feature selection and hyperparameter tuning were also applied.

### XGBoost Hyperparameters:
```bash
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.03,  # Learning rate
    'max_depth': 12,
    'subsample': 0.85,
    'colsample_bytree': 0.45,
    'alpha': 0.1,
    'gamma': 1e-6,
    'random_state': 42
}
```
### Results
The model achieved a validation **AUC score of approximately 0.866**, indicating strong performance in distinguishing between customers likely to respond positively to insurance offers.
### Key Insights:
**Vehicle Damage:** Customers who reported vehicle damage are more likely to respond positively to new insurance offers.
**Previously Insured:** Customers who are already insured tend to respond negatively to new offers.
### Conclusion
This project demonstrated the effectiveness of boosting models, specifically XGBoost, in handling imbalanced insurance datasets. The insights gained from this model could help insurance companies optimize their marketing efforts and target customers more efficiently.

### Future Work
- Further fine-tuning using Grid Search or Bayesian Optimization.
- Try alternative models like LightGBM or CatBoost.
- Incorporate more external features, such as customer income levels or geographical data, for better prediction accuracy.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.
