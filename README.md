
# Loan Status Prediction

This project aims to predict loan statuses using machine learning techniques, specifically leveraging the LightGBM classifier. The model processes both categorical and numerical features from the dataset to deliver predictions.


## Installation

To run this project, ensure you have Python 3.x installed. You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn lightgbm
```

## Usage

1. Place your `train.csv` and `test.csv` files in the root directory of the project.
2. Run the script:

```bash
python main.py
```

This will generate a `submission.csv` file containing the predicted loan statuses.

## Data

- **train.csv**: Contains the training data with features and target labels.
- **test.csv**: Contains the test data for which predictions are made.

The following features are utilized in the model:

### Categorical Features

- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `cb_person_default_on_file`

### Numerical Features

- `person_age`
- `person_income`
- `person_emp_length`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_cred_hist_length`

## Modeling

The project employs the following techniques:

- **Preprocessing**: 
  - OneHotEncoding for categorical features.
  - StandardScaler for numerical features.

- **Modeling**: 
  - LightGBM Classifier.
  - RandomizedSearchCV for hyperparameter tuning.
  - Early stopping and cross-validation to prevent overfitting.

- **Feature Importance**: 
  - The model evaluates feature importance to optimize predictions using only the most significant features.

## Results

After running the model, the best hyperparameters and ROC-AUC scores will be displayed. The predictions for the test dataset are saved in `submission.csv`.


```
