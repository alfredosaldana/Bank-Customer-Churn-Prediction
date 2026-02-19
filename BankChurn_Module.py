"""
BankChurn_Module.py
====================
Production-ready module for Bank Customer Churn prediction.

This module encapsulates all preprocessing and prediction logic,
ensuring consistency between the training pipeline and inference pipeline.

Classes
-------
CustomScaler  : A sklearn-compatible scaler that scales only selected columns.
CustomerChurn : Loads a saved model + scaler, preprocesses new data, and predicts churn.

Usage Example
-------------
    from BankChurn_Module import CustomerChurn

    churn_model = CustomerChurn(
        model_file='model_file.pkl',
        scaler_file='Scaler_file.pkl'
    )
    results = churn_model.load_and_clean_data('new_customers.csv')
    predictions = churn_model.predict_churn()
    print(predictions[['CustomerId', 'Surname', 'Predicted_Exited']].head())
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# CustomScaler
# =============================================================================

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom sklearn-compatible scaler that applies StandardScaler
    only to a specified subset of columns, leaving the rest unchanged.

    This is essential when a DataFrame contains both numerical features
    (that need standardisation) and binary/categorical features
    (that should NOT be scaled).

    Parameters
    ----------
    columns : list of str
        Column names to scale.
    copy, with_mean, with_std : bool
        Passed directly to sklearn's StandardScaler.

    Attributes
    ----------
    mean_ : float  – mean of the scaled columns (informational)
    std_  : float  – std  of the scaled columns (informational)

    Examples
    --------
    >>> scaler = CustomScaler(columns=['Age', 'Balance', 'CreditScore'])
    >>> scaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    """

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler    = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns   = columns
        self.with_mean = with_mean
        self.with_std  = with_std
        self.copy      = copy
        self.mean_     = None
        self.std_      = None

    def fit(self, X, y=None):
        """Learn mean and std from the training data (only for selected columns)."""
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.std_  = np.std(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        """
        Apply standardisation to the selected columns.
        The original column order is preserved in the output.
        """
        init_col_order = X.columns
        X_scaled    = pd.DataFrame(
            self.scaler.transform(X[self.columns]),
            columns=self.columns,
            index=X.index          # preserve index so concat aligns correctly
        )
        X_notscaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_notscaled, X_scaled], axis=1)[init_col_order]


# =============================================================================
# CustomerChurn
# =============================================================================

class CustomerChurn:
    """
    End-to-end inference pipeline for the Bank Customer Churn model.

    Loads a pre-trained Random Forest model and a fitted CustomScaler
    (both saved as .pkl files), preprocesses raw customer data exactly
    as done during training, and returns predictions appended to the
    original dataframe for easy interpretation.

    Parameters
    ----------
    model_file  : str – path to the saved model  (.pkl)
    scaler_file : str – path to the saved scaler (.pkl)

    Methods
    -------
    load_and_clean_data(data_file)
        Reads a CSV, drops irrelevant columns, scales numerical features,
        and one-hot encodes categorical features.
    predict_churn()
        Runs the loaded model on the preprocessed data and returns
        the original DataFrame augmented with a 'Predicted_Exited' column.
    """

    # Columns expected by the trained model (in this exact order)
    FEATURE_COLUMNS = [
        'HasCrCard', 'IsActiveMember', 'CreditScore', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score',
        'Point Earned', 'Geography_Germany', 'Geography_Spain', 'Gender_Male',
        'Card Type_GOLD', 'Card Type_PLATINUM', 'Card Type_SILVER'
    ]

    # Numerical columns that were standardised during training
    NUMERICAL_COLS = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'EstimatedSalary', 'Satisfaction Score', 'Point Earned'
    ]

    # Columns to drop before modelling (identifiers + target leakage)
    COLS_TO_DROP = ['RowNumber', 'CustomerId', 'Surname', 'Complain']

    # Categorical columns to one-hot encode
    CATEGORICAL_COLS = ['Geography', 'Gender', 'Card Type']

    def __init__(self, model_file, scaler_file):
        # ---------- load model ----------
        with open(model_file, 'rb') as f:
            self.model_selected = pickle.load(f)

        # ---------- load scaler ----------
        with open(scaler_file, 'rb') as f:
            self.scaler_selected = pickle.load(f)

        self.data        = None   # will hold the preprocessed feature matrix
        self.df_original = None   # will hold the raw data (for readable output)

    # ------------------------------------------------------------------
    def load_and_clean_data(self, data_file):
        """
        Load raw customer data from a CSV and preprocess it for inference.

        Steps performed (mirror the training pipeline exactly):
            1. Read CSV.
            2. Keep a copy of the original data for human-readable output.
            3. Drop identifier and leakage columns.
            4. Scale numerical features with the pre-trained CustomScaler.
            5. One-hot encode categorical features (drop_first=True to avoid
               the dummy-variable trap / multicollinearity).
            6. Reindex to the exact column order expected by the model.

        Parameters
        ----------
        data_file : str – path to the CSV file with new customer records.

        Returns
        -------
        pd.DataFrame – preprocessed feature matrix (shape: n_customers × 16).
        """
        df = pd.read_csv(data_file, delimiter=',')

        # Keep original for attaching predictions later
        self.df_original = df.copy()

        # --- Step 1: drop non-modelling columns ---
        df = df.drop(columns=self.COLS_TO_DROP, errors='ignore')

        # --- Step 2: scale numerical features (using pre-trained scaler) ---
        # IMPORTANT: we use transform(), NOT fit_transform()
        # Using fit_transform() here would re-learn mean/std from the *new* data,
        # producing different scaling than what the model was trained on.
        df[self.NUMERICAL_COLS] = self.scaler_selected.transform(df[self.NUMERICAL_COLS])

        # --- Step 3: one-hot encode categorical features ---
        # drop_first=True removes one category per feature to avoid the dummy trap
        #   Geography : France (dropped), Germany, Spain
        #   Gender    : Female (dropped), Male
        #   Card Type : DIAMOND (dropped), GOLD, PLATINUM, SILVER
        df = pd.get_dummies(df, columns=self.CATEGORICAL_COLS, drop_first=True, dtype='int')

        # --- Step 4: align column order with what the model expects ---
        # pd.get_dummies may produce columns in a different order, or may be
        # missing columns if a category is absent in this batch of data.
        df = df.reindex(columns=self.FEATURE_COLUMNS, fill_value=0)

        self.data = df.copy()
        return self.data

    # ------------------------------------------------------------------
    def predict_churn(self):
        """
        Generate churn predictions and attach them to the original data.

        Returns
        -------
        pd.DataFrame – original customer data with an added column:
            'Predicted_Exited'  :  1 = predicted to churn,  0 = predicted to stay.

        Raises
        ------
        ValueError – if load_and_clean_data() has not been called first.
        """
        if self.data is None:
            raise ValueError(
                "No data loaded. Call load_and_clean_data(data_file) first."
            )

        predictions = self.model_selected.predict(self.data)

        # Convert to a Series with reset index to align with df_original
        predictions_series = pd.Series(predictions, name='Predicted_Exited').reset_index(drop=True)
        output = self.df_original.reset_index(drop=True).copy()
        output['Predicted_Exited'] = predictions_series

        return output
