import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Class to remove outliers using upper_quantile and lower_quantile."""

    def __init__(self, variable, upper_quantile=1.0, lower_quantile=0.0):
        if not isinstance(variable, str):
            raise ValueError("Variable should be a string data type")
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.variable = variable

    def fit(self, X, y=None):
        """Learns the upper_quantile and lower quantile value of a variable."""
        self.upper_quantile = X[self.variable].quantile(self.upper_quantile)
        self.lower_quantile = X[self.variable].quantile(self.lower_quantile)
        return self

    def transform(self, X):
        """Cap the outliers with upper_quantile & lower_quantile values of a variable."""
        X = X.copy()
        X[self.variable] = np.where(
            X[self.variable] > self.upper_quantile,
            self.upper_quantile,
            np.where(
                X[self.variable] < self.lower_quantile,
                self.lower_quantile,
                X[self.variable],
            ),
        )
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Class to find the rare labels of feature and combine them into a category called Rare."""

    def __init__(self, variables, tolerance=0.05, rare=-1):
        self.tolerance = tolerance
        self.rare = rare
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """Learns the rare values based on the threshold of each variables passed in variables."""
        self.encoder_dict = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float64(len(X)))
            self.encoder_dict[var] = list(t[t > self.tolerance].index)
        return self

    def transform(self, X):
        """Removes the rare labels for each variables passed into the variables list."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict[feature]), X[feature], self.rare
            )
        return X


class CustomPowerTransformer(BaseEstimator, TransformerMixin):
    """Custom Power Transformer to class."""

    def __init__(self, variable, power=0.5):
        if not isinstance(power, (float, int)):
            raise ValueError("Power must be either Float or Int")
        self.variable = variable
        self.power = power

    def fit(self, X, y=None):
        """Return the object & does nothing."""

        return self

    def transform(self, X):
        """Transforms the variable after applying the power transformation."""

        X = X.copy()
        X.loc[:, self.variable + f"_{round(self.power,3)}"] = np.power(
            X.loc[:, self.variable], self.power
        )
        return X


class TotalNumberOfDues(BaseEstimator, TransformerMixin):
    """Custom class to Extract Total Number of Dues using NumberOfTime30-59DaysPastDueNotWorse,
    NumberOfTime60-89DaysPastDueNotWorse, NumberOfTimes90DaysLate features."""

    def __init__(self, variables):
        if isinstance(variables, list) and len(variables) == 3:
            self.variables = variables
        else:
            raise ValueError("Variables must be a list.")

    def fit(self, X, y=None):
        """Return the object & does nothing."""

        return self

    def transform(self, X):
        """Extracts new feature using the variables passed into the list."""

        X = X.copy()
        X["TotalNumberOfDues"] = (
            X[self.variables[0]] + X[self.variables[1]] + X[self.variables[2]]
        )
        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    """Custom class to Drop unnecessary features from the dataset."""

    def __init__(self, variables_to_drop=None):
        if not isinstance(variables_to_drop, list):
            self.variables_to_drop = [variables_to_drop]
        else:
            self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        """Return the object & does nothing."""

        return self

    def transform(self, X):
        """Drops the features passed."""

        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X
