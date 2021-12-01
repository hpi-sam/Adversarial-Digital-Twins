from os import stat
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge


class RidgeUtilityPredictor():

    def __init__(self) -> None:

        self.train_cols = {
            'features': [
                'CRITICALITY',
                'CONNECTIVITY',
                'RELIABILITY',
                'IMPORTANCE',
                'PROVIDED_INTERFACE',
                'REQUIRED_INTERFACE',
                'ADT',
                ' PMax',
                'alpha',
                'In Use REPLICA',
                'LOAD',
            ],
            'y': 'OPTIMAL_UTILITY'
        }

        self.mrubis_features_to_train_map = {
            'criticality': 'CRITICALITY',
            'connectivity': 'CONNECTIVITY',
            'reliability': 'RELIABILITY',
            'importance': 'IMPORTANCE',
            'provided_interface': 'PROVIDED_INTERFACE',
            'required_interface': 'REQUIRED_INTERFACE',
            'adt': 'ADT',
            'perf_max': ' PMax',
            'sat_point': 'alpha',
            'replica': 'In Use REPLICA',
            'request': 'LOAD',
        }

    def load_train_data(self, path=Path('../data/TrainingmRUBiS_Theta0.05_NonStationary.csv')):
        data = pd.read_csv(
            path,
            usecols=self.train_cols['features'] + [self.train_cols['y']]
        ).dropna()
        self.data = self.calculate_interaction_terms(data)
        self.data = data
        self.train_X = self.data[self.train_cols['features'] + ['beta', 'Linear', 'Saturating', 'Discontinuous', 'Combined']].drop(['IMPORTANCE', 'CONNECTIVITY', 'REQUIRED_INTERFACE'], axis=1)
        self.train_y = self.data[self.train_cols['y']]

    @staticmethod
    def calculate_interaction_terms(data):
        data['beta'] = data.apply(lambda row: row['PROVIDED_INTERFACE'] + (0 if row['PROVIDED_INTERFACE'] >= 2 else 1), axis=1)
        data['Linear'] = data['RELIABILITY'] * data['CRITICALITY'] * (data['PROVIDED_INTERFACE'] + data['REQUIRED_INTERFACE'])
        data['Saturating'] = data['RELIABILITY'] * \
                            data['CRITICALITY'] * \
                            data[' PMax'] * \
                            np.tanh(data['alpha'] * \
                            data['In Use REPLICA'] / data['LOAD']) * \
                            (data['PROVIDED_INTERFACE'] + data['REQUIRED_INTERFACE'])
        data['Discontinuous'] = data['RELIABILITY'] * data['CRITICALITY'] * (data['REQUIRED_INTERFACE'] + 1) * data['IMPORTANCE'] * data['beta'] * data['PROVIDED_INTERFACE'] - 10 * data['ADT']
        data['Combined'] = data['RELIABILITY'] * data['CRITICALITY'] * data['IMPORTANCE'] * data['beta'] * data[' PMax'] * np.tanh(data['alpha'] * data['In Use REPLICA'] / data['LOAD']) *  (data['REQUIRED_INTERFACE'] + 1) - 10 * data['ADT']
        return data

    def train_on_batch_file(self):
        self.model = Ridge()
        self.model.fit(self.train_X, self.train_y)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_on_mrubis_output(self, component_features):
        feature_cols = self.train_cols['features']
        renamed_features = component_features.rename(columns=self.mrubis_features_to_train_map)[feature_cols].astype(float)
        features_with_interaction_terms = self.calculate_interaction_terms(renamed_features)
        required_cols_for_prediction = feature_cols + ['beta', 'Linear', 'Saturating', 'Discontinuous', 'Combined']
        relevant_features = features_with_interaction_terms[required_cols_for_prediction].drop(columns=['IMPORTANCE', 'CONNECTIVITY', 'REQUIRED_INTERFACE'])
        return self.model.predict(relevant_features)
