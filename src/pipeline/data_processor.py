"""
Data preprocessing module for Spam predictor ML Pipeline.

This module handles data loading, cleaning, and preprocessing.
Handles time-series data with geographical groupings properly.
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split

from utils.config import (
    DATA_PATH, SMS_FILE, EMAIL_FILE, LABEL_COL, TRAIN_TEST_SPLIT_SIZE, RANDOM_STATE
)

class DataProcessor:
    """
    Data processor for stock market datasets

    Handles loading, cleaning and preprocessing of data
    """

    def __init__(self):
        """ Initialize the data processor """
        self.sms_data = None
        self.email_data = None
        self.train_data = None
        self.test_data = None
    
    def load_data(self):
        """ Load data """
        self.sms_data = pd.read_csv(DATA_PATH / SMS_FILE, sep=";")
        self.email_data = pd.read_csv(DATA_PATH / EMAIL_FILE, sep=",")

        self.sms_data[LABEL_COL] = self.sms_data[LABEL_COL].astype('Int64')
        self.email_data[LABEL_COL] = self.email_data[LABEL_COL].astype('Int64')

        return self.sms_data.copy(), self.email_data.copy()

    def drop_duplicates(self, df):
        """ Drop duplicate entries"""

        data = df.copy()
        data.drop_duplicates(inplace=True, ignore_index=True)

        return data
    
    def splitter(self, sms_data, email_data, method):
        """ Split data into train and test sets"""

        sms_labels = sms_data[LABEL_COL].astype('Int64')
        email_labels = email_data[LABEL_COL].astype('Int64')

        sms_messages = sms_data['message'].astype('string')
        email_messages = email_data['message'].astype('string')

        if method == "SMS_ONLY":

            return train_test_split(
                sms_messages,
                sms_labels,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=sms_labels
            )
        
        elif method == "EMAIL_ONLY":

            return train_test_split(
                email_messages,
                email_labels,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=email_labels
            )
        
        elif method == "COMBINED":
            data = pd.concat([self.sms_data, self.email_data], ignore_index=True)
            messages = data['message'].astype('string')
            labels = data[LABEL_COL].astype('Int64')

            return train_test_split(
                messages,
                labels,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=labels
            )

        elif method == "TRANSFERT":
            return (
                sms_messages,
                email_messages,
                sms_labels,
                email_labels
        )

        else:
            raise ValueError(f"Unknown train-test split method: {method}")
        
    def balance(self, training_messages, training_labels):
        """ Balance date using oversampling"""
        
        counts = training_labels.value_counts()
        diff = 0
        if counts[1] > counts[0]:
            label_to_oversample = 0
            diff = counts[1] - counts[0]
        else:
            label_to_oversample = 1
            diff = counts[0] - counts[1]
        
        training_data = pd.concat([training_messages, training_labels], axis=1)
        draw_from = training_data[training_data["label"] == label_to_oversample]
        
        for i in range(diff):
            sample = draw_from.sample(random_state=RANDOM_STATE)
            training_data = pd.concat([training_data, sample])
        
        training_messages = training_data["message"]
        training_labels = training_data["label"]
        
        return training_messages, training_labels

    def preprocess_data(self, drop_duplicates=True):
        """ Preprocess data """

        if self.sms_data is None or self.email_data is None:
            self.load_data()

        sms_data = self.sms_data.copy()
        email_data = self.email_data.copy()

        if drop_duplicates:
            sms_data = self.drop_duplicates(sms_data)
            email_data = self.drop_duplicates(email_data)

        train_messages, test_messages, train_labels, test_labels = self.splitter(sms_data, email_data, method="COMBINED")
        train_messages, train_labels = self.balance(train_messages, train_labels)

        return train_messages, train_labels, test_messages, test_labels
    
    def load_and_preprocess(self, drop_duplicates=True):
        """ Load and preprocess data """

        self.load_data()
        
        return self.preprocess_data(drop_duplicates=drop_duplicates)