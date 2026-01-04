""" 

FEATURE ENGINEER

"""

import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils.config import (
    TOKEN_REGEX, NB_FEATURES
    )

class FeatureEngineer:
    """
    Docstring for FeatureEngineer
    """

    def __init__(self):
        """ Initialize the feature engineer """
        pass

    def tokenizer(self, text):
        """
        Docstring for tokenizer
        
        :param self: Description
        :param text: Description
        """

        message = text.lower()

        message = message.translate(str.maketrans('', '', string.punctuation))

        message = re.sub(r'\d+', '<NUM>', message)

        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(message)
        filtered_tokens = [word for word in tokens if word not in stop_words]

        return ''.join(filtered_tokens)
    
    def count_vectorizer(self, tokenizer, training_messages, testing_messages):
        """
        Docstring for count_vectorizer

        :param self: Description
        :param messages: Description
        """

        count_vectorize = CountVectorizer(
            max_features=NB_FEATURES,
            token_pattern=TOKEN_REGEX,
            preprocessor=tokenizer
        )

        training_vectors = count_vectorize.fit_transform(training_messages)
        testing_vectors = count_vectorize.transform(testing_messages)
            
        return training_vectors, testing_vectors
    
    def encoder(self, training_messages, testing_messages):
        """
        Docstring for encoder

        :param self: Description
        :param messages: Description
        """

        print(training_messages)

        training_vectors, testing_vectors = self.count_vectorizer(self.tokenizer, training_messages, testing_messages)

        return training_vectors, testing_vectors