"""
Docstring for pipeline.model_trainer
"""

from sklearn.linear_model import LinearRegression

from .evaluator import Evaluator
from utils.config import RANDOM_STATE, NB_ITERATIONS

class ModelTrainer:
    """
    Docstring for ModelTrainer
    """

    def __init__(self):
        """ Initialize the model trainer """
        self.trained_models = {}
        self.evaluator = Evaluator()
        self.best_model = None
        self.best_model_name=None

    def create_model(self, model_type, **params):
        """
        Docstring for create_model

        :param self: Description
        :param model_type: Description
        :param params: Description
        """

        # Clean string input
        model_type = model_type.lower().strip()

        # --- Linear Models ---
        if model_type == 'linear':
            model = LinearRegression(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def train_single_model(self, model, model_type, X_train, y_train, **model_params):
        """
        Docstring for train_model

        :param self: Description
        :param X_train: Description
        :param y_train: Description
        """

        model.fit(X_train, y_train)

        self.trained_models[model_type] = model
        predictions = self.model_predictor(model, X_train)

        #metrics = self.evaluator.calculate_metrics(y_train, predictions)

        return model
    
    def model_predictor(self, model, X):
        """
        Docstring for model_predictor

        :param self: Description
        :param model: Description
        :param X_test: Description
        """

        predictions = model.predict(X)

        return predictions