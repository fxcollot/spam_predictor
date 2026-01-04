"""
Model evaluation modeule for Spam Detection Pipeline
"""

import numpy  as np


class Evaluator:
    """
    Docstring for Evaluator
    """

    def __init__(self):
        """ Initialize the evaluator """
        pass

    def accuracy_score(self, truth, pred, pos_label):
        """
        Calculate accuracy as the proportion of correct predictions.
        
        Accuracy measures overall correctness but may not reflect performance
        on individual classes, especially with imbalanced datasets.
        
        Parameters
        ----------
        truth : array-like
            Ground truth labels (0 for ham, 1 for spam)
        pred : array-like  
            Predicted labels from the model
            
        Returns
        -------
        float
            Accuracy score between 0 and 1, where 1 indicates perfect accuracy
            
        Notes
        -----
        Accuracy alone may be misleading for imbalanced datasets where
        a model could achieve high accuracy by always predicting the majority class.
        """
        FN, FP, TN, TP = 0, 0, 0, 0

        for i in range(len(truth)):
            if truth[i] != pred[i]:
                if truth[i] == pos_label:
                    FN += 1
                else:
                    FP += 1
            else:
                if truth[i] == pos_label:
                    TP += 1
                else:
                    TN += 1 

        return (TP + TN) / (TP + TN + FP + FN)
    
    def precision_score(self, truth, pred, pos_label):
        """
        Calculate precision for the positive class (spam).
        
        Precision measures the proportion of predicted spam messages that are
        actually spam. High precision indicates few false positive errors
        (legitimate messages incorrectly classified as spam).
        
        Parameters
        ----------
        truth : array-like
            Ground truth labels
        pred : array-like
            Predicted labels from the model  
        pos_label : int or str
            Label that represents the positive class (spam)
            
        Returns
        -------
        float
            Precision score between 0 and 1, where 1 indicates perfect precision
            
        Notes
        -----
        Precision is especially important in spam detection to minimize
        false positives that could cause users to miss important messages.
        """
        FN, FP, TN, TP = 0, 0, 0, 0

        for i in range(len(truth)):
            if truth[i] != pred[i]:
                if truth[i] == pos_label:
                    FN += 1
                else:
                    FP += 1
            else:
                if truth[i] == pos_label:
                    TP += 1
                else:
                    TN += 1 
                    
        return TP / (TP + FP)

    def recall_score(self, truth, pred, pos_label):
        """
        Calculate recall for the positive class (spam).
        
        Recall measures the proportion of actual spam messages that the model
        correctly identifies. High recall indicates the model catches most
        spam with few false negative errors.
        
        Parameters
        ----------
        truth : array-like
            Ground truth labels
        pred : array-like
            Predicted labels from the model
        pos_label : int or str  
            Label that represents the positive class (spam)
            
        Returns
        -------
        float
            Recall score between 0 and 1, where 1 indicates perfect recall
            
        Notes
        -----
        Recall is crucial in spam detection to ensure most unwanted messages
        are filtered out, protecting users from spam content.
        """
        FN, FP, TN, TP = 0, 0, 0, 0

        for i in range(len(truth)):
            if truth[i] != pred[i]:
                if truth[i] == pos_label:
                    FN += 1
                else:
                    FP += 1
            else:
                if truth[i] == pos_label:
                    TP += 1
                else:
                    TN += 1 
        return TP / (TP + FN)

    def calculate_metrics(self, testing_labels, predictions):
        """
        Docstring for calculate_metrics

        :param self: Description
        :param y_true: Description
        :param y_pred: Description
        """
        # Placeholder implementation
        mse = np.mean((testing_labels - predictions) ** 2)

        rec = self.recall_score(testing_labels.values, predictions, 1)
        prec = self.precision_score(testing_labels.values, predictions, 1)
        acc = self.accuracy_score(testing_labels.values, predictions, 1)
        
        return {'rec': rec, 'prec': prec, 'acc': acc, 'mse': mse}