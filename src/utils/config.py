"""
Configuration for Stock Prediction ML Pipeline.

This module contains all configuration constants used throughout
the pipeline. 
"""

from pathlib import Path
import numpy as np

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

SMS_FILE = "sms_spam.csv"
EMAIL_FILE = "email_spam.csv"

# COLUMN NAMES
MESSAGE_COL = "message"
LABEL_COL = "label"

TRAIN_TEST_SPLIT_SIZE = 0.2

RANDOM_STATE = 42

# The following example extracts series of non-whitespace characters.
TOKEN_REGEX = r"(\S+)"
NB_FEATURES = 5000
NB_ITERATIONS = 1000

MODEL_TYPES = ['linear']