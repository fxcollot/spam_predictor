import argparse

from utils.config import MODEL_TYPES


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Stock Market ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', type=str, default='linear',
        choices=MODEL_TYPES,
        help='Model type to train'
    )

    parser.add_argument(
        '--n-features', type=int, default=8,
        help='Number of features to select'
    )

    parser.add_argument(
        '--method', type=str, default='rfe',
        choices=['selectkbest', 'rfe', 'all'],
        help='Feature selection method'
    )

    parser.add_argument(
        '--optimize', type=bool, default=False,
        help='Use of hyperparameter optimization ' 
    )

    parser.add_argument(
        '--compare', action='store_true',
        help='Compare multiple models instead of training single model'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )

    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )

    # Add MLflow tracking argument --mlflow (Workshop 4)
    parser.add_argument(
        '--mlflow', type=bool, default=True,
        help='Use mlflow dashboard for history. Defaults to True'
    )

    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name given to the run, defaults to general parameters'
    )

    parser.add_argument(
        '--run_tags',
        nargs='*',  # Allow 0 or more tags
        help='Tags to give to the run (format key=value or just key)'
    )

    return parser.parse_args()

def parse_tags(tags_list):
    tags_dict = {}
    if tags_list:
        for tag in tags_list:
            if '=' in tag:
                # On coupe seulement au premier '=' pour autoriser les '=' dans les valeurs
                key, value = tag.split('=', 1)
                tags_dict[key] = value
            else:
                # Si pas de '=', on considère que c'est un flag booléen
                tags_dict[tag] = "True"
    return tags_dict