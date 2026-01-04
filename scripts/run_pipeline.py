"""
Pipeline to train NLP models for spam detection.
"""

import mlflow
import datetime as time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.data_processor import DataProcessor
from pipeline.feature_engineer import FeatureEngineer
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import DATA_PATH

from utils.parse_args import parse_arguments, parse_tags
from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """

    start_time = time.time()
    #logger = get_logger()

    #warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

    #Reduce the amount of noise shown by mlflow
    #warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    #warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    # Creating Eperiment name (easier to separate this way)
    

    if args.mlflow:
        # Reduce the amount of noise shown by mlflow
        #warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
        #warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

        # Add MLflow setup and run start (Workshop 4)
            # Configuration MLflow simple
        #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        #mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Create descriptive run name
        if args.run_name == '':
            run_name = f"stock_market_{args.model}"

            if args.optimize:
                run_name += "_opti"

        else:
            run_name = args.run_name

        timestamp = time.datetime.now().strftime("%Hh%M")

        mlflow.start_run(run_name=f"{run_name}_{timestamp}")

            # Set tags for Dataset and Model columns in MLflow UI
        # PARAMS
        mlflow.log_param("model.type", args.model)
        mlflow.log_param("model.optimize", args.optimize)
        mlflow.log_param("features.n_features", args.n_features)
        mlflow.log_param("features.selection_method", args.method)

        # TAGS
        run_tags = parse_tags(args.run_tags)
        if run_tags:
            mlflow.set_tags(run_tags)

        # Adding default context
        if "context" not in run_tags.items():
            mlflow.set_tag("context", "SimpleRun")

        mlflow.set_tag("dataset.path", DATA_PATH)
        mlflow.set_tag("pipeline.mlflow_enabled", args.mlflow)
        mlflow.set_tag("mlflow.note.content", f"Pipeline with {args.model} model and {args.n_features} features")

        run = mlflow.active_run()
        #logger.info("Run started:", run is not None)

    try:
        print("Run finished.")
        #logger.info(" SPAM DETECTION ML PIPELINE")
        
        processor = DataProcessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        evaluator = Evaluator()

        # STEP 1
        train_messages, train_labels, test_messages, test_labels = processor.load_and_preprocess()

        # STEP 2
        training_vectors, testing_vectors = engineer.encoder(train_messages, test_messages)

        # STEP 3
        model = trainer.create_model(
            model_type=args.model
        )

        model = trainer.train_single_model(model, args.model, training_vectors, train_labels)

        predictions = trainer.model_predictor(model, testing_vectors)

        metrics = evaluator.calculate_metrics(test_labels, predictions)

        print("metrics", metrics)
    
    finally:
        print("Run finished.")
        

def main():
    """Main entry point."""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging level
        if args.verbose:
            # Support legacy --verbose flag
            log_level = LogLevel.VERBOSE
        else:
            log_level = log_level_from_string(args.log_level)
        
        set_log_level(log_level)
        
        # Run pipeline
        run_pipeline(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
if __name__ == "__main__":
    exit(main())
