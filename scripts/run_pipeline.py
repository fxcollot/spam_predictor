"""
Pipeline to train NLP models for spam detection.
"""

import mlflow
import datetime as time
from pathlib import Path
import warnings
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mlflow.models.signature import infer_signature
from mlflow.data import from_pandas

from pipeline.data_processor import DataProcessor
from pipeline.feature_engineer import FeatureEngineer
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import DATA_PATH, MODEL_TYPES, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

from utils.parse_args import parse_arguments, parse_tags
from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

def run_pipeline(args):
    """
    Run the complete spam prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    if args.mlflow:
        # Reduce the amount of noise shown by mlflow
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

        # Add MLflow setup and run start (Workshop 4)
            # Configuration MLflow simple
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

        # TAGS
        run_tags = parse_tags(args.run_tags)
        if run_tags:
            mlflow.set_tags(run_tags)

        # Adding default context
        if "context" not in run_tags.items():
            mlflow.set_tag("context", "SimpleRun")

        mlflow.set_tag("dataset.path", DATA_PATH)
        mlflow.set_tag("pipeline.mlflow_enabled", args.mlflow)
        mlflow.set_tag("mlflow.note.content", f"Pipeline with {args.model} model")

        run = mlflow.active_run()
        logger.info("Run started:", run is not None)

    try:
        logger.info(" SPAM DETECTION ML PIPELINE")
        
        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
            logger.info(f"MLflow tracking: {'Enabled' if args.mlflow else 'Disabled'}")
            if args.mlflow:
                logger.info("Final model will be retrained on all data and registered in MLflow")

        processor = DataProcessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        evaluator = Evaluator()

        # STEP 1
        logger.step("Data Loading and Preprocessing", 1)
        with logger.timer("Data loading and preprocessing"):
            train_messages, train_labels, test_messages, test_labels = processor.load_and_preprocess()

        from mlflow.data import from_pandas

        mlflow_train_messages = train_messages.to_frame(name="message")
        mlflow_test_messages = test_messages.to_frame(name="message")



        if mlflow.active_run():
            # Log dataset inline (no separate function)
            mlflow.log_input(from_pandas(mlflow_train_messages, source="training_data"), context="training")
            mlflow.log_input(from_pandas(mlflow_test_messages, source="test_data"), context="test")

            # Log dataset metrics
            mlflow.log_metric("dataset.train_rows", (train_messages.shape[0]))
            mlflow.log_metric("dataset.test_rows", (test_messages.shape[0]))


        # STEP 2
        logger.step("Encoding", 2)
        with logger.timer("Encoding"):
            training_vectors, testing_vectors = engineer.encoder(train_messages, test_messages)

        # STEP 3
        if args.model in MODEL_TYPES:
            model = trainer.create_model(args.model)
        else:
            raise Exception(f"The input model '{args.model}' is not recognised")

        model = trainer.train_single_model(model, args.model, training_vectors, train_labels)

        predictions = trainer.model_predictor(model, testing_vectors)

        metrics = evaluator.calculate_metrics(test_labels, predictions)

        logger.info("Metrics : ")
        logger.info("Recall : ", metrics['rec'])
        logger.info("Precision : ", metrics['prec'])
        logger.info("Accuracy : ", metrics['acc']) 

        if mlflow.active_run():
            # Log cross-validation results
            mlflow.log_metric("precision", metrics['prec'])
            mlflow.log_metric("recall", metrics['rec'])
            mlflow.log_metric("accuracy", metrics['acc'])

        model_name_mlflow = f"{args.model}_model"
        signature = infer_signature(training_vectors, train_labels)
        input_example = training_vectors[:5]

        mlflow.sklearn.log_model(
            model,
            model_name_mlflow,
            signature=signature,
            input_example=input_example,
        )

    finally:
        if mlflow.active_run():
            mlflow.end_run()
        

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
