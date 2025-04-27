"""
Main script for dog breed classification using MobileNetV2.
"""
import argparse
import os
import tensorflow as tf
import numpy as np
from src.data_loading import load_dataset
from src.models import (
    create_baseline_model,
    create_model_with_unfrozen_layers,
    create_optimized_model,
    create_model_with_inverted_residual,
    print_model_summary,
    verify_trainable_layers
)
from src.training import train_model, plot_training_history
from src.evaluation import evaluate_model, print_evaluation_results, visualize_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dog breed classification using MobileNetV2")
    
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory. If not provided, will try to load from tensorflow_datasets.")
    
    parser.add_argument("--model_type", type=str, default="optimized",
                        choices=["baseline", "unfrozen", "optimized", "residual"],
                        help="Type of model to create.")
    
    parser.add_argument("--unfrozen_layers", type=int, default=50,
                        help="Number of layers to unfreeze from the bottom of the model.")
    
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Alpha value for ELU activation function.")
    
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                        help="Dropout rate for regularization.")
    
    parser.add_argument("--learning_rate", type=float, default=0.00001,
                        help="Learning rate for the optimizer.")
    
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="Kernel size for depthwise convolution in inverted residual block.")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and testing.")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train.")
    
    parser.add_argument("--save_model", type=str, default=None,
                        help="Path to save the trained model.")
    
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for TensorBoard logs.")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up TensorFlow to use a reasonable amount of GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    
    # Load the dataset
    print("Loading dataset...")
    train_ds, test_ds = load_dataset(args.data_dir, args.batch_size)
    
    if train_ds is None or test_ds is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Create the model based on the specified type
    print(f"Creating {args.model_type} model...")
    
    if args.model_type == "baseline":
        model = create_baseline_model()
    elif args.model_type == "unfrozen":
        model = create_model_with_unfrozen_layers(
            unfrozen_layers=args.unfrozen_layers,
            alpha=args.alpha,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate
        )
    elif args.model_type == "optimized":
        model = create_optimized_model(
            unfrozen_layers=args.unfrozen_layers,
            alpha=args.alpha,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate
        )
    elif args.model_type == "residual":
        model = create_model_with_inverted_residual(
            unfrozen_layers=args.unfrozen_layers,
            alpha=args.alpha,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            kernel_size=args.kernel_size
        )
    
    # Print model summary and trainable layers
    print_model_summary(model)
    verify_trainable_layers(model)
    
    # Train the model
    print(f"Training the model for {args.epochs} epochs...")
    history = train_model(
        model, train_ds, test_ds,
        epochs=args.epochs,
        use_early_stopping=True,
        patience=5,
        save_model_path=args.save_model,
        log_dir=args.log_dir
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    print("Evaluating the model...")
    evaluation_results = evaluate_model(model, test_ds)
    
    # Print evaluation results
    print_evaluation_results(evaluation_results)
    
    # Visualize results
    visualize_results(history, evaluation_results)
    
    print("Done!")

if __name__ == "__main__":
    main()
