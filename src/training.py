"""
Training functions for dog breed classification models.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import os
import time

def train_model(model, train_ds, test_ds, epochs=50, use_early_stopping=True, patience=5,
               save_model_path=None, log_dir=None):
    """
    Train a model for dog breed classification.
    
    Args:
        model: TensorFlow Keras model to train.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        epochs (int): Number of epochs to train.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        save_model_path (str): Path to save the best model.
        log_dir (str): Directory for TensorBoard logs.
        
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    callbacks = []
    
    # Add early stopping callback if requested
    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # Add model checkpoint callback if save_model_path is provided
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            filepath=save_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    # Add TensorBoard callback if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join(log_dir, f"run_{int(time.time())}"),
            histogram_freq=1
        )
        callbacks.append(tensorboard_callback)
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Training completed.")
    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history: Training history from model.fit().
    """
    # Get training and validation metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Get number of epochs
    epochs = range(1, len(acc) + 1)
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def continue_training(model, train_ds, test_ds, additional_epochs=10, use_early_stopping=True,
                     patience=5, save_model_path=None, log_dir=None):
    """
    Continue training a previously trained model.
    
    Args:
        model: TensorFlow Keras model to continue training.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        additional_epochs (int): Number of additional epochs to train.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        save_model_path (str): Path to save the best model.
        log_dir (str): Directory for TensorBoard logs.
        
    Returns:
        tf.keras.callbacks.History: Training history.
    """
    print(f"Continuing training for {additional_epochs} more epochs...")
    return train_model(
        model, train_ds, test_ds, 
        epochs=additional_epochs, 
        use_early_stopping=use_early_stopping,
        patience=patience,
        save_model_path=save_model_path,
        log_dir=log_dir
    )

def train_with_different_learning_rates(model_fn, train_ds, test_ds, learning_rates=[1e-3, 1e-4, 1e-5],
                                      epochs=10, use_early_stopping=True, patience=3):
    """
    Train a model with different learning rates to find the optimal one.
    
    Args:
        model_fn: Function that creates a model.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        learning_rates (list): List of learning rates to try.
        epochs (int): Number of epochs to train for each learning rate.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        
    Returns:
        dict: Dictionary mapping learning rates to training history.
    """
    results = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Create a new model instance
        model = model_fn(learning_rate=lr)
        
        # Train the model
        history = train_model(
            model, train_ds, test_ds,
            epochs=epochs,
            use_early_stopping=use_early_stopping,
            patience=patience
        )
        
        # Store the results
        results[lr] = {
            'history': history,
            'final_accuracy': history.history['val_accuracy'][-1],
            'model': model
        }
        
        print(f"Final validation accuracy with learning rate {lr}: {results[lr]['final_accuracy']:.4f}")
    
    # Find the best learning rate
    best_lr = max(results.keys(), key=lambda lr: results[lr]['final_accuracy'])
    print(f"\nBest learning rate: {best_lr} with validation accuracy: {results[best_lr]['final_accuracy']:.4f}")
    
    return results, best_lr

def experiment_with_alpha_values(model_fn, train_ds, test_ds, alpha_values=[0.5, 1.0, 1.5, 2.0],
                               epochs=10, use_early_stopping=True, patience=3, learning_rate=1e-4):
    """
    Experiment with different alpha values for the ELU activation function.
    
    Args:
        model_fn: Function that creates a model.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        alpha_values (list): List of alpha values to try.
        epochs (int): Number of epochs to train for each alpha value.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        dict: Dictionary mapping alpha values to training history.
    """
    results = {}
    
    for alpha in alpha_values:
        print(f"\nTraining with alpha value: {alpha}")
        
        # Create a new model instance with the current alpha value
        model = model_fn(alpha=alpha, learning_rate=learning_rate)
        
        # Train the model
        history = train_model(
            model, train_ds, test_ds,
            epochs=epochs,
            use_early_stopping=use_early_stopping,
            patience=patience
        )
        
        # Store the results
        results[alpha] = {
            'history': history,
            'final_accuracy': history.history['val_accuracy'][-1],
            'model': model
        }
        
        print(f"Final validation accuracy with alpha {alpha}: {results[alpha]['final_accuracy']:.4f}")
    
    # Find the best alpha value
    best_alpha = max(results.keys(), key=lambda a: results[a]['final_accuracy'])
    print(f"\nBest alpha value: {best_alpha} with validation accuracy: {results[best_alpha]['final_accuracy']:.4f}")
    
    return results, best_alpha

def experiment_with_dropout_rates(model_fn, train_ds, test_ds, dropout_rates=[0.2, 0.3, 0.5],
                                epochs=10, use_early_stopping=True, patience=3, 
                                learning_rate=1e-5, alpha=1.5):
    """
    Experiment with different dropout rates for regularization.
    
    Args:
        model_fn: Function that creates a model.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        dropout_rates (list): List of dropout rates to try.
        epochs (int): Number of epochs to train for each dropout rate.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        learning_rate (float): Learning rate for the optimizer.
        alpha (float): Alpha value for the ELU activation function.
        
    Returns:
        dict: Dictionary mapping dropout rates to training history.
    """
    results = {}
    
    for dropout_rate in dropout_rates:
        print(f"\nTraining with dropout rate: {dropout_rate}")
        
        # Create a new model instance with the current dropout rate
        model = model_fn(dropout_rate=dropout_rate, learning_rate=learning_rate, alpha=alpha)
        
        # Train the model
        history = train_model(
            model, train_ds, test_ds,
            epochs=epochs,
            use_early_stopping=use_early_stopping,
            patience=patience
        )
        
        # Store the results
        results[dropout_rate] = {
            'history': history,
            'final_accuracy': history.history['val_accuracy'][-1],
            'model': model
        }
        
        print(f"Final validation accuracy with dropout rate {dropout_rate}: {results[dropout_rate]['final_accuracy']:.4f}")
    
    # Find the best dropout rate
    best_dropout_rate = max(results.keys(), key=lambda dr: results[dr]['final_accuracy'])
    print(f"\nBest dropout rate: {best_dropout_rate} with validation accuracy: {results[best_dropout_rate]['final_accuracy']:.4f}")
    
    return results, best_dropout_rate

def experiment_with_kernel_sizes(model_fn, train_ds, test_ds, kernel_sizes=[2, 3, 4, 5, 6],
                               epochs=10, use_early_stopping=True, patience=3, 
                               learning_rate=1e-5, alpha=1.5, dropout_rate=0.5):
    """
    Experiment with different kernel sizes for depthwise convolution.
    
    Args:
        model_fn: Function that creates a model.
        train_ds: Training dataset.
        test_ds: Validation dataset.
        kernel_sizes (list): List of kernel sizes to try.
        epochs (int): Number of epochs to train for each kernel size.
        use_early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        learning_rate (float): Learning rate for the optimizer.
        alpha (float): Alpha value for the ELU activation function.
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        dict: Dictionary mapping kernel sizes to training history.
    """
    results = {}
    
    for kernel_size in kernel_sizes:
        print(f"\nTraining with kernel size: {kernel_size}")
        
        # Create a new model instance with the current kernel size
        model = model_fn(
            kernel_size=kernel_size, 
            learning_rate=learning_rate, 
            alpha=alpha, 
            dropout_rate=dropout_rate
        )
        
        # Train the model
        history = train_model(
            model, train_ds, test_ds,
            epochs=epochs,
            use_early_stopping=use_early_stopping,
            patience=patience
        )
        
        # Store the results
        results[kernel_size] = {
            'history': history,
            'final_accuracy': history.history['val_accuracy'][-1],
            'model': model
        }
        
        print(f"Final validation accuracy with kernel size {kernel_size}: {results[kernel_size]['final_accuracy']:.4f}")
    
    # Find the best kernel size
    best_kernel_size = max(results.keys(), key=lambda ks: results[ks]['final_accuracy'])
    print(f"\nBest kernel size: {best_kernel_size} with validation accuracy: {results[best_kernel_size]['final_accuracy']:.4f}")
    
    return results, best_kernel_size
