"""
Evaluation and visualization functions for dog breed classification models.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model, test_ds):
    """
    Evaluate a model on a test dataset.
    
    Args:
        model: TensorFlow Keras model to evaluate.
        test_ds: Test dataset.
        
    Returns:
        dict: Dictionary containing evaluation results.
    """
    # Evaluate the model
    results = model.evaluate(test_ds)
    
    # Get predictions
    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    y_true = []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    
    # Compute classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Return evaluation results
    evaluation_results = {
        'accuracy': results[1],
        'loss': results[0],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    
    return evaluation_results

def print_evaluation_results(evaluation_results, class_names=None):
    """
    Print evaluation results.
    
    Args:
        evaluation_results (dict): Dictionary containing evaluation results.
        class_names (list): List of class names.
    """
    print(f"Test Accuracy: {evaluation_results['accuracy'] * 100:.2f}%")
    print(f"Test Loss: {evaluation_results['loss']:.4f}")
    print("\nClassification Report:")
    
    # Convert the dictionary-based classification report to a formatted string
    report_dict = evaluation_results['classification_report']
    
    # If class_names is provided, replace numeric indices with class names
    if class_names:
        # Create a new dictionary with class names as keys
        formatted_report = {}
        for key, value in report_dict.items():
            if key.isdigit():
                class_idx = int(key)
                if class_idx < len(class_names):
                    formatted_report[class_names[class_idx]] = value
                else:
                    formatted_report[key] = value
            else:
                formatted_report[key] = value
    else:
        formatted_report = report_dict
    
    # Print metrics for each class
    for class_name, metrics in formatted_report.items():
        if isinstance(metrics, dict):  # Skip the aggregated metrics
            print(f"{class_name.ljust(30)}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1-score={metrics['f1-score']:.2f}, support={metrics['support']}")
    
    # Print aggregated metrics
    print("\nAggregated Metrics:")
    for metric_name in ['accuracy', 'macro avg', 'weighted avg']:
        if metric_name in formatted_report:
            metrics = formatted_report[metric_name]
            if isinstance(metrics, dict):
                print(f"{metric_name.ljust(30)}: precision={metrics.get('precision', '-'):.2f}, recall={metrics.get('recall', '-'):.2f}, f1-score={metrics.get('f1-score', '-'):.2f}, support={metrics.get('support', '-')}")
            else:
                print(f"{metric_name.ljust(30)}: {metrics:.2f}")

def plot_confusion_matrix(evaluation_results, class_names=None, figsize=(12, 10)):
    """
    Plot the confusion matrix.
    
    Args:
        evaluation_results (dict): Dictionary containing evaluation results.
        class_names (list): List of class names.
        figsize (tuple): Figure size (width, height).
    """
    conf_matrix = evaluation_results['confusion_matrix']
    
    # If there are too many classes, plot a summary confusion matrix
    if conf_matrix.shape[0] > 20:
        print("Too many classes for a detailed confusion matrix. Plotting a summary instead.")
        
        # Calculate metrics per class
        class_metrics = []
        for i in range(conf_matrix.shape[0]):
            true_positives = conf_matrix[i, i]
            false_positives = conf_matrix[:, i].sum() - true_positives
            false_negatives = conf_matrix[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'class': class_names[i] if class_names else f"Class {i}",
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })
        
        # Sort classes by F1 score
        class_metrics = sorted(class_metrics, key=lambda x: x['f1_score'], reverse=True)
        
        # Plot metrics for top and bottom 10 classes
        num_classes_to_show = min(10, len(class_metrics) // 2)
        
        plt.figure(figsize=figsize)
        
        # Top classes
        plt.subplot(2, 1, 1)
        top_classes = class_metrics[:num_classes_to_show]
        x = np.arange(len(top_classes))
        width = 0.25
        
        plt.bar(x - width, [m['precision'] for m in top_classes], width, label='Precision')
        plt.bar(x, [m['recall'] for m in top_classes], width, label='Recall')
        plt.bar(x + width, [m['f1_score'] for m in top_classes], width, label='F1 Score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Top Classes by F1 Score')
        plt.xticks(x, [m['class'] for m in top_classes], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bottom classes
        plt.subplot(2, 1, 2)
        bottom_classes = class_metrics[-num_classes_to_show:]
        x = np.arange(len(bottom_classes))
        
        plt.bar(x - width, [m['precision'] for m in bottom_classes], width, label='Precision')
        plt.bar(x, [m['recall'] for m in bottom_classes], width, label='Recall')
        plt.bar(x + width, [m['f1_score'] for m in bottom_classes], width, label='F1 Score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Bottom Classes by F1 Score')
        plt.xticks(x, [m['class'] for m in bottom_classes], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        # Plot the full confusion matrix
        plt.figure(figsize=figsize)
        
        if class_names:
            # If there are too many classes, use numeric indices
            if len(class_names) > 30:
                tick_labels = [f"{i}" for i in range(conf_matrix.shape[0])]
            else:
                tick_labels = class_names
        else:
            tick_labels = [f"{i}" for i in range(conf_matrix.shape[0])]
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.show()

def visualize_results(history, evaluation_results, class_names=None):
    """
    Visualize training history and evaluation results.
    
    Args:
        history: Training history from model.fit().
        evaluation_results (dict): Dictionary containing evaluation results.
        class_names (list): List of class names.
    """
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print evaluation metrics
    print_evaluation_results(evaluation_results, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(evaluation_results, class_names)

def plot_model_comparison(model_results, metric='val_accuracy'):
    """
    Plot a comparison of training histories for different models.
    
    Args:
        model_results (dict): Dictionary mapping model names to training histories.
        metric (str): Metric to plot ('val_accuracy', 'val_loss', 'accuracy', 'loss').
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, history in model_results.items():
        if isinstance(history, dict) and 'history' in history:
            history = history['history']
        
        if metric in history.history:
            plt.plot(history.history[metric], label=model_name)
    
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_single_image(model, image_path, class_names=None, target_size=(224, 224)):
    """
    Make a prediction for a single image.
    
    Args:
        model: TensorFlow Keras model to use for prediction.
        image_path (str): Path to the image file.
        class_names (list): List of class names.
        target_size (tuple): Target image size (height, width).
        
    Returns:
        tuple: (predicted_class, confidence) - Predicted class index/name and confidence.
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get the class name if available
    if class_names and predicted_class_idx < len(class_names):
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = predicted_class_idx
    
    return predicted_class, confidence

def visualize_prediction(image_path, predicted_class, confidence, true_class=None, target_size=(224, 224)):
    """
    Visualize a prediction for a single image.
    
    Args:
        image_path (str): Path to the image file.
        predicted_class: Predicted class index or name.
        confidence (float): Prediction confidence.
        true_class: True class index or name (optional).
        target_size (tuple): Target image size (height, width).
    """
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    
    # Plot the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    
    if true_class:
        plt.title(f"True: {true_class}, Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    else:
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    
    plt.axis('off')
    plt.show()

def visualize_model_activations(model, image_path, layer_names=None, target_size=(224, 224)):
    """
    Visualize activations of intermediate layers of a model for a single image.
    
    Args:
        model: TensorFlow Keras model.
        image_path (str): Path to the image file.
        layer_names (list): List of layer names to visualize. If None, will use the last 5 layers.
        target_size (tuple): Target image size (height, width).
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0  # Normalize
    
    # Create a model that outputs the activations of selected layers
    if layer_names is None:
        # If no layer names provided, use the last 5 layers (excluding the output layer)
        layer_names = [layer.name for layer in model.layers[:-1][-5:]]
    
    activation_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(name).output for name in layer_names]
    )
    
    # Get activations
    activations = activation_model.predict(img_array)
    
    # Plot the activations
    for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        plt.figure(figsize=(12, 6))
        
        # If the activation is 4D (batch_size, height, width, channels)
        if len(activation.shape) == 4:
            # Compute the number of channels to show
            num_channels = min(16, activation.shape[-1])
            
            # Determine the grid size for subplots
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            
            # Plot activations for selected channels
            for j in range(num_channels):
                plt.subplot(grid_size, grid_size, j + 1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')
                plt.axis('off')
            
            plt.suptitle(f"Activations for layer: {layer_name} (showing {num_channels} of {activation.shape[-1]} channels)")
        
        # If the activation is 2D (batch_size, features)
        elif len(activation.shape) == 2:
            # Plot a histogram of activation values
            plt.hist(activation.flatten(), bins=50, color='blue', alpha=0.7)
            plt.title(f"Activation Distribution for Layer {layer_name}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
