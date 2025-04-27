"""
Data loading and preprocessing functions for dog breed classification.
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def load_dataset(data_dir=None, batch_size=64, image_size=(224, 224)):
    """
    Load and prepare the Stanford Dogs Dataset or use TensorFlow datasets if available.
    
    Args:
        data_dir (str): Path to dataset directory. If None, will try to load from tensorflow_datasets.
        batch_size (int): Batch size for training and testing.
        image_size (tuple): Target image size (height, width).
        
    Returns:
        tuple: (train_dataset, test_dataset) - Preprocessed TensorFlow datasets for training and testing.
    """
    if data_dir is None:
        # Try to load the dataset from tensorflow_datasets
        try:
            import tensorflow_datasets as tfds
            train_dataset, info = tfds.load("stanford_dogs", split="train", with_info=True, as_supervised=False)
            test_dataset = tfds.load("stanford_dogs", split="test", as_supervised=False)
            print(f"Dataset loaded with {info.splits['train'].num_examples} training examples and "
                  f"{info.splits['test'].num_examples} test examples.")
            
            # Get class names
            class_names = info.features['label'].names
            num_classes = len(class_names)
            print(f"Dataset contains {num_classes} classes.")
            
            return preprocess_tfds_dataset(train_dataset, test_dataset, batch_size, image_size)
            
        except Exception as e:
            print(f"Error loading from tensorflow_datasets: {e}")
            print("Please provide a valid data_dir or ensure tensorflow_datasets is installed.")
            return None, None
    else:
        # Load from directory
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        return preprocess_directory_dataset(train_ds, test_ds)

def preprocess_tfds_dataset(train_dataset, test_dataset, batch_size=64, image_size=(224, 224)):
    """
    Preprocess the Stanford Dogs Dataset loaded from tensorflow_datasets.
    
    Args:
        train_dataset: TensorFlow dataset for training.
        test_dataset: TensorFlow dataset for testing.
        batch_size (int): Batch size for training and testing.
        image_size (tuple): Target image size (height, width).
        
    Returns:
        tuple: (train_dataset, test_dataset) - Preprocessed datasets ready for training.
    """
    def process_and_crop_image(data):
        """Crop the image using the bounding box and apply preprocessing."""
        image = tf.cast(data['image'], tf.float32)
        bbox = data['objects']['bbox'][0]  # Get the first bounding box
        label = tf.one_hot(data['label'], depth=120)  # One-hot encode the label (120 dog breeds)
        
        # Extract bounding box coordinates
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        y_min, x_min, y_max, x_max = bbox[0] * tf.cast(height, tf.float32), bbox[1] * tf.cast(width, tf.float32), \
                                     bbox[2] * tf.cast(height, tf.float32), bbox[3] * tf.cast(width, tf.float32)
        
        # Convert to integers
        y_min, x_min, y_max, x_max = tf.cast(y_min, tf.int32), tf.cast(x_min, tf.int32), \
                                     tf.cast(y_max, tf.int32), tf.cast(x_max, tf.int32)
        
        # Crop the image
        cropped_image = tf.image.crop_to_bounding_box(
            image, y_min, x_min, y_max - y_min, x_max - x_min
        )
        
        # Resize image
        resized_image = tf.image.resize(cropped_image, image_size)
        
        # Normalize to [0, 1]
        normalized_image = resized_image / 255.0
        
        return normalized_image, label
    
    # Create augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    
    # Apply preprocessing to training dataset with augmentation
    train_ds = train_dataset.map(process_and_crop_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Apply preprocessing to test dataset (no augmentation)
    test_ds = test_dataset.map(process_and_crop_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

def preprocess_directory_dataset(train_ds, test_ds):
    """
    Apply additional preprocessing to datasets loaded from directories.
    
    Args:
        train_ds: Training dataset.
        test_ds: Testing dataset.
        
    Returns:
        tuple: (train_ds, test_ds) - Preprocessed datasets.
    """
    # Create normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # Create data augmentation layer for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.2)
    ])
    
    # Apply normalization and data augmentation to training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(normalization_layer(x)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    # Apply only normalization to test dataset
    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

def visualize_samples(dataset, class_names=None, num_samples=5):
    """
    Visualize samples from the dataset.
    
    Args:
        dataset: TensorFlow dataset to visualize.
        class_names (list): List of class names.
        num_samples (int): Number of samples to visualize.
    """
    plt.figure(figsize=(12, 12))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].numpy())
            
            if class_names is not None:
                if isinstance(labels[i], tf.Tensor) and len(labels[i].shape) > 0:
                    # For one-hot encoded labels
                    index = tf.argmax(labels[i]).numpy()
                    class_name = class_names[index]
                else:
                    # For index labels
                    class_name = class_names[int(labels[i].numpy())]
                plt.title(class_name)
            
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def crop_using_bounding_box(image, bbox):
    """
    Crop an image using bounding box coordinates.
    
    Args:
        image (numpy.ndarray): Input image.
        bbox (list): Bounding box coordinates [y_min, x_min, y_max, x_max] in normalized format.
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    height, width, _ = image.shape
    y_min, x_min, y_max, x_max = int(bbox[0] * height), int(bbox[1] * width), \
                                int(bbox[2] * height), int(bbox[3] * width)
    
    # Crop the image using PIL
    pil_image = Image.fromarray(image.astype('uint8'))
    cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
    
    return np.array(cropped_image)

def process_and_save_cropped_images(dataset, output_dir, class_names=None, max_samples=None):
    """
    Process and save cropped images to disk.
    
    Args:
        dataset: TensorFlow dataset containing images with bounding box information.
        output_dir (str): Output directory to save cropped images.
        class_names (list): List of class names.
        max_samples (int): Maximum number of samples to process per class.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    
    for data in dataset:
        if max_samples is not None and sample_count >= max_samples:
            break
        
        image = np.array(data['image'])
        bbox = data['objects']['bbox'][0]  # Get the first bounding box
        label_index = data['label']
        
        # Get class name if available
        if class_names is not None:
            label_name = class_names[label_index]
        else:
            label_name = f"class_{label_index}"
        
        # Create directory for this class
        class_dir = os.path.join(output_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Crop the image
        cropped_image = crop_using_bounding_box(image, bbox)
        
        # Save the cropped image
        file_name = f"{label_name}_{label_index}_{sample_count}.jpg"
        file_path = os.path.join(class_dir, file_name)
        
        pil_image = Image.fromarray(cropped_image)
        pil_image.save(file_path)
        
        print(f"Saved {file_path}")
        sample_count += 1
