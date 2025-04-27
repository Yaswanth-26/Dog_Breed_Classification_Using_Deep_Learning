"""
Model architecture definitions for dog breed classification.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ELU

def create_baseline_model(input_shape=(224, 224, 3), num_classes=120):
    """
    Create a baseline model using MobileNetV2 with a simple top layer.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
        num_classes (int): Number of output classes.
        
    Returns:
        tf.keras.Model: Baseline model.
    """
    # Load MobileNetV2 without the top layer
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    mobilenet.trainable = False
    
    # Build the model with MaxPooling2D and Flatten
    model = models.Sequential([
        mobilenet,
        layers.MaxPooling2D(pool_size=(7, 7)),  # Reduce 7x7 spatial dimensions to 1x1
        layers.Flatten(),  # Flatten the feature map to a 1D vector
        layers.Dense(num_classes, activation='softmax')  # Add output layer for classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_with_unfrozen_layers(input_shape=(224, 224, 3), num_classes=120, 
                                    unfrozen_layers=10, alpha=1.5, dropout_rate=0.2,
                                    learning_rate=0.0001):
    """
    Create a model with unfrozen layers for fine-tuning.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
        num_classes (int): Number of output classes.
        unfrozen_layers (int): Number of layers to unfreeze from the bottom of the model.
        alpha (float): Alpha value for ELU activation function.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        tf.keras.Model: Model with unfrozen layers.
    """
    # Load MobileNetV2 without the top layer
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Replace ReLU with ELU in the last <unfrozen_layers> layers
    for layer in mobilenet.layers[-unfrozen_layers:]:
        if hasattr(layer, 'activation') and layer.activation == tf.nn.relu:
            layer.activation = ELU(alpha=alpha)
    
    # Make only the last <unfrozen_layers> layers trainable
    for layer in mobilenet.layers[:-unfrozen_layers]:
        layer.trainable = False
    for layer in mobilenet.layers[-unfrozen_layers:]:
        layer.trainable = True
    
    # Build the model
    model = models.Sequential([
        mobilenet,
        layers.MaxPooling2D(pool_size=(7, 7)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with a smaller learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_optimized_model(input_shape=(224, 224, 3), num_classes=120, 
                         unfrozen_layers=50, alpha=1.5, dropout_rate=0.5,
                         learning_rate=0.00001):
    """
    Create an optimized model with custom layers and fine-tuning.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
        num_classes (int): Number of output classes.
        unfrozen_layers (int): Number of layers to unfreeze from the bottom of the model.
        alpha (float): Alpha value for ELU activation function.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        tf.keras.Model: Optimized model.
    """
    # Load MobileNetV2 without the top layer
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Replace ReLU with ELU in the last <unfrozen_layers> layers
    for layer in mobilenet.layers[-unfrozen_layers:]:
        if hasattr(layer, 'activation') and layer.activation == tf.nn.relu:
            layer.activation = ELU(alpha=alpha)
    
    # Make only the last <unfrozen_layers> layers trainable
    for layer in mobilenet.layers[:-unfrozen_layers]:
        layer.trainable = False
    for layer in mobilenet.layers[-unfrozen_layers:]:
        layer.trainable = True
    
    # Build the model with additional Conv2D layer
    model = models.Sequential([
        mobilenet,
        layers.Conv2D(512, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.ELU(alpha=alpha),
        layers.MaxPooling2D(pool_size=(7, 7)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with a smaller learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def inverted_residual_block(inputs, expansion_factor=6, output_channels=512, stride=1, kernel_size=5, alpha=1.5):
    """
    Create an inverted residual block (a key component of MobileNetV2 architecture).
    
    Args:
        inputs: Input tensor.
        expansion_factor (int): Expansion factor for the block.
        output_channels (int): Number of output channels.
        stride (int): Stride for depthwise convolution.
        kernel_size (int): Kernel size for depthwise convolution.
        alpha (float): Alpha value for ELU activation function.
        
    Returns:
        tf.Tensor: Output tensor from the block.
    """
    # Get the input channels
    input_channels = inputs.shape[-1]
    
    # Expansion phase: 1x1 convolution to increase the number of channels
    x = layers.Conv2D(expansion_factor * input_channels, (1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha=alpha)(x)
    
    # Depthwise separable convolution phase
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha=alpha)(x)
    
    # Projection phase: 1x1 convolution to reduce the number of channels to the output size
    x = layers.Conv2D(output_channels, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Add residual connection if input and output channels match
    if input_channels == output_channels:
        x = layers.Add()([x, inputs])
        
    return x

def create_model_with_inverted_residual(input_shape=(224, 224, 3), num_classes=120, 
                                       unfrozen_layers=50, alpha=1.5, dropout_rate=0.5,
                                       learning_rate=0.00001, kernel_size=5):
    """
    Create a model with an inverted residual block.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels).
        num_classes (int): Number of output classes.
        unfrozen_layers (int): Number of layers to unfreeze from the bottom of the model.
        alpha (float): Alpha value for ELU activation function.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        kernel_size (int): Kernel size for depthwise convolution in the inverted residual block.
        
    Returns:
        tf.keras.Model: Model with inverted residual block.
    """
    # Load MobileNetV2 without the top layer
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Replace ReLU with ELU in the last <unfrozen_layers> layers
    for layer in mobilenet.layers[-unfrozen_layers:]:
        if hasattr(layer, 'activation') and layer.activation == tf.nn.relu:
            layer.activation = ELU(alpha=alpha)
    
    # Make only the last <unfrozen_layers> layers trainable
    for layer in mobilenet.layers[:-unfrozen_layers]:
        layer.trainable = False
    for layer in mobilenet.layers[-unfrozen_layers:]:
        layer.trainable = True
    
    # Functional API: Define the input layer
    inputs = layers.Input(shape=input_shape)
    
    # Use MobileNetV2 as the base model
    x = mobilenet(inputs)
    
    # Apply the inverted residual block
    x = inverted_residual_block(x, output_channels=512, kernel_size=kernel_size, alpha=alpha)
    
    # Add the rest of the layers
    x = layers.MaxPooling2D(pool_size=(7, 7))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model using the Functional API
    model = models.Model(inputs, x)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def print_model_summary(model):
    """
    Print a summary of the model's architecture.
    
    Args:
        model: TensorFlow Keras model.
    """
    model.summary()
    
    # Count trainable and non-trainable parameters
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    
    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")

def verify_trainable_layers(model, base_model_name='mobilenet'):
    """
    Verify which layers are trainable in the model.
    
    Args:
        model: TensorFlow Keras model.
        base_model_name (str): Name of the base model layer.
    """
    # Find the base model
    base_model = None
    for layer in model.layers:
        if base_model_name.lower() in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print(f"Base model with name containing '{base_model_name}' not found.")
        return
    
    # Print trainable status for each layer in the base model
    for i, layer in enumerate(base_model.layers):
        print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")
