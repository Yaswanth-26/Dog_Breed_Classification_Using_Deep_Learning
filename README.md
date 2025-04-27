# Dog Breed Recognition for Lost/Found Services and Pet Shop Management

## Overview

This project addresses the challenge of efficiently and accurately classifying dog breeds in resource-constrained environments, such as mobile devices or lightweight platforms. Using the MobileNetV2 architecture, the solution provides practical and accessible dog breed classification capabilities.

The model achieves 80% accuracy on the Stanford Dogs Dataset, while maintaining a lightweight architecture suitable for deployment on mobile devices and other resource-limited environments.

## Dataset

The project uses the Stanford Dogs Dataset for training and evaluation:

- **Dataset Link**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Number of Images**: 20,580 images across 120 different dog breeds
- **Training Set**: 12,000 images
- **Testing Set**: 8,580 images
- **Features**: Includes bounding box annotations for precise dog localization

## Key Features

- **Lightweight Architecture**: Based on MobileNetV2, optimized for mobile and embedded devices
- **Improved Activation Function**: Uses ELU (Exponential Linear Unit) activation with optimized alpha parameter to prevent dead neurons
- **Data Preprocessing**: Implements bounding box cropping, data augmentation, and standardization
- **Custom Layers**: Adds convolutional layer with batch normalization and maxpooling for enhanced feature extraction
- **Regularization**: Utilizes dropout with optimized rate to prevent overfitting
- **Fine-tuning**: Various layers unfrozen and fine-tuned for improved performance

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Added Layers**: 
  - Conv2D (512 filters, 5x5 kernel, 'same' padding)
  - BatchNormalization
  - ELU (alpha=1.5)
  - MaxPooling2D (7x7)
  - Flatten
  - Dropout (0.5)
  - Dense (120, softmax)

## Performance

- **Accuracy**: 80% on test set
- **Training**: 50 epochs with early stopping (patience=5)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dog-breed-recognition.git
cd dog-breed-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

```python
from src.data_loading import load_dataset, preprocess_data

# Load and preprocess the data
train_ds, test_ds = load_dataset('path/to/dataset')
```

### Training

```python
from src.models import create_model
from src.training import train_model

# Create the model
model = create_model()

# Train the model
history = train_model(model, train_ds, test_ds, epochs=50)
```

### Evaluation

```python
from src.evaluation import evaluate_model, visualize_results

# Evaluate the model
results = evaluate_model(model, test_ds)

# Visualize results
visualize_results(history, results)
```

## Project Structure

```
dog-breed-recognition/
├── README.md                 # Project documentation
├── data/                     # Data directory
├── notebooks/                # Jupyter notebooks
│   └── original_analysis.ipynb
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loading.py       # Data loading and preprocessing functions
│   ├── models.py             # Model architecture definitions
│   ├── training.py           # Training functions
│   └── evaluation.py         # Evaluation and visualization functions
├── requirements.txt          # Project dependencies
└── setup.py                  # Package installation
```

## Citation

If you use this code or model in your research, please cite:

```
@article{Stanford_Dogs_Dataset,
  author = {Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
  title = {Novel Dataset for Fine-Grained Image Categorization},
  journal = {First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2011}
}
```

## License

MIT

## Contributors

- Your Name