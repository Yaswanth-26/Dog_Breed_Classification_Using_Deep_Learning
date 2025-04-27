"""
Tests for dog breed recognition model functionality.
"""
import unittest
import numpy as np
import tensorflow as tf

# Import modules from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import (
    create_baseline_model, 
    create_model_with_unfrozen_layers, 
    create_optimized_model,
    create_model_with_inverted_residual
)

class TestDogBreedModels(unittest.TestCase):
    """Test cases for dog breed recognition model creation and basic functionality."""

    def test_baseline_model_creation(self):
        """Test that the baseline model can be created."""
        model = create_baseline_model()
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape for 120 dog breeds
        self.assertEqual(model.output_shape, (None, 120))
    
    def test_model_with_unfrozen_layers(self):
        """Test that the model with unfrozen layers can be created."""
        model = create_model_with_unfrozen_layers(
            unfrozen_layers=10, 
            alpha=1.5, 
            dropout_rate=0.2
        )
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 120))
    
    def test_optimized_model(self):
        """Test that the optimized model can be created."""
        model = create_optimized_model(
            unfrozen_layers=50,
            alpha=1.5,
            dropout_rate=0.5
        )
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 120))
    
    def test_inverted_residual_model(self):
        """Test that the model with inverted residual block can be created."""
        model = create_model_with_inverted_residual(
            unfrozen_layers=50,
            alpha=1.5,
            dropout_rate=0.5,
            kernel_size=5
        )
        self.assertIsInstance(model, tf.keras.Model)
        # Check that the model has the expected output shape
        self.assertEqual(model.output_shape, (None, 120))
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        model = create_baseline_model()
        
        # Create a small batch of dummy data
        batch_size = 2
        dummy_input = np.random.random((batch_size, 224, 224, 3))
        
        # Perform a forward pass
        output = model(dummy_input, training=False)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 120))
        
        # Check that output is a probability distribution (sums to 1)
        for i in range(batch_size):
            self.assertAlmostEqual(np.sum(output[i]), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
