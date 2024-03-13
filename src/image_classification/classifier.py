import os
from typing import Tuple, Any, Union
import numpy as np

from image_classification.tensorflow_classifier import ImageClassifier


def train_predictor_model(
    train_data: Any,
    hyperparameters: dict,
    num_classes: int,
    valid_data: Union[Any, None] = None,
) -> ImageClassifier:
    """
    Instantiate and train the classifier model.

    Args:
        train_data (Any): The training data loader object.
        hyperparameters (dict): Hyperparameters for the model.
        num_classes (int): Number of classes in the classificatiion problem.
        valid_data (Union[Any, None]): The validation data object or None.

    Returns:
        'ImageClassifier': The ImageClassifier model
    """
    model = ImageClassifier(
        num_classes=num_classes,
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
    )
    return model


def predict_with_model(
    model: ImageClassifier, test_data: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions.

    Args:
        model (ImageClassifier): The ImageClassifier model.
        test_data (DataLoader): The test input data for model.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (predicted class labels, predicted class probabilites).
    """
    labels, probabilites = model.predict(test_data)
    return labels, probabilites


def save_predictor_model(model: ImageClassifier, predictor_dir_path: str) -> None:
    """
    Save the ImageClassifier model to disk.

    Args:
        model (ImageClassifier): The Classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> ImageClassifier:
    """
    Load the ImageClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        ImageClassifier: A new instance of the loaded ImageClassifier model.
    """
    return ImageClassifier.load(predictor_dir_path)


def evaluate_predictor_model(model: ImageClassifier, test_data: Any) -> float:
    """
    Evaluate the ImageClassifier model and return the loss.

    Args:
        model (ImageClassifier): The Classifier model.
        test_data (Any): The dataset to be evaluate the model on.

    Returns:
        float: The computed loss on the dataset.
    """
    return model.evaluate(test_data)
