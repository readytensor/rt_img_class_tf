import os
import keras
import joblib
import warnings
import numpy as np
from typing import Tuple
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from logger import get_logger
from keras.optimizers.schedules import (
    ExponentialDecay,
    LearningRateSchedule,
    PolynomialDecay,
    PiecewiseConstantDecay,
    CosineDecay,
)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


IS_GPU_AVAI = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)
logger.info(IS_GPU_AVAI)


def get_optimizer(optimizer: str):
    supported_optimizers = {"adam": optimizers.Adam, "sgd": optimizers.SGD}

    if optimizer not in supported_optimizers:
        raise ValueError(
            f"{optimizer} is not a supported optimizer. "
            f"Supported: {supported_optimizers}"
        )
    return supported_optimizers[optimizer]


def get_lr_scheduler(scheduler: str) -> LearningRateSchedule:

    supported_schedulers = {
        "exponential": ExponentialDecay,
        "polynomial": PolynomialDecay,
        "cosine": CosineDecay,
        "piecewise_constant": PiecewiseConstantDecay,
    }
    if scheduler not in supported_schedulers.keys():
        raise ValueError(
            f"{scheduler} is not a supported scheduler. Supported: {supported_schedulers}"
        )
    return supported_schedulers[scheduler]


class ImageClassifier:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        model: keras.models.Model,
        lr: float = 0.01,
        optimizer: str = "adam",
        dense_layer_units: int = 512,
        max_epochs: int = 10,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: dict = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model = model
        self.lr = lr
        self.optimizer_str = optimizer
        self.dense_layer_units = dense_layer_units
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_str = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.kwargs = kwargs

        if self.lr_scheduler_str is not None:
            scheduler = get_lr_scheduler(self.lr_scheduler_str)(
                **self.lr_scheduler_kwargs
            )
            self.optimizer = get_optimizer(self.optimizer_str)(learning_rate=scheduler)
        else:
            self.optimizer = get_optimizer(self.optimizer_str)(learning_rate=self.lr)

        self.model.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(
        self, train_data: tf.data.Dataset, valid_data: tf.data.Dataset = None
    ) -> dict:
        callbacks = []
        if self.early_stopping:
            early_stopper = EarlyStopping(
                monitor="val_loss" if valid_data is not None else "loss",
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_delta,
                verbose=1,
                mode="auto",
            )
            callbacks.append(early_stopper)

        history = self.model.fit(
            train_data,
            validation_data=valid_data,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def predict(self, data):
        """
        Predicts outputs based on the input data using the trained model.

        Args:
        - data: Input data for prediction. This can be a `tf.data.Dataset`, a NumPy array,
                or a TensorFlow tensor.

        Returns:
        - predictions: The predicted values.
        """
        probabilities = self.model.predict(data)
        labels = probabilities.argmax(axis=-1)

        return labels, probabilities

    def save(self, predictor_dir_path: str) -> None:
        """
        Saves the model's state dictionary and training parameters to the specified path.

        This method saves two files:
        one with the model's parameters (such as learning rate, number of classes, etc.
        and another with the model's state dictionary. The parameters are
        saved in a joblib file, and the model's state is saved in a PyTorch file.

        Args:
        - predictor_path (str): The directory path where the model parameters
          and state are to be saved.
        """
        model_params = {
            "model_name": self.model_name,
            "lr": self.lr,
            "optimizer": self.optimizer_str,
            "lr_scheduler": self.lr_scheduler_str,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "max_epochs": self.max_epochs,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_delta": self.early_stopping_delta,
            "num_classes": self.num_classes,
        }
        params_path = os.path.join(predictor_dir_path, "model_params.joblib")
        model_path = os.path.join(predictor_dir_path, "model_state.keras")
        joblib.dump(model_params, params_path)
        self.model.save(model_path)

    @classmethod
    def load(cls, predictor_dir_path: str) -> "ImageClassifier":
        """
        Loads a pretrained model and its training configuration from a specified path.

        Args:
        - predictor_dir_path (str): Path to the directory with model's parameters and state.

        Returns:
        - ImageClassifier: A trainer object with the loaded model and training configuration.
        """
        params_path = os.path.join(predictor_dir_path, "model_params.joblib")
        model_path = os.path.join(predictor_dir_path, "model_state.keras")
        params = joblib.load(params_path)
        model = load_model(model_path)

        classifier = ImageClassifier(model=model, **params)
        classifier.model = model
        return classifier


def train_predictor_model(
    model_name: str,
    train_data: tf.data.Dataset,
    hyperparameters: dict,
    num_classes: int,
    valid_data: tf.data.Dataset = None,
) -> ImageClassifier:
    """
    Instantiate and train the classifier model.

    Args:
        train_data (DataLoader): The training data.
        hyperparameters (dict): Hyperparameters for the model.
        num_classes (int): Number of classes in the classificatiion problem.
        valid_data (DataLoader): The validation data.

    Returns:
        'ImageClassifier': The ImageClassifier model
    """

    if model_name.startswith("resnet"):
        from models.resnet import ResNet

        constructor = ResNet
    elif model_name.startswith("inception"):
        from models.inception import Inception

        constructor = Inception

    model = constructor(
        model_name=model_name,
        num_classes=num_classes,
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
    )
    return model


def predict_with_model(
    model: ImageClassifier, test_data: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions.

    Args:
        model (ImageClassifier): The ImageClassifier model.
        test_data (tf.data.Dataset): The test input data for model.

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
