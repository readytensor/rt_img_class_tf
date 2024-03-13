import os
import keras
import joblib
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from logger import get_logger

logger = get_logger(__name__)


IS_GPU_AVAI = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)
logger.info(IS_GPU_AVAI)
print(tf.config.list_physical_devices("GPU"))


def get_optimizer(optimizer: str):
    supported_optimizers = {"adam": optimizers.Adam, "sgd": optimizers.SGD}

    if optimizer not in supported_optimizers:
        raise ValueError(
            f"{optimizer} is not a supported optimizer. "
            f"Supported: {supported_optimizers}"
        )
    return supported_optimizers[optimizer]


class ImageClassifier:
    def __init__(
        self,
        num_classes: int,
        lr: float = 0.01,
        optimizer: str = "adam",
        max_epochs: int = 10,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: dict = None,
        **kwargs,
    ):
        self.lr = lr
        self.optimizer_str = optimizer
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_str = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.kwargs = kwargs

        self.optimizer = get_optimizer(self.optimizer_str)(learning_rate=self.lr)

        self.model = self.build_model()

        self.model.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def build_model(self) -> keras.models.Model:
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(256, activation="relu")(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)
        return models.Model(inputs=base_model.input, outputs=output)

    def fit(
        self, train_data: tf.data.Dataset, valid_data: tf.data.Dataset = None
    ) -> dict:
        callbacks = []
        if self.early_stopping_patience > 0:
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

    @staticmethod
    def load(predictor_dir_path: str) -> "ImageClassifier":
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
        trainer = ImageClassifier(**params)
        trainer.model = model
        return trainer
