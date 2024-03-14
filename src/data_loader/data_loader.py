import os
import joblib
import numpy as np
import pandas as pd
from data_loader.base_loader import AbstractDataLoaderFactory
from pathlib import Path
from typing import Tuple, List, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TensorFlowDataLoaderFactory(AbstractDataLoaderFactory):
    def __init__(
        self,
        batch_size: int = 64,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        image_size: Tuple[int, int] = (224, 224),
        validation_size: float = 0.0,
        shuffle_train: bool = True,
        random_state: int = 42,
    ):
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.image_size = image_size
        self.validation_size = validation_size
        self.shuffle_train = shuffle_train
        self.random_state = random_state

    def _preprocess_function(self, x):
        x = (x - self.mean) / self.std
        return x

    @staticmethod
    def stratified_split(dataset_path: str, valid_size: float):
        class_names = [i for i in os.listdir(dataset_path) if not i.startswith(".")]
        file_paths = []
        labels = []

        # Generate file paths and labels
        for class_name in class_names:
            class_dir = os.path.join(dataset_path, class_name)
            class_file_paths = [
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if not fname.startswith(".")
            ]
            class_labels = [class_name] * len(class_file_paths)

            file_paths.extend(class_file_paths)
            labels.extend(class_labels)

        # Convert to numpy arrays for compatibility with train_test_split
        file_paths = np.array(file_paths)
        labels = np.array(labels).astype(str)

        # Split data into training and validation sets in a stratified manner
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, labels, test_size=valid_size, stratify=labels
        )

        return train_files, val_files, train_labels, val_labels

    def create_train_and_valid_data_loaders(
        self,
        train_dir_path: str,
        validation_dir_path: Optional[str] = None,
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """
        Creates TensorFlow data loaders (tf.data.Dataset) for training and, if
        specified, validation datasets.

        Args:
            train_data_dir_path: The path to the training dataset directory.
            validation_dir_path: Optional; the path to the validation dataset
                                 directory.

        Returns:
            A tuple of tf.data.Datasets for the training and validation datasets.
            The validation dataset is None if no validation dataset is provided or
            created.
        """

        val_loader = None
        if validation_dir_path is None and self.validation_size > 0:
            train_files, val_files, train_labels, val_labels = (
                TensorFlowDataLoaderFactory.stratified_split(
                    dataset_path=train_dir_path, valid_size=self.validation_size
                )
            )
            datagen = ImageDataGenerator(
                preprocessing_function=self._preprocess_function
            )
            train_loader = datagen.flow_from_dataframe(
                dataframe=pd.DataFrame(
                    {"filename": train_files, "class": train_labels}
                ),
                directory=None,  # 'directory' is None because paths are absolute
                x_col="filename",
                y_col="class",
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="sparse",
                shuffle=self.shuffle_train,
            )

            val_loader = datagen.flow_from_dataframe(
                dataframe=pd.DataFrame({"filename": val_files, "class": val_labels}),
                directory=None,
                x_col="filename",
                y_col="class",
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="sparse",
            )
        else:

            datagen = ImageDataGenerator(
                preprocessing_function=self._preprocess_function
            )

            train_loader = datagen.flow_from_directory(
                train_dir_path,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode="sparse",
                shuffle=self.shuffle_train,
            )
            if validation_dir_path is not None:
                val_loader = datagen.flow_from_directory(
                    validation_dir_path,
                    target_size=self.image_size,
                    batch_size=self.batch_size,
                    class_mode="sparse",
                    shuffle=True,
                )

        self.num_classes = len(train_loader.class_indices)
        self.class_to_idx = train_loader.class_indices
        return train_loader, val_loader

    def create_test_data_loader(self, data_dir_path: str) -> tf.data.Dataset:
        """
        Create a TensorFlow DataLoader for test data.

        Args:
            data_dir_path: Path to the test dataset directory.

        Returns:
            A tf.data.Dataset for test data.
        """
        test_datagen = ImageDataGenerator(
            preprocessing_function=self._preprocess_function
        )
        test_loader = test_datagen.flow_from_directory(
            data_dir_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False,  # No need to shuffle test data
        )
        return test_loader

    @staticmethod
    def get_file_names(loader):
        return [i.split("/")[1] for i in loader.filenames]

    def save(self, file_path: str) -> None:
        """
        Save the data loader factory to a file.

        Args:
            file_path (str): The path to the file where the data loader factory will
                             be saved.
        """
        path = Path(file_path)
        directory_path = path.parent
        os.makedirs(directory_path, exist_ok=True)
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> "TensorFlowDataLoaderFactory":
        """
        Load the data loader factory from a file.
        """
        return joblib.load(file_path)


def get_data_loader(model_name: str) -> TensorFlowDataLoaderFactory:
    ordinary = {
        "resnet50",
        "resnet101",
        "resnet152",
    }
    inception = {"inceptionV3"}
    supported = ordinary | inception
    if model_name in ordinary:
        return OrdinaryDataLoader
    if model_name in inception:
        return InceptionV3DataLoader

    raise ValueError(f"Invalid model name. supported model names: {supported}")


class OrdinaryDataLoader(TensorFlowDataLoaderFactory):
    mean: List[float] = [0.485, 0.456, 0.406]
    std: List[float] = [0.229, 0.224, 0.225]
    image_size: Tuple[int, int] = (224, 224)

    def __init__(
        self,
        batch_size: int = 64,
        validation_size: float = 0.0,
        shuffle_train=True,
        random_state: int = 42,
    ):
        super().__init__(
            batch_size=batch_size,
            mean=self.mean,
            std=self.std,
            image_size=self.image_size,
            validation_size=validation_size,
            shuffle_train=shuffle_train,
            random_state=random_state,
        )


class InceptionV3DataLoader(TensorFlowDataLoaderFactory):
    mean: List[float] = [0.485, 0.456, 0.406]
    std: List[float] = [0.229, 0.224, 0.225]
    image_size: Tuple[int, int] = (299, 299)

    def __init__(
        self,
        batch_size: int = 64,
        validation_size: float = 0.0,
        shuffle_train=True,
        random_state: int = 42,
    ):
        super().__init__(
            batch_size=batch_size,
            mean=self.mean,
            std=self.std,
            image_size=self.image_size,
            validation_size=validation_size,
            shuffle_train=shuffle_train,
            random_state=random_state,
        )
