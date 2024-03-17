import keras
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras import layers, models
from prediction.predictor_model import ImageClassifier


def get_model(
    model_name: str, num_classes: int, pretrained=True, dense_layer_units: int = 2048
) -> keras.models.Model:
    """
    Retrieves a specified ResNet model by name, optionally loading it with pretrained weights,
    and adjusts its fully connected layer to match the specified number of output classes.

    Args:
    - model_name (str): Name of the ResNet model to retrieve ('resnet50', 'resnet101', 'resnet152').
    - num_classes (int): Number of classes for the new fully connected layer.
    - pretrained (bool, optional): Whether to load the model with pretrained weights. Defaults to True.
    - dense_layer_units (int, optional): Number of units in the fully connected layer. Defaults to 512.

    Returns:
    - keras.models.Model: The modified ResNet model with the updated fully connected layer.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
    supprotred_models = {
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
    }

    if model_name in supprotred_models.keys():
        model = supprotred_models[model_name]
        weights = "imagenet" if pretrained else None
        model = model(weights=weights, include_top=False, input_shape=(224, 224, 3))
        x = layers.GlobalAveragePooling2D()(model.output)
        x = layers.Dense(dense_layer_units, activation="relu")(x)
        output = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs=model.input, outputs=output)
    raise ValueError(f"Invalid model name. Supported models {supprotred_models.keys()}")


class ResNet(ImageClassifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float = 0.01,
        optimizer: str = "adam",
        dense_layer_units: int = 2048,
        max_epochs: int = 10,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.05,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: dict = None,
        **kwargs,
    ):
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            dense_layer_units=dense_layer_units,
        )
        super().__init__(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            lr=lr,
            optimizer=optimizer,
            dense_layer_units=dense_layer_units,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            early_stopping_delta=early_stopping_delta,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **kwargs,
        )
