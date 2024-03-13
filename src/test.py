from data_loader.tensorflow_data_loader import TensorFlowDataLoaderFactory
from config import paths
from image_classification.tensorflow_classifier import TensorFlowImageClassifier

x = TensorFlowDataLoaderFactory()

train, val = x.create_train_and_valid_data_loaders(train_data_dir_path=paths.TRAIN_DIR)

# print(train)

model = TensorFlowImageClassifier(3)
model.fit(train_data=train)
