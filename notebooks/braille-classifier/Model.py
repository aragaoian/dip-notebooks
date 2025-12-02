import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2


class ConvolutionalModel:
    def __init__(self, img_size=(28, 28, 1)) -> None:
        self.input_shape = img_size
        self.history = None

    def upload_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def upload_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def save_model(self, export_path):
        self.model.save(export_path)

    def model_def(self, summary=False):
        self.model = models.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                # layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                # layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.4),
                layers.Dense(26, activation="softmax"),
            ]
        )

        self.model.summary() if summary else None

    def model_train(
        self, opt="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    ):
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=30,
            validation_data=(self.X_val, self.y_val),
            batch_size=32,
        )

    def model_stats(self):
        plt.figure()
        plt.plot(self.history.history["loss"], label="Treino")
        plt.plot(self.history.history["val_loss"], label="Validação")
        plt.title("Loss")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.history.history["accuracy"], label="Treino")
        plt.plot(self.history.history["val_accuracy"], label="Validação")
        plt.title("Acurácia")
        plt.xlabel("Época")
        plt.ylabel("Acurácia")
        plt.legend()
        plt.show()

    def model_predict(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.expand_dims(img, axis=-1)  #  (28,28) -> (28,28,1)
        img = np.expand_dims(img, axis=0)  # (28,28,1) -> (1,28,28,1)
        predictions = self.model.predict(img)
        class_id = np.argmax(predictions, axis=1)[0]
        print("Predicted class:", class_id)
