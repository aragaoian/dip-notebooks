from PreProcessing import Preprocessing
from Model import ConvolutionalModel
import os
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")

if __name__ == "__main__":
    # preprocess_data = Preprocessing(
    #     dt_path=f"{BASE_PATH}\\BrailleDataset.zip",
    #     formatted_path=f"{BASE_PATH}\\Braille Dataset",
    #     dt_name="Braille Dataset",
    # )
    # preprocess_data.unzip_folder()
    # X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data.pipeline()

    # model = ConvolutionalModel()
    # model.upload_data(X_train, X_val, X_test, y_train, y_val, y_test)
    # model.model_def(summary=True)
    # model.model_train()
    # model.save_model(
    #     export_path=r"braille-classifier\braille_cnn.keras"
    # )

    model = ConvolutionalModel()
    model.upload_model(f"{BASE_PATH}\\braille_cnn.keras")
    model.model_predict(path=f"{BASE_PATH}\\images\\y.png")
