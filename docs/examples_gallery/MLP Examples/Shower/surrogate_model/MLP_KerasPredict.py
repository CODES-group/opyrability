import os
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class KerasPipelineModel:
    def __init__(self, model_path, scalerX_path, scalerY_path):
        # Carrega o modelo do Keras
        self.model = load_model(model_path)

        # Carrega o scaler dos dados de entrada
        with open(scalerX_path, 'rb') as file:
            self.scalerX = pickle.load(file)

        # Carrega o scaler dos dados de saída
        with open(scalerY_path, 'rb') as file:
            self.scalerY = pickle.load(file)

    def predict(self, input_data):
        # 1. Escalar os dados de entrada
        #print(input_data)

        X_valid = self.scalerX.transform(input_data)
        #print(X_valid)

        # 2. Fazer a predição com o modelo
        ypred_Scaled = self.model.predict(X_valid)
        #print(ypred_Scaled)

        # 3. Inverter a escala dos dados de saída
        ypred = self.scalerY.inverse_transform(ypred_Scaled)
        #print(ypred)

        return ypred


def PredictValues(input_data, Tensor=False):
    model_path = 'surrogate_model/Keras_MLP_Surrogate.keras'
    scalerX_path = 'surrogate_model/scalerX.pkl'
    scalerY_path = 'surrogate_model/scalerY.pkl'

    # Criar uma instância do KerasPipelineModel
    pipeline_model = KerasPipelineModel(model_path, scalerX_path, scalerY_path)

    ypred = pipeline_model.predict(input_data)

    if Tensor == True:
        return ypred
    else:
        return ypred[0]