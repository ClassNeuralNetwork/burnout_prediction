import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Carregar os dados de treinamento
print("Carregando dados de treinamento...")
input_train = pd.read_csv('../dataset/train/input_train.csv')
output_train = pd.read_csv('../dataset/train/output_train.csv')

# Carregar os dados de teste
print("Carregando dados de teste...")
input_test = pd.read_csv('../dataset/test/input_test.csv')
output_test = pd.read_csv('../dataset/test/output_test.csv')

# Normalizando os dados
print("Normalizando os dados...")
scaler = StandardScaler()
input_train_scaled = scaler.fit_transform(input_train)
input_test_scaled = scaler.transform(input_test)

# Salvando os dados normalizados
pd.DataFrame(input_train_scaled).to_csv('../dataset/train/input_train_standard.csv', index=False)
pd.DataFrame(input_test_scaled).to_csv('../dataset/test/input_test_standard.csv', index=False)

# Criando e Treinando o Modelo
mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1)

print("Treinando o modelo MLPClassifier...")
# Para o sklearn, o target de output_train precisa ser 1D, por isso values.ravel()
mlp.fit(input_train_scaled, output_train.values.ravel())

print("Accuracy on training set: {:.2f}".format(mlp.score(input_train_scaled, output_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(input_test_scaled, output_test)))

# Plot do Loss
plt.figure()
plt.plot(mlp.loss_curve_)
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
os.makedirs('../plots', exist_ok=True)
plt.savefig('../plots/loss_curve.png')
plt.close()

# Salvando o Loss no model folder como custo.csv
pd.DataFrame({'loss': mlp.loss_curve_}).to_csv('../model/custo.csv', index=False)

# Salvando o modelo
print("Salvando o modelo...")
joblib.dump(mlp, '../model/model.pkl')

print("Finalizado com sucesso.")
