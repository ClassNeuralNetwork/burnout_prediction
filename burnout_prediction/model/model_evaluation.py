from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Carregar o modelo
print("Carregando o modelo...")
model = joblib.load('../model/model.pkl')

# Carregar dados
print("Carregando dados de teste...")
input_test = pd.read_csv('../dataset/test/input_test_standard.csv')
output_test = pd.read_csv('../dataset/test/output_test.csv')
y_test = output_test.values.flatten()

# Previsões no conjunto de teste
output_model_ = model.predict(input_test)

print('\nResultados da Avaliação:')
print('Acurácia:', accuracy_score(y_test, output_model_))
print('Precisão:', precision_score(y_test, output_model_, average='weighted'))
print('Sensibilidade:', recall_score(y_test, output_model_, average='weighted'))
print('F1-Score:', f1_score(y_test, output_model_, average='weighted'))

# Matriz de Confusão
cm = confusion_matrix(y_test, output_model_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sem Risco (0)', 'Com Risco (1)'])
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_title('Matriz de Confusão')
disp.ax_.set_xlabel('Classificação Prevista')
disp.ax_.set_ylabel('Classificação Real')

os.makedirs('../plots', exist_ok=True)
plt.savefig('../plots/confusion_matrix.png')
print("Matriz de confusão salva em 'plots/confusion_matrix.png'.")
