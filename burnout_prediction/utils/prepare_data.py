import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prepare_data():
    dataset_path = '../dataset/mental_health_burnout_prediction_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Remover colunas desnecessárias
    df = df.drop(columns=['Person_ID', 'AI_Wellness_Recommendation'])

    # Binarizar o Target
    df['Burnout_Risk_Binary'] = df['Burnout_Risk'].apply(lambda x: 1 if x in ['High', 'Critical'] else 0)
    
    X = df.drop(columns=['Burnout_Risk', 'Burnout_Score', 'Burnout_Risk_Binary'])
    y = df['Burnout_Risk_Binary']

    # Identificar colunas numéricas e categóricas
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

    # Tratar valores nulos (estritamente pandas)
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
    
    # Converter categóricas (One-Hot Encoding)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Divisão Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Salvar dados originais (antes de normalizar)
    os.makedirs('../dataset/train', exist_ok=True)
    os.makedirs('../dataset/test', exist_ok=True)
    X_train.to_csv('../dataset/train/input_train.csv', index=False)
    y_train.to_csv('../dataset/train/output_train.csv', index=False)
    X_test.to_csv('../dataset/test/input_test.csv', index=False)
    y_test.to_csv('../dataset/test/output_test.csv', index=False)
    
    print("Dados preparados e salvos na pasta dataset.")

if __name__ == '__main__':
    prepare_data()
