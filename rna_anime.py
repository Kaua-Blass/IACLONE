# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import ast

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =====================================================
# 0) CRIAR PASTA DE SAÍDA
# =====================================================
if not os.path.exists("resultados"):
    os.makedirs("resultados")

# =====================================================
# FUNÇÃO PARA CONVERTER AS LISTAS DO CSV
# =====================================================
def parse_list(texto):
    try:
        valor = ast.literal_eval(texto)
        if isinstance(valor, list):
            return " ".join(valor)   
        return str(texto)
    except:
        return str(texto)

# =====================================================
# 1) CARREGAR DATASET
# =====================================================
df = pd.read_csv("anime_dataset.csv")
print("Formato original:", df.shape)

# =====================================================
# 2) PROCESSAR GÊNEROS E ESTÚDIOS
# =====================================================
df['genres'] = df['genres'].apply(parse_list)
df['studios'] = df['studios'].apply(parse_list)

df = df[['score', 'episodes', 'members', 'year', 'genres', 'studios']]
df = df.dropna()

df = pd.get_dummies(df, columns=['genres', 'studios'])

print("Formato após processamento:", df.shape)

# =====================================================
# 3) SEPARAR X E y
# =====================================================
y = df['score'].values
X = df.drop(['score'], axis=1)
colunas_X = X.columns.tolist()
X = X.values

# =====================================================
# 4) NORMALIZAR X e y
# =====================================================
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  

# =====================================================
# 5) TREINO / TESTE
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 6) MODELO MLP (com dropout)
# =====================================================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

print(model.summary())

# =====================================================
# 7) TREINAMENTO 
# =====================================================
history = model.fit(
    X_train,
    y_train,
    epochs=200,  
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 8) GRÁFICO MSE
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.legend()
plt.xlabel("Épocas")
plt.ylabel("Erro MSE")
plt.title("Evolução do Erro MSE")
plt.savefig("resultados/treinamento_mse.png", dpi=300)
plt.close()

# =====================================================
# 9) AVALIAÇÃO
# =====================================================
pred = model.predict(X_test).flatten()

pred_desnorm = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
y_test_desnorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test_desnorm, pred_desnorm)
r2 = r2_score(y_test_desnorm, pred_desnorm)
mae = mean_absolute_error(y_test_desnorm, pred_desnorm)

print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.4f}")

# =====================================================
# 10) GRÁFICO REAL vs PREDITO
# =====================================================
plt.figure(figsize=(6,6))
plt.scatter(y_test_desnorm, pred_desnorm, s=10)
plt.xlabel("Score real")
plt.ylabel("Score predito")
plt.title("Real vs Predito")
plt.savefig("resultados/real_vs_predito.png", dpi=300)
plt.close()

# =====================================================
# 11) SALVAR MODELO E SCALERS
# =====================================================
model.save("resultados/modelo_prever_score.h5")
joblib.dump(scaler_X, "resultados/scaler_X.joblib")
joblib.dump(scaler_y, "resultados/scaler_y.joblib")
joblib.dump(colunas_X, "resultados/labels_X.joblib")

print("\nModelo e scalers salvos na pasta 'resultados/'")

# =====================================================
# 12) TESTE MANUAL (MENU INTERATIVO)
# =====================================================
def prever_score_terminal():

    print("\n==== TESTE MANUAL DE PREDIÇÃO DO SCORE ====\n")

    scaler_X = joblib.load("resultados/scaler_X.joblib")
    scaler_y = joblib.load("resultados/scaler_y.joblib")
    labels_X = joblib.load("resultados/labels_X.joblib")

    vetor = {col: 0 for col in labels_X}

    # =================== VALORES NUMÉRICOS ===================
    print("\nDigite valores numéricos:\n")
    for col in labels_X:
        if col in ["episodes", "members", "year"]:
            while True:
                try:
                    v = float(input(f"{col}: "))
                    vetor[col] = v
                    break
                except:
                    print("Valor inválido.")

    # =================== GÊNEROS POR TEXTO ===================
    print("\nDigite os gêneros separados por vírgula (ex: action, romance, comedy)")
    entrada_generos = input("Gêneros: ").strip().lower()
    generos_usuario = [g.strip() for g in entrada_generos.split(",") if g.strip()]

    generos_setados = []
    for g in generos_usuario:
        matched = False
        for col in labels_X:
            if col.startswith("genres_"):
                nome = col.replace("genres_", "").lower()
                if g == nome:  
                    vetor[col] = 1
                    generos_setados.append(col.replace("genres_", ""))
                    matched = True
                    break
        if not matched:
            print(f"Gênero '{g}' não encontrado. Ignorado.")

    print(f"Gêneros setados: {', '.join(generos_setados) if generos_setados else 'Nenhum'}")

    # =================== ESTÚDIOS POR MENU ===================
    colunas_studios = [c for c in labels_X if c.startswith("studios_")]

    print("\n=== ESTÚDIOS DISPONÍVEIS ===")
    for i, col in enumerate(colunas_studios):
        print(f"{i+1}. {col.replace('studios_', '')}")

    escolha = input("\nEscolha o(s) estúdio(s) (ex: 1,4): ").strip()

    studios_setados = []
    if escolha:
        try:
            indices = [int(x.strip()) - 1 for x in escolha.split(",")]
            for idx in indices:
                if 0 <= idx < len(colunas_studios):
                    vetor[colunas_studios[idx]] = 1
                    studios_setados.append(colunas_studios[idx].replace("studios_", ""))
                else:
                    print(f"Índice {idx+1} inválido. Ignorado.")
        except:
            print("Entrada inválida para estúdios.")

    print(f"Estúdios setados: {', '.join(studios_setados) if studios_setados else 'Nenhum'}")

    # =================== Montar vetor final na ordem correta ===================
    vetor_lista = np.array([[vetor[col] for col in labels_X]])

    vetor_lista = scaler_X.transform(vetor_lista)

    pred_norm = model.predict(vetor_lista).flatten()[0]

    pred = scaler_y.inverse_transform(np.array([[pred_norm]])).flatten()[0]

    print("\n===== RESULTADO =====")
    print(f"Score previsto: {pred:.2f}")

    return pred

# =====================================================
# 13) EXECUTAR TESTE MANUAL
# =====================================================
import sys
if len(sys.argv) > 1 and sys.argv[1] == "--teste":
    prever_score_terminal()
    exit()