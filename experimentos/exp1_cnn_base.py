import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import kagglehub

# 1. Configurações Iniciais e Download do Dataset
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 10

# Caminho local conhecido para evitar falhas de conexão com o Kaggle
LOCAL_DATASET_PATH = r"C:\Users\Usuario\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"

if os.path.exists(LOCAL_DATASET_PATH):
    print("Usando dataset da pasta cache local...")
    DATASET_PATH = LOCAL_DATASET_PATH
else:
    print("Verificando e baixando dataset do Kaggle...")
    DATASET_PATH = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print(f"Caminho do dataset: {DATASET_PATH}")

# 2. Carregamento e Pré-processamento dos Dados
print("Carregando imagens das pastas...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
  os.path.join(DATASET_PATH, 'Training'),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
  os.path.join(DATASET_PATH, 'Training'),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
  os.path.join(DATASET_PATH, 'Testing'),
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE,
  shuffle=False # Importante não embaralhar para avaliar as métricas corretamente depois
)

# Normalização dos pixels (escala de 0 a 1)
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# 3. Definição da Arquitetura do Modelo CNN (Base)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax') # 4 classes de saída
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("\nResumo da Arquitetura CNN:")
model.summary()

# 4. Treinamento do Modelo
print("\nIniciando o treinamento...")
history = model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=EPOCHS
)

# 5. Avaliação e Cálculo das Métricas (Scikit-Learn)
print("\nRealizando predições no conjunto de teste...")
y_true = []
y_pred_probs = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

# Pega a classe com maior probabilidade
y_pred = np.argmax(y_pred_probs, axis=1)

# Cálculo exigido pelo trabalho
acuracia = accuracy_score(y_true, y_pred)
# Como é um problema multiclasse, usamos macro ou weighted. "macro" é o padrão para média simples das classes.
precisao = precision_score(y_true, y_pred, average='macro') 
recall = recall_score(y_true, y_pred, average='macro')
f_measure = f1_score(y_true, y_pred, average='macro')

print("\n--- RESULTADOS FINAIS DO EXPERIMENTO 1 ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-Mesure:  {f_measure:.4f}")
print("------------------------------------------")
