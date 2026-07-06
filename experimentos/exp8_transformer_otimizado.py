import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import kagglehub

# 1. Configurações Iniciais e Download do Dataset
BATCH_SIZE = 32
IMG_HEIGHT = 300 # Aumentado para 300x300 como na referência
IMG_WIDTH = 300
EPOCHS = 10
PATCH_SIZE = 30 # Aumentado para 30 para manter o grid de 10x10 patches (100 total)
NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)
PROJECTION_DIM = 128
NUM_HEADS = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_UNITS = [256, 128]
DROPOUT_RATE = 0.1 # Dropout dentro do bloco do Transformer
L2_REG = 1e-4 # Regularização L2 suave para pesos

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
  shuffle=False
)

# Camada de Aumento de Dados de Brilho (RandomBrightness) - simula calibração do scanner de RM
data_augmentation = tf.keras.Sequential([
    layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0), seed=123)
])

# Normalização dos pixels (escala de 0 a 1)
normalization_layer = layers.Rescaling(1./255)

# Aplicando normalização e data augmentation apenas para treino
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# 3. Componentes da Arquitetura Vision Transformer (ViT) Otimizada

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, dropout_rate=0.0, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(
            units=projection_dim,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return self.dropout(encoded)

def mlp(x, hidden_units, dropout_rate=0.0, l2_reg=1e-4):
    for units in hidden_units:
        x = layers.Dense(
            units,
            activation=tf.nn.gelu,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)
    return x

# 4. Construção do Modelo ViT Otimizado
def create_vit_classifier():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Gerar patches e codificá-los com embeddings, L2 e dropout inicial
    patches = Patches(PATCH_SIZE)(inputs)
    encoded_patches = PatchEncoder(
        NUM_PATCHES, 
        PROJECTION_DIM, 
        dropout_rate=DROPOUT_RATE, 
        l2_reg=L2_REG
    )(patches)
    
    x = encoded_patches
    # Empilhar os blocos Transformer
    for _ in range(TRANSFORMER_LAYERS):
        # Primeiro sub-bloco: LayerNorm + MultiHeadAttention + Conexão Residual
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM,
            dropout=DROPOUT_RATE,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Segundo sub-bloco: LayerNorm + MLP + Conexão Residual
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)
        x = layers.Add()([x3, x2])
        
    # Normalização e achatamento (Flatten) para classificação final
    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.Flatten()(representation)
    
    # Cabeçote Densa Final Duplo (Dropout e Densa intermediária) com base na referência do Kaggle
    representation = layers.Dropout(0.3)(representation)
    representation = layers.Dense(
        128, 
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(representation)
    representation = layers.Dropout(0.25)(representation)
    
    # Camada densa de saída com L2 e 4 classes (Softmax)
    logits = layers.Dense(
        4, 
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(representation)
    
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

model = create_vit_classifier()

# Compilação utilizando o otimizador Adamax com lr=0.001 como na referência de 99% do Kaggle
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

print("\nResumo da Arquitetura ViT Otimizada:")
model.summary()

# 5. Treinamento do Modelo
print("\nIniciando o treinamento...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# 6. Avaliação e Cálculo das Métricas (Scikit-Learn)
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
precisao = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f_measure = f1_score(y_true, y_pred, average='macro')

print("\n--- RESULTADOS FINAIS DO EXPERIMENTO 8 ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-Mesure:  {f_measure:.4f}")
print("------------------------------------------")
