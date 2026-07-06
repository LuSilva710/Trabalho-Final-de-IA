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
PATCH_SIZE = 25
NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_UNITS = [128, 64]

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

# Normalização dos pixels (escala de 0 a 1)
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# 3. Componentes da Arquitetura Vision Transformer (ViT)

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
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

def mlp(x, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

# 4. Construção do Modelo ViT Base (Sem Regularizações adicionais)
def create_vit_classifier():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Gerar patches e codificá-los com embeddings de posição
    patches = Patches(PATCH_SIZE)(inputs)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)
    
    x = encoded_patches
    # Empilhar os blocos Transformer
    for _ in range(TRANSFORMER_LAYERS):
        # Primeiro sub-bloco: LayerNorm + MultiHeadAttention + Conexão Residual
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        
        # Segundo sub-bloco: LayerNorm + MLP + Conexão Residual
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS)
        x = layers.Add()([x3, x2])
        
    # Normalização e achatamento (Flatten) para classificação final
    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.Flatten()(representation)
    
    # Camada densa de saída com 4 classes (Softmax)
    logits = layers.Dense(4, activation="softmax")(representation)
    
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

model = create_vit_classifier()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

print("\nResumo da Arquitetura ViT Base:")
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

print("\n--- RESULTADOS FINAIS DO EXPERIMENTO 5 ---")
print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-Mesure:  {f_measure:.4f}")
print("------------------------------------------")
