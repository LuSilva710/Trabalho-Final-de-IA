# Relatório Técnico: Detecção de Tumores Cerebrais via Aprendizado de Máquina

**Nome:** Ludmila Silva  
**Disciplina:** Inteligência Artificial  
**Tema:** Detecção de Tumores em Imagens de Ressonância Magnética (RM) do Cérebro  
**Link do vídeo:** [Assista no YouTube](https://youtu.be/vePPzw5Lcxo)  
**Repositório Github:** [Trabalho-Final-de-IA](https://github.com/LuSilva710/Trabalho-Final-de-IA.git)  

---

## 1. Introdução e Contexto
O diagnóstico precoce e preciso de tumores cerebrais é crucial na oncologia médica. O uso de redes neurais artificiais e visão computacional tem se mostrado um aliado poderoso para auxiliar médicos radiologistas na triagem e classificação de imagens de ressonância magnética (RM).

Este relatório documenta os experimentos realizados para classificar RMs do cérebro humano em quatro categorias distintas utilizando o dataset **Brain Tumor MRI Dataset**:
* **Glioma** (Tumor das células gliais)
* **Meningioma** (Tumor nas membranas que envolvem o cérebro)
* **Tumor Hipofisário** (Tumor na glândula pituitária)
* **Sem Tumor** (RM saudável)

O conjunto de dados possui um total de 7.200 imagens, divididas entre treino (*Training*) e teste (*Testing*). Implementamos e avaliamos duas famílias fundamentais de arquiteturas de Aprendizado de Máquina: Redes Neurais Convolucionais (CNNs) e Vision Transformers (ViTs), testando 8 variações de parâmetros e técnicas de redução de overfitting (Dropout e L2).

---

## 2. Metodologia e Pré-processamento dos Dados
Para todos os experimentos, adotou-se o seguinte pipeline de processamento de dados em Python com Keras/TensorFlow e Scikit-Learn:
* **Divisão Treino/Validação:** A pasta *Training* foi subdividida em 80% para treinamento (5.712 imagens) e 20% para validação local (1.428 imagens) utilizando uma semente aleatória fixa (`seed=123`).
* **Avaliação Independente:** O conjunto de dados da pasta *Testing* (1.311 imagens) foi mantido estritamente para o cálculo das métricas de teste final, garantindo que o modelo fosse testado em dados não vistos.
* **Normalização de Pixels:** Os valores originais dos pixels das imagens foram normalizados para a escala [0, 1] dividindo por 255 (`layers.Rescaling(1./255)`).
* **Dimensões da Imagem:**
  * **Experimentos 1 a 7:** Imagens redimensionadas para 150×150×3.
  * **Experimento 8:** Imagens redimensionadas para 300×300×3 (com o grid de patches adaptado para manter a consistência dimensional).

---

## 3. Descrição Detalhada dos 8 Experimentos
Abaixo estão explicadas as variações estruturais e os hiperparâmetros de cada um dos 8 experimentos realizados:

### Grupo A: Redes Neurais Convolucionais (CNN)
* **Experimento 1 (CNN Base):** Arquitetura baseline com 3 camadas convolucionais (`Conv2D` com ativação ReLU e filtros `[32, 64, 64]`), cada uma seguida de uma camada de pooling máximo (`MaxPooling2D`). Os mapas de características são achatados (`Flatten`) e conectados a uma camada densa de 64 neurônios, seguida da camada de saída Softmax (4 classes). Sem qualquer regularização de pesos ou dropout.
* **Experimento 2 (CNN com Dropout):** Mesma arquitetura do baseline, mas incorpora uma camada de `Dropout(0.5)` logo após a camada densa de 64 neurônios. O objetivo é desativar aleatoriamente 50% dos neurônios a cada iteração de treino, forçando a rede a aprender caminhos redundantes e mitigar o overfitting.
* **Experimento 3 (CNN com Regularização L2):** Aplicação de regularização nos pesos ($L_2$) com fator de penalização `0.01` em todos os kernels convolucionais e nas camadas densas. A regularização L2 penaliza pesos de valores elevados inserindo a norma quadrada do vetor de pesos no cálculo do loss, incentivando o modelo a manter pesos pequenos e uniformes.
* **Experimento 4 (CNN com Aumento de Neurônios):** Mesma estrutura convolucional da CNN Base, porém a camada densa oculta foi expandida de 64 para 256 neurônios, aumentando significativamente a capacidade não linear do classificador final de características.

### Grupo B: Vision Transformers (ViT)
Dada a ampla utilização e o avanço dos Transformers em visão computacional, implementamos uma arquitetura ViT customizada em Keras contendo camadas de extração de patches e encoders posicionais.
* **Experimento 5 (ViT Base):** Imagens de 150×150 divididas em patches de 25×25 pixels (36 patches por imagem). Os patches são projetados linearmente para uma dimensão de 64 e somados a embeddings posicionais aprendíveis. Passam por 4 blocos de Transformer Encoder (LayerNorm, Multi-Head Attention com 4 cabeças, conexões residuais e sub-bloco MLP de dimensões `[128, 64]`). Sem dropout ou L2.
* **Experimento 6 (ViT com Dropout):** Adição de taxas de dropout estruturadas no Vision Transformer: `Dropout(0.1)` logo após os embeddings de patches, `dropout=0.1` nas camadas de atenção, `Dropout(0.3)` nas camadas internas do MLP do bloco Transformer, e `Dropout(0.5)` no cabeçote de saída.
* **Experimento 7 (ViT com Regularização L2):** Aplicação de penalização de peso L2 (`0.01`) nas projeções de patches, no Multi-Head Attention e em todas as camadas lineares da rede, mantendo a estrutura dimensional básica do ViT Base.
* **Experimento 8 (ViT Otimizado):** Modelo expandido com maior resolução espacial e hiperparâmetros inspirados em técnicas topo de ranking:
  * **Resolução de imagem:** 300×300 pixels.
  * **Tamanho de patch:** 30×30 (gerando 100 patches no total, melhorando a granularidade espacial).
  * **Dimensão de projeção:** 128 (8 cabeças de atenção).
  * **Otimizador:** Adamax (learning rate = 0.001) para maior estabilidade.
  * **Aumento de dados (Data Augmentation):** nativo focado em brilho (`layers.RandomBrightness`), simulando variações de calibração entre diferentes scanners de ressonância magnética reais.
  * **Cabeçote Densa duplo com dropouts intercalados:** `Dropout(0.3) -> Dense(128, relu) -> Dropout(0.25) -> Dense(4, softmax)`.
  * **Regularização L2 mais suave:** (1e-4).

---

## 4. Tabela de Resultados Consolidados
Após executar todos os experimentos no conjunto de teste independente (1.311 imagens), as métricas macro foram compiladas na tabela a seguir:

| Experimento | Arquitetura | Variação / Parâmetros | Acurácia | Precisão | Recall | F-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | CNN | Baseline (3 Conv, 64 Neurônios, Sem Regularização) | 0.8719 | 0.8736 | 0.8719 | 0.8698 |
| **Exp 2** | CNN | Dropout (0.5) na camada densa final | 0.8650 | 0.8697 | 0.8650 | 0.8630 |
| **Exp 3** | CNN | Regularização L2 (0.01) nas camadas Conv e Densa | 0.8006 | 0.8044 | 0.8006 | 0.7902 |
| **Exp 4** | CNN | Aumento de neurônios na Densa (256) | **0.8781** | **0.8827** | **0.8781** | **0.8753** |
| **Exp 5** | ViT | Baseline (Patch 25x25, 4 blocos, Sem Regularização) | 0.8356 | 0.8387 | 0.8356 | 0.8323 |
| **Exp 6** | ViT | Dropout (0.1 no Transformer, 0.3 no MLP, 0.5 no Cabeçote) | 0.7819 | 0.7818 | 0.7819 | 0.7747 |
| **Exp 7** | ViT | Regularização L2 (0.01) nas projeções e MLP | 0.7706 | 0.7742 | 0.7706 | 0.7600 |
| **Exp 8** | ViT | Otimizado (300x300, Adamax, Brilho, Densa Dupla, L2 1e-4) | 0.7612 | 0.7565 | 0.7613 | 0.7471 |

---

## 5. Análise Comparativa e Conjecturas Acadêmicas

### Qual foi a melhor abordagem aplicada?
A melhor abordagem geral foi o **Experimento 4 (CNN com Aumento de Neurônios na Densa)**, alcançando **87.81% de Acurácia** e **87.53% de F-Score** no conjunto de teste. No grupo dos Vision Transformers, a melhor configuração foi o **Experimento 5 (ViT Base)**, com **83.56% de Acurácia**.

### Conjectura 1: Eficiência de Dados e Vieses Indutivos (CNN vs. ViT)
O fato de as CNNs terem superado sistematicamente os Vision Transformers no teste reflete um princípio fundamental do aprendizado profundo: os **vieses indutivos (*inductive biases*)**.
* **CNNs:** Possuem o viés indutivo de localidade (pixels adjacentes são correlacionados) e invariância à translação (uma característica detectada no topo esquerdo é igual no canto inferior). Isso torna a CNN altamente eficiente ao aprender com poucos dados (o dataset tem 5.712 imagens de treino).
* **Vision Transformers (ViTs):** Não possuem esse viés embutido. Eles dependem do mecanismo de auto-atenção global para aprender as relações de distância entre os patches do zero. Por isso, ViTs são tradicionalmente conhecidos por sofrer de subajuste (*underfitting*) quando treinados do zero em datasets pequenos/médios, necessitando de bases muito maiores (como ImageNet) ou pré-treinamento maciço para superar as CNNs.

### Conjectura 2: O Efeito da Regularização (Dropout e L2) no Tempo de Treinamento
Em ambos os modelos (CNN e ViT), a introdução de Dropout ou L2 diminuiu as métricas obtidas na 10ª época (comparando Exp 2 e Exp 3 contra Exp 1; e Exp 6 e Exp 7 contra Exp 5). Isso ocorre porque a regularização adiciona ruído ou restringe os valores de parâmetros para prevenir o overfitting no longo prazo. Consequentemente, o modelo treina de forma mais lenta e exige muito mais épocas (ex: 30 a 50 épocas) para atingir sua convergência e superioridade. Em um limite curto de apenas 10 épocas, as versões regularizadas sofrem de subajuste relativo (*underfit* temporal), não superando os modelos sem regularização.

### Conjectura 3: Impacto do Experimento 8 (ViT Otimizado)
O Experimento 8 possui uma capacidade teórica muito maior (4.3 milhões de parâmetros, imagens a 300×300 e data augmentation de brilho). No entanto, sua acurácia de teste foi a menor do grupo ViT (76.12%). Conjectura-se que a dimensão de entrada maior aumenta drasticamente a complexidade do espaço de busca dos pesos. Para um Vision Transformer com cerca de 4.3M de parâmetros, treinar de forma eficaz em CPU por apenas 10 épocas é insuficiente para convergir. Modelos maiores necessitam de taxas de aprendizado agendadas (warmup + decay) e de períodos de otimização muito mais longos.

---

## 6. Códigos Utilizados como Base
Abaixo estão expostos os scripts base em Python/Keras para os experimentos realizados, que servem de modelo estrutural para a reprodução dos experimentos.

### Código Base 1: Pipeline CNN (Exemplo Baseado no Experimento 1)

```python
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
```

### Código Base 2: Pipeline Vision Transformer (Exemplo Baseado no Experimento 5)

```python
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
```
