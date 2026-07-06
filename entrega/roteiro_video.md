# Roteiro do Vídeo — Apenas Código (sem slides)

**Disciplina:** Projeto Prático de Inteligência Artificial — 2026  
**Instituição:** IFMG — Campus Ouro Branco  
**Curso:** Bacharelado em Sistemas de Informação  
**Aluno(a):** [SEU NOME]  
**Matrícula:** [SUA MATRÍCULA]  
**Duração estimada:** 12 a 15 minutos  
**Formato:** gravação de tela mostrando exclusivamente o código, arquivos do projeto e saída do terminal

> **Como usar:** abra os arquivos indicados em cada bloco, role até as linhas mencionadas e leia as frases sugeridas. Substitua `[SEU NOME]` e `[SUA MATRÍCULA]` antes de gravar.

---

## Configuração antes de gravar

- **Editor:** VS Code ou Cursor com fonte 16–18 pt (legível no vídeo)
- **Tela:** mostre só o projeto — sem abas pessoais ou notificações
- **Terminal:** deixe preparada a saída de um experimento já executado (não rode os 8 ao vivo)
- **Arquivos abertos antes de iniciar:**
  - `deteccao_tumores_ia/` (explorador de arquivos)
  - `experimentos/exp1_cnn_base.py`
  - `outputs/tabelas/resultados_experimentos.csv`

---

## BLOCO 1 — Abertura e estrutura do projeto (~2 min)

### Tela: explorador de arquivos + `README.md`

**O que falar:**
> "Olá, meu nome é [SEU NOME], matrícula [SUA MATRÍCULA], do Bacharelado em Sistemas de Informação do IFMG Campus Ouro Branco. Este vídeo apresenta o Projeto Prático de Inteligência Artificial de 2026: classificação automática de tumores cerebrais em imagens de ressonância magnética usando Aprendizado de Máquina em Python, com Keras e Scikit-Learn.
>
> O objetivo é classificar imagens de RM em quatro categorias — glioma, meningioma, tumor hipofisário e sem tumor — comparando duas arquiteturas: Rede Neural Convolucional e Vision Transformer. Realizei oito experimentos variando técnicas de regularização e parâmetros."

**Mostre a pasta `deteccao_tumores_ia/` e explique cada item:**

| Pasta / arquivo | O que falar |
|-----------------|-------------|
| `experimentos/` | "Oito scripts Python — um por experimento, de `exp1` a `exp8`." |
| `executar_todos.py` | "Script orquestrador que executa todos e gera a tabela de resultados." |
| `outputs/tabelas/` | "Saída consolidada em CSV e Markdown." |
| `entrega/` | "Relatório final e links de entrega." |
| `requirements.txt` | "Dependências: TensorFlow, Scikit-Learn, kagglehub." |
| `brain-tumor-mri-dataset/` | "Dataset do Kaggle — pastas Training e Testing, cada classe em uma subpasta." |

**Abra `brain-tumor-mri-dataset/Training/` e mostre as subpastas:**
> "O Kaggle organiza as imagens por classe: glioma, meningioma, pituitary e notumor. O Keras lê o nome da pasta e converte automaticamente em rótulo numérico. São cerca de 7.200 imagens no total — Training para treino e validação, Testing com 1.311 imagens reservadas exclusivamente para avaliação final."

---

## BLOCO 2 — Pipeline comum de dados (~3 min)

### Tela: `experimentos/exp1_cnn_base.py` — linhas 1 a 57

> "Todos os experimentos compartilham o mesmo pipeline de dados nas primeiras 57 linhas. A partir da linha 59 cada script diverge na arquitetura. Vou usar o Experimento 1 como referência."

---

### Linhas 1–6 — Imports

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import kagglehub
```

**O que falar:**
> "TensorFlow e Keras constroem e treinam a rede. O Scikit-Learn entra **somente na avaliação final** — Keras treina, Sklearn mede. O `kagglehub` baixa o dataset se não estiver em cache local."

---

### Linhas 9–12 — Hiperparâmetros globais

```python
BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 10
```

**O que falar:**
> "Hiperparâmetros iguais nos experimentos 1 a 7: imagens 150×150, lotes de 32, 10 épocas. Manter constante garante comparação justa — só muda o modelo, não o pipeline."

---

### Linhas 15–22 — Carregamento do dataset

```python
LOCAL_DATASET_PATH = r"C:\Users\Usuario\.cache\kagglehub\..."
if os.path.exists(LOCAL_DATASET_PATH):
    DATASET_PATH = LOCAL_DATASET_PATH
else:
    DATASET_PATH = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

**O que falar:**
> "Primeiro verifico se o dataset já está no cache local do Kaggle. Se não estiver, chamo `dataset_download` para baixar automaticamente."

---

### Linhas 28–51 — Divisão treino / validação / teste

**Destaque linha por linha:**

| Linhas | Código | O que falar |
|--------|--------|-------------|
| 28–35 | `subset="training"`, `validation_split=0.2`, `seed=123` | "80% da pasta Training vira treino. O `seed=123` garante sempre a mesma divisão — reprodutibilidade." |
| 37–44 | `subset="validation"` | "20% da Training vira validação — monitoro overfitting durante o treino." |
| 46–51 | `'Testing'`, `shuffle=False` | "A pasta Testing nunca entra no treino. `shuffle=False` porque depois preciso alinhar predições com rótulos reais." |

**Frase de síntese:**
> "Separação correta: Training dividido em treino e validação; Testing guardado exclusivamente para as quatro métricas finais do trabalho."

---

### Linhas 53–57 — Normalização

```python
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
```

**O que falar:**
> "A camada `Rescaling(1./255)` divide cada pixel por 255, levando valores de [0,255] para [0,1]. Redes neurais convergem melhor nessa escala. O `.map()` aplica a transformação em cada batch sem alterar os rótulos `y`. Idêntico em todos os oito experimentos."

---

## BLOCO 3 — CNN: baseline e variações (~4 min)

### Tela: `exp1_cnn_base.py` — linhas 59–87

---

### Linhas 60–72 — Arquitetura CNN (Exp 1)

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])
```

**O que falar (linha a linha):**

| Linha | O que falar |
|-------|-------------|
| 60 | "`Sequential` empilha camadas — saída de uma vira entrada da próxima." |
| 61 | "1ª convolução: 32 filtros 3×3 com ReLU. Detecta bordas e texturas. Entrada RGB 150×150." |
| 62 | "MaxPooling reduz pela metade — 150→75. Diminui computação e dá invariância espacial." |
| 64–65 | "2ª convolução com 64 filtros — padrões mais complexos. Outro pooling: 75→37." |
| 67–68 | "3ª convolução sem pooling. `Flatten()` transforma o mapa 3D em vetor 1D." |
| 70 | "Camada densa com 64 neurônios — combina as features extraídas." |
| 71 | "Saída: 4 neurônios com Softmax — uma probabilidade por classe." |

---

### Linhas 74–87 — Compilação e treino

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
```

**O que falar:**
> "Adam como otimizador. `SparseCategoricalCrossentropy` porque os rótulos são inteiros 0, 1, 2, 3 — não preciso de one-hot encoding. O `fit` treina por 10 épocas usando `validation_data` para monitorar desempenho em dados que não entram no gradiente."

---

### Tela: alternar entre Exp 2, 3 e 4 — só as linhas que mudam

**Frase de transição:**
> "Do Experimento 1 ao 4, as linhas 1 a 57 são idênticas. A única diferença está no bloco da arquitetura — princípio do experimento controlado."

| Exp | Arquivo | Linha | Mudança | O que falar |
|-----|---------|-------|---------|-------------|
| 2 | `exp2_cnn_dropout.py` | **71** | `layers.Dropout(0.5)` | "Inseri Dropout entre a densa e a saída. Durante o treino, 50% dos neurônios são desligados aleatoriamente — técnica anti-overfitting." |
| 3 | `exp3_cnn_l2.py` | **62, 67, 71, 75** | `kernel_regularizer=l2(0.01)` | "Penalização L2 em todas as camadas com peso. Adiciona λ·Σw² à função de perda, desencorajando pesos muito grandes." |
| 4 | `exp4_cnn_neuronios.py` | **70** | `Dense(256)` | "Expandi de 64 para 256 neurônios. Maior capacidade não-linear. **Melhor resultado: 87,81% de acurácia.**" |

**Abra cada arquivo e mostre só a linha que difere** — use split screen ou alterne rapidamente.

---

## BLOCO 4 — Vision Transformer: baseline e variações (~4 min)

### Tela: `exp5_transformer_base.py`

**Frase de transição:**
> "Agora o Vision Transformer. Do Experimento 5 ao 8, o pipeline de dados é igual, mas a arquitetura muda a partir da linha 13."

---

### Linhas 13–18 — Hiperparâmetros exclusivos do ViT

```python
PATCH_SIZE = 25
NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)  # 36 patches
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_UNITS = [128, 64]
```

**O que falar:**
> "Patch de 25×25 → grid 6×6 = 36 patches por imagem. Projeção para dimensão 64, 4 cabeças de atenção, 4 blocos Transformer, MLP interno com camadas 128 e 64."

---

### Linhas 67–83 — Classe `Patches`

```python
class Patches(layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
```

**O que falar:**
> "Camada customizada que substitui a convolução da CNN. `extract_patches` fatia a imagem em blocos 25×25 sem sobreposição. Cada patch vira um token — como uma palavra num texto."

---

### Linhas 85–96 — Classe `PatchEncoder`

```python
self.projection = layers.Dense(units=projection_dim)
self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
# ...
return self.projection(patch) + self.position_embedding(positions)
```

**O que falar:**
> "Duas operações: projeta o conteúdo visual do patch para dimensão 64, e soma um embedding posicional aprendível. Sem a posição, o Transformer não saberia onde cada patch está na imagem."

---

### Linhas 104–134 — Blocos Transformer

```python
for _ in range(TRANSFORMER_LAYERS):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM)(x1, x1)
    x2 = layers.Add()([attention_output, x])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS)
    x = layers.Add()([x3, x2])
```

**O que falar:**
> "Quatro blocos idênticos. Cada um tem dois sub-blocos: primeiro, LayerNorm + Multi-Head Attention + conexão residual — cada patch observa todos os outros simultaneamente. Depois, LayerNorm + MLP com GELU + conexão residual. As skip connections evitam degradação do gradiente."

---

### Tela: alternar Exp 6, 7 e 8 — linhas que mudam

| Exp | Arquivo | Linhas | O que falar |
|-----|---------|--------|-------------|
| 6 | `exp6_transformer_dropout.py` | 19–20, 100, 123, 129, 135 | "Dropout graduado: 0.1 nos embeddings e atenção, 0.3 no MLP, 0.5 no cabeçote de classificação." |
| 7 | `exp7_transformer_l1.py` | 19, 92, 107, 127, 144 | "Apesar do nome do arquivo, usa L2 com coeficiente 0.01 em projeções, MLP, atenção e saída." |
| 8 | `exp8_transformer_otimizado.py` | 10–11, 62–71, 160–167, 182–183 | "Único com 300×300, 100 patches, RandomBrightness no treino, cabeçote duplo e Adamax. Modelo mais complexo, mas pior resultado — 76,12% — por não convergir em 10 épocas." |

**Destaque no Exp 8 — linhas 62–71 (data augmentation):**
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0), seed=123)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
```
> "Varia o brilho ±20% **apenas no treino**, simulando calibrações diferentes de aparelhos de RM. Validação e teste ficam limpos."

---

## BLOCO 5 — Avaliação e métricas (~2 min)

### Tela: `exp1_cnn_base.py` — linhas 89–113 (igual em todos os experimentos)

---

### Linhas 94–100 — Loop de predição

```python
for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_pred = np.argmax(y_pred_probs, axis=1)
```

**O que falar:**
> "Itero batch a batch no conjunto de **teste** — nunca visto no treino. Guardo o rótulo real e as probabilidades Softmax do modelo. O `argmax` escolhe a classe com maior probabilidade."

---

### Linhas 103–113 — Cálculo das métricas

```python
acuracia = accuracy_score(y_true, y_pred)
precisao = precision_score(y_true, y_pred, average='macro')
recall   = recall_score(y_true, y_pred, average='macro')
f_measure = f1_score(y_true, y_pred, average='macro')

print(f"Acurácia:  {acuracia:.4f}")
print(f"Precisão:  {precisao:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F-Mesure:  {f_measure:.4f}")
```

**O que falar:**
> "Quatro métricas exigidas pelo trabalho, calculadas com Scikit-Learn. Uso `average='macro'` — calcula a métrica por classe e faz a média simples, tratando glioma, meningioma, pituitary e notumor com peso igual. O Keras só mostra acurácia durante o treino; as métricas finais vêm exclusivamente daqui."

**Mostre a saída no terminal** de um experimento já executado (ex.: Exp 4):
```
--- RESULTADOS FINAIS DO EXPERIMENTO 4 ---
Acurácia:  0.8781
Precisão:  0.8827
Recall:    0.8781
F-Mesure:  0.8753
```

---

## BLOCO 6 — Orquestração e resultados (~2 min)

### Tela: `executar_todos.py`

---

### Linhas 7–56 — Lista de experimentos

**O que falar:**
> "Lista de dicionários — cada um mapeia um experimento ao seu arquivo, arquitetura e descrição da variação. Centraliza a configuração dos oito experimentos."

---

### Linhas 68–90 — Cache de resultados

**O que falar:**
> "Antes de executar, verifica se o CSV já tem resultados válidos. Se sim, pula aquele experimento — cada treino leva minutos ou horas. Só reexecuta o que falhou ou ainda não foi rodado."

---

### Linhas 114–133 — Execução e regex

```python
process = subprocess.run([python_executable, script_path], capture_output=True, text=True)
acuracia_match = acuracia_regex.search(output)  # r"Acurácia:\s*([0-9.]+)"
```

**O que falar:**
> "Executa cada script como processo separado via `subprocess`, captura o stdout e usa regex para extrair as métricas dos prints finais. Se um script falhar, registra 'ERRO' na tabela sem interromper os demais."

---

### Linhas 168–187 — Geração das tabelas

**O que falar:**
> "Ao final, grava automaticamente o CSV e o Markdown em `outputs/tabelas/`. Foi assim que montei a tabela do relatório."

---

### Tela: `outputs/tabelas/resultados_experimentos.csv`

**Abra o CSV e percorra linha a linha comentando:**

| Exp | Acurácia | O que falar |
|-----|----------|-------------|
| 1 | 0,8719 | "CNN baseline — referência." |
| 2 | 0,8650 | "Dropout reduziu levemente — convergência mais lenta em 10 épocas." |
| 3 | 0,8006 | "L2 forte (0.01) penalizou demais neste prazo." |
| **4** | **0,8781** | "**Melhor resultado geral.** Mais neurônios na camada densa." |
| 5 | 0,8356 | "Melhor ViT — baseline sem regularização." |
| 6 | 0,7819 | "Dropout no Transformer." |
| 7 | 0,7706 | "L2 no Transformer." |
| 8 | 0,7612 | "Modelo mais complexo, pior resultado — subajuste por falta de épocas." |

**Análise verbal (sem slide — fale olhando o CSV):**
> "A CNN superou o ViT em todos os casos. Acredito que isso se deve ao viés indutivo da CNN — localidade e invariância à translação — que a torna mais eficiente com datasets médios como o nosso. O ViT precisa aprender relações espaciais do zero e normalmente exige bases muito maiores ou pré-treinamento.
>
> A regularização — dropout e L2 — reduziu métricas em 10 épocas porque desacelera a convergência. Com 30 a 50 épocas, poderia superar os baselines. O Exp 8, apesar de ser o mais ambicioso, não convergiu a tempo."

---

## BLOCO 7 — Encerramento (~45 s)

### Tela: volte ao explorador de arquivos mostrando a estrutura completa

**O que falar:**
> "Em resumo: oito experimentos em Python com Keras e Scikit-Learn, comparando CNN e Vision Transformer com variações de dropout, L2 e parâmetros. A melhor abordagem foi a CNN com 256 neurônios — Experimento 4 — com 87,81% de acurácia. Como limitações: apenas 10 épocas, sem transfer learning e execução em CPU. Trabalhos futuros incluiriam mais épocas, modelos pré-treinados e data augmentation mais agressivo. O relatório completo está em `entrega/relatorio_final.md`. Obrigado pela atenção."

---

## Mapa de gravação — ordem dos arquivos na tela

```
1.  explorador → estrutura do projeto + dataset
2.  exp1_cnn_base.py      → linhas 1–57   (pipeline comum)
3.  exp1_cnn_base.py      → linhas 59–87  (CNN baseline)
4.  exp2_cnn_dropout.py   → linha 71      (Dropout)
5.  exp3_cnn_l2.py        → linhas 62–75  (L2)
6.  exp4_cnn_neuronios.py → linha 70      (256 neurônios)
7.  exp5_transformer_base.py → linhas 13–18, 67–134 (ViT)
8.  exp6_transformer_dropout.py → linhas 19–20, 100, 123, 129, 135
9.  exp7_transformer_l1.py      → linhas 19, 92, 107, 127, 144
10. exp8_transformer_otimizado.py → linhas 10–11, 62–71, 160–167, 182
11. exp1_cnn_base.py      → linhas 89–113 (métricas)
12. terminal              → saída do Exp 4
13. executar_todos.py     → linhas 7–56, 68–90, 114–133, 168–187
14. resultados_experimentos.csv → tabela completa
15. explorador            → encerramento
```

---

## Cronograma

| Bloco | Conteúdo | Tempo |
|-------|----------|-------|
| 1 | Abertura + estrutura do projeto | ~2 min |
| 2 | Pipeline comum (exp1, linhas 1–57) | ~3 min |
| 3 | CNN baseline + variações (exp1–4) | ~4 min |
| 4 | ViT baseline + variações (exp5–8) | ~4 min |
| 5 | Métricas + saída do terminal | ~2 min |
| 6 | executar_todos.py + CSV de resultados | ~2 min |
| 7 | Encerramento | ~45 s |
| **Total** | | **~12–15 min** |

---

## Dicas para gravação só com código

1. **Use o minimap ou números de linha** do editor — o professor precisa ver onde você está.
2. **Destaque com o cursor** a linha enquanto explica; evite scroll rápido demais.
3. **Split screen** ao comparar experimentos: Exp 1 à esquerda, Exp 4 à direita na linha 70.
4. **Terminal pré-gravado**: tenha a saída do Exp 4 copiada ou execute só esse (mais rápido).
5. **Não rode `executar_todos.py` ao vivo** — leva horas. Mostre o CSV já gerado.
6. **Zoom 125–150%** no editor se a fonte ainda parecer pequena na gravação.

---

## Checklist pós-gravação

- [ ] Nome e matrícula falados no início
- [ ] Estrutura do projeto mostrada
- [ ] Pipeline comum explicado linha a linha
- [ ] Diferença de cada experimento apontada pela linha exata
- [ ] Métricas explicadas (macro, teste separado)
- [ ] Tabela CSV comentada
- [ ] Melhor abordagem e conjecturas verbalizadas
- [ ] Link atualizado em `entrega/links_entrega.txt`
- [ ] Duração entre 10 e 15 minutos
