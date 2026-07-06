# Tabela Comparativa de Resultados dos Experimentos


| Experimento | Arquitetura              | Variação / Parâmetros                                       | Acurácia | Precisão | Recall | F-Score |
| ----------- | ------------------------ | ----------------------------------------------------------- | -------- | -------- | ------ | ------- |
| Exp 1       | CNN                      | Baseline (3 Conv, 64 Neurônios na Densa, Sem Regularização) | 0.8719   | 0.8736   | 0.8719 | 0.8698  |
| Exp 2       | CNN                      | Dropout (0.5) na camada densa final                         | 0.8650   | 0.8697   | 0.8650 | 0.8630  |
| Exp 3       | CNN                      | Regularização L2 (0.01) nas camadas Conv e Densa            | 0.8006   | 0.8044   | 0.8006 | 0.7902  |
| Exp 4       | CNN                      | Aumento de neurônios na Densa (256)                         | 0.8781   | 0.8827   | 0.8781 | 0.8753  |
| Exp 5       | Vision Transformer (ViT) | Baseline (Patch 25x25, 4 blocos, Sem Regularização)         | 0.8356   | 0.8387   | 0.8356 | 0.8323  |
| Exp 6       | Vision Transformer (ViT) | Dropout (0.1 no Transformer, 0.3 no MLP, 0.5 no Cabeçote)   | 0.7819   | 0.7818   | 0.7819 | 0.7747  |
| Exp 7       | Vision Transformer (ViT) | Regularização L2 (0.01) nas projeções e MLP do Transformer  | 0.7706   | 0.7742   | 0.7706 | 0.7600  |
| Exp 8       | Vision Transformer (ViT) | Otimizado (300x300, Adamax, Brilho, Densa Dupla, L2 1e-4)   | 0.7612   | 0.7565   | 0.7613 | 0.7471  |


