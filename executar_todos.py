import os
import subprocess
import re
import csv

# Definição dos experimentos a serem executados
experimentos = [
    {
        "id": "Exp 1",
        "nome": "exp1_cnn_base.py",
        "modelo": "CNN",
        "variacao": "Baseline (3 Conv, 64 Neurônios na Densa, Sem Regularização)"
    },
    {
        "id": "Exp 2",
        "nome": "exp2_cnn_dropout.py",
        "modelo": "CNN",
        "variacao": "Dropout (0.5) na camada densa final"
    },
    {
        "id": "Exp 3",
        "nome": "exp3_cnn_l2.py",
        "modelo": "CNN",
        "variacao": "Regularização L2 (0.01) nas camadas Conv e Densa"
    },
    {
        "id": "Exp 4",
        "nome": "exp4_cnn_neuronios.py",
        "modelo": "CNN",
        "variacao": "Aumento de neurônios na Densa (256)"
    },
    {
        "id": "Exp 5",
        "nome": "exp5_transformer_base.py",
        "modelo": "Vision Transformer (ViT)",
        "variacao": "Baseline (Patch 25x25, 4 blocos, Sem Regularização)"
    },
    {
        "id": "Exp 6",
        "nome": "exp6_transformer_dropout.py",
        "modelo": "Vision Transformer (ViT)",
        "variacao": "Dropout (0.1 no Transformer, 0.3 no MLP, 0.5 no Cabeçote)"
    },
    {
        "id": "Exp 7",
        "nome": "exp7_transformer_l1.py",
        "modelo": "Vision Transformer (ViT)",
        "variacao": "Regularização L2 (0.01) nas projeções e MLP do Transformer"
    },
    {
        "id": "Exp 8",
        "nome": "exp8_transformer_otimizado.py",
        "modelo": "Vision Transformer (ViT)",
        "variacao": "Otimizado (300x300, Adamax, Brilho, Densa Dupla, L2 1e-4)"
    }
]

# Caminho para o Python do virtual environment (Windows)
python_executable = os.path.join("venv", "Scripts", "python.exe")
if not os.path.exists(python_executable):
    print(f"Aviso: Executável da venv não encontrado em '{python_executable}'. Usando 'python' global.")
    python_executable = "python"

print("="*60)
print("INICIANDO A EXECUÇÃO DOS 8 EXPERIMENTOS DE IA")
print("="*60)

resultados_salvos = {}
csv_path = os.path.join("outputs", "tabelas", "resultados_experimentos.csv")
if os.path.exists(csv_path):
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 7:
                    exp_id, modelo, variacao, ac, pr, rc, f1 = row
                    if ac not in ("ERRO", "N/A", ""):
                        resultados_salvos[exp_id] = {
                            "id": exp_id,
                            "modelo": modelo,
                            "variacao": variacao,
                            "acuracia": ac,
                            "precisao": pr,
                            "recall": rc,
                            "f1_score": f1
                        }
        print(f"Carregados {len(resultados_salvos)} resultados anteriores com sucesso.")
    except Exception as e:
        print(f"Aviso: Não foi possível ler resultados anteriores ({e}). Executando tudo de novo.")

resultados = []

# Regex para extrair as métricas das saídas dos terminais
acuracia_regex = re.compile(r"Acurácia:\s*([0-9.]+)")
precisao_regex = re.compile(r"Precisão:\s*([0-9.]+)")
recall_regex = re.compile(r"Recall:\s*([0-9.]+)")
f1_regex = re.compile(r"F-Mesure:\s*([0-9.]+)")

for exp in experimentos:
    if exp["id"] in resultados_salvos:
        print(f"\n---> {exp['id']} ({exp['nome']}) já possui resultados válidos salvos. Pulando...")
        resultados.append(resultados_salvos[exp["id"]])
        continue
    script_path = os.path.join("experimentos", exp["nome"])
    print(f"\n---> Executando {exp['id']}: {exp['nome']} ({exp['variacao']})...")
    
    if not os.path.exists(script_path):
        print(f"Erro: Arquivo do experimento não encontrado em: {script_path}")
        continue
        
    try:
        # Executa o script do experimento capturando a saída
        process = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = process.stdout
        
        # Procura as métricas na saída impressa
        acuracia_match = acuracia_regex.search(output)
        precisao_match = precisao_regex.search(output)
        recall_match = recall_regex.search(output)
        f1_match = f1_regex.search(output)
        
        # Extrai os valores ou define como N/A se não encontrados
        ac = acuracia_match.group(1) if acuracia_match else "N/A"
        pr = precisao_match.group(1) if precisao_match else "N/A"
        rc = recall_match.group(1) if recall_match else "N/A"
        f1 = f1_match.group(1) if f1_match else "N/A"
        
        print(f"Concluído! Acurácia: {ac} | Precisão: {pr} | Recall: {rc} | F1-Score: {f1}")
        
        resultados.append({
            "id": exp["id"],
            "modelo": exp["modelo"],
            "variacao": exp["variacao"],
            "acuracia": ac,
            "precisao": pr,
            "recall": rc,
            "f1_score": f1
        })
        
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {exp['id']} ({exp['nome']}):")
        print(e.stderr)
        resultados.append({
            "id": exp["id"],
            "modelo": exp["modelo"],
            "variacao": exp["variacao"],
            "acuracia": "ERRO",
            "precisao": "ERRO",
            "recall": "ERRO",
            "f1_score": "ERRO"
        })

# Criação das tabelas de saída
print("\n" + "="*60)
print("GRAVANDO RESULTADOS CONSOLIDADOS")
print("="*60)

# Garantir que a pasta de tabelas existe
os.makedirs(os.path.join("outputs", "tabelas"), exist_ok=True)

# 1. Gravar em formato CSV
csv_path = os.path.join("outputs", "tabelas", "resultados_experimentos.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Experimento", "Modelo", "Variacao", "Acuracia", "Precisao", "Recall", "F-Score"])
    for res in resultados:
        writer.writerow([res["id"], res["modelo"], res["variacao"], res["acuracia"], res["precisao"], res["recall"], res["f1_score"]])

print(f"Tabela CSV salva em: {csv_path}")

# 2. Gravar em formato Markdown (pronta para o relatório)
md_path = os.path.join("outputs", "tabelas", "resultados_experimentos.md")
with open(md_path, mode="w", encoding="utf-8") as f:
    f.write("# Tabela Comparativa de Resultados dos Experimentos\n\n")
    f.write("| Experimento | Arquitetura | Variação / Parâmetros | Acurácia | Precisão | Recall | F-Score |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for res in resultados:
        f.write(f"| {res['id']} | {res['modelo']} | {res['variacao']} | {res['acuracia']} | {res['precisao']} | {res['recall']} | {res['f1_score']} |\n")

print(f"Tabela Markdown salva em: {md_path}")
print("\nProcesso concluído com sucesso!")
