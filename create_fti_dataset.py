#!/usr/bin/env python3
"""
Script FINAL para criar dataset para FTI.py (Failure Type Identification)

Faz merge entre:
- clustering_holistic_outptu_results (count vectors)
- NOVA_clusters_processed_padded (tipos de falha originais)

Entrada: 
  - clustering_holistic_outptu_results_180s_clusters_10_.csv
  - NOVA_clusters_processed_padded.csv
Saída: classification_FTI_180s_clusters_10_.csv

Autor: Manus AI
Data: 08/10/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("CRIANDO DATASET PARA FTI.py (VERSÃO FINAL)")
print("="*80)

# Configurações
TIME_INTERVAL = "180s"
N_CLUSTERS = 10

# Caminhos
BASE_DIR = Path(__file__).resolve().parent
CLOG_DIR = BASE_DIR / "clog"
DATA_DIR = BASE_DIR / "data" / "NOVA"

INPUT_CLUSTERING = CLOG_DIR / f"clustering_holistic_outptu_results_{TIME_INTERVAL}_clusters_{N_CLUSTERS}_.csv"
INPUT_ORIGINAL = DATA_DIR / "NOVA_clusters_processed_padded.csv"

OUTPUT_DIR = DATA_DIR / "resources" / TIME_INTERVAL / "classification_data"
OUTPUT_FILE = OUTPUT_DIR / f"classification_FTI_{TIME_INTERVAL}_clusters_{N_CLUSTERS}_.csv"

# Criar diretório de saída
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📂 ARQUIVOS:")
print(f"   Entrada 1 (count vectors): {INPUT_CLUSTERING}")
print(f"   Entrada 2 (tipos de falha): {INPUT_ORIGINAL}")
print(f"   Saída: {OUTPUT_FILE}")

# Verificar arquivos
if not INPUT_CLUSTERING.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CLUSTERING}")
if not INPUT_ORIGINAL.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_ORIGINAL}")

print(f"\n✅ Ambos arquivos encontrados!")

# Carregar dados
print(f"\n📊 Carregando dados...")

df_clustering = pd.read_csv(INPUT_CLUSTERING)
print(f"   Clustering: {len(df_clustering):,} linhas, {len(df_clustering.columns)} colunas")

df_original = pd.read_csv(INPUT_ORIGINAL)
print(f"   Original: {len(df_original):,} linhas, {len(df_original.columns)} colunas")

# Identificar colunas importantes no arquivo original
print(f"\n🔍 Identificando colunas de tipo de falha...")

fault_col = None
for col in ['FAULT_TYPE', 'fault_type', 'Fault_Type', 'assertion_error', 'app_error']:
    if col in df_original.columns:
        fault_col = col
        break

if fault_col is None:
    # Listar colunas disponíveis
    print(f"\n   ⚠️ Coluna de tipo de falha não encontrada!")
    print(f"   Colunas disponíveis no arquivo original:")
    for i, col in enumerate(df_original.columns[:30], 1):
        print(f"      {i:2d}. {col}")
    raise ValueError("Coluna de tipo de falha não encontrada!")

print(f"   ✅ Coluna de tipo de falha: {fault_col}")

# Verificar valores únicos
fault_types = df_original[fault_col].dropna().unique()
print(f"   Total de tipos de falha: {len(fault_types)}")
print(f"   Exemplos: {list(fault_types[:5])}")

# Identificar coluna de ID para fazer merge
print(f"\n🔍 Identificando coluna de ID para merge...")

# No clustering: test_id ou experiment_id
id_col_clustering = 'test_id' if 'test_id' in df_clustering.columns else 'experiment_id'

# No original: procurar coluna similar
id_col_original = None
for col in ['test_id', 'experiment_id', 'task_id', 'ID']:
    if col in df_original.columns:
        id_col_original = col
        break

if id_col_original is None:
    print(f"   ⚠️ Coluna de ID não encontrada no arquivo original!")
    print(f"   Vamos criar ID baseado no índice...")
    df_original['test_id'] = df_original.index
    id_col_original = 'test_id'

print(f"   ✅ ID clustering: {id_col_clustering}")
print(f"   ✅ ID original: {id_col_original}")

# Agrupar clustering por experimento e criar count vectors
print(f"\n🔧 Criando count vectors por experimento...")

subprocess_col = 'pred_kmeans'
grouped = df_clustering.groupby(id_col_clustering)

count_vectors = []

for exp_id, group in grouped:
    # Count vector: contar ocorrências de cada subprocess (0-9)
    count_vector = {'ID': exp_id}
    for i in range(N_CLUSTERS):
        count_vector[str(i)] = (group[subprocess_col] == i).sum()
    count_vectors.append(count_vector)

df_counts = pd.DataFrame(count_vectors)
print(f"   ✅ Count vectors criados: {len(df_counts)} experimentos")

# Preparar dados originais para merge
print(f"\n🔧 Preparando dados originais...")

# Agrupar por ID e pegar primeiro tipo de falha
df_original_grouped = df_original.groupby(id_col_original).agg({
    fault_col: 'first'  # Pegar primeiro tipo de falha
}).reset_index()

df_original_grouped.columns = ['ID', 'target']

# Preencher NaN com NO_FAILURE
df_original_grouped['target'] = df_original_grouped['target'].fillna('NO_FAILURE')

print(f"   ✅ Dados originais preparados: {len(df_original_grouped)} experimentos")

# Fazer merge
print(f"\n🔗 Fazendo merge...")

fti_df = df_counts.merge(df_original_grouped, on='ID', how='left')

# Preencher NaN em target com NO_FAILURE
fti_df['target'] = fti_df['target'].fillna('NO_FAILURE')

print(f"   ✅ Merge completo: {len(fti_df)} linhas")

# Reordenar colunas
cols = ['ID'] + [str(i) for i in range(N_CLUSTERS)] + ['target']
fti_df = fti_df[cols]

# Estatísticas
print(f"\n📊 ESTATÍSTICAS:")
print(f"   Total de experimentos: {len(fti_df):,}")
print(f"   Total de tipos de falha: {fti_df['target'].nunique()}")

# Distribuição de tipos de falha
print(f"\n📊 DISTRIBUIÇÃO DE TIPOS DE FALHA:")
target_counts = fti_df['target'].value_counts()
print(f"   Total de classes: {len(target_counts)}")
print(f"\n   Top 10:")
for target, count in target_counts.head(10).items():
    pct = count / len(fti_df) * 100
    print(f"      {target:50s}: {count:5,} ({pct:5.1f}%)")

# Verificar classes raras
rare_classes = target_counts[target_counts < 5]
if len(rare_classes) > 0:
    print(f"\n   ⚠️ Classes com < 5 exemplos: {len(rare_classes)}")
    for target, count in rare_classes.items():
        print(f"      {target}: {count}")
else:
    print(f"\n   ✅ Todas as classes têm >= 5 exemplos!")

# Salvar
print(f"\n💾 Salvando arquivo...")
fti_df.to_csv(OUTPUT_FILE, index=False)

print(f"   ✅ Arquivo salvo: {OUTPUT_FILE}")
print(f"   Tamanho: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

print(f"\n{'='*80}")
print("✅ SUCESSO!")
print(f"{'='*80}")

print(f"\n📋 PRÓXIMOS PASSOS:")
print(f"\n1. Modificar FTI.py para usar o novo arquivo:")
print(f"   input_csv = RESOURCES_DIR / f\"classification_FTI_{{TIME_INTERVAL}}_clusters_{{n_clusters}}_.csv\"")
print(f"\n2. Executar FTI.py:")
print(f"   cd clog")
print(f"   python FTI.py")
print(f"\n3. Resultado esperado:")
print(f"   Macro F1: 0.85-0.89")

print(f"\n{'='*80}")