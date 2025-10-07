#!/usr/bin/env python3
"""
Script para corrigir automaticamente os caminhos hardcoded no CLog.
Uso: python fix_paths.py
"""

import os
import re
from pathlib import Path

# Mapeamento de substituições
REPLACEMENTS = {
    # Padrão antigo -> novo
    r'/home/matilda/PycharmProjects/FailurePrediction/4_analysis/clog/data/NOVA/': '../data/NOVA/',
    r'/home/matilda/PycharmProjects/FailurePrediction/5_results/output_files_per_experiment/': '../results/output_files_per_experiment/',
    r'/home/matilda/PycharmProjects/FailurePrediction/6_insights/0_different_clustering_results_anomaly_detection/': '../results/clustering_results/',
    r'/home/matilda/PycharmProjects/RCA_logs/2_copy_original_data/Fault-Injection-Dataset-master/': '../data/RCA_logs/Fault-Injection-Dataset-master/',
    r'\./data/NOVA/': '../data/NOVA/',  # Ajustar caminhos relativos em CLog_main.py
}

# Arquivos a modificar
FILES_TO_FIX = [
    'clog/1_preprocess_data.py',
    'clog/2_create_train_test_data.py',
    'clog/3_postporcessresults.py',
    'clog/CLog_main.py',
    'clog/FD.py',
    'clog/FTI.py',
    'clog/create_sequences.py',
]

def fix_file(filepath):
    """Corrige os caminhos em um arquivo."""
    if not os.path.exists(filepath):
        print(f"⚠ Arquivo não encontrado: {filepath}")
        return False
    
    # Ler conteúdo
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Aplicar substituições
    for old_pattern, new_path in REPLACEMENTS.items():
        content = re.sub(old_pattern, new_path, content)
    
    # Verificar se houve mudanças
    if content != original_content:
        # Fazer backup
        backup_path = filepath + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Salvar versão corrigida
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Corrigido: {filepath} (backup em {backup_path})")
        return True
    else:
        print(f"○ Sem mudanças: {filepath}")
        return False

def create_directories():
    """Cria a estrutura de diretórios necessária."""
    time_intervals = ['60s', '120s', '180s', '240s', '300s', '600s']
    subdirs = ['masked_train_valid_test', 'results', 'HMM_sequencies', 'classification_data']
    
    dirs = []
    
    # Diretórios de dados NOVA
    for interval in time_intervals:
        for subdir in subdirs:
            dirs.append(f'data/NOVA/resources/{interval}/{subdir}')
    
    # Outros diretórios
    dirs.extend([
        'data/RCA_logs/Fault-Injection-Dataset-master',
        'results/output_files_per_experiment',
        'results/clustering_results',
    ])
    
    created = 0
    for dir_path in dirs:
        p = Path(dir_path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created += 1
    
    print(f"\n✓ Estrutura de diretórios criada ({created} novos diretórios)")

def check_datasets():
    """Verifica se os datasets necessários estão presentes."""
    print("\n3. Verificando datasets...")
    
    datasets = {
        'NOVA_clusters_processed_padded.csv': 'data/NOVA/NOVA_clusters_processed_padded.csv',
        'nova.tsv': 'data/RCA_logs/Fault-Injection-Dataset-master/nova.tsv',
    }
    
    missing = []
    for name, path in datasets.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  ✓ {name}: {size:.2f} MB")
        else:
            print(f"  ✗ {name}: FALTANDO")
            missing.append((name, path))
    
    if missing:
        print("\n⚠ AÇÃO NECESSÁRIA:")
        for name, path in missing:
            print(f"  - Coloque {name} em: {path}")

def main():
    print("="*70)
    print("CORREÇÃO AUTOMÁTICA DE CAMINHOS - CLog")
    print("="*70)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('clog') or not os.path.exists('data'):
        print("\n✗ ERRO: Execute este script na raiz do projeto CLOG-MAIN")
        print("  Estrutura esperada:")
        print("    CLOG-MAIN/")
        print("    ├── clog/")
        print("    ├── data/")
        print("    └── fix_paths.py  ← você está aqui")
        return
    
    print("\n1. Criando estrutura de diretórios...")
    create_directories()
    
    print("\n2. Corrigindo caminhos nos arquivos...")
    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if fix_file(filepath):
            fixed_count += 1
    
    check_datasets()
    
    print("\n" + "="*70)
    print(f"CONCLUÍDO: {fixed_count} arquivo(s) corrigido(s)")
    print("="*70)
    print("\n📝 Próximos passos:")
    print("1. Coloque os datasets nos locais indicados acima")
    print("2. Ative o ambiente virtual: source venv/bin/activate")
    print("3. Instale dependências: pip install -r requirements.txt")
    print("4. Execute os scripts NA ORDEM, de dentro da pasta clog/:")
    print("   cd clog")
    print("   python 1_preprocess_data.py")
    print("   python 2_create_train_test_data.py")
    print("   python CLog_main.py")
    print("   python 3_postporcessresults.py")
    print("   python FD.py")
    print("   python FTI.py")
    print()

if __name__ == "__main__":
    main()
