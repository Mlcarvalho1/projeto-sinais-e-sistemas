# CNN-DAE — Filtragem de Ruído em ECG

Autoencoder convolucional (CNN-DAE) para filtragem de ruído em sinais de ECG
do MIT-BIH Arrhythmia Database. Parte do projeto ES413.

## Estrutura

```
cnn_dae/
├── src/
│   ├── data_io.py               # Leitura dos 48 registros mitdb + nstdb
│   ├── noise/
│   │   └── contaminate.py       # Contaminação com SNR controlado (5 tipos)
│   └── models/
│       ├── dataset.py           # Pipeline de dados (pools, geradores, tabela de teste)
│       └── model.py             # Arquitetura CNN-DAE + callbacks
├── notebooks/
│   ├── treino_cnn_dae.ipynb     # Treino no Google Colab (T4 GPU)
│   └── validacao_cnn_dae.ipynb  # Avaliação de um best_model.keras salvo
├── checkpoints/                 # Modelos salvos (best_model.keras)
├── data/                        # Criada automaticamente pelo data_io
│   ├── mitdb/                   # 48 registros MIT-BIH (baixados via wfdb)
│   └── nstdb/                   # bw, ma, em (baixados via wfdb)
├── train.py                     # Script de treino (linha de comando)
└── requirements.txt
```

## Relação com o restante do projeto

Este subprojeto é **independente** dos filtros convencionais (`src/filters/`).
- `src/data_io.py` aqui cobre os 48 registros do mitdb (o `data_io.py` principal
  usa apenas 5 registros de DEFAULT_RECORDS para os filtros convencionais).
- Os dados são baixados em `cnn_dae/data/`, separados de `data/` na raiz.
- Não há dependência de código do projeto principal — pode ser executado isoladamente.

## Protocolo de dados

| Conjunto | Registros | Critério |
|---|---|---|
| Treino + Val | 100–219 (38 registros) | 80% inicial de cada → treino; 20% final → val |
| Teste | 220–234 (10 registros) | Nunca vistos durante treino |

**Ruído:** `bw`, `ma`, `em` (real, do NSTDB) + `60hz`, `50hz` (sintético).
**Treino:** SNR contínuo sorteado em [−2, 20] dB.
**Teste:** grade fixa em {0, 6, 12, 18} dB, seed 2026.

## Uso local

```bash
cd cnn_dae
pip install -r requirements.txt
python train.py                      # treino com defaults
python train.py --epochs 200 --batch 64
```

## Uso no Google Colab

1. Abra `notebooks/treino_cnn_dae.ipynb` no Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Execute as células em ordem — faça upload dos 4 arquivos quando pedido:
   `data_io.py`, `contaminate.py`, `dataset.py`, `model.py`

Para validar um modelo já treinado, use `notebooks/validacao_cnn_dae.ipynb`
(upload adicional de `best_model.keras`).
