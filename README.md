# Projeto ES413 — Filtros Digitais para ECG

Estudo comparativo de filtros digitais na remoção de ruído de ECG (rede 60 Hz,
variação de linha de base 0,05–0,5 Hz e ruído muscular >40 Hz), usando MIT-BIH
`mitdb` (sinal limpo) + `nstdb` (ruído real) do PhysioNet.

Esta fatia do repositório cobre: **infraestrutura de dados, contaminação
controlada, filtros convencionais, métricas compartilhadas, visualização, demo e
relatório (introdução + fundamentação)**. Os filtros avançados/inteligentes
(LMS/NLMS, DWT, CNN-DAE) e o runner do estudo comparativo são de colegas — a
infra aqui (`contaminate` + `metrics`) é o contrato sobre o qual eles plugam.

## Estrutura

```
src/
  data_io.py            # download/leitura mitdb+nstdb (wfdb)
  noise/contaminate.py  # contaminate(signal, noise_type, snr_db) — bw/ma/em/60hz
  filters/              # notch 60Hz, passa-alta 0.5Hz, passa-baixa FIR 40Hz
  metrics/metrics.py    # SNR/RMSE/PRD/correlação (compartilhado)
  viz/plots.py          # polos-zeros, |H(e^jω)|, atraso de grupo, FFT, overlay
notebooks/              # 01_dados, 02_filtros_convencionais, 03_demo
report/                 # fontes LaTeX (intro + fundamentação)
figures/                # figuras geradas para o relatório
tests/                  # smoke tests (offline)
```

## Setup (ambiente principal — Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> O CNN-DAE usa TensorFlow, que **não** tem wheels para Python 3.14. Use um venv
> separado em Python 3.12 com `requirements-dae.txt` (ver o próprio arquivo).

## Baixar os dados (PhysioNet)

```python
from src.data_io import ensure_datasets
ensure_datasets()   # baixa mitdb (100,103,105,115,215) + nstdb (bw,ma,em) em data/
```

## Uso rápido

```python
from src.data_io import load_ecg, ensure_datasets
from src.noise.contaminate import contaminate
from src.filters.notch import apply_notch
from src.metrics.metrics import summary

ensure_datasets()
clean, fs = load_ecg("100")
noisy = contaminate(clean, "60hz", snr_db=6)
filtered = apply_notch(noisy)
print(summary(clean, noisy, filtered))
```

## Testes

```bash
pip install pytest
pytest -q            # smoke tests offline (não baixam dados)
```

## Notebooks

- `01_dados.ipynb` — download, visualização de ECG limpo + ruídos, demo de `contaminate`.
- `02_filtros_convencionais.ipynb` — projeto/validação dos 3 filtros e **geração das
  figuras de formalização** (polos/zeros, |H(e^jω)|, FFT antes/depois) para o relatório.
- `03_demo.ipynb` — dashboard interativo (ipywidgets) filtrando um registro ao vivo.

## Contrato para o time (filtros avançados)

Qualquer filtro novo deve ter assinatura `f(noisy) -> filtered` e ser avaliado com
`src.metrics.metrics.summary(clean, noisy, filtered)` sobre os pares gerados por
`src.noise.contaminate.contaminate(...)` nos SNRs 0/6/12/18 dB. Assim os resultados
são diretamente comparáveis aos filtros convencionais.

### ⚠️ Dois cuidados no protocolo de avaliação (validados em dados reais)

1. **Atraso de grupo do FIR causal.** O FIR passa-baixa causal (`lfilter`) desloca
   o sinal em `(numtaps-1)/2 = 30` amostras. Comparar a saída crua contra o `clean`
   dá `corr ≈ 0` e SNR enganosamente ruim. **Avalie o FIR com fase-zero**
   (`apply_lowpass_fir(..., zero_phase=True)`) **ou compense o atraso** antes das
   métricas. Filtros `filtfilt`/IIR-`filtfilt` não têm esse problema.
2. **Referência para deriva de linha de base.** O ECG do mitdb já contém conteúdo
   <0,5 Hz. Ao avaliar o passa-alta contra deriva (`bw`), compare contra uma
   referência também passa-alta (`ref = apply_highpass(clean)`), senão o filtro é
   penalizado por remover conteúdo legítimo do próprio `clean`. Com referência
   consistente, o passa-alta dá Δ ≈ +25 dB no `bw` real.

> Observação física esperada: contra `ma`/`em` (ruído muscular banda-larga), o
> passa-baixa de 40 Hz melhora SNR ≈ 0 dB — é a limitação dos filtros convencionais
> que **motiva** os avançados (LMS/DWT/DAE) da Entrega 2.
