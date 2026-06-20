# TODO — Manoel · Projeto ES413 (Filtros Digitais para ECG)

> Board: Tasks — ES413 Fase 3 · Entrega final: **26/06/2026 (antes das 10h, sem atraso)**

## Etapa A — Dados ⚠️ ATRASADAS (caminho crítico)

- [ ] **Setup do repositório** — Alta · prazo 11/06
  - Git + estrutura: `data/`, `src/filters/`, `src/metrics/`, `src/noise/`, `notebooks/`, `report/`
  - `requirements`: wfdb, numpy, scipy, pywavelets, tensorflow, matplotlib
- [ ] **Download mitdb + nstdb via wfdb e seleção dos registros** — Alta · prazo 12/06
  - Registros sugeridos: 100, 103, 105, 115, 215 · Ruídos nstdb: `bw`, `ma`, `em`
- [ ] **Módulo de contaminação controlada** — Alta · prazo 13/06
  - `contaminate(signal, noise_type, snr_db)` para bw/ma/em + senoide 60 Hz
  - SNRs alvo: 0, 6, 12, 18 dB · (coração do estudo comparativo)

## Etapa B — Filtros Convencionais

- [ ] **Filtro Notch IIR 60 Hz** (`scipy.signal.iirnotch`) — Alta · prazo 15/06
  - Q ajustado para banda ~2 Hz · plotar polos/zeros + |H(e^jω)|
- [ ] **Filtro Passa-Alta IIR Butterworth 0,5 Hz** (`butter` + `filtfilt`) — Alta · prazo 15/06
  - 2ª ordem, transformação bilinear · comparar `filtfilt` (fase zero) vs `lfilter` (causal)
- [ ] **Filtro Passa-Baixa FIR Hamming 40 Hz** (`firwin`, 61 taps) — Alta · prazo 16/06
  - Verificar fase linear e atraso de grupo de (M-1)/2 amostras
- [ ] **Formalização: plano-z, H(z) e resposta em frequência dos 3 filtros** — Média · prazo 16/06
  - Figuras: diagrama polos/zeros, |H(e^jω)| teórica vs medida, espectro FFT antes/depois → vão pro relatório

## Etapa E — Relatório / Apresentação

- [ ] **Relatório: Introdução + Fundamentação Teórica** — Alta · prazo 24/06
  - IIR/FIR, transformada-z, DTFT, adaptativo, wavelet, DAE · Overleaf self-hosted
- [ ] **Demo ao vivo / dashboard interativo** — Média · prazo 25/06
  - Notebook que filtra um registro na hora ou dashboard · diferencial em criatividade

## Tasks da equipe (incluem o Manoel)

- [ ] **🧊 CODE FREEZE** — Alta · prazo 23/06 · só relatório, slides e ensaio a partir daqui
- [ ] **Slides da apresentação** (~12–15, tempo igual por pessoa) — Alta · prazo 24/06
- [ ] **Ensaio cronometrado da apresentação** — Alta · prazo 25/06
- [ ] **📤 Upload do relatório final + apresentação** — Alta · prazo 26/06 · ⚠️ sem atraso permitido