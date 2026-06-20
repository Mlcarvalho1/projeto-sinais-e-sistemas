# Filtros Digitais para ECG — ES413

## Introdução

Vou fazer uma comparação de um sinal original de ECG e, depois, desse mesmo sinal com a adição de ruídos. A ideia é observar como cada interferência altera o comportamento do sinal ao longo do tempo.

---

## Fórmulas no Domínio do Tempo

O ECG sintético foi modelado como a soma de pulsos gaussianos centrados em cada batimento:

$$x_{ECG}(t)=\sum_{k} \exp\left(-\frac{(t-t_k)^2}{2\sigma^2}\right)$$

Os ruídos adicionados foram:

$$n_1(t)=\sin(2\pi\cdot 60\,t), \quad n_2(t)=\sin(2\pi\cdot 0.2\,t), \quad n_3(t)\sim \mathcal{N}(0,1)$$

E o sinal final ruidoso foi definido por:

$$x_{r}(t)=x_{ECG}(t)+n_1(t)+n_2(t)+n_3(t)$$

---

## Geração do Sinal Sintético

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy.signal import iirnotch, filtfilt
import pywt

frequencia_amostragem = 360 # 360 hz
duracao = 10 # 10 segundos
frequencia_cardiaca = 1.2 # frequencia cardiaca (hz) 72 bpm

tempo = np.arange(0.5, duracao, 1/frequencia_amostragem)

ecg = np.zeros_like(tempo)

# Instantes dos batimentos
batimentos = np.arange(0.5, duracao, 1/frequencia_cardiaca)

# Adiciona uma gaussiana em cada batimento
for centro in batimentos:
    ecg += np.exp(-((tempo - centro)**2)/(2*0.02**2))

plt.figure(figsize=(10,4))
plt.plot(tempo, ecg)
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("ECG sintético simples")
plt.grid()
plt.show()
```

```python
ruido   = np.sin(2*np.pi*60*tempo)           # senoide 60 Hz
ruido_2 = np.sin(2*np.pi*0.2*tempo)          # senoide 0.2 Hz
ruido_3 = np.random.randn(len(tempo))        # ruído aleatório

# ecg_ruidoso = ecg + ruido + ruido_2 + ruido_3  # sinal com os 3 ruídos
ecg_ruidoso = ecg + ruido  # sinal apenas com o ruído de 60 Hz

plt.figure(figsize=(10,4))
plt.plot(tempo, ecg_ruidoso)
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("ECG com ruído de 60 Hz")
plt.grid()
plt.show()
```

---

## Análise em Frequência

Agora vamos analisar os sinais e os ruídos no domínio da frequência. Isso ajuda a identificar quais componentes dominam cada sinal e facilita a comparação entre eles.

### Transformada de Fourier

Para analisar os componentes espectrais dos sinais, usamos a transformada discreta de Fourier (via FFT):

$$X(f)=\sum_{n=0}^{N-1} x[n]e^{-j2\pi fn/N}$$

A magnitude espectral exibida nos gráficos é:

$$|X(f)|$$

```python
fft_ecg      = np.abs(np.fft.rfft(ecg))
freqs        = np.fft.rfftfreq(len(ecg), d=1/frequencia_amostragem)

fft_60hz     = np.abs(np.fft.rfft(ruido))
fft_02hz     = np.abs(np.fft.rfft(ruido_2))
fft_aletorio = np.abs(np.fft.rfft(ruido_3))
fft_ruidoso  = np.abs(np.fft.rfft(ecg_ruidoso))

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(freqs, fft_60hz)
plt.title("FFT do ruído de 60 Hz")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(freqs, fft_02hz)
plt.title("FFT do ruído de 0.2 Hz")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(freqs, fft_aletorio)
plt.title("FFT do ruído aleatório")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(freqs, fft_ruidoso)
plt.title("FFT do ECG + ruído de 60 Hz")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
```

---

## LMS

O método LMS (Least Mean Squares) é uma técnica adaptativa usada para reduzir ruídos de forma iterativa. A ideia é ajustar os coeficientes do filtro para minimizar o erro entre o sinal desejado e a saída filtrada.

No contexto deste trabalho, ele pode ser aplicado para tentar remover interferências do ECG ruidoso sem depender apenas de um filtro fixo. Isso é útil quando o ruído muda ao longo do tempo ou quando se quer acompanhar variações do sinal em tempo real.

Em termos práticos, o LMS atualiza os pesos do filtro a cada amostra, usando o erro instantâneo como base para a correção.

```python
# Sinal de referência: correlacionado com a interferência de 60 Hz + pequeno ruído
sinal_ref = ruido + 0.05 * np.random.randn(len(tempo))
```

### Algoritmo LMS

O filtro adaptativo LMS atualiza seus coeficientes a cada amostra através das seguintes equações:

**Saída do filtro:**

$$y[n] = \mathbf{w}^T[n-1] \mathbf{x}[n]$$

**Erro de estimação:**

$$e[n] = d[n] - y[n]$$

**Atualização dos pesos:**

$$\mathbf{w}[n] = \mathbf{w}[n-1] + 2\mu e[n] \mathbf{x}[n]$$

onde:
- $\mathbf{w}[n]$ são os coeficientes do filtro na amostra $n$
- $\mathbf{x}[n]$ é o vetor de entrada (sinal de referência)
- $d[n]$ é o sinal desejado (primário/ruidoso)
- $e[n]$ é o erro instantâneo
- $\mu$ é o passo de adaptação (taxa de aprendizado)

```python
def lms_filter(d, x, mu, M):
    """
    d: sinal primário (ruidoso)
    x: sinal de referência (correlacionado com o ruído)
    mu: passo de adaptação
    M: ordem do filtro (nº de coeficientes)
    """
    N = len(d)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_n = x[n-M:n][::-1]       # janela de M amostras mais recentes
        y[n] = np.dot(w, x_n)      # saída do filtro
        e[n] = d[n] - y[n]         # erro instantâneo
        w = w + 2 * mu * e[n] * x_n  # atualização dos pesos
        w_history[n] = w

    return e, y, w_history

mu = 0.01
M  = 5

ecg_lms, ruido_estimado_lms, pesos_lms = lms_filter(ecg_ruidoso, sinal_ref, mu, M)

plt.figure(figsize=(10,4))
plt.plot(tempo, ecg, label="ECG limpo (referência)")
plt.plot(tempo, ecg_lms, label="ECG após LMS", alpha=0.7)
plt.legend()
plt.title("LMS - cancelamento do ruído de 60 Hz")
plt.grid()
plt.show()

# Curva de convergência
plt.figure(figsize=(10,3))
plt.plot(tempo, (ecg_lms - ecg)**2)
plt.title("Erro quadrático (convergência do LMS)")
plt.xlabel("Tempo (s)")
plt.grid()
plt.show()

# Evolução dos pesos
plt.figure(figsize=(10,4))
for i in range(M):
    plt.plot(tempo, pesos_lms[:, i], label=f"w[{i}]")
plt.legend()
plt.title("Evolução dos pesos do LMS (convergência real)")
plt.xlabel("Tempo (s)")
plt.grid()
plt.show()

# FFT antes/depois
fft_antes  = np.abs(np.fft.rfft(ecg_ruidoso))
fft_depois = np.abs(np.fft.rfft(ecg_lms))
freqs      = np.fft.rfftfreq(len(ecg_ruidoso), d=1/frequencia_amostragem)

plt.figure(figsize=(10,4))
plt.plot(freqs, fft_antes, label="Antes do LMS", alpha=0.6)
plt.plot(freqs, fft_depois, label="Depois do LMS", alpha=0.8)
plt.xlim(0, 80)
plt.legend()
plt.title("FFT antes/depois do LMS — note o pico de 60 Hz desaparecendo")
plt.grid()
plt.show()
```

---

### NLMS (Normalized Least Mean Squares)

O algoritmo NLMS é uma versão normalizada do LMS que adapta automaticamente o passo de adaptação com base na potência do sinal de entrada. Isso oferece melhor convergência e estabilidade em comparação com o LMS padrão.

A diferença principal está na atualização dos pesos:

$$\mathbf{w}[n] = \mathbf{w}[n-1] + \frac{\mu}{\|\mathbf{x}[n]\|^2 + \epsilon} e[n] \mathbf{x}[n]$$

onde:
- O denominador $\|\mathbf{x}[n]\|^2$ normaliza o passo de adaptação pela potência do sinal de entrada
- $\epsilon$ é um pequeno valor (geralmente $10^{-6}$) para evitar divisão por zero
- Isso torna o NLMS menos sensível à variação da potência do sinal de referência

```python
def nlms_filter(d, x, mu, M, eps=1e-6):
    N = len(d)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_n  = x[n-M:n][::-1]
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        norm = np.dot(x_n, x_n) + eps
        w    = w + (mu / norm) * e[n] * x_n
        w_history[n] = w

    return e, y, w_history

mu_nlms = 0.5
M       = 5

ecg_nlms, ruido_estimado_nlms, pesos_nlms = nlms_filter(ecg_ruidoso, sinal_ref, mu_nlms, M)

plt.figure(figsize=(10,4))
plt.plot(tempo, ecg, label="ECG limpo (referência)")
plt.plot(tempo, ecg_nlms, label="ECG após NLMS", alpha=0.7)
plt.legend()
plt.title("NLMS - cancelamento do ruído de 60 Hz")
plt.grid()
plt.show()

# Erro quadrático
plt.figure(figsize=(10,3))
plt.plot(tempo, (ecg_nlms - ecg)**2)
plt.title("Erro quadrático (convergência do NLMS)")
plt.xlabel("Tempo (s)")
plt.grid()
plt.show()

# FFT antes/depois
fft_depois_nlms = np.abs(np.fft.rfft(ecg_nlms))
plt.figure(figsize=(10,4))
plt.plot(freqs, fft_antes, label="Antes", alpha=0.6)
plt.plot(freqs, fft_depois_nlms, label="Depois do NLMS", alpha=0.8)
plt.xlim(0, 80)
plt.legend()
plt.title("FFT antes/depois do NLMS")
plt.grid()
plt.show()

# Comparação de magnitude em 60 Hz e convergência dos pesos
freqs  = np.fft.rfftfreq(len(ecg_ruidoso), d=1/frequencia_amostragem)
idx60  = np.argmin(np.abs(freqs - 60))

mag_antes = np.abs(np.fft.rfft(ecg_ruidoso))[idx60]
mag_lms   = np.abs(np.fft.rfft(ecg_lms))[idx60]
mag_nlms  = np.abs(np.fft.rfft(ecg_nlms))[idx60]

print(f"Magnitude em 60Hz — antes: {mag_antes:.1f} | LMS: {mag_lms:.1f} | NLMS: {mag_nlms:.1f}")

plt.figure(figsize=(10,4))
plt.plot(tempo[:300], pesos_lms[:300, 0],  label="LMS - w[0]")
plt.plot(tempo[:300], pesos_nlms[:300, 0], label="NLMS - w[0]")
plt.legend()
plt.title("Velocidade de convergência: LMS vs NLMS")
plt.xlabel("Tempo (s)")
plt.grid()
plt.show()
```

---

## Métrica de Qualidade (SNR)

A relação sinal-ruído foi calculada pela razão entre a potência do sinal limpo e a potência do erro:

$$P_s=\frac{1}{N}\sum_{n=0}^{N-1}x_{limpo}^2[n], \quad P_e=\frac{1}{N}\sum_{n=0}^{N-1}(x_{ruidoso}[n]-x_{limpo}[n])^2$$

$$\mathrm{SNR}_{dB}=10\log_{10}\left(\frac{P_s}{P_e}\right)$$

```python
def comparar_sinais(sinal_limpo, sinal_ruidoso):
    potencia_sinal = np.mean(sinal_limpo**2)
    erro           = sinal_ruidoso - sinal_limpo
    potencia_erro  = np.mean(erro**2)
    snr            = 10 * np.log10(potencia_sinal / potencia_erro)
    print("Potência sinal:", potencia_sinal)
    print("Potência erro:", potencia_erro)
    return snr
```

---

## DWT (Discrete Wavelet Transform)

A DWT decompõe o sinal em coeficientes de aproximação (baixa frequência) e de detalhe (alta frequência), em múltiplos níveis.

Para remover ruído, aplica-se um threshold nos coeficientes de detalhe:
- Coeficientes pequenos (ruído) → zerados
- Coeficientes grandes (sinal real) → mantidos

Reconstruindo o sinal com os coeficientes modificados, obtemos o sinal limpo.

```python
wavelet = 'db4'  # Daubechies ordem 4, boa para ECG
nivel   = 5      # níveis de decomposição

# Decomposição
coeffs = pywt.wavedec(ecg_ruidoso, wavelet, level=nivel)
# coeffs[0]    → aproximação (nível mais grosso)
# coeffs[1..5] → detalhes do nível 5 ao 1

# Visualiza os coeficientes de detalhe
fig, axs = plt.subplots(nivel, 1, figsize=(12, 10))
for i, c in enumerate(coeffs[1:], 1):
    axs[i-1].plot(c)
    axs[i-1].set_title(f"Detalhe nível {nivel+1-i}")
    axs[i-1].grid()
plt.tight_layout()
plt.show()

# Threshold universal de Donoho & Johnstone
sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
threshold = sigma * np.sqrt(2 * np.log(len(ecg_ruidoso)))

print(f"Threshold calculado: {threshold:.4f}")

# Aplica threshold nos coeficientes de detalhe (mantém a aproximação intacta)
coeffs_filtrados = [coeffs[0]]
for c in coeffs[1:]:
    c_thresh = pywt.threshold(c, threshold, mode='soft')
    coeffs_filtrados.append(c_thresh)

# Reconstrução
ecg_dwt = pywt.waverec(coeffs_filtrados, wavelet)
ecg_dwt = ecg_dwt[:len(ecg)]  # garante mesmo tamanho

plt.figure(figsize=(10,4))
plt.plot(tempo, ecg, label="ECG limpo (referência)")
plt.plot(tempo, ecg_dwt, label="ECG após DWT", alpha=0.7)
plt.legend()
plt.title("DWT - denoising do ECG")
plt.grid()
plt.show()

# Erro quadrático
plt.figure(figsize=(10,3))
plt.plot(tempo, (ecg_dwt - ecg)**2)
plt.title("Erro quadrático (DWT)")
plt.xlabel("Tempo (s)")
plt.grid()
plt.show()

# FFT antes/depois
fft_dwt = np.abs(np.fft.rfft(ecg_dwt))
plt.figure(figsize=(10,4))
plt.plot(freqs, fft_antes, label="Antes", alpha=0.6)
plt.plot(freqs, fft_dwt,   label="Depois do DWT", alpha=0.8)
plt.xlim(0, 80)
plt.legend()
plt.title("FFT antes/depois do DWT")
plt.grid()
plt.show()
```

---

## Comparação entre os sinais

Nesta etapa, a função `comparar_sinais` é usada para comparar o ECG original com os sinais obtidos após a filtragem adaptativa. A métrica utilizada é a SNR, que relaciona a potência do sinal limpo com a potência do erro entre o sinal processado e o sinal de referência. Assim, quanto maior for o valor de SNR, mais próximo o sinal filtrado está do ECG original e melhor foi a remoção do ruído.

```python
snr_ruidoso = comparar_sinais(ecg, ecg_ruidoso)
snr_lms     = comparar_sinais(ecg, ecg_lms)
snr_nlms    = comparar_sinais(ecg, ecg_nlms)
snr_dwt     = comparar_sinais(ecg, ecg_dwt)

resultados = pd.DataFrame({
    'Método':   ['Sinal ruidoso', 'LMS', 'NLMS', 'DWT'],
    'SNR (dB)': [snr_ruidoso, snr_lms, snr_nlms, snr_dwt]
})

print(resultados.to_string(index=False))
```
