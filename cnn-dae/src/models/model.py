"""CNN-DAE para filtragem de ruído em sinais de ECG.

Arquitetura encoder–bottleneck–decoder totalmente convolucional.
Entrada e saída: (batch, 1024, 1) — janelas z-score normalizadas.

Uso típico::

    from src.models.model import build_model, build_callbacks

    model = build_model()
    model.summary()
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    UpSampling1D,
)

# ---------------------------------------------------------------------------
# Constante central — deve bater com WINDOW de dataset.py
# ---------------------------------------------------------------------------
WINDOW = 1024


# ---------------------------------------------------------------------------
# Blocos de construção
# ---------------------------------------------------------------------------

def _conv_bn_relu(x: tf.Tensor, filters: int, kernel: int) -> tf.Tensor:
    """Conv1D → BatchNorm → ReLU. Padding 'same' preserva o comprimento temporal.

    BatchNorm depois da Conv e antes da ativação é o padrão mais estável para
    redes de filtragem: normaliza as ativações por batch, reduzindo a
    sensibilidade ao learning rate e acelerando a convergência.
    """
    x = Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def _residual_block(x: tf.Tensor, filters: int, kernel: int) -> tf.Tensor:
    """Dois Conv-BN-ReLU com skip connection.

    Por que conexão residual?
      A tarefa é "sinal ruidoso → sinal limpo". O sinal limpo é muito parecido
      com a entrada — a rede precisa aprender apenas a *diferença* (o ruído).
      Conexões residuais tornam isso explícito: a saída é F(x) + x, então F(x)
      aprende apenas a correção incremental. Também evita gradiente
      desvanecente em redes mais profundas.
    """
    skip = x
    x = _conv_bn_relu(x, filters, kernel)
    x = Conv1D(filters, kernel, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    # Ajusta canais do skip se necessário (projeção 1×1)
    if skip.shape[-1] != filters:
        skip = Conv1D(filters, 1, padding="same", use_bias=False)(skip)
        skip = BatchNormalization()(skip)

    x = Add()([x, skip])
    x = Activation("relu")(x)
    return x


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

def build_model(window: int = WINDOW) -> Model:
    """Constrói e retorna a CNN-DAE compilada.

    Arquitetura
    -----------
    Entrada    : (batch, 1024, 1)

    Encoder    : 4 blocos Conv-BN-ReLU + MaxPool(2)
                 1024 → 512 → 256 → 128 → 64 amostras
                 filtros: 64 → 128 → 256 → 256

    Bottleneck : 2 blocos residuais, 512 filtros, sem pooling
                 mantém 64 amostras; máxima capacidade representacional

    Decoder    : 4 blocos UpSampling(2) + bloco residual
                 64 → 128 → 256 → 512 → 1024 amostras
                 filtros: 256 → 128 → 64 → 32

    Saída      : Conv1D(1, kernel=1, ativação linear)

    Loss       : MAE  — robusto a outliers (batimentos ectópicos, artefatos)
    Métrica    : MSE  — proxy de SNR de reconstrução
    """
    inp = Input(shape=(window, 1), name="ecg_ruidoso")

    # --- ENCODER ---
    # Kernels maiores nas primeiras camadas (campo receptivo amplo na
    # resolução original), menores nas profundas (resolução já reduzida).

    # E1: 1024×1 → 1024×64 → 512×64
    e1 = _conv_bn_relu(inp, filters=64,  kernel=7)
    p1 = MaxPooling1D(pool_size=2, name="pool_1")(e1)

    # E2: 512×64 → 512×128 → 256×128
    e2 = _conv_bn_relu(p1, filters=128, kernel=5)
    p2 = MaxPooling1D(pool_size=2, name="pool_2")(e2)

    # E3: 256×128 → 256×256 → 128×256
    e3 = _conv_bn_relu(p2, filters=256, kernel=3)
    p3 = MaxPooling1D(pool_size=2, name="pool_3")(e3)

    # E4: 128×256 → 128×256 → 64×256
    e4 = _conv_bn_relu(p3, filters=256, kernel=3)
    p4 = MaxPooling1D(pool_size=2, name="pool_4")(e4)

    # --- BOTTLENECK (64 amostras × 512 filtros) ---
    # Dois blocos residuais: o primeiro expande para 512 filtros, o segundo mantém.
    bn = _residual_block(p4, filters=512, kernel=3)
    bn = _residual_block(bn, filters=512, kernel=3)

    # --- DECODER ---
    # UpSampling + bloco residual espelha o encoder.
    # Kernels crescem conforme a resolução é restaurada.

    # D1: 64×512 → 128×256
    d1 = UpSampling1D(size=2, name="up_1")(bn)
    d1 = _residual_block(d1, filters=256, kernel=3)

    # D2: 128×256 → 256×128
    d2 = UpSampling1D(size=2, name="up_2")(d1)
    d2 = _residual_block(d2, filters=128, kernel=3)

    # D3: 256×128 → 512×64
    d3 = UpSampling1D(size=2, name="up_3")(d2)
    d3 = _residual_block(d3, filters=64,  kernel=5)

    # D4: 512×64 → 1024×32
    d4 = UpSampling1D(size=2, name="up_4")(d3)
    d4 = _residual_block(d4, filters=32,  kernel=7)

    # --- SAÍDA ---
    # Conv1D(1, kernel=1): projeção linear de 32 canais → 1 canal.
    # Ativação linear porque o ECG tem valores negativos (onda Q, ST deprimido).
    out = Conv1D(
        1, kernel_size=1, padding="same",
        activation="linear", name="ecg_limpo",
    )(d4)

    model = Model(inputs=inp, outputs=out, name="CNN_DAE")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"],
    )
    return model


# ---------------------------------------------------------------------------
# Callbacks recomendados
# ---------------------------------------------------------------------------

def build_callbacks(
    checkpoint_path: str = "checkpoints/best_model.keras",
) -> list:
    """Retorna lista de callbacks prontos para ``model.fit()``.

    - ModelCheckpoint  : salva o melhor modelo por val_loss
    - EarlyStopping    : interrompe se val_loss não melhorar por 15 épocas
    - ReduceLROnPlateau: reduz o LR pela metade se val_loss estagnar por 7 épocas
    - TensorBoard      : logs para visualização
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=0,
        ),
    ]


if __name__ == "__main__":
    import numpy as np

    model = build_model()
    model.summary(line_length=80)

    x_dummy = np.random.randn(4, WINDOW, 1).astype("float32")
    y_dummy = model(x_dummy, training=False)
    print(f"\nentrada : {x_dummy.shape}")
    print(f"saída   : {y_dummy.shape}")
    assert y_dummy.shape == x_dummy.shape, "Forma de saída incorreta!"
    print("Sanity check OK — entrada e saída têm a mesma forma.")
    print(f"\nParâmetros treináveis : {model.count_params():,}")
