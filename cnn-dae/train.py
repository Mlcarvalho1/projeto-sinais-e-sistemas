"""Script de treino da CNN-DAE."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.models.dataset import build_data_pools, make_train_dataset, make_val_dataset
from src.models.model import build_callbacks, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina a CNN-DAE de filtragem de ECG.")
    p.add_argument("--epochs",     type=int, default=150,
                   help="Épocas máximas (EarlyStopping pode interromper antes)")
    p.add_argument("--batch",      type=int, default=32,  help="Batch size")
    p.add_argument("--steps",      type=int, default=400, help="Steps por época (treino)")
    p.add_argument("--val-steps",  type=int, default=100, help="Steps por época (val)")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/best_model.keras",
                   help="Caminho para salvar o melhor modelo")
    p.add_argument("--seed-train", type=int, default=42)
    p.add_argument("--seed-val",   type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Carregando pools de dados...")
    pools = build_data_pools()
    print(f"  janelas treino : {pools.ecg_train_windows.shape}")
    print(f"  janelas val    : {pools.ecg_val_windows.shape}")
    print(f"  janelas teste  : {pools.ecg_test_windows.shape}")

    train_ds = make_train_dataset(pools, batch_size=args.batch, seed=args.seed_train)
    val_ds   = make_val_dataset(pools,   batch_size=args.batch, seed=args.seed_val)

    print("\nConstruindo modelo...")
    model = build_model()
    model.summary(line_length=80)
    print(f"  Parâmetros treináveis: {model.count_params():,}\n")

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    print(f"Iniciando treino: {args.epochs} épocas, {args.steps} steps/época\n")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=val_ds,
        validation_steps=args.val_steps,
        callbacks=build_callbacks(args.checkpoint),
        verbose=1,
    )

    best_val_loss = min(history.history["val_loss"])
    best_epoch    = history.history["val_loss"].index(best_val_loss) + 1
    print(f"\nMelhor val_loss: {best_val_loss:.6f}  (época {best_epoch})")
    print(f"Modelo salvo em: {args.checkpoint}")


if __name__ == "__main__":
    main()
