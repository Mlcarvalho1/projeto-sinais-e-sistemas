"""Garante que a raiz do repositório está no sys.path para ``import src...``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
