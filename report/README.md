# Relatório (LaTeX)

Esqueleto do relatório: **Introdução + Fundamentação Teórica + Metodologia**, com as
seções de Resultados/Conclusão marcadas para a equipe completar (Etapas C/D).

- `main.tex` — documento principal.
- `referencias.bib` — bibliografia (BibTeX, estilo `ieeetr`).
- Figuras vêm de `../figures/` (gerar com `./.venv/bin/python scripts/gerar_figuras.py`).

## Compilar

Local (TeX Live) ou Overleaf self-hosted:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Compila sem dependências extras (sem `siunitx`); unidades em texto simples para
máxima portabilidade. No Overleaf, basta subir `main.tex`, `referencias.bib` e a
pasta `figures/`.
