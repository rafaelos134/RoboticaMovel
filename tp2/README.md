# TP2 — Planejamento e Navegação

Rafael Santos Oliveira

Enunciado completo em [`insrucoesPDF/TP2.pdf`](insrucoesPDF/TP2.pdf).

## Objetivo

Implementar e comparar algoritmos de planejamento de caminhos para dois tipos de robô:

- **Roadmap** — robô holonômico
- **Campos Potenciais** — robô diferencial
- **RRT** — robô holonômico

Para cada algoritmo são feitos ao menos dois experimentos em cenários/mapas diferentes.

## Estrutura

- `src/1_Roadmap/`, `src/2_CamposPotenciais/`, `src/3_RRT/` — notebook e cena do CoppeliaSim de cada algoritmo.
- `src/mapas_meus/`, `src/mapas_meus_prov/`, `src/mapas_moodle/` — mapas usados para gerar os cenários (originais do Moodle e versões próprias/invertidas/redimensionadas).
- `src/imagens_pdf/` — figuras usadas no relatório final.
- `src/reescale.py` — utilitário para redimensionar/ajustar os mapas.
- `testes/` — notebooks de aula reaproveitados como teste/rascunho.
- `photoshop/` — arquivos-fonte (`.xcf`) dos mapas editados.

### Versões

Esta pasta tem duas subpastas com versões alternativas do trabalho, mantidas como histórico:

- [`tp2-prov/`](tp2-prov/README.md) — versão provisória/rascunho, anterior à entrega final.
- [`tp2_fim/`](tp2_fim/README.md) — versão final entregue.

## Como executar

1. Abra a cena `.ttt` do algoritmo desejado (em `src/<algoritmo>/`) no CoppeliaSim.
2. `pip install -r requirements.txt`
3. Rode o `main_*.ipynb` correspondente no Jupyter.
