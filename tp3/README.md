# TP3 — Exploração e Mapeamento

Rafael Santos Oliveira

Enunciado completo em [`TP3.pdf`](TP3.pdf).

## Objetivo

Implementar o algoritmo de Occupancy Grid para mapeamento, com uma estratégia de navegação/exploração (wall-following) para o robô diferencial Kobuki equipado com um sensor laser (Hokuyo), considerando a localização do robô conhecida (via RemoteAPI) e ruído aleatório na leitura do laser.

## Estrutura

- `cenas/` — cenas do CoppeliaSim: `cena-tp3-estatico.ttt` (obstáculos estáticos), `cena-tp3-dinamico.ttt` (obstáculos dinâmicos) e `cena_Testes_Simples.ttt` (cena de teste).
- `src/estatico/` — implementação usada no cenário estático:
  - `python/controleWallFollowing.py` — controlador de wall-following.
  - `python/HokuyoSensorSim.py` — simulação do sensor laser Hokuyo.
  - `main.ipynb` — notebook principal (execução completa).
  - `grid*.png`, `output.png`, `python/occupancy_grid_final.png`, `python/scatter_plot_final.png` — mapas de ocupação e resultados gerados.
  - `lua*.lua` — scripts Lua originais/modificados usados na cena.
- `melhor.png` — melhor resultado obtido, usado no relatório.
- `TP3.pdf` — relatório entregue.

## Como executar

1. Abra `cenas/cena-tp3-estatico.ttt` (ou a variante desejada) no CoppeliaSim.
2. `pip install -r requirements.txt`
3. Rode `src/estatico/main.ipynb` no Jupyter.
