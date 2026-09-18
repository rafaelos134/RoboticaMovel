# TP1 — Ferramentas e Transformações

Rafael Santos Oliveira

Enunciado completo em [`insrucoesPDF/TP1.pdf`](insrucoesPDF/TP1.pdf).

## Objetivo

Familiarização com o CoppeliaSim e com descrição espacial / transformações homogêneas: criar uma cena com um robô móvel e outros elementos, definir seus referenciais e representar as transformações entre eles, calculando a pose dos elementos no referencial local do robô.

## Estrutura

- `src/` — notebooks (`trabalho.ipynb`, `notebook-tp1.ipynb`) e cenas do CoppeliaSim (`cena1.ttt`, `cena2.ttt`, `t1.ttt`).
- `imgs/` — capturas das cenas e das nuvens de pontos usadas na documentação.
- `testes/` — notebooks e scripts de teste/rascunho.
- `documentacao.pdf` — relatório entregue.
- `requirements.txt` — dependências Python (Jupyter + `coppeliasim-zmqremoteapi-client`).

## Como executar

1. Abra `src/cena1.ttt` (ou `cena2.ttt`) no CoppeliaSim e inicie a simulação.
2. `pip install -r requirements.txt`
3. Rode `src/trabalho.ipynb` (ou `notebook-tp1.ipynb`) no Jupyter.
