# Robótica Móvel

Repositório da disciplina **Robótica Móvel** (2º semestre de 2025), Programa de Pós-Graduação em Ciência da Computação — UFMG, ministrada pelo Prof. Douglas G. Macharet.

**Aluno:** Rafael Santos Oliveira

## Estrutura

- [`aulas/`](aulas) — slides, notebooks e cenas do CoppeliaSim usados em cada aula (aula1 a aula14).
- [`tp1/`](tp1/README.md) — TP1: Ferramentas e Transformações (descrição espacial, transformações homogêneas).
- [`tp2/`](tp2/README.md) — TP2: Planejamento e Navegação (Roadmap, Campos Potenciais, RRT).
- [`tp3/`](tp3/README.md) — TP3: Exploração e Mapeamento (Occupancy Grid com o robô Kobuki).

Cada trabalho prático (`tpX/`) tem seu próprio README com detalhes de execução, além do PDF com o enunciado original.

## Ambiente

Todos os trabalhos usam:

- [CoppeliaSim](https://www.coppeliarobotics.com/) para simulação dos robôs e cenas (arquivos `.ttt`).
- Python 3 + Jupyter Notebook, comunicando com o CoppeliaSim via `coppeliasim-zmqremoteapi-client`.

Para rodar o código de um TP:

```bash
cd tpX
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Abra a cena `.ttt` correspondente no CoppeliaSim antes de executar o notebook.
