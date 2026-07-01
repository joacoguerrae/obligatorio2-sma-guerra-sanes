# Obligatorio 2 — Sistemas Multiagente: Juegos Alternados

**Materia:** Sistemas Multiagente — Universidad ORT Uruguay
**Equipo:** Ramiro Sanes & Joaquín Guerra

Implementación y validación de algoritmos de aprendizaje/búsqueda para juegos
alternados multiagente, sobre el template de la cátedra (`base/`).

## Algoritmos

| Algoritmo | Tipo | Información | Archivo |
|---|---|---|---|
| **MCTS** — Monte Carlo Tree Search | Búsqueda + simulación | Perfecta | `agents/mcts.py` |
| **CFR** — Counterfactual Regret Minimization | Aprendizaje por *regret* | Imperfecta | `agents/cfr.py` |
| **ISMCTS** — Information Set MCTS | Búsqueda + determinización | Imperfecta | `agents/ismcts.py` |
| MiniMax (provisto) | Búsqueda exhaustiva | Perfecta | `agents/minimax.py` |
| Random (provisto) | — | — | `agents/agent_random.py` |

## Juegos

| Juego | Información | Jugadores | Archivo |
|---|---|---|---|
| Tic-Tac-Toe | Perfecta | 2 | `games/tictactoe/` |
| Nocca-Nocca | Perfecta | 2 | `games/nocca_nocca/` |
| Kuhn Poker | Imperfecta | 2 | `games/kuhn.py` |
| Kuhn Poker | Imperfecta | 3 | `games/kuhn3.py` |
| Leduc Poker | Imperfecta | 2 | `games/leduc.py` |

## Estructura

```
├── base/                  # AlternatingGame, Agent (template de la cátedra)
├── agents/                # minimax (provisto), agent_random, mcts, cfr, ismcts
├── games/                 # tictactoe, nocca_nocca, kuhn, kuhn3p, leduc
├── utils.py               # utilidades de experimentación y gráficas
├── requirements.txt
├── 01_tictactoe.ipynb     # MCTS + Minimax (info perfecta)
├── 02_nocca_nocca.ipynb   # MCTS + Minimax (info perfecta, juego grande)
├── 03_kuhn.ipynb          # CFR + ISMCTS (Kuhn 2 jugadores)
├── 04_kuhn3p.ipynb        # CFR + ISMCTS (Kuhn 3 jugadores)
├── 05_leduc.ipynb         # CFR + ISMCTS (Leduc 2 jugadores)
└── INFORME.pdf            # Informe detallado: teoría, experimentación y conclusiones
```

## Cómo correr (VS Code)

1. Abrir esta carpeta como **workspace** en VS Code.
2. Crear el entorno e instalar dependencias. **Opción rápida (Windows PowerShell):**
   ```powershell
   .\create_venv.ps1            # crea .venv e instala el proyecto (pip install -e .)
   .\.venv\Scripts\Activate.ps1
   ```
   **Opción manual (cualquier SO):**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\Activate.ps1   |   Linux/Mac: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Abrir cualquier notebook y, arriba a la derecha, **seleccionar el kernel** del
   `.venv` recién creado.
4. Ejecutar las celdas. La primera celda de cada notebook agrega la raíz del
   proyecto al `sys.path`, así que los `import base/agents/games` funcionan tanto si
   se instalo el proyecto (`pip install -e .`) como si no.


