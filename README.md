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
   instalaste el proyecto (`pip install -e .`) como si no.

> Cada notebook es autocontenido; pueden correrse en cualquier orden. El
> recomendado para leer es `01 → 05` y luego `INFORME.pdf`.

## Notas

- **Tiempos de ejecución.** Algunos experimentos son costosos (especialmente
  `02_nocca_nocca`, por el gran espacio de acciones, y los barridos de ISMCTS).
  Para acelerar, bajar los parámetros `NG` / `n_games` / `simulations` al inicio de
  cada notebook. Cada notebook indica su costo aproximado.
- **MiniMax** se usa tal cual lo provee la cátedra (sin poda alfa-beta); por eso en
  Tic-Tac-Toe se acota la profundidad y en Nocca-Nocca se trabaja a profundidad
  baja con la función de evaluación.
- **Kuhn 2P** (`games/kuhn.py`), **Kuhn 3P** (`games/kuhn3.py`) y **Leduc**
  (`games/leduc.py`) son las versiones de la **cátedra**. La Leduc envuelve
  `pettingzoo.classic.leduc_holdem_v4` (rlcard), por eso `requirements.txt` usa
  `pettingzoo[classic]`. A las tres se les agregó solo `random_change`
  (determinización para ISMCTS); a Leduc además un `clone()` correcto (el
  `deepcopy` del entorno de PettingZoo reinicia el estado interno del juego).
- **Costo:** CFR e ISMCTS sobre la Leduc de la cátedra (rlcard) son **bastante más
  lentos** que sobre Kuhn, por el costo de clonar el entorno. `05_leduc.ipynb` usa
  pocas iteraciones/simulaciones por eso; subirlas mejora la calidad a más tiempo.
- **Uso de IA generativa:** ver sección 11 de `INFORME.pdf`.
