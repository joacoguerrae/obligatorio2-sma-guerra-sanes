from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable


class ISMCTSNode:
    """Un nodo del árbol de Information Set MCTS.

    A diferencia de MCTS clásico, donde cada nodo representa un estado puntual
    del juego, en ISMCTS cada nodo representa un information set — lo que el
    jugador que actúa puede observar.

    Los hijos se indexan por acción (no por el estado resultante), porque la
    misma acción tomada desde el mismo information set debe agruparse junto,
    sin importar cuál sea el estado oculto real.
    """
    def __init__(self, parent: 'ISMCTSNode', action: ActionType, agent: AgentID):
        self.parent = parent
        self.action = action          # accion que llevo a este nodo
        self.agent = agent            # quien juega en este nodo
        self.children: dict[ActionType, 'ISMCTSNode'] = {}  # accion -> nodo hijo
        self.visits = 0
        self.cum_rewards = {}         # agent_id -> recompensa acumulada
        self.availability_count = 0   # cuantas veces estuvo disponible este nodo


def ismcts_ucb(node: ISMCTSNode, parent_agent: AgentID, C: float = sqrt(2)) -> float:
    """Variante de UCB1 para ISMCTS.

    Evalua la recompensa desde la perspectiva del agente del padre, y usa el
    availability_count (en vez de las visitas del padre) para el termino de
    exploracion, porque no todos los hijos estan disponibles en cada
    determinizacion.
    """
    if node.visits == 0:
        return float('inf')

    reward = node.cum_rewards.get(parent_agent, 0) / node.visits
    exploration = C * sqrt(log(node.availability_count) / node.visits)
    return reward + exploration


class ISMCTS(Agent):
    """Information Set Monte Carlo Tree Search.

    Extiende MCTS a juegos de información imperfecta (donde los jugadores no
    conocen el estado completo del juego).

    Diferencias clave respecto a MCTS clásico:
    - Antes de cada simulación se "determiniza" el juego: se sortea al azar la
      información oculta, de forma consistente con lo que el agente sabe
    - El árbol se construye sobre information sets, no sobre estados puntuales
    - En la selección solo se consideran los hijos cuya acción es legal en la
      determinización actual

    Esta es la variante Single-Observer ISMCTS (SO-ISMCTS): el árbol se
    construye desde la perspectiva de un solo jugador.
    """

    def __init__(self, game: AlternatingGame, agent: AgentID,
                 simulations: int = 1000, C: float = sqrt(2)) -> None:
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.C = C

    def action(self) -> ActionType:
        a, _ = self.ismcts()
        return a

    def ismcts(self) -> tuple[ActionType, float]:
        """Corre ISMCTS y devuelve la mejor acción.

        Por cada simulación:
        1. Determinizar: sortear la información oculta
        2. Seleccionar: bajar por el árbol usando UCB (solo acciones legales)
        3. Expandir: agregar un hijo nuevo para una acción no probada
        4. Rollout: jugar al azar hasta el final
        5. Backpropagation: subir actualizando estadísticas

        Al final, devuelve la acción más visitada desde la raíz.
        """
        root = ISMCTSNode(parent=None, action=None, agent=self.game.agent_selection)

        for _ in range(self.simulations):
            # Determinizar: version del juego con la info oculta sorteada
            det_game = self.game.random_change(self.agent)

            # Seleccion + expansion
            node, det_game = self._select_and_expand(root, det_game)

            # Rollout
            rewards = self._rollout(det_game)

            # Backpropagation
            self._backprop(node, rewards)

        # Elegir la accion mas visitada
        return self._best_action(root)

    def _select_and_expand(self, node: ISMCTSNode, game: AlternatingGame) -> tuple[ISMCTSNode, AlternatingGame]:
        """Recorre el árbol y expande un nodo nuevo.

        En cada nodo:
        1. Se calculan las acciones legales en la determinizacion actual
        2. Si hay alguna sin probar, se expande
        3. Si no, se elige entre los hijos ya probados usando UCB
        """
        while not game.game_over():
            available = game.available_actions()
            current_agent = game.agent_selection

            # Acciones legales que todavia no tienen hijo
            untried = [a for a in available if a not in node.children]

            if untried:
                # Expandir: elegir una accion sin probar al azar
                action = untried[np.random.randint(len(untried))]

                # Crear el nodo hijo nuevo
                game_copy = game.clone()
                game_copy.step(action)

                next_agent = game_copy.agent_selection if not game_copy.game_over() else current_agent
                child = ISMCTSNode(parent=node, action=action, agent=next_agent)
                node.children[action] = child

                # Actualizar disponibilidad de los hijos legales
                for a in available:
                    if a in node.children:
                        node.children[a].availability_count += 1

                return child, game_copy
            else:
                # Ya se probaron todas las acciones legales — elegir con UCB
                for a in available:
                    if a in node.children:
                        node.children[a].availability_count += 1

                best_action = max(
                    available,
                    key=lambda a: ismcts_ucb(node.children[a], current_agent, self.C)
                )
                node = node.children[best_action]
                game.step(best_action)

        return node, game

    def _rollout(self, game: AlternatingGame) -> dict[AgentID, float]:
        """Juega al azar hasta llegar a un estado terminal."""
        rollout_game = game.clone()

        while not rollout_game.game_over():
            available = rollout_game.available_actions()
            action = available[np.random.randint(len(available))]
            rollout_game.step(action)

        return {agent: rollout_game.reward(agent) for agent in rollout_game.agents}

    def _backprop(self, node: ISMCTSNode, rewards: dict[AgentID, float]) -> None:
        """Propaga las recompensas desde el nodo hasta la raíz."""
        current = node
        while current is not None:
            current.visits += 1
            for agent_id, reward in rewards.items():
                current.cum_rewards[agent_id] = current.cum_rewards.get(agent_id, 0) + reward
            current = current.parent

    def _best_action(self, root: ISMCTSNode) -> tuple[ActionType, float]:
        """Elige la mejor acción desde la raíz según la cantidad de visitas."""
        if not root.children:
            return None, 0.0

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        best_child = root.children[best_action]

        value = best_child.cum_rewards.get(self.agent, 0) / best_child.visits if best_child.visits > 0 else 0.0

        return best_action, value

    def policy(self):
        """ISMCTS no mantiene una política persistente; usar action() para jugar."""
        raise NotImplementedError("ISMCTS no mantiene una politica persistente. Usar action() para obtener una accion.")
