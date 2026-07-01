from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable


class MCTSNode:
    """Un nodo del árbol de búsqueda de MCTS.

    Cada nodo guarda:
    - Una referencia al estado del juego en ese nodo
    - Relaciones padre/hijos para recorrer el árbol
    - Visitas y recompensas acumuladas para calcular UCB
    """
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action          # accion que llevo a este nodo
        self.children = []            # lista de MCTSNode hijos
        self.explored_children = 0    # cuantos hijos ya se visitaron al menos una vez
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))  # recompensa acumulada por agente
        self.agent = self.game.agent_selection          # a quien le toca jugar en este nodo


def ucb(node: MCTSNode, C: float = sqrt(2)) -> float:
    """Upper Confidence Bound (UCB1).

    UCB1 = Q(nodo) / N(nodo) + C * sqrt(ln(N(padre)) / N(nodo))

    El termino de explotacion usa el agente del PADRE porque es el padre quien
    esta eligiendo entre sus hijos, y quiere maximizar su propia recompensa.
    """
    parent_agent_idx = node.game.agent_name_mapping[node.parent.agent]
    return node.cum_rewards[parent_agent_idx] / node.visits + C * sqrt(log(node.parent.visits) / node.visits)


def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    """Elige el hijo con mayor valor de UCB."""
    child = max(node.children, key=ucb)
    return child


class MonteCarloTreeSearch(Agent):
    """Agente de Monte Carlo Tree Search para juegos alternados.

    Cada simulacion repite 4 fases:
    1. Seleccion: bajar por el arbol usando UCB
    2. Expansion: agregar un nodo hijo nuevo
    3. Rollout: jugar movimientos al azar hasta el final
    4. Backpropagation: subir por el arbol actualizando estadisticas

    Al terminar todas las simulaciones, se elige la accion que lleva al hijo
    mas visitado (politica "robusta").
    """
    def __init__(self, game: AlternatingGame, agent: AgentID,
                 simulations: int = 100, rollouts: int = 10,
                 selection: Callable[[MCTSNode, AgentID], MCTSNode] = uct) -> None:
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection

    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> tuple[ActionType, float]:
        root = MCTSNode(parent=None, game=self.game, action=None)

        for i in range(self.simulations):
            node = root
            node.game = self.game.clone()

            # Fase 1: Seleccion — recorrer el arbol usando UCB
            node = self.select_node(node=node)

            # Fase 2: Expansion — agregar hijos al nodo seleccionado
            self.expand_node(node)

            # Fase 3: Rollout — simular una partida al azar
            rewards = self.rollout(node)

            # Fase 4: Backpropagation — actualizar estadisticas hacia la raiz
            self.backprop(node, rewards)

        action, value = self.action_selection(root)
        return action, value

    def backprop(self, node: MCTSNode, rewards: np.ndarray) -> None:
        """Propaga las recompensas desde la hoja hasta la raiz.

        Sube por el arbol siguiendo los punteros a padre, sumando visitas y
        recompensas en cada nodo del camino.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.cum_rewards += rewards
            current = current.parent

    def rollout(self, node: MCTSNode) -> np.ndarray:
        """Juega partidas al azar desde el estado del nodo.

        Juega self.rollouts partidas aleatorias desde esta posicion y promedia
        las recompensas finales.
        """
        rewards = np.zeros(len(self.game.agents))

        for _ in range(self.rollouts):
            # Clonar el estado para no modificar el del nodo
            rollout_game = node.game.clone()

            # Jugar al azar hasta que termine la partida
            while not rollout_game.game_over():
                available = rollout_game.available_actions()
                action = available[np.random.randint(len(available))]
                rollout_game.step(action)

            # Acumular la recompensa final de cada agente
            for i, agent_id in enumerate(rollout_game.agents):
                rewards[i] += rollout_game.reward(agent_id)

        # Promediar entre todos los rollouts
        rewards /= self.rollouts
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Baja por el árbol siguiendo la política de selección.

        En cada nodo interno:
        - Si hay hijos sin visitar, se elige el siguiente (así todos los hijos
          se visitan al menos una vez antes de usar UCB)
        - Si ya se visitaron todos, se usa la seleccion por UCB

        Devuelve una hoja (un nodo sin hijos todavia).
        """
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                # Elegir el siguiente hijo sin visitar
                child = curr_node.children[curr_node.explored_children]
                curr_node.explored_children += 1
                curr_node = child
            else:
                # Todos los hijos ya se visitaron al menos una vez — usar UCB
                curr_node = self.selection(curr_node, self.agent)
        return curr_node

    def expand_node(self, node: MCTSNode) -> None:
        """Expande el nodo creando un hijo por cada acción legal.

        Si la partida no terminó, crea un MCTSNode por cada acción disponible,
        guardando el estado resultante de jugarla.
        """
        if node.game.game_over():
            return

        for action in node.game.available_actions():
            child_game = node.game.clone()
            child_game.step(action)
            child_node = MCTSNode(parent=node, game=child_game, action=action)
            node.children.append(child_node)

    def action_selection(self, node: MCTSNode) -> tuple[ActionType, float]:
        """Elige la mejor acción desde la raíz, una vez terminadas las simulaciones.

        Usa el criterio del "hijo robusto": el hijo mas visitado. Es mas estable
        que elegir por valor promedio, sobre todo con pocas simulaciones.
        """
        if not node.children:
            return None, 0.0

        # Hijo robusto: el mas visitado
        best_child = max(node.children, key=lambda c: c.visits)

        agent_idx = node.game.agent_name_mapping[self.agent]
        value = best_child.cum_rewards[agent_idx] / best_child.visits if best_child.visits > 0 else 0.0

        return best_child.action, value
