import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent


class Node():
    """Representa un único information set en el árbol.

    Cada nodo corresponde a una observación única (lo que un jugador puede ver).
    Almacena los regrets acumulados y las sumas de estrategia necesarias para
    el regret matching y el cálculo de la estrategia promedio.
    """

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def regret_matching(self):
        """Actualiza la política actual usando regret matching.

            El algoritmo de regret matching:
            1. Toma solo los regrets positivos (los negativos se ignoran)
            2. Si la suma de regrets positivos > 0, normaliza para obtener la política
            3. De lo contrario, usa una política uniforme aleatoria

            Esto asegura que la política converge al equilibrio de Nash
            a lo largo de muchas iteraciones.
        """
        positive_regrets = np.maximum(self.cum_regrets, 0)
        total = positive_regrets.sum()
        if total > 0:
            self.curr_policy = positive_regrets / total
        else:
            self.curr_policy = np.full(self.num_actions, 1 / self.num_actions)

    def update(self, utility, node_utility, probability) -> None:
        """Actualiza los regrets y la estrategia promedio de este nodo.

        Args:
            utility: utilidad contrafactual de cada acción posible
            node_utility: utilidad esperada del nodo (ponderada por la política actual)
            probability: probabilidad de que el/los rivales lleguen a este nodo
        """
        # Actualizar los regrets acumulados: regret[a] = utility[a] - node_utility
        for a in range(self.num_actions):
            regret = utility[a] - node_utility
            self.cum_regrets[a] += probability * regret

        # Actualizar la suma de politicas (para la estrategia promedio)
        self.sum_policy += self.curr_policy
        self.niter += 1

        # Calcular la estrategia promedio (politica aprendida)
        total = self.sum_policy.sum()
        if total > 0:
            self.learned_policy = self.sum_policy / total
        else:
            self.learned_policy = np.full(self.num_actions, 1 / self.num_actions)

        # Actualizar la politica actual via regret matching
        self.regret_matching()

    def policy(self):
        return self.learned_policy


class CounterFactualRegret(Agent):
    """Agente de Counterfactual Regret Minimization (CFR).

    CFR es un algoritmo iterativo que aproxima equilibrios de Nash en juegos
    secuenciales de información imperfecta.

    Idea central: para cada information set, se lleva la cuenta del "regret"
    de no haber jugado cada acción. Con suficientes iteraciones, la estrategia
    promedio converge al equilibrio de Nash.

    El algoritmo recorre todo el árbol del juego una vez por iteración,
    calculando utilidades contrafactuales y actualizando los regrets.
    """

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        """Elige una acción usando la política aprendida.

        Filtra la política a las acciones disponibles y renormaliza.
        """
        try:
            obs = self.game.observe(self.agent)
            node = self.node_dict[obs]
            available = self.game.available_actions()
            policy = node.policy()

            # Filtrar la politica a las acciones disponibles
            avail_probs = np.array([policy[a] for a in available])
            total = avail_probs.sum()
            if total > 0:
                avail_probs = avail_probs / total
            else:
                avail_probs = np.ones(len(available)) / len(available)

            # Sortear una accion entre las disponibles segun la politica filtrada
            a = np.random.choice(available, p=avail_probs)
            return a
        except KeyError:
            return np.random.choice(self.game.available_actions())

    def train(self, niter=1000):
        """Corre CFR durante niter iteraciones."""
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        """Una iteración completa de CFR.

        Para cada agente, recorre el árbol completo desde la raíz calculando
        utilidades contrafactuales y actualizando regrets.
        """
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        """Recorrido recursivo de CFR sobre el árbol del juego.

        Args:
            game: estado actual del juego (se clona para cada rama)
            agent: agente para el cual se calcula la utilidad en este recorrido
            probability: probabilidades de alcance — probability[i] es el
                producto de las probabilidades de accion que tomo el jugador i
                para llegar a este estado

        Returns:
            node_utility: la utilidad contrafactual de este nodo para `agent`
        """
        # Estado terminal: devolver la recompensa
        if game.game_over():
            return game.reward(agent)

        # Jugador que actua en este nodo
        current_player = game.agent_selection
        current_idx = game.agent_name_mapping[current_player]

        # Buscar o crear el nodo de information set para esta observacion
        obs = game.observe(current_player)
        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)
        node = self.node_dict[obs]

        # Acciones disponibles (solo las legales)
        available = game.available_actions()

        # Politica actual en este nodo
        policy = node.curr_policy.copy()

        # Normalizar la politica sobre las acciones disponibles
        avail_probs = np.array([policy[a] for a in available])
        prob_sum = avail_probs.sum()
        if prob_sum > 0:
            avail_probs = avail_probs / prob_sum
        else:
            avail_probs = np.ones(len(available)) / len(available)

        # Calcular la utilidad de cada accion disponible
        num_actions = node.num_actions
        action_utilities = np.zeros(num_actions)

        for i, a in enumerate(available):
            # Clonar el juego y tomar la accion a
            child_game = game.clone()
            child_game.step(a)

            # Actualizar la probabilidad de alcance del jugador actual
            new_probability = probability.copy()
            new_probability[current_idx] *= avail_probs[i]

            # Recursion
            action_utilities[a] = self.cfr_rec(child_game, agent, new_probability)

        # Utilidad esperada del nodo (ponderada por la politica sobre acciones disponibles)
        node_utility = sum(avail_probs[i] * action_utilities[a] for i, a in enumerate(available))

        # Actualizar regrets solo para los nodos del jugador actual
        if current_player == agent:
            # Probabilidad de alcance de los rivales = producto de la de todos los demas
            opp_probability = 1.0
            for idx in range(len(probability)):
                if idx != current_idx:
                    opp_probability *= probability[idx]

            node.update(
                utility=action_utilities,
                node_utility=node_utility,
                probability=opp_probability
            )

        return node_utility
