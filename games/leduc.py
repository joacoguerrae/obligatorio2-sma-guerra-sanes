from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from pettingzoo.classic import leduc_holdem_v4 as leduc
from base.game import AlternatingGame, AgentID, ActionType
import numpy as np
import copy
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

# Atributos de runtime que el raw_env de PettingZoo excluye de __getstate__
# (y por lo tanto se pierden al hacer copy.deepcopy). Hay que restaurarlos a mano
# para que un clon sea jugable.
_RUNTIME_ATTRS = [
    '_cumulative_rewards', '_last_obs', 'agent_selection', 'infos',
    'next_legal_moves', 'rewards', 'terminations', 'truncations',
]


class Leduc(AlternatingGame):
    """Leduc Hold'em — versión de la cátedra, basada en `pettingzoo.classic.leduc_holdem_v4`.

    A la versión provista se le agregaron únicamente:
    - `clone()`: copia funcional del juego (el `deepcopy` del entorno de PettingZoo
      no preserva el estado de runtime; ver `_RUNTIME_ATTRS`).
    - `random_change()`: determinización del estado oculto para ISMCTS.
    """

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = leduc.raw_env(render_mode=render_mode)
        # Reset inicial para poblar agents / espacios antes de usarse.
        self.env.reset()
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.render_mode = render_mode
        self._hist = ''
        self._moves = ['c', 'r', 'f', 'k']
        self._update()
        self._hist = str(self.env._name_to_int(self.agent_selection))

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def observe(self, agent: AgentID) -> ObsType:
        state = self.env.env.game.get_state(self.env._name_to_int(agent))
        hand = state['hand'][1]
        public_card = '#' if state['public_card'] is None else state['public_card'][1]
        chips = '_'.join([str(x) for x in state['all_chips']])
        obs = hand + '_' + public_card + '_' + chips + '_' + self._hist
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.env.reset(seed, options)
        self._update()
        self._hist = str(self.env._name_to_int(self.agent_selection))

    def render(self) -> ndarray | str | list | None:
        return self.env.render()

    def step(self, action: ActionType) -> None:
        self._hist += self._moves[action]
        self.env.step(action)
        self._update()

    def available_actions(self):
        return list(self.env.next_legal_moves)

    def clone(self):
        """Copia independiente y jugable del juego.

        El `__getstate__` del `raw_env` de PettingZoo no solo excluye el estado de
        runtime, sino que al reconstruirse **reinicia el juego rlcard interno**
        (p. ej. el contador de subidas `have_raised`), lo que rompería la búsqueda.
        Por eso: (1) injertamos un `deepcopy` del env rlcard interno (`self.env.env`),
        que sí preserva todo el estado del juego, y (2) restauramos a mano los
        atributos de runtime del `raw_env` (ver `_RUNTIME_ATTRS`).
        """
        new_game = copy.deepcopy(self)
        new_game.env.env = copy.deepcopy(self.env.env)
        for attr in _RUNTIME_ATTRS:
            if hasattr(self.env, attr):
                setattr(new_game.env, attr, copy.deepcopy(getattr(self.env, attr)))
        new_game._update()
        return new_game

    def random_change(self, agent: AgentID):
        """Determinización para ISMCTS: re-muestrea la carta oculta del rival.

        El agente conoce su carta y la carta comunitaria (si fue revelada). Las
        cartas no vistas son la del rival más las que quedan en el mazo; tomamos
        una de ellas para el rival y devolvemos el resto al mazo, manteniendo la
        consistencia del estado.
        """
        new_game = self.clone()
        game = new_game.env.env.game
        me = self.env._name_to_int(agent)
        opp = 1 - me
        candidates = [game.players[opp].hand] + list(game.dealer.deck)
        if len(candidates) > 0:
            j = np.random.randint(len(candidates))
            chosen = candidates[j]
            game.players[opp].hand = chosen
            game.dealer.deck = [c for k, c in enumerate(candidates) if k != j]
        return new_game

    def close(self):
        self.env.close()
