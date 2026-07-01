# noqa: D212, D415
"""
# Tic Tac Toe

```{figure} classic_tictactoe.gif
:width: 140px
:name: tictactoe
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import tictactoe_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= ['player_1', 'player_2']`            |
| Agents             | 2                                             |
| Action Shape       | (1)                                           |
| Action Values      | [0, 8]                                        |
| Observation Shape  | (3, 3, 2)                                     |
| Observation Values | [0,1]                                         |


Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation is 2 planes of the 3x3 board. For player_1, the first plane represents the placement of Xs, and the second plane shows the placement of Os. The possible values for each cell are 0 or 1; in the first plane, 1 indicates that an X has been placed in that cell, and 0 indicates
that X is not in that cell. Similarly, in the second plane, 1 indicates that an O has been placed in that cell, while 0 indicates that an O has not been placed. For player_2, the observation is the same, but Xs and Os swap positions, so Os are encoded in plane 1 and Xs in plane 2. This allows for
self-play.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:


 ```
0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8
 ```

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.

### Version History

* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.classic.tictactoe.board import TTT_GAME_NOT_OVER, TTT_TIE, Board
from pettingzoo.utils import AgentSelector, wrappers

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "tictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self, render_mode: str | None = None, screen_height: int | None = 1000
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, screen_height)
        self.board = Board()

        self.agents = ["X", "O"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None
        
    def observe(self, agent):
        board_vals = np.array(self.board.squares).reshape(3, 3)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        observation = np.empty((3, 3, 2), dtype=np.int8)
        # this will give a copy of the board that is 1 for player 1's
        # marks and zero for every other square, whether empty or not.
        observation[:, :, 0] = np.equal(board_vals, cur_player + 1)
        observation[:, :, 1] = np.equal(board_vals, opp_player + 1)

        action_mask = self._get_mask(agent)

        return {"observation": observation, "action_mask": action_mask}

    def _get_mask(self, agent):
        action_mask = np.zeros(9, dtype=np.int8)

        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        if agent == self.agent_selection:
            for i in self.board.legal_moves():
                action_mask[i] = 1

        return action_mask

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # action in this case is a value from 0 to 8 indicating position to move on tictactoe board
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self.board.play_turn(self.agents.index(self.agent_selection), action)

        status = self.board.game_status()
        if status != TTT_GAME_NOT_OVER:
            if status == TTT_TIE:
                pass
            else:
                winner = status  # either TTT_PLAYER1_WIN or TTT_PLAYER2_WIN
                loser = winner ^ 1  # 0 -> 1; 1 -> 0
                self.rewards[self.agents[winner]] += 1
                self.rewards[self.agents[loser]] -= 1

            # once either play wins or there is a draw, game over, both players are done
            self.terminations = {i: True for i in self.agents}

        self.agent_selection = self.agents[(self.agents.index(self.agent_selection) + 1) % 2]

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.board.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self.agent_selection = self.agents[0]

        if self.render_mode == "human":
            self.render()

    def close(self):
        pass

    def render(self):
        raise NotImplementedError(
            "Rendering is not implemented for this environment."
        )
