import random
from itertools import product
from gymnasium.spaces import Discrete, Tuple
from base.game import AlternatingGame, AgentID, ActionType
from games.nocca_nocca.board import Board, MOVES, MAX_STACK, ROWS, COLS
from games.nocca_nocca.board import Player, BLACK, WHITE, EMPTY
from games.nocca_nocca.board import BLACK_START, WHITE_START, BLACK_GOAL, WHITE_GOAL
from games.nocca_nocca.board import Action

class NoccaNocca(AlternatingGame):
    def __init__(self, initial_player=None, max_steps=None, seed=None, render_mode='human'):
        super().__init__()

        self.metadata = {
            "name": "nocca_nocca_v0",
            "render_mode": "human",
            "agents": ["black", "white"],
            "agent_order": "random"
        }

        self.render_mode = render_mode
        self.initial_player = initial_player
        self.max_steps = max_steps
        self.seed = seed
        random.seed(seed)

        # board
        self.board = None

        # agents
        self.agents: list[AgentID] = ["Black", "White"]
        self.players: list[Player] = [BLACK, WHITE]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))
        self.agent_selection = None

        # actions
        self.action_board_dict = dict(map(lambda x: x, enumerate((product(range(ROWS), range(COLS), MOVES)))))
        self.board_action_dict = dict(map(lambda x: (x[1], x[0]), self.action_board_dict.items()))
        self.action_list = list(self.action_board_dict.keys())
        self.n_actions = len(self.action_list)
        self.action_spaces = {agent: Discrete(self.n_actions) for agent in self.agents}

        # observations
        self.observation_spaces = {agent: Discrete(self.n_actions) for agent in self.agents}

    def available_actions(self) -> list[ActionType]:
        player = self.agent_name_mapping[self.agent_selection]
        board_actions = self.board.legal_moves(player=player)
        actions = list(map(lambda x: self.board_action_dict[x], board_actions))
        return actions
    
    def step(self, action: ActionType) -> None:
        
        # check for termination
        if self.terminated():
            raise ValueError(f"Game has already finished - Call reset() if you want to play again")
    
        player = self.agent_name_mapping[self.agent_selection]
        board_action = self.action_board_dict[action]

        # check if action is valid
        valid_action, message = self.board.is_legal_move(player=player, action=board_action)
        if not valid_action:
            raise ValueError(f"Invalid board action {board_action} ({action}) for agent {self.agent_selection}({player}) - {message}.")
        
        # play turn
        self.board.play_turn(player=player, action=board_action)

        self.steps += 1

        # check for game over or max steps
        _game_over = self.board.check_game_over()
        _truncated = self._check_truncated()
        if _game_over or _truncated:
            # set termination
            self.terminations = dict(map(lambda agent: (agent, True), self.agents))
            self.truncations = dict(map(lambda agent: (agent, _truncated), self.agents))
            # set rewards
            self._set_rewards()
        else:
            # select next player
            next_player = self.board._opponent(player=player)
            self.agent_selection = self.agents[next_player]

        # set observations    
        self.observations = dict(map(lambda agent: (agent, self.board.squares), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def _check_truncated(self):
        return (self.max_steps is not None and self.steps >= self.max_steps)

    def _set_rewards(self):
        winner = self.board.check_for_winner()
        if winner is not None:
            for p in self.players:
                agent = self.agents[p]
                if p == winner:
                    self.rewards[agent] = 1
                else:
                    self.rewards[agent] = -1
        else:
            for agent in self.agents:
                self.rewards[agent] = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        # reset board
        self.board = Board()

        # reset agent selection
        if self.initial_player is None:
            # select random player
            self.agent_selection = self.agents[random.choice(self.players)]
        else:
            # select initial player
            self.agent_selection = self.agents[self.initial_player]

        # reset steps
        self.steps = 0

        # reset observations    
        self.observations = dict(map(lambda agent: (agent, self.board.squares), self.agents))
        self.rewards = dict(map(lambda agent: (agent, 0), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self):
        self.board.render()

    def check_for_winner(self):
        winner = self.board.check_for_winner()
        if winner is not None:
            return self.agents[winner]
        else:
            return None
    
    def clone(self):
        #return super().clone()
        self_clone = NoccaNocca(initial_player=self.initial_player, max_steps=self.max_steps, seed=self.seed, render_mode=self.render_mode)
        self_clone.board = Board()
        self_clone.board.set_board(self.board)
        self_clone.rewards = self.rewards.copy()
        self_clone.terminations = self.terminations.copy()
        self_clone.truncations = self.truncations.copy()
        self_clone.infos = self.infos.copy()
        self_clone.agent_selection = self.agent_selection
        self_clone.steps = self.steps
        return self_clone
    
    def eval(self, agent: AgentID) -> float:
        """ Función de evaluación heurística para Nocca-Nocca. 
        
        Considera:
        1. Progreso de las piezas hacia la fila objetivo (más importante)
        2. Control de la pila: las piezas en la parte superior pueden moverse, las otras no
        3. Movilidad: más movimientos legales = más flexibilidad"""
    
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")

        if self.terminated():
            return self.rewards[agent]
    
        player = self.agent_name_mapping[agent]
        opponent = 1 - player
        
        # Goal rows for each player
        goal_row = {BLACK: BLACK_GOAL, WHITE: WHITE_GOAL}
        start_row = {BLACK: BLACK_START, WHITE: WHITE_START}
        max_dist = abs(goal_row[BLACK] - start_row[BLACK])  # = 6
        
        def player_score(p):
            """Compute a score for player p based on piece positions and control."""
            import numpy as np
            
            pieces = np.argwhere(self.board.squares == p)
            if len(pieces) == 0:
                return -1.0  # All pieces gone/blocked
            
            progress = 0.0
            top_control = 0
            total_pieces = 0
            
            for x, y, k in pieces:
                total_pieces += 1
                # Progress: how close to goal (normalized 0..1)
                dist_to_goal = abs(x - goal_row[p])
                progress += 1.0 - (dist_to_goal / max_dist)
                
                # Stack control: is this piece on top of its stack?
                stack = self.board.squares[x][y]
                # Find highest occupied position in stack
                highest = -1
                for h in range(MAX_STACK):
                    if stack[h] != EMPTY:
                        highest = h
                if k == highest:
                    top_control += 1
            
            # Normalize progress by number of pieces (max 5 pieces)
            progress_score = progress / 5.0
            
            # Top control ratio
            control_score = top_control / max(total_pieces, 1)
            
            # Mobility (legal moves count, normalized)
            mobility = len(self.board.legal_moves(p))
            mobility_score = min(mobility / 20.0, 1.0)
            
            # Weighted combination
            return 0.5 * progress_score + 0.25 * control_score + 0.25 * mobility_score
        
        score = player_score(player) - player_score(opponent)
        return max(-1.0, min(1.0, score))
