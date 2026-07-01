"""Utilidades de experimentacion para el Obligatorio 2 (juegos alternados).

Funciones para correr partidas, armar matchups y graficar resultados,
reutilizadas en los distintos notebooks.
"""

import numpy as np
import time
from collections import defaultdict
from typing import Callable
from tqdm import tqdm

from base.game import AlternatingGame, AgentID
from base.agent import Agent


def run_games(game: AlternatingGame, agents: dict[AgentID, Agent],
              n_games: int = 100, verbose: bool = False) -> dict:
    """Juega n_games entre los agentes dados y junta estadisticas.

    Devuelve un dict con:
    - 'rewards': recompensa de cada agente en cada partida
    - 'results': cuantas veces salio cada valor de recompensa (agente 0)
    - 'mean_reward': recompensa media del agente 0
    - 'total_time' / 'avg_time_per_game': tiempos
    - 'game_lengths' / 'mean_game_length': duracion de las partidas en pasos
    """
    all_rewards = {agent: [] for agent in game.agents}
    game_lengths = []

    t0 = time.time()
    for i in tqdm(range(n_games), disable=not verbose, desc="Jugando partidas"):
        game.reset()
        steps = 0
        while not game.game_over():
            action = agents[game.agent_selection].action()
            game.step(action)
            steps += 1

        for agent in game.agents:
            all_rewards[agent].append(game.reward(agent))
        game_lengths.append(steps)

    total_time = time.time() - t0

    # Resumen de resultados para el agente 0
    agent0 = game.agents[0]
    rewards_0 = all_rewards[agent0]
    v, c = np.unique(rewards_0, return_counts=True)

    return {
        'rewards': all_rewards,
        'results': dict(zip(v, c)),
        'mean_reward': np.mean(rewards_0),
        'total_time': total_time,
        'avg_time_per_game': total_time / n_games,
        'game_lengths': game_lengths,
        'mean_game_length': np.mean(game_lengths),
    }


def win_draw_loss(results: dict, agent_idx: int = 0) -> dict:
    """Resume un dict de resultados (de run_games) en tasas de victoria/empate/derrota.

    Toma las recompensas del agente `agent_idx` y cuenta cuántas veces ganó
    (reward > 0), empató (reward == 0) o perdió (reward < 0).

    Returns:
        dict con 'win', 'draw', 'loss' (fracciones que suman 1) y 'n'.
    """
    agent = list(results['rewards'].keys())[agent_idx]
    rewards = np.array(results['rewards'][agent])
    n = len(rewards)
    wins = int(np.sum(rewards > 0))
    draws = int(np.sum(rewards == 0))
    losses = int(np.sum(rewards < 0))
    return {
        'win': wins / n, 'draw': draws / n, 'loss': losses / n,
        'wins': wins, 'draws': draws, 'losses': losses, 'n': n,
    }


def cfr_policy_agent(trained_cfr) -> Callable:
    """Devuelve una función `make(game, agent) -> CFR` que reutiliza la política ya
    entrenada (`node_dict`) enlazada al juego y agente concretos.

    Un agente CFR lee `self.game`/`self.agent` fijados al construirse, por lo que no
    puede reutilizarse tal cual en otra instancia de juego o en otra posición. Este
    helper crea un CFR liviano (sin reentrenar) que comparte la tabla de información
    ya aprendida, para evaluar el mismo CFR en ambas posiciones.
    """
    from agents.cfr import CounterFactualRegret

    def make(game, agent):
        ag = CounterFactualRegret(game, agent)
        ag.node_dict = trained_cfr.node_dict
        return ag
    return make


def matchup_both_sides(game_factory: Callable, make_a: Callable, make_b: Callable,
                       n_games: int = 100) -> dict:
    """Enfrenta dos agentes jugando AMBAS posiciones y agrega desde la óptica de A.

    Juega `n_games` con A como jugador 0 y `n_games` con A como jugador 1, para
    neutralizar la ventaja de la posición inicial. `make_a`/`make_b` son funciones
    `(game, agent_id) -> Agent` (permiten entrenar dentro si hace falta).

    Returns:
        dict con 'mean_reward_A' (promedio sobre ambas posiciones), 'as_p0',
        'as_p1' (tasas win/draw/loss de A en cada posición) y 'mean_p0'/'mean_p1'.
    """
    # A como jugador 0
    g0 = game_factory()
    agents0 = {g0.agents[0]: make_a(g0, g0.agents[0]), g0.agents[1]: make_b(g0, g0.agents[1])}
    res0 = run_games(g0, agents0, n_games)

    # A como jugador 1
    g1 = game_factory()
    agents1 = {g1.agents[0]: make_b(g1, g1.agents[0]), g1.agents[1]: make_a(g1, g1.agents[1])}
    res1 = run_games(g1, agents1, n_games)

    mean_p0 = float(np.mean(res0['rewards'][g0.agents[0]]))
    mean_p1 = float(np.mean(res1['rewards'][g1.agents[1]]))
    return {
        'mean_reward_A': (mean_p0 + mean_p1) / 2,
        'mean_p0': mean_p0, 'mean_p1': mean_p1,
        'as_p0': win_draw_loss(res0, 0),
        'as_p1': win_draw_loss(res1, 1),
        'time': res0['total_time'] + res1['total_time'],
    }


def run_matchup(game_factory: Callable, agent_configs: list[dict],
                n_games: int = 100, verbose: bool = False) -> dict:
    """Arma un matchup entre varias configuraciones de agentes y lo juega.

    `agent_configs` es una lista de dicts, uno por agente, con:
    - 'name': nombre para mostrar
    - 'class': clase del agente
    - 'kwargs': argumentos del constructor (sin contar game/agent)
    - 'train_kwargs': opcional, argumentos para entrenar (p. ej. niter en CFR)
    """
    game = game_factory()

    # Crear los agentes
    agents = {}
    for i, config in enumerate(agent_configs):
        agent_id = game.agents[i]
        agent = config['class'](game, agent_id, **config.get('kwargs', {}))

        # Entrenar si corresponde
        if 'train_kwargs' in config:
            agent.train(**config['train_kwargs'])

        agents[agent_id] = agent

    # Jugar las partidas
    results = run_games(game, agents, n_games, verbose)
    results['agent_configs'] = agent_configs

    return results


def run_parameter_sweep(game_factory: Callable, agent_class, agent_id_idx: int,
                        param_name: str, param_values: list,
                        opponent_configs: list[dict],
                        n_games: int = 50, base_kwargs: dict = None) -> list[dict]:
    """Barre un parámetro de un agente manteniendo fijos a los rivales.

    Devuelve una lista de resultados, uno por cada valor del parámetro.
    """
    results = []
    base_kwargs = base_kwargs or {}

    for val in param_values:
        game = game_factory()

        # Crear los agentes
        agents = {}
        for i, config in enumerate(opponent_configs):
            agent_id = game.agents[i]
            if i == agent_id_idx:
                kwargs = {**base_kwargs, param_name: val}
                agent = agent_class(game, agent_id, **kwargs)
                if hasattr(agent, 'train') and 'train_kwargs' in config:
                    agent.train(**config['train_kwargs'])
            else:
                agent = config['class'](game, agent_id, **config.get('kwargs', {}))
                if 'train_kwargs' in config:
                    agent.train(**config['train_kwargs'])
            agents[agent_id] = agent

        result = run_games(game, agents, n_games)
        result['param_value'] = val
        result['param_name'] = param_name
        results.append(result)

        print(f"  {param_name}={val}: mean_reward={result['mean_reward']:.3f}, "
              f"time={result['total_time']:.1f}s")

    return results


def run_tournament(game_factory: Callable, agent_configs: list[dict],
                   n_games: int = 50) -> np.ndarray:
    """Torneo round-robin entre todas las configuraciones de agentes.

    Cada par juega n_games con cada agente como jugador 0 y como jugador 1.

    Returns:
        reward_matrix: matriz donde reward_matrix[i][j] es la recompensa media
        del agente i jugando contra el agente j.
    """
    n = len(agent_configs)
    reward_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            game = game_factory()

            # Agente i como jugador 0, agente j como jugador 1
            agent_0 = agent_configs[i]['class'](
                game, game.agents[0], **agent_configs[i].get('kwargs', {})
            )
            agent_1 = agent_configs[j]['class'](
                game, game.agents[1], **agent_configs[j].get('kwargs', {})
            )

            # Entrenar si corresponde
            if 'train_kwargs' in agent_configs[i]:
                agent_0.train(**agent_configs[i]['train_kwargs'])
            if 'train_kwargs' in agent_configs[j]:
                agent_1.train(**agent_configs[j]['train_kwargs'])

            agents = {game.agents[0]: agent_0, game.agents[1]: agent_1}
            results = run_games(game, agents, n_games)
            reward_matrix[i][j] = results['mean_reward']

            print(f"  {agent_configs[i]['name']} vs {agent_configs[j]['name']}: "
                  f"{results['mean_reward']:.3f}")

    return reward_matrix


def cfr_convergence_data(game_factory: Callable, n_iters: int = 5000,
                         check_interval: int = 100, n_eval_games: int = 200) -> dict:
    """Sigue la convergencia de CFR a lo largo del entrenamiento.

    Cada check_interval iteraciones, evalúa la recompensa media y guarda la
    política aprendida en cada information set, para poder graficar su evolución.
    """
    from agents.cfr import CounterFactualRegret

    game = game_factory()
    cfr0 = CounterFactualRegret(game, game.agents[0])
    cfr1 = CounterFactualRegret(game, game.agents[1])

    iterations = []
    mean_rewards = []
    policies = defaultdict(list)

    for i in range(0, n_iters, check_interval):
        # Entrenar check_interval iteraciones mas
        cfr0.train(niter=check_interval)
        cfr1.train(niter=check_interval)

        # Evaluar
        agents = {game.agents[0]: cfr0, game.agents[1]: cfr1}
        results = run_games(game, agents, n_eval_games)

        iterations.append(i + check_interval)
        mean_rewards.append(results['mean_reward'])

        # Guardar la politica de cada information set
        for obs, node in cfr0.node_dict.items():
            policies[obs].append(node.learned_policy.copy())

    return {
        'iterations': iterations,
        'mean_rewards': mean_rewards,
        'policies': dict(policies),
        'final_cfr0': cfr0,
        'final_cfr1': cfr1,
    }


# ============================================================================
# Graficas
# ============================================================================

def plot_tournament_heatmap(reward_matrix: np.ndarray, agent_names: list[str],
                            game_name: str, ax=None):
    """Grafica los resultados de un torneo como un heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(reward_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=agent_names, yticklabels=agent_names, ax=ax)
    ax.set_title(f'Resultados del torneo — {game_name}')
    ax.set_xlabel('Rival (jugador 1)')
    ax.set_ylabel('Agente (jugador 0)')

    return ax


def plot_convergence(convergence_data: dict, game_name: str, ax=None):
    """Grafica la convergencia de CFR (recompensa media a lo largo del entrenamiento)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(convergence_data['iterations'], convergence_data['mean_rewards'],
            'b-', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteraciones de entrenamiento')
    ax.set_ylabel('Recompensa media (agente 0)')
    ax.set_title(f'Convergencia de CFR — {game_name}')
    ax.grid(True, alpha=0.3)

    return ax


def plot_policy_evolution(convergence_data: dict, obs_list: list[str],
                          action_names: list[str], game_name: str, ax=None):
    """Grafica cómo evoluciona la política aprendida para information sets puntuales."""
    import matplotlib.pyplot as plt

    n_obs = len(obs_list)
    if ax is None:
        fig, axes = plt.subplots(1, n_obs, figsize=(5 * n_obs, 4))
        if n_obs == 1:
            axes = [axes]
    else:
        axes = [ax]

    for i, obs in enumerate(obs_list):
        if obs not in convergence_data['policies']:
            continue

        policies = np.array(convergence_data['policies'][obs])
        iters = convergence_data['iterations'][:len(policies)]

        for a_idx, a_name in enumerate(action_names):
            if a_idx < policies.shape[1]:
                axes[i].plot(iters, policies[:, a_idx], label=a_name, linewidth=1.5)

        axes[i].set_title(f'Information set: {obs}')
        axes[i].set_xlabel('Iteraciones')
        axes[i].set_ylabel('Probabilidad')
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    return axes


def plot_parameter_sweep(sweep_results: list[dict], param_name: str,
                          game_name: str, ax=None):
    """Grafica el resultado de barrer un parámetro (desempeño y costo)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    values = [r['param_value'] for r in sweep_results]
    means = [r['mean_reward'] for r in sweep_results]
    times = [r['avg_time_per_game'] for r in sweep_results]

    axes[0].plot(values, means, 'bo-', linewidth=1.5)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Recompensa media')
    axes[0].set_title(f'{game_name} — desempeño vs {param_name}')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(values, times, 'ro-', linewidth=1.5)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Tiempo medio por partida (s)')
    axes[1].set_title(f'{game_name} — tiempo vs {param_name}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return axes
