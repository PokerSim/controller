import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from poker_env import PokerEnv
from all_in_player import AllInPlayer
from stats_player import StatsPlayer
from random_player import RandomPlayer

class PPOPlayer:
    def __init__(self, name="PPO", chips=1000):
        self.name = name
        self.chips = chips
        self.hand = []
        self.active = True
        self.current_bet = 0

    def receive_card(self, card):
        self.hand.append(card)

    def take_action(self, current_bet):
        return None  # PPO is controlled by model externally

# --- Create new players ---
def make_players():
    return [
        PPOPlayer("PPO"),
        AllInPlayer("AllInBot"),
        StatsPlayer("StatsBot"),
        RandomPlayer("RandomBot")
    ]

# --- Setup training environment ---
players = make_players()
env = PokerEnv(players, agent_index=0)
check_env(env, warn=True)

# --- Train PPO ---
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=30000)

# --- Play multiple hands ---
num_hands = 10
for hand in range(1, num_hands + 1):
    print(f"\n===== Hand {hand} =====")
    players = make_players()
    env = PokerEnv(players, agent_index=0)

    obs = env.reset()
    done = False

    while not done:
        player = players[env.action_pointer]
        if env.action_pointer == env.agent_index:
            action, _ = model.predict(obs)
        else:
            action = None
        obs, reward, done, info = env.step(action)

    print("===== End of Hand =====")
    for player in players:
        print(f"{player.name}: Chips: {player.chips}")
