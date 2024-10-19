import gym
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.current_step = 0
        self.initial_balance = 10000  # Start with $10,000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.done = False

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(len(stock_data.columns),), dtype=np.float32
        )

    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]["Close"]

        if action == 0:  # Buy
            self.shares_held += self.balance // current_price
            self.balance -= self.shares_held * current_price
        elif action == 1:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        # Calculate the new total profit and reward
        total_value = self.balance + (self.shares_held * current_price)
        reward = total_value - self.initial_balance

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.stock_data):
            self.done = True

        return self._next_observation(), reward, self.done, {}

    def _next_observation(self):
        return self.stock_data.iloc[self.current_step].values

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.done = False
        return self._next_observation()
