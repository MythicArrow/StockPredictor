import random
import numpy as np
import jax
import jax.numpy as jnp
from env import StockTradingEnv
from agent import Agent
from model import TradingModel
import optax

# Load stock data
import yfinance as yf
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
env = StockTradingEnv(stock_data)

# Initialize model and optimizer
model = TradingModel(features=[128, 64])
rng = jax.random.PRNGKey(0)
sample_input = jnp.ones((1, len(stock_data.columns)))
params = model.init(rng, sample_input)
optimizer = optax.adam(learning_rate=0.001)

agent = Agent(model, params, optimizer)
memory = []  # Replay buffer

# Training loop
num_episodes = 1000
batch_size = 32
gamma = 0.99  # Discount factor

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        q_values = agent.predict(jnp.array(state))
        action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Store transition in memory
        memory.append((state, action, reward, next_state, done))

        # Sample a mini-batch from memory
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = map(np.array, zip(*batch))

            # Compute target Q-values
            next_q_values = agent.predict(batch_next_states)
            targets = batch_rewards + (1 - batch_dones) * gamma * np.max(next_q_values, axis=1)

            # Update the agent
            agent.update(batch_states, batch_actions, targets)

        state = next_state

    print(f'Episode {episode}: Reward: {episode_reward}')
