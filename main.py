
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models

ACTIONS = [0,1,2]  # 0=flat, 1=long, 2=hold-position
ACTION_NAMES = {0:"flat",1:"enter_long",2:"hold"}

class TradingEnv:
    def __init__(self, prices, window=30):
        self.prices = prices.values.astype("float32")
        self.returns = pd.Series(self.prices).pct_change().fillna(0.0).values.astype("float32")
        self.window = window
        self.reset()

    def reset(self):
        self.t = self.window
        self.position = 0  # 0 flat, 1 long
        self.done = False
        return self._obs()

    def _obs(self):
        # price window + position flag
        window = self.prices[self.t-self.window:self.t]
        return np.concatenate([window/window[0]-1.0, [self.position]]).astype("float32")

    def step(self, action):
        reward = 0.0
        if action == 1: # enter long
            self.position = 1
        elif action == 0: # flat
            self.position = 0
        # hold keeps position as-is
        r = self.returns[self.t]
        reward = r * self.position
        self.t += 1
        if self.t >= len(self.prices):
            self.done = True
        return self._obs(), reward, self.done, {}

def build_qnet(obs_dim, n_actions):
    model = models.Sequential([
        layers.Input(shape=(obs_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_actions, activation=None),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_dqn(env, episodes=10, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.95, lr=0.001):
    obs_dim = env.reset().shape[0]
    qnet = build_qnet(obs_dim, len(ACTIONS))
    target = build_qnet(obs_dim, len(ACTIONS))
    target.set_weights(qnet.get_weights())
    memory = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if np.random.rand() < eps:
                a = np.random.choice(ACTIONS)
            else:
                q = qnet.predict(s[None,:], verbose=0)[0]
                a = int(np.argmax(q))
            ns, r, done, _ = env.step(a)
            memory.append((s,a,r,ns,done))
            # learn
            if len(memory) > 512:
                batch = memory[-512:]
                S = np.array([b[0] for b in batch])
                A = np.array([b[1] for b in batch])
                R = np.array([b[2] for b in batch])
                NS = np.array([b[3] for b in batch])
                D = np.array([b[4] for b in batch])
                q_next = target.predict(NS, verbose=0).max(axis=1)
                y = qnet.predict(S, verbose=0)
                y[np.arange(len(A)), A] = R + gamma * q_next * (1-D)
                qnet.fit(S, y, epochs=1, verbose=0)
            s = ns
            ep_reward += r
        target.set_weights(qnet.get_weights())
        eps = max(eps_min, eps*eps_decay)
        print(f"Episode {ep+1}: reward={ep_reward:.4f}, eps={eps:.3f}")
    return qnet

def main(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    prices = df["Close"].dropna()
    env = TradingEnv(prices, window=30)
    train_dqn(env, episodes=5)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()
    main(args.ticker)
