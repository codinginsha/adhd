import numpy as np
import pandas as pd

class SimpleAdaptiveAgent:
    def __init__(self, n_actions=3, epsilon=0.2):
        self.n = n_actions
        self.epsilon = epsilon
        self.q = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n)
        return np.argmax(self.q)
    def update(self, a, r):
        self.counts[a] += 1
        self.q[a] += (r - self.q[a]) / self.counts[a]

def simulate_episode(agent, attention_seq, n_steps=100):
    rewards = []
    for i in range(n_steps):
        a = agent.act()
        att = attention_seq[i % len(attention_seq)]
        difficulty = a / 2.0
        if att == 1:
            reward = 1.0 - 0.2*abs(1 - difficulty)
        else:
            reward = 1.0 - 0.5*difficulty
        noise = np.random.normal(0,0.05)
        r = max(0, reward + noise)
        agent.update(a, r)
        rewards.append(r)
    return np.mean(rewards)

def main():
    df = pd.read_csv('wearable_timeseries.csv')
    attention_seq = df[df['subject']==0]['attention'].values[:300]
    agent = SimpleAdaptiveAgent()
    mean_reward = simulate_episode(agent, attention_seq, n_steps=300)
    print('Mean reward after simulation:', mean_reward)
    print('Learned Q-values:', agent.q)
    pd.DataFrame({'action':[0,1,2],'q':agent.q,'counts':agent.counts}).to_csv('rl_agent_summary.csv', index=False)
    print('RL summary saved to rl_agent_summary.csv')

if __name__ == '__main__':
    main()
