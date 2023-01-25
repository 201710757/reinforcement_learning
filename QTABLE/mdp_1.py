import numpy as np
import matplotlib.pyplot as plt
import gym
env = gym.make('FrozenLake-v1')
# trans_model = 

class ValueIteration:
    def __init__(self, reward_function, transition_model, gamma):
        # self.num_states = transition_model.shape[0]
        self.num_states = env.observation_space.n # transition_model.shape[0]
        # self.num_actions = transition_model.shape[1]
        self.num_actions = env.action_space.n # transition_model.shape[1]
        
        # env.P[S][A] = p, n_s, r, _
        # self.reward_function = np.nan_to_num(reward_function)
        # self.transition_model = transition_model
        
        self.gamma = gamma
        self.values = np.zeros(self.num_states)
        self.policy = None

    def one_iteration(self):
        delta = 0
        for s in range(self.num_states):
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                # p = self.transition_model[s, a]
                # v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
                for p, n_s, r, _ in env.P[s][a]:
                    v_list[a] += (r + self.gamma * p * self.values[n_s]) # np.sum(p * self.values)

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                # p = self.transition_model[s, a]
                # v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)
                for p, n_s, r, _ in env.P[s][a]:
                    v_list[a] += (r + self.gamma * p * self.values[n_s]) # np.sum(p * self.values)

            max_index = []
            max_val = np.max(v_list)
            for a in range(self.num_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            pi[s] = np.random.choice(max_index)
        return pi.astype(int)

    def train(self, tol=1e-3):
        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]
        while delta > tol:
            epoch += 1
            delta = self.one_iteration()
            delta_history.append(delta)
            if delta < tol:
                break
        self.policy = self.get_policy()

        print(f'# iterations of policy improvement: {len(delta_history)}')
        print(f'delta = {delta_history}')

        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
        ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Delta')
        ax.legend()
        plt.tight_layout()
        plt.show()


# from GridWorld import GridWorld
# from ValueIteration import ValueIteration

# problem = GridWorld('data/world00.csv')

# solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver = ValueIteration(None, None, gamma=0.9)#problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

# problem.visualize_value_policy(policy=solver.policy, values=solver.values)
# problem.random_start_policy(policy=solver.policy)









def first_visit_mc_prediction(policy, env, n_episodes):

    # First, we initialize the empty value table as a dictionary for storing the values of each state
    value_table = defaultdict(float)

    N = defaultdict(int)


    for _ in range(n_episodes):

        # Next, we generate the epsiode and store the states and rewards
        states, _, rewards = generate_episode(policy, env)
        returns = 0

        # Then for each step, we store the rewards to a variable R and states to S, and we calculate
        # returns as a sum of rewards

        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]

            # MC 에서의 R 누적 합
            returns += R

            # Now to perform first visit MC, we check if the episode is visited for the first time, if yes,
            # we simply take the average of returns and assign the value of the state as an average of returns
            
            # 현재 ep기준, 방문 안했을 경우만 가지고 value func 업데이트.
            if S not in states[:t]:
                N[S] += 1
                # N[S] : 여러 에피소드를 진행하기 때문에, 각 에피소드에 대한 평균을 적용.
                value_table[S] += (returns - value_table[S]) / N[S] 

    return value_table
