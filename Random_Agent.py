from Hoverer3DEnv_noisy import Hoverer3DEnv
import numpy as np


def main():
    np.random.seed(0)
    env = Hoverer3DEnv()

    # Try some random actor trials
    print("Starting Random Actor Trials:")
    episodes = 1000
    gamma = 0.99
    total_rewards = np.zeros((episodes,))
    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        num_steps = 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            num_steps += 1
            #total_reward += reward * (gamma**num_steps)
            total_reward += reward
        total_rewards[i] = total_reward
        print('Episode: {}, Total Reward: {}, with {} steps until failure.'.format(i+1, total_reward, num_steps))
    print("Finished Random Trials.")
    print("Average Reward: {}".format(total_rewards.mean()))
    print("Std Dev: {}".format(np.std(total_rewards)))


if __name__ == '__main__':
    main()
