from Hoverer3DEnv_noisy import Hoverer3DEnv
import numpy as np
import sys
from stable_baselines.common.policies import MlpPolicy
import stable_baselines as sb

def loadModel(folder, modelName):
    if modelName == 'A2C':
        return sb.A2C.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'ACKTR':
        return sb.ACKTR.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'DDPG':
        return sb.DDPG.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'PPO2':
        return sb.PPO2.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'SAC':
        return sb.SAC.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'TD3':
        return sb.TD3.load("{}hover3d_noisy_{}".format(folder, modelName))
    if modelName == 'TRPO':
        return sb.TRPO.load("{}hover3d_noisy_{}".format(folder, modelName))
    assert False, "ERROR: Invalid model name given as argument."

def loadBestModel(folder, modelName):
    if modelName == 'A2C':
        return sb.A2C.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'ACKTR':
        return sb.ACKTR.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'DDPG':
        return sb.DDPG.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'PPO2':
        return sb.PPO2.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'SAC':
        return sb.SAC.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'TD3':
        return sb.TD3.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    if modelName == 'TRPO':
        return sb.TRPO.load("{}trainingLogNoisy{}/best_model".format(folder, modelName))
    assert False, "ERROR: Invalid model name given as argument."

def main(modelName):
    seed = 0
    np.random.seed(seed)
    env = Hoverer3DEnv()

    folder = "Data/Data_Noisy_5e6_seed{}/".format(seed)
    #model = loadModel(folder, modelName)
    model = loadBestModel(folder, modelName)

    # test the model
    print("\nTrained {} Trials: ".format(modelName))
    episodes = 2
    total_rewards = np.zeros((episodes,))
    gamma = 0.99
    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        num_steps = 0
        while not done:
            action, _next_state = model.predict(state)
            state, reward, done, info = env.step(action)
            num_steps += 1
            #total_reward += reward * (gamma**num_steps)
            total_reward += reward
        total_rewards[i] = total_reward
        print('Episode: {}, Total Reward: {}, with {} steps until failure.'.format(i+1, total_reward, num_steps))
    print("Finished {} Trials.\n".format(modelName))
    print("Average Reward: {}".format(total_rewards.mean()))
    print("Std Dev: {}".format(np.std(total_rewards)))


if __name__ == '__main__':
    main(sys.argv[1])
