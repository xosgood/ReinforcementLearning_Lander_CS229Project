#from Hoverer3DEnv import Hoverer3DEnv
from Hoverer3DEnv_noisy import Hoverer3DEnv
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3_MlpPolicy
import stable_baselines as sb
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from CallbackPlotting_Example import SaveOnBestTrainingRewardCallback as Plt_Callback


def getModel(modelName, env, verbose=0):
    if modelName == 'A2C':
        return sb.A2C(MlpPolicy, env, verbose=verbose)
    if modelName == 'ACKTR':
        return sb.ACKTR(MlpPolicy, env, verbose=verbose)
    if modelName == 'DDPG':
        return sb.DDPG(DDPG_MlpPolicy, env, verbose=verbose)
    if modelName == 'PPO2':
        return sb.PPO2(MlpPolicy, env, verbose=verbose)
    if modelName == 'SAC':
        return sb.SAC(SAC_MlpPolicy, env, verbose=verbose)
    if modelName == 'TD3':
        return sb.TD3(TD3_MlpPolicy, env, verbose=verbose)
    if modelName == 'TRPO':
        return sb.TRPO(MlpPolicy, env, verbose=verbose)


def main(modelName):
    # modelName = 'A2C', 'ACKTR', 'DDPG', 'PPO2', 'SAC', 'TD3', 'TRPO'
    np.random.seed(0)
    startTime = time.time()

    log_dir = "trainingLog{}/".format(modelName)
    os.makedirs(log_dir, exist_ok=True)

    env = Hoverer3DEnv()
    env = Monitor(env, log_dir) # wrap the envinornment
    model = getModel(modelName, env, verbose=1)
    callback = Plt_Callback(check_freq=1000, log_dir=log_dir)
    totTimeSteps = 10000
    model.learn(total_timesteps=totTimeSteps, callback=callback)
    model.save("hover3d_{}".format(modelName))

    results_plotter.plot_results([log_dir], totTimeSteps, results_plotter.X_TIMESTEPS, 'Training Data (Learning Curve) for {} Agent'.format(modelName))
    plt.savefig("{}/trainData_{}".format(log_dir, modelName))
    #plt.show()
    plt.close()

    '''# test the model
    print("\nTrained {} Trials: ".format(modelName))
    episodes = 10
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
            total_reward += reward * (gamma**num_steps)
        print('Episode: {}, Total Reward: {}, with {} steps until failure.'.format(i+1, total_reward, num_steps))
    print("Finished {} Trials.\n".format(modelName))'''

    print("Finished Training {} Agent Model!".format(modelName))
    totTime = time.time() - startTime
    print("Total Run Time: {}s".format(totTime))


if __name__ == '__main__':
    main(sys.argv[1])
