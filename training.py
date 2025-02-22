import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.blockBreakerEnv import *


def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


def create_model(args, env):
    if args.algo == "ppo":
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.num_epochs,
            n_steps=512,
            clip_range=0.2,
            seed=args.seed,
            verbose=1,
        )
    elif args.algo == "dqn":
        model = DQN(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            verbose=1,
        )
    elif args.algo == "sac":
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=1,
        )
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model


def load_model(args, env):
    env = VecNormalize.load(
        "./models/vecNormalize{}.pkl".format(args.env), env)
    env.training = False
    env.norm_reward = False

    if args.algo == "ppo":
        model = PPO.load(
            "./model_checkpoints/rl_model_5000000_steps.zip", env=env)
    elif args.algo == "dqn":
        model = DQN.load("./models/{}".format(args.algo), env=env)
    elif args.algo == "sac":
        model = SAC.load(
            "./model_checkpoints/rl_model_5000000_steps.zip", env=env)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model, env


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, args, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x[x <= args.total_timesteps], y[x <= args.total_timesteps])
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed ({})".format(args.algo))
    plt.savefig("./images/trainingResults{}.png".format(args.algo))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None, type=str,
                        help="Test a trained policy")
    parser.add_argument(
        "--env",
        type=str,
        default="BlockBreaker-v0",
        help="Training/Testing environment",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=25000,
        help="The total number of samples to train on",
    )
    parser.add_argument(
        "--render_test", action="store_true", help="Render test")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--algo", default="ppo", type=str, help="RL Algo [ppo, dqn, sac]"
    )
    parser.add_argument("--lr", default=0.0003,
                        type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=10,
                        type=int, help="Training epochs")
    parser.add_argument(
        "--test_episodes", default=100, type=int, help="# episodes for test evaluations"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make(args.env, render_mode="human")

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])

    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None:
        try:
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            model = create_model(args, env)
            callback = CheckpointCallback(
                save_freq=1000000, save_path="./model_checkpoints/"
            )
            # Policy training
            model.learn(total_timesteps=args.total_timesteps,
                        callback=callback)
            # Saving model and env
            env.save("./models/vecNormalize{}.pkl".format(args.env))
            model.save("./models/{}".format(args.algo))
            plot_results(log_dir, args)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        print("Testing...")
        model, env = load_model(args, env)
        # Policy evaluation
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=args.test_episodes, render=args.render_test
        )

        print(
            f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}"
        )

    env.close()


if __name__ == "__main__":
    main()
