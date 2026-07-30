'''
Trains PPO policy (Stable Baseline3/SB3) on custom Jetbot gym env
'''

import argparse
import os
 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor 

from sim.sim_env import JetBotSimEnv

def make_env():
    env = JetBotSimEnv(render_mode=None, domain_randomization=True)
    return Monitor(env)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="models/jetbot_ppo")
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs")
    args = parser.parse_args()
 
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
  
    env = make_vec_env(make_env, n_envs=args.n_envs)
 
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log=args.tensorboard_log,
    )
 
    checkpoint_cb = CheckpointCallback(
        save_freq=max(20_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix="jetbot_ppo_ckpt",
    )
 
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(args.save_path)
    print(f"Saved final model to {args.save_path}.zip")
 
 
if __name__ == "__main__":
    main()
