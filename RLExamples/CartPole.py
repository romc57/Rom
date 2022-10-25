import gym
from pyvirtualdisplay import Display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.air.checkpoint import Checkpoint


ray.init(
  num_cpus=3,
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)


config = {
    "env": "CartPole-v1",
}
stop = {"episode_reward_mean": 400}
env = gym.make("CartPole-v1")
display = Display(visible=True, size=(1400, 900))
display.start()


def train():
    # execute training
    analysis = ray.tune.run(
        "PPO",
        config=config,
        stop=stop,
        checkpoint_at_end=True,
    )
    trial = analysis.get_best_logdir("episode_reward_mean", "max")
    checkpoint = analysis.get_best_checkpoint(
        trial,
        "training_iteration",
        "max",
    )
    print('Best checkpoint: {}'.format(checkpoint))


def run_random_trial():
    p = r'/home/rom/PycharmProjects/Rom/outputs/RLCartPole/testing.mp4'
    video = VideoRecorder(env, p)
    # returns an initial observation
    env.reset()
    for i in range(200):
        env.render()
        video.capture_frame()
        # env.action_space.sample() produces either 0 (left) or 1 (right).
        observation, reward, done, info = env.step(env.action_space.sample())
        # Not printing this time
        # print("step", i, observation, reward, done, info)
    video.close()
    env.close()


def run_trained_trial(checkpoint):
    checkpoint = Checkpoint(local_path=checkpoint)
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint)
    after_training = r'/home/rom/PycharmProjects/Rom/outputs/RLCartPole/after_training_from_checkpoint.mp4'
    print('Saving trial at {}'.format(after_training))
    after_video = VideoRecorder(env, after_training)
    observation = env.reset()
    done = False
    while not done:
        env.render()
        after_video.capture_frame()
        action = trainer.compute_action(observation)
        observation, reward, done, info = env.step(action)
    after_video.close()
    env.close()


if __name__ == '__main__':
    last_checkpoint = r'/home/rom/ray_results/PPO/PPO_CartPole-v1_23583_00000_0_2022-10-10_01-45-56/checkpoint_000013/'
    run_trained_trial(last_checkpoint)


