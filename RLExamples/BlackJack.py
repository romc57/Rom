import gym
from pyvirtualdisplay import Display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.air.checkpoint import Checkpoint
from gym.spaces import Discrete
from gym.spaces import Tuple
from ray.tune.registry import register_env


ACTION_LIST = ['stand', 'hit', 'double', 'surrender']


BASIC_STRATEGY = {
    21: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'},
        'Ace': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    20: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'},
        'Ace': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    19: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'},
        'Ace': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'double', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    18: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'},
        'Ace': {2: 'double', 3: 'double', 4: 'double', 5: 'double', 6: 'double', 7: 'stand', 8: 'stand', 9: 'hit', 10: 'hit', 1: 'hit'}},
    17: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'},
        'Ace': {2: 'hit', 3: 'double', 4: 'double', 5: 'double', 6: 'double', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    16: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'},
        'Ace': {2: 'hit', 3: 'hit', 4: 'double', 5: 'double', 6: 'double', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    15: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'},
        'Ace': {2: 'hit', 3: 'hit', 4: 'double', 5: 'double', 6: 'double', 7: 'stand', 8: 'stand', 9: 'stand', 10: 'stand', 1: 'stand'}},
    14: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'},
        'Ace': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'double', 6: 'double', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    13: {
        'NoAce': {2: 'stand', 3: 'stand', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'},
        'Ace': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'double', 6: 'double', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    12: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'stand', 5: 'stand', 6: 'stand', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'},
        'Ace': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    11: {
        'NoAce': {2: 'double', 3: 'double', 4: 'double', 5: 'double', 6: 'double', 7: 'double', 8: 'double', 9: 'double', 10: 'stand', 1: 'double'}},
    10: {
        'NoAce': {2: 'double', 3: 'double', 4: 'double', 5: 'double', 6: 'double', 7: 'double', 8: 'double', 9: 'double', 10: 'hit', 1: 'hit'}},
    9: {
        'NoAce': {2: 'stand', 3: 'double', 4: 'double', 5: 'double', 6: 'double', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    8: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    7: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    6: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    5: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    4: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    3: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
    2: {
        'NoAce': {2: 'hit', 3: 'hit', 4: 'hit', 5: 'hit', 6: 'hit', 7: 'hit', 8: 'hit', 9: 'hit', 10: 'hit', 1: 'hit'}},
}


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, double, surrender):
        super().__init__(env)
        self.action_lst = ACTION_LIST
        self.action_space = Discrete(len(self.action_lst))
        self.double = double
        self.surrender = surrender
        self.first = False

    def action(self, action):
        if action == 0 or action == 1:
            return action
        elif action == 2:
            self.double[0] = True
            return 1
        elif action == 3:
            self.surrender[0] = True
            return 0


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shared_obs):
        super().__init__(env)
        self.shared_obs = shared_obs

    def observation(self, obs):
        self.shared_obs[0] = obs
        return obs


class MoneyReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward, double, surrender):
        super().__init__(env)
        self.double = double
        self.surrender = surrender
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (self.min_reward, self.max_reward)

    def reward(self, reward):
        if self.surrender[0]:
            self.surrender[0] = False
            return -2.5
        if reward == 1:
            if self.double[0]:
                self.double[0] = False
                return 10
            else:
                return 5
        elif reward == -1:
            if self.double[0]:
                self.double[0] = False
                return -10
            else:
                return -5
        elif reward == 1.5:
            return 7.5
        else:
            return 0

    def check_action_with_basic_strategy(self, action):
        b_strategy = check_basic_strategy(obs[0], obs[2], obs[1])
        return ACTION_LIST[action] == b_strategy



class CustomEnv(gym.Env):
    def __init__(self):
        self.base_env = gym.make("Blackjack-v1", natural=True)
        self.double = [False]
        self.surrender = [False]
        self.shared_obs = [None]
        self.base_env = DiscreteActions(self.base_env, self.double, self.surrender, self.shared_obs)
        self.base_env = ObservationWrapper(self.base_env, self.shared_obs)
        self.base_env = MoneyReward(self.base_env, -10, 10, self.double, self.surrender, self.shared_obs)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        return self.base_env.step(action)

    def render(self):
        return self.base_env.render()


ray.init(
  num_cpus=3,
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)


register_env('ImprovedBlackJack', lambda _: CustomEnv())
stop = {"training_iteration": 500}

config = {
    "env": 'ImprovedBlackJack'
}


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
    return checkpoint


def run_random_trial(env):
    p = r'/home/rom/PycharmProjects/Rom/outputs/RLBlackJack/testing.mp4'
    video = VideoRecorder(env, p)
    # returns an initial observation
    env.reset()
    for i in range(20):
        env.render()
        video.capture_frame()
        # env.action_space.sample() produces either 0 (left) or 1 (right).
        observation, reward, done, info = env.step(env.action_space.sample())
        # Not printing this time
        # print("step", i, observation, reward, done, info)
    video.close()
    env.close()


def check_basic_strategy(player_sum, has_ace, dealer_card):
    player_sum = int(player_sum)
    has_ace = 'Ace' if has_ace else 'NoAce'
    dealer_card = int(dealer_card)
    if player_sum in BASIC_STRATEGY:
        if has_ace in BASIC_STRATEGY[player_sum]:
            if dealer_card in BASIC_STRATEGY[player_sum][has_ace]:
                return BASIC_STRATEGY[player_sum][has_ace][dealer_card]


def run_trained_trial(checkpoint, wrapped_env, games_to_play=100):
    trainer = PPOTrainer(config=config)
    #trainer = DQNTrainer(config=config)
    #trainer = ImpalaTrainer(config=config)
    trainer.restore(checkpoint)
    after_training = r'/home/rom/PycharmProjects/Rom/outputs/RLBlackJack/after_training_from_checkpoint.mp4'
    print('Saving trial at {}'.format(after_training))
    observation = wrapped_env.reset()
    player_money = 0
    overall_games = games_to_play
    like_basic_s = 0
    initial_cards = True
    print('-' * 10,  ' Starting a play ', '-' * 10)
    print('Player score: {}, Has Ace: {}, DealerCard: {}'.format(observation[0], observation[2],
                                                                 'A' if observation[1] == 1 else observation[1]))
    while True:
        wrapped_env.render()
        action = trainer.compute_action(observation)
        print('Player played: {}'.format(ACTION_LIST[int(action)]))
        if initial_cards:
            basic_strategy = check_basic_strategy(observation[0], observation[2], observation[1])
            if basic_strategy is not None:
                print('Basic strategy: {} | Model: {}'.format(basic_strategy, ACTION_LIST[int(action)]))
                if basic_strategy == ACTION_LIST[int(action)]:
                    like_basic_s += 1
                initial_cards = False
        observation, reward, done, info = wrapped_env.step(action)
        print('Model reward: {}'.format(reward))
        player_money += reward
        print('Model overall earnings: {}'.format(player_money))
        if done:
            games_to_play -= 1
            print('-' * 10, ' Ended ', '-' * 10, end='\n\n')
            if games_to_play == 0:
                break
            observation = wrapped_env.reset()
            print('-' * 10, ' Starting a play ', '-' * 10)
            initial_cards = True
        print('Player score: {}, Has Ace: {}, Dealers Card: {}'.format(observation[0], observation[2],
                                                                       'A' if observation[1] == 1 else observation[1]))
    wrapped_env.close()
    print('Model played {}% correctly'.format((like_basic_s / overall_games) * 100))


if __name__ == '__main__':
    #checkpoint = train()
    #checkpoint = Checkpoint(r'/home/rom/ray_results/PPO/PPO_ImprovedBlackJack_f211c_00000_0_2022-10-10_11-24-23/checkpoint_000150/')
    checkpoint = Checkpoint(r'/home/rom/ray_results/PPO/PPO_ImprovedBlackJack_0e12f_00000_0_2022-10-10_12-15-16/checkpoint_000500/')
    run_trained_trial(checkpoint, CustomEnv(), 200)
    # r'/home/rom/ray_results/PPO/PPO_ImprovedBlackJack_f211c_00000_0_2022-10-10_11-24-23/checkpoint_000150/'
