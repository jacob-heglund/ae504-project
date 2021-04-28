from gym.envs.registration import register

register(
    id='cpn-v0',
    entry_point='cartpole_noise.envs:CartPoleNoiseEnv',
)

register(
    id='cpl-v0',
    entry_point='cartpole_noise.envs:CartPoleLinearEnv',
)