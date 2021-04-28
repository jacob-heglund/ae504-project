# AE 504 Project

## Using CartPole Noise Gym Environment
In the cartpole-noise directory run `pip install -e . `

Then you can use this environment with

``` 
import gym
import cartpole_noise
env = gym.make('cpn-v0') # cartpole with noise
env = gym.make('cpl-v0') # linearlized cartpole
env = gym.make('CartPole-v0') # standard (nonlinear) cartpole
```
