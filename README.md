# baseline_ddpg
baseline DDPG implementation less than 400 lines

After seeing a few sample implementation on DDPG, I have decided to implement a baseline DDPG within a single python scripts. And this is done in less than 400 lines, including (hopefully) intuitive comments. This implementation is inpired by the original DDPG paper, as well as the following github repos:

Original DDPG paper: https://arxiv.org/pdf/1509.02971.pdf

Git hub repos:

https://github.com/pemami4911/deep-rl/tree/master/ddpg

https://github.com/floodsung/DDPG

https://github.com/openai/baselines

To keep it a short implementation, I have simplify the following:
1. I only implemented a very basic replay buffer, without any clever tricks such as priorities.
2. I didn't implement any model load/save codes, as the training is relatively quick for simple mujoco envs such as InvertedPendulum-v2
3. The tensorboard implementation only contains basic functions, I track the test performance every 100 episode as the scores during training are heavily affected by the exploration noise.
