![Neovim](https://img.shields.io/badge/NeoVim-%2357A143.svg?&style=for-the-badge&logo=neovim&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

Block Breaker is an arcade game in which the goal is to break all the blocks in a vertical map, by making a ball bounce on a moving paddle. After creating the non-RL version using Pygame, an RL environvent of the game is created and used to train an agent. For the training, after unsuccesful attempts obtained with DQN/PPO, a long training using SAC showed good results on the test phase.
The RL environment is made using [Gymnasium](https://gymnasium.farama.org/index.html) and the SAC's implementation used is the one provided by [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html).
