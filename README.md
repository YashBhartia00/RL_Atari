# RL_Atari
 using DQN to learn atari games
 
 **Required Dependencies :** gym, torch, torchvision, cv2, numpy
 
 **main**: Training loop, just run this to get results
 * change env_name for different environment, use '{env_name}NoFrameskip-v4' format
 * change episode numbers in train-loop execution
 
 **environments**: preprocessing
 
 **model**: The DQN model
 
 **memory**: Replay memory
 
 results on 400 episodes of training, test wins all 21 points
 ![data](https://github.com/YashBhartia00/RL_Atari/blob/master/train_stats.png)
 
 ![test](https://github.com/YashBhartia00/RL_Atari/blob/master/test_result.gif)
 
