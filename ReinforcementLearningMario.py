print('\n\nHello, this is the start of RL_Mario\n\n')
#pip install gym_super_mario_bros==7.3.0 nes_py --user
#pip install gym==0.26.2 --user
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 tqdm==4.66.1 typing_extensions==4.8.0 urllib3==2.0.6 numpy==1.26.1
#pip install stable-baselines3[extra] --user

#----------------------------------------------------------------------------------------------------------------------------------
#PART 1: SETUP MARIO GAME
#----------------------------------------------------------------------------------------------------------------------------------

# Import the game
import gym #gym 0.26.2
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
##
## Setup game
#env = gym_super_mario_bros.make('SuperMarioBros-v0') #gym==0.24.1
#env = gym.make('SuperMarioBros-1-1-v0',apply_api_compatibility=True, render_mode="human") #gym==0.26.2
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human") #gym==0.26.2 (https://stackoverflow.com/a/77003659)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print('\nFinished basic setup of game!\nLooping through each frame of game.\n')

## Create a flag - restart or not
done = True
## Loop through each frame in the game
for step in range(1000): 
    ### Start the game to begin with 
    if done: 
        ### Start the gamee
        env.reset()
    ### Do random actions
    #state, reward, done, info = env.step(env.action_space.sample()) #gym==0.24.1
    state, reward, done, truncated, info = env.step(env.action_space.sample()) #gym==0.26.2 [[0.24.1] obs, reward, done, info -->[0.26.2] obs, reward, terminated, truncated, info] (https://stackoverflow.com/a/73765266)
    ### Show the game on the screen
    env.render()
## Close the game
env.close()

print('\nFinished looping through game. Closing window.\n')

#----------------------------------------------------------------------------------------------------------------------------------
#PART 2: Preprocess Environment 
#----------------------------------------------------------------------------------------------------------------------------------

print("\nPreprocessing Environment.\n")
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# 1. Create the base environment
#env = gym_super_mario_bros.make('SuperMarioBros-v0') #gym==0.24.1
env = gym_super_mario_bros.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="human") #gym==0.26.2 (https://stackoverflow.com/a/77003659)
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs) #gym==0.26.2 (https://stackoverflow.com/a/76562664)
state = env.reset()

#state, reward, done, info = env.step([5]) #gym==0.24.1
state, reward, done, info, = env.step([5]) 

print('\nCreating plot.\n')

plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()

print('\nPlot show called!\n')

#----------------------------------------------------------------------------------------------------------------------------------
#STEP 3: TRAIN THE REINFORCEMENT LEARNING MODEL
#----------------------------------------------------------------------------------------------------------------------------------

print('\n\nBEGINNING PROCESS TO TRAIN THE REINFORCEMENT LEARNING MODEL!\n\n')
# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

#Inspired by: https://github.com/nicknochnack/MarioRL/tree/main
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        print('\nTrain and logging Callback - init called')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            print('\nTrain and logging Callback - init_callback called')

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            print('\nTrain and Logging Callback - on_step called')

        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100, save_path=CHECKPOINT_DIR) #NOTE: changed check frequency from 10,000 to just 100

# This is the AI model started (PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, #NOTE: changed learning rate from 0.000001 to just 0.0001
            n_steps=512) 

# Train the AI model, this is where the AI model starts to learn
print('\n\n\nBEGINNING LEARNING PROCESS FOR MODEL!\n\n\n')
model.learn(total_timesteps=10000, callback=callback) #NOTE: changed total_timesteps from 1,000,000 to just 10,000
print('\n\nFinished learning. Going to save model!\n\n')

model.save('thisisatestmodel') #TODO: i think we need to do smth with this
print('\n\nTRAINED AND SAVED THE TRAINED REINFORCEMENT LEARNING MODEL!\n\n')

#----------------------------------------------------------------------------------------------------------------------------------
#STEP 4: TEST THE MODEL!
#----------------------------------------------------------------------------------------------------------------------------------
# Load model
model = PPO.load('./train/best_model_10000') #TODO: i don't think this is correct...

print('\nLoaded model for testing!\n')

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs) #gym==0.26.2 (https://stackoverflow.com/a/76562664)
state = env.reset()

# Start the game 
print('\nStarting game to test model!\n')
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
    print('Looped through game...repeating loop.')