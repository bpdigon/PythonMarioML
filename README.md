README
In this github you will find two files.
The first is 'CS595a_SuperMarioBros.ipynb', which is the model we created for our project. This file is a Jupyter Notebook/ JupyterLab file.

The second file is 'mario.yml'

There are two ways to run the model:
1. Uncomment all lines with a pip install. This will make it so when the cells are run they will install the packages they need as you go
2. Using the 'mario.yml' file, create a python environment with the necessary packages to run the model. A command such as 'conda create' or simply using 'yum install PACKAGENAME' works.
    mario_env.txt serves as an alternative list with all packages

Notes and considerations
1. Microsoft Visual Studio needed to be installed and updated for the GUI containing Mario to display.
2. The model with the configurations in the provided code will take anywhere from 18-24 hours to fully train. Adjust accordingly!
3. Tips to adjust: reduce stepsize and total timesteps in the following lines:
   'model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=512)'
   'model.learn(total_timesteps=1000000, callback=callback)'
4. The code will save the model at certain intervals. To reduce the number of models saved (and PC storage taken up) Adjust the following line according to the adjustments made in previous code:
   'callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)'
