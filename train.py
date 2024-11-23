import os
from tqdm import tqdm
from keras import optimizers

from env import Hangman
from network import Network, NNAgent
from utils import *


class HangmanTrainer:
    def __init__(self, player=None, word_src='words.txt', save_path='policy_v2.h5'):
        self.word_src = word_src
        self.save_path = save_path

        # Load words from file
        if type(word_src) == list:
            self.words = word_src
        else:
            with open(word_src, 'r') as f:
                self.words = f.read().splitlines()
        
        # Initialize game parameters and network
        len_list = list(map(len, self.words))
        maxlen = max(len_list)
        print('Max length of words is', maxlen)

        if player is not None:
            if maxlen <= player.maxlen:
                self.player = player
            else:
                raise ValueError('The maximum length of words in the word source is greater than the maximum length of words in the player.')
        else:
            self.player = NNAgent(Network(maxlen=maxlen), maxlen=maxlen)
        self.player.model.summary()


    def train(self, env, max_lives=6, n_trials=70000, warmup=True, max_lives_warmup=8, n_trials_warmup=20000, save_episode=5000, view_episode=500, update_episode=5):
        """
        Train the agent using the given environment.
        
        Parameters
        ----------
        env : Hangman
            The environment to train the agent on.
        max_lives : int 
            The maximum number of lives.
        n_trials : int
            The number of episodes to train the agent.
        warmup : bool
            Whether to start with a warmup phase with extended max_lives.
        max_lives_warmup : int
            The maximum number of lives for the warmup phase.
        n_trials_warmup : int
            The number of episodes for the warmup phase.
        save_episode : int
            The number of episodes before saving the model.
        view_episode : int
            The number of episodes before printing the average correct per episode.
        update_episode : int
            The number of episodes before updating the model.
        """
        avg_correct = 0
        wins_avg = 0
        progbar = tqdm(range(n_trials))

        if warmup:
            # Warmup training phase with extended max_lives for better data collection.
            env.set_max_lives(max_lives_warmup)
            print('Warmup Training Start ...', end='\n\n')
            self.player = self.train(env, max_lives=max_lives_warmup, n_trials=n_trials_warmup, warmup=False, save_episode=save_episode, view_episode=view_episode, update_episode=update_episode)
            env.set_max_lives(max_lives)
            print('Warmup Training Done', end='\n\n')
        
        for episode_set in progbar:
            for _ in range(update_episode):
                state = env.reset()
                done = False
                correct_count = 0
                while not done:
                    guess = self.player.select_action(state)
                    state, reward, done, ans = env.step(guess)
                    if reward > 0:
                        correct_count += 1.0
                    if reward == env.win_reward:
                        wins_avg += 1.0
                self.player.finalize_episode(ans['ans'])
                avg_correct += correct_count
            loss = self.player.train_model()
            progbar.set_description("Loss : {:.3f}              ".format(loss))
            
            if (episode_set + 1) % view_episode == 0:
                views = (episode_set + 1,avg_correct/(update_episode*view_episode), view_episode*update_episode, wins_avg/(update_episode*view_episode))
                print('Episode {} -------- Average Correct Count : {:.3f}     Last {} winrate : {:.3f}'.format(*views))
                if loss is not None:
                    print('Loss :', loss)
                    print()
                    avg_correct = 0
                    wins_avg = 0

            if (episode_set + 1) % save_episode == 0:
                self.player.model.save(self.save_path, include_optimizer=False)

        print('Training Done')
        return self.player