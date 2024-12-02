import os

import numpy as np

from keras.layers import LSTM, Embedding, Input, Dense, GlobalAveragePooling1D, Concatenate, Bidirectional
from keras.models import Model
from keras import optimizers, backend as K

from warnings import filterwarnings

filterwarnings('ignore')
import random
from utils import letter_dict, letters, pad_sequences, cheat_letters

class Network(object) :
    """Define the network
    network consists of two input that reads the current state 
    and one hot encoded matrix of guessed letters
    This class includes some helper function to ease training and inference
    """
    def __init__(self, maxlen = 29) :
        state_embedding = self.get_state_embedding(maxlen)
        guessed_embedding = self.get_guessed_embedding()
        x = Concatenate(name='concatenate')([state_embedding.output, guessed_embedding.output])
        x = Dense(100, activation = 'tanh', name='dense')(x)
        x = Dense(26, activation = 'softmax', name='output')(x)
        self.full_model = Model([state_embedding.input, guessed_embedding.input], x, name = 'fullmodel')
        self.compile()
        
    def get_state_embedding(self, maxlen = 29) :
        inp = Input(shape = (maxlen,), name='state_input')
        x = Embedding(30, 100, mask_zero = True, name='state_embedding')(inp)
        x = Bidirectional(LSTM(100 , dropout = 0.2, return_sequences=True), name='state_bilstm_1')(x)
        x = Bidirectional(LSTM(100, dropout = 0.2 , return_sequences=True), name='state_bilstm_2')(x)
        x = GlobalAveragePooling1D(name='state_pooling')(x)
        x = Dense(100, activation = 'tanh', name='state_dense')(x)
        return Model(inp, x, name='state_model')
    
    def get_guessed_embedding(self) :
        inp = Input(shape = (26,), name='guessed_input')
        x = Dense(60, activation = 'tanh', name='guessed_dense_1')(inp)
        x = Dense(60, activation = 'tanh', name='guessed_dense_2')(x)
        return Model(inp, x, name='guessed_model')

    def __call__(self, state, guessed) :
        return self.full_model.predict([state,guessed], verbose=0).flatten()
    
    def fit(self, *args, **kwargs) :
        return self.full_model.fit(*args, **kwargs)
    
    def train_on_batch(self, *args, **kwargs) :
        return self.full_model.train_on_batch(*args, **kwargs)        
    
    def summary(self) :
        self.full_model.summary()

    def save(self, *args, **kwargs) :
        self.full_model.save(*args, **kwargs)

    def load_weights(self, *args, **kwargs) :
        self.full_model.load_weights(*args, **kwargs)
        self.compile()

    def compile(self, optimizer= None) : 
        if optimizer is not None :
            self.full_model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
        else :
            self.full_model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(1e-3, clipnorm=1))

class Agent(object) :

    """Agent definition.
    Agent is embedded with a model and policy
    Agent can use stochastic policy i.e. choose action randomly from computed probability 
    or greedy i.e choose the most probable action out of unused actions.
    
    Agent is trained off-policy, after a set amount of episode (in my case I trained with 3 episodes), 
    and after each episode during training must be finalized with finalize_episode method 
    to compute the correct course of actions.

    train_model method will collect accumulated episodes and perform one iteration of gradient descent
    with the collected episode data.

    Tried both with stochastic and greedy. Greedy policy performs converges better.
    """

    def __init__(self, model, policy = 'greedy') :
        self.reset_guessed()
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = 'greedy'
        self.policy = property(self.get_policy, self.set_policy)
        self.reset_guessed()
        self.is_training = True
        self.model = model

    @staticmethod
    def guessed_mat(guessed) :
        mat = np.empty([1,26])
        for i, l in enumerate(letters) :
            mat[0,i] = 1 if l in guessed else 0
        return mat
    
    def get_guessed_mat(self) :
        return self.guessed_mat(self.guessed)

    def reset_guessed(self) :
        self.guessed = []

    def get_probs(self, state) :
        raise NotImplementedError()

    def get_policy(self) :
        return self._policy

    def set_policy(self, policy) :
        if policy not in ['stochastic', 'greedy'] :
            raise ValueError('Policy can only be stochastic or greedy')
        self._policy = policy

    def select_action(self,state) :            
        probs = self.get_probs(state)
        if self._policy == 'greedy' :            
            i = 1
            sorted_probs = probs.argsort()
            while letters[sorted_probs[-i]] in self.guessed :
                i+= 1
            idx_act = sorted_probs[-i]
        elif self._policy == 'stochastic' :
            idx_act = np.random.choice(np.arange(probs.shape[0]), p = probs)
        guess = letters[idx_act]
        if guess not in self.guessed :
            self.guessed.append(guess)
            
        return guess
        
    def eval(self) :
        self.is_training = False
        self.set_policy('greedy') 
    
    def train(self) :
        self.is_training = True

class NNAgent(Agent) :
    def __init__(self, model, maxlen=29, policy='greedy') :
        super().__init__(model, policy)
        self.episode_memory = []
        self.states_history = []
        self.maxlen = maxlen

    def train_model(self):
        inp_1, inp_2, obj = zip(*self.states_history)
        inp_1 = np.vstack(list(inp_1)).astype(float)
        inp_2 = np.vstack(list(inp_2)).astype(float)
        obj = np.vstack(list(obj)).astype(float)
        loss = self.model.train_on_batch([inp_1,inp_2], obj)
        self.states_history = []
        return loss

    def get_probs(self, state) :
        state = self.preprocess_input(state)
        probs = self.model(*state)
        probs /= probs.sum()
        return probs

    def finalize_episode(self, answer) :
        inp_1, inp_2 = zip(*self.episode_memory)
        inp_1 = np.vstack(list(inp_1)).astype(float)      #stack the game state matrix
        inp_2 = np.vstack(list(inp_2)).astype(float)      #stack the one hot-encoded guessed matrix
        obj = 1.0 - inp_2                                 #compute the unused letters one-hot encoded
        len_ep = len(self.episode_memory)                 #length of episode
        correct_mask = np.array([[1 if l in answer else 0 for l in letters]]) # get mask from correct answer
        correct_mask = np.repeat(correct_mask, len_ep, axis = 0).astype(float)
        obj = correct_mask * obj  #the correct action is choosing the letters that are both unused AND exist in the word
        obj /= obj.sum(axis = 1).reshape(-1,1) #normalize so it sums to one
        self.states_history.append((inp_1, inp_2,obj))
        self.episode_memory = []
        self.reset_guessed()
    
    def preprocess_input(self, state) :
        new_input = []
        for l in state :
            new_input.append(letter_dict[l])
        state = pad_sequences([new_input], maxlen = self.maxlen, padding="post")
        if self.is_training :
            self.episode_memory.append((state,self.get_guessed_mat()))
        return state, self.get_guessed_mat()


class CheatAgent(NNAgent) :

    def __init__(self, model, maxlen=29, policy='eps-greedy', epsilon=0.1) :
        super().__init__(model, maxlen)

        # override policy
        if policy not in ['eps-greedy', 'greedy', 'stochastic'] :
            raise ValueError('Policy can only be eps-greedy, greedy, or stochastic')
        self.policy = policy
        
        # epsilon greedy parameters
        self.epsilon = epsilon


    def load_pretrained_agent(self, model_path) :
        self.model.load_weights(model_path)

        # rebuild the output layer
        new_output_dim = 27
        x = self.model.full_model.get_layer('dense').output
        x = Dense(new_output_dim, activation='softmax', name='new_output')(x)
        self.model.full_model = Model(self.model.full_model.input, x, name='fullmodel')

        # compile the model
        self.model.compile()

    def load_weights(self, model_path) :
        # change the output layer to have 27 output dimensions
        new_output_dim = 27
        x = self.model.full_model.get_layer('dense').output
        x = Dense(new_output_dim, activation='softmax', name='new_output')(x)
        self.model.full_model = Model(self.model.full_model.input, x, name='fullmodel')

        # load the weights
        self.model.load_weights(model_path)

        # compile the model
        self.model.compile()

    def finalize_episode(self, answer):
        # 0. extract answer and episode information
        cheated = answer['cheated'] if 'cheated' in answer else None
        cheated_step = answer['cheated_step'] if 'cheated_step' in answer else None
        cheated_letter = answer['cheated_letter'] if 'cheated_letter' in answer else None
        doubted = answer['doubted'] if 'doubted' in answer else None
        doubted_step = answer['doubted_step'] if 'doubted_step' in answer else None
        answer = answer['ans']
        len_ep = len(self.episode_memory)
        # 1. unzip the episode memory
        inp_1, inp_2 = zip(*self.episode_memory)
        # 2. stack the game state matrix
        inp_1 = np.vstack(list(inp_1)).astype(float)
        # 3. stack the one hot-encoded guessed matrix
        inp_2 = np.vstack(list(inp_2)).astype(float)
        # 4. compute the unused letters one-hot encoded
        obj = 1.0 - inp_2
        # 5. update obj to assign 1 to the letter on which the env cheated
        if cheated:
            # update obj from the step the environment cheated
            for i in range(cheated_step, len_ep):
                obj[i][cheat_letters.index(cheated_letter)] = 1
        # 6. add a column for the doubt action
        obj = np.hstack([obj, np.ones((obj.shape[0], 1))])
        if doubted:
            obj[doubted_step][-1] = 0
        # 7. get mask from correct answer
        correct_mask = np.array([[1 if l in answer else 0 for l in letters]])
        # 8. repeat the mask for each step in the episode
        correct_mask = np.repeat(correct_mask, len_ep, axis=0).astype(float)
        # 9. update mask based on whether the environment cheated
        correct_mask = np.hstack([correct_mask, np.zeros((len_ep, 1))])
        if cheated:
            # update mask from the step the environment cheated
            for i in range(cheated_step, len_ep):
                correct_mask[i][-1] = 1
        # 10. the correct action is choosing the letters that are both unused AND exist in the word
        obj = correct_mask * obj
        # 11. normalize so it sums to one
        obj /= obj.sum(axis=1).reshape(-1, 1)
        # 12. if doubted correctly, obj for this last step should be all zeros
        if cheated and doubted and doubted_step == len_ep - 1:
            obj[-1] = np.zeros(obj.shape[1])
        self.states_history.append((inp_1, inp_2, obj))
        self.episode_memory = []
        self.reset_guessed()
        # debug
        if np.isnan(obj).any():
            print(answer)
            print(inp_1)
            print(inp_2)
            print(obj)
            print(cheated, cheated_step, cheated_letter, doubted, doubted_step)

    def train_model(self):
        # account for new target output dimension
        inp_1, inp_2, obj = zip(*self.states_history)
        inp_1 = np.vstack(list(inp_1)).astype(float)
        inp_2 = np.vstack(list(inp_2)).astype(float)
        obj = np.vstack(list(obj)).astype(float)
        loss = self.model.train_on_batch([inp_1, inp_2], obj)
        self.states_history = []
        return loss

    def select_action(self, state):
        probs = self.get_probs(state)
        if self.policy == 'eps-greedy' and np.random.rand() < self.epsilon:
            guess = random.choice(cheat_letters)
            idx_act = cheat_letters.index(guess)
        elif self.policy == 'stochastic':
            idx_act = np.random.choice(np.arange(probs.shape[0]), p = probs)
        else:
            # greedy
            i = 1
            sorted_probs = probs.argsort()
            while cheat_letters[sorted_probs[-i]] in self.guessed:
                i+= 1
            idx_act = sorted_probs[-i]
        guess = cheat_letters[idx_act]
        if guess not in self.guessed and guess != '?':
            self.guessed.append(guess)
        return guess
    
    def train(self) :
        self.is_training = True
        self.set_policy('eps-greedy')



if __name__ =='__main__' :
    pass
