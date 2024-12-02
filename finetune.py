from train import HangmanTrainer
from env import HangmanCheat
from network import Network, CheatAgent

step_size = 500
step = 3500
# create the cheat environment
cheat_env = HangmanCheat(word_src='words/top_1000.txt')

# create the cheat agent
cheat_agent = CheatAgent(Network(maxlen=cheat_env.maxlen), maxlen=cheat_env.maxlen)

# load pre-trained weights
cheat_agent.load_weights(f'models/finetuned/policy_finetuned_{step}.h5')

# create the trainer
cheat_trainer = HangmanTrainer(player=cheat_agent, word_src='words/top_1000.txt', save_path='models/finetuned/policy_finetuned')

# fine-tune the agent
cheat_trainer.fine_tune(cheat_env, n_trials=step_size, save_episode_init=step)