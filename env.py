import numpy as np

class Hangman(object) :
    """ Hangman Game.
    Implemented Hangman game with almost gym-like environment
    Requires word source(word_src) to build dictionary
    Reward parameter can be set manually.
    Game board will refers to the word that we are asked to guess.

    Brief Description :
    reset : starts the game, cleared guessed letters memory, 
            select random word from word collection,
            returns a masked version of the word e.g. 'alpaca' = '------'

    step : accepts an alphabet as input, 
           Evaluation of this input will occur in these steps :
           1. Check if the input is used already, if used return similar state, 
              if there is repeated_guessing_penalty, will return this penalty as reward.
              If not used go to next step.

           2. if the letter exists in the game board,
              will reveal every existence of the letter in the gameboard.
              if not the last letter of the problem will return correct_reward,
              otherwise return win_reward as reward

            3. If the letter does not exist in the game board. Return similar game board.
               Will reduce live by 1. If lives reaches 0. Game over.
               Return false_reward if not game over, else return lose_reward 
    """
    def __init__(self , 
                 word_src, 
                 max_lives = 6 , 
                 win_reward = 30,
                 correct_reward = 1,
                 repeated_guessing_penalty = -100,
                 lose_reward = -0, 
                 false_reward = -0,
                 verbose = False) :
        if type(word_src) == list :
            self.words = word_src
        else :
            with open(word_src, 'r') as f :
                self.words = f.read().splitlines()
        # maximum word length
        self.maxlen = max(map(len, self.words))
        self.max_lives = max_lives
        self.win_reward = win_reward
        self.correct_reward = correct_reward
        self.lose_reward = lose_reward
        self.false_reward = false_reward
        self.verbose = verbose
        self.repeated_guessing_penalty = repeated_guessing_penalty
        self.game_won = False
        
    def pick_random(self) :
        self.guess_word = np.random.choice(self.words)
    
    def get_current_live(self) :
        return self.curr_live
    
    def get_guess_word(self) :
        return self.guess_word
    
    def set_max_lives(self, max_lives) :
        self.max_lives = max_lives
    
    def is_game_won(self) :
        return self.game_won
        
    def reset(self, guess_word = None) :
        self.curr_live = self.max_lives
        self.game_won = False
        if guess_word is not None :
            self.guess_word = guess_word
        else :
            self.pick_random()
        self.guessing_board = ['-' for i in range(len(self.guess_word))]
        self.correct_guess = 0
        self.guessed = []
        self.done = False
        # if self.verbose :
        #     print('Game Starting')
        #     print('Current live :', self.curr_live)
        return self.get_gameboard()
        

    def get_gameboard(self) :
        return ''.join(self.guessing_board)
    
    def show_status(self, add_info = '') :
        if self.verbose:
            gameboard = self.get_gameboard()
            if add_info != '' :
                print(f'>> {gameboard} | {self.curr_live} ! {add_info}')
            else :
                print(f'>> {gameboard} | {self.curr_live}')
        
    def step(self, letter) :
        # check if the letter is in alphabet
        if not(letter.isalpha()) :
            raise TypeError('Can only accept alphabet')
        
        # convert to lowercase
        letter = letter.lower()

        # check if the letter is used already
        if letter not in self.guessed :
            self.guessed.append(letter)
        else :
            self.show_status('Word used already')
            return self.get_gameboard(), self.repeated_guessing_penalty, self.done, {}

        # check if the guessed letter is in the word
        if letter in self.guess_word :
            # reveal the letter in the game board
            for i in range(len(self.guess_word)) :
                if letter == self.guess_word[i] :
                    self.guessing_board[i] = letter
                    self.correct_guess += 1
            # check if the game is won
            if self.correct_guess == len(self.guess_word) :
                self.done = True
                self.game_won = True
                self.show_status('You Win')
                if self.verbose :
                    print(f"Word is '{self.guess_word}'")
                return self.guess_word, self.win_reward, self.done, {'ans' : self.guess_word}
            else :                
                self.show_status()
                return self.get_gameboard(), self.correct_reward, self.done, {}
        else :
            self.curr_live -= 1
            # check if the game is lost
            if self.curr_live == 0 :
                self.done = True
                self.show_status('You Lose')
                if self.verbose :
                    print(f"Word is '{self.guess_word}'")
                return self.get_gameboard(), self.lose_reward, self.done, {'ans' : self.guess_word}
            else :
                self.show_status()
                return self.get_gameboard(), self.false_reward, self.done, {}
        

class HangmanCheat(Hangman):
    """ Hangman Cheat Game.
    Similar to the Hangman game, but with the ability to cheat.
    If the environment is in cheat mode, it will reject the agent's guess in a given share of steps, regardless of whether the guess is correct or not.
    The agent can still guess the word, but the environment will not accept the guess.    
    """
    def __init__(self , 
                 word_src, 
                 max_lives = 6 , 
                 win_reward = 30,
                 correct_reward = 1,
                 repeated_guessing_penalty = -100,
                 lose_reward = -0, 
                 false_reward = -0,
                 reject_rate = 0.1,
                 verbose = False) :
        super().__init__(word_src, max_lives, win_reward, correct_reward, repeated_guessing_penalty, lose_reward, false_reward, verbose)
        self.reject_rate = reject_rate
        self.cheated = False
        self.cheated_step = 0
        self.cheated_letter = ''
        self.doubted = False
        self.doubted_step = 0


    def reset(self, guess_word = None) :
        self.cheated = False
        self.cheated_step = -1
        self.cheated_letter = ''
        self.doubted = False
        self.doubted_step = -1
        return super().reset(guess_word)


    def step(self, letter) :
        # count the step number to know at which env started cheating
        if not self.cheated:
            self.cheated_step += 1
        if not self.doubted:
            self.doubted_step += 1

        # check if agent is doubting
        if letter == '?':
            self.doubted = True
            # check if the agent is correct in doubting
            if self.cheated:
                # agent is correct in doubting and wins the game
                self.done = True
                self.game_won = True
                self.show_status('Doubt Correct')
                if self.verbose :
                    print(f"Word is '{self.guess_word}'")
                # print("Doubted correctly")
                return self.get_gameboard(), self.win_reward, self.done, {'ans' : self.guess_word, 'cheated' : self.cheated, 'cheated_step' : self.cheated_step, 'cheated_letter' : self.cheated_letter,'doubted' : self.doubted, 'doubted_step' : self.doubted_step}
            else:
                # agent is wrong in doubting, game continues if lives are left
                self.curr_live -= 1
                # print("Doubted falsely")
                # check if the game is lost
                if self.curr_live == 0 :
                    self.done = True
                    self.show_status('Doubt False, You Lose')
                    if self.verbose :
                        print(f"Word is '{self.guess_word}'")
                    return self.get_gameboard(), self.lose_reward, self.done, {'ans' : self.guess_word, 'cheated' : self.cheated, 'doubted' : self.doubted, 'doubted_step' : self.doubted_step}
                else :
                    self.show_status('Doubt False, Game Continues')
                    return self.get_gameboard(), self.false_reward, self.done, {}

        # check if the letter is in alphabet
        if not(letter.isalpha()) :
            raise TypeError('Can only accept alphabet')
        
        # convert to lowercase
        letter = letter.lower()

        # check if the letter is used already
        if letter not in self.guessed :
            self.guessed.append(letter)
        else :
            self.show_status('Word used already')
            return self.get_gameboard(), self.repeated_guessing_penalty, self.done, {}

        # check if the guessed letter is in the word
        if letter in self.guess_word :
            # NEW: Reject the guess with a certain probability
            if not self.cheated and np.random.rand() < self.reject_rate:
                self.cheated = True
                self.cheated_letter = letter
                # remove the letter from the guessed list
                self.guessed.remove(letter)
                self.show_status('Rejected guess')
                return self.get_gameboard(), self.false_reward, self.done, {}
            # reveal the letter in the game board
            for i in range(len(self.guess_word)) :
                if letter == self.guess_word[i] :
                    self.guessing_board[i] = letter
                    self.correct_guess += 1
            # check if the game is won
            if self.correct_guess == len(self.guess_word) :
                self.done = True
                self.game_won = True
                self.show_status('You Win')
                if self.verbose :
                    print(f"Word is '{self.guess_word}'")
                return self.guess_word, self.win_reward, self.done, {'ans' : self.guess_word, 'cheated' : self.cheated, 'cheated_step': self.cheated_step, 'cheated_letter' : self.cheated_letter, 'doubted' : self.doubted, 'doubted_step' : self.doubted_step}
            else :                
                self.show_status()
                return self.get_gameboard(), self.correct_reward, self.done, {}
        else :
            self.curr_live -= 1
            # check if the game is lost
            if self.curr_live == 0 :
                self.done = True
                self.show_status('You Lose')
                if self.verbose :
                    print(f"Word is '{self.guess_word}'")
                return self.get_gameboard(), self.lose_reward, self.done, {'ans' : self.guess_word, 'cheated' : self.cheated, 'cheated_step': self.cheated_step, 'cheated_letter' : self.cheated_letter, 'doubted' : self.doubted, 'doubted_step' : self.doubted_step}
            else :
                self.show_status()
                return self.get_gameboard(), self.false_reward, self.done, {}