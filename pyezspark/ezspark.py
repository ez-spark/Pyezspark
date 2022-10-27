from . import protocol_host as host
from . import protocol_trainer as trainer 
import requests
import json
import gym
import warnings

class EzSpark:
    def __init__(self, training_public_key, training_private_key = None, max_number_of_genomes_per_client = 100):
        warnings.filterwarnings('ignore')
        self.API_BASE_URL = 'https://api.ezspark.ai/trainings/'
        self.training_public_key = training_public_key
        self.training_private_key = training_private_key
        self.remote_ip = '62.112.9.80'
        self.remote_port = 3389
        self.max_number_of_genomes_per_client = max_number_of_genomes_per_client
        if self.max_number_of_genomes_per_client < 1:
            print("max number of genomes must be >= 1")
            exit(1)
        
    def execute(self, username = 'erre4', threads = 4):
        try:
            if self.training_private_key == None:
                # trainer
                ret = requests.get(self.API_BASE_URL+self.training_public_key)
                ret = json.loads(ret.text)
                gym_game_name = ret['gym_game_name']
                input_size = ret['input_size']
                output_size = ret['output_size']
                max_number_of_games = ret['max_number_of_games']
                max_number_of_steps = ret['max_number_of_steps']
                t = trainer.Trainer(gym_game_name, self.training_public_key, input_size, output_size, max_number_of_games, max_number_of_steps, threads, username = username)
                t.connect(self.remote_ip, self.remote_port)
                t.train()
            else:
                # host
                headers = {'authorization':'private_key '+self.training_private_key}
                ret = requests.get(self.API_BASE_URL+self.training_public_key, headers = headers)
                ret = json.loads(ret.text)
                gym_game_name = ret['gym_game_name']
                max_number_of_games = ret['max_number_of_games']
                max_number_of_steps = ret['max_number_of_steps']
                alone_training_iterations = ret['alone_training_iterations']
                training_public_key1 = ret['training_public_key1']
                training_public_key2 = ret['training_public_key2']
                training_public_key3 = ret['training_public_key3']
                training_private_key1 = ret['training_private_key1']
                training_private_key2 = ret['training_private_key2']
                training_private_key3 = ret['training_private_key3']
                
                #check the
                # 1 game must exists
                # 2 action space must be discrete
                # 3 action space must be a list of lists or a numpy array
                
                try:
                    env = gym.make(gym_game_name)
                    env.close()
                
                except:
                    print("the game does not exists")
                    exit(1)
                
                
                try:
                    a = env.action_space
                
                
                except:
                    print("action space is not defined")
                    exit(1)
                
                
                s = str(a)
                if 'Discrete' not in s:
                    print("Your action space is not discrete")
                    exit(1)
                
                
                h = host.Host(gym_game_name, alone_training_iterations,ret,max_number_of_games,max_number_of_steps,self.training_public_key,self.training_private_key,
                training_public_key1,training_private_key1,training_public_key2,training_private_key2,training_public_key3,
                training_private_key3)
                h.alone_training()
                h.distributed_training(self.remote_ip, self.remote_port,self.max_number_of_genomes_per_client,3)
        except:
            print("something went wrong")
            exit(1)
