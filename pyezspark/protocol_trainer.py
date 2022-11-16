import ezclient
import requests
import time
import json
from datetime import datetime

class Trainer:
    def __init__(self, gym_game_name, training_public_key, input_size, output_size, max_number_of_games, max_number_of_steps, threads, username=None):
        if max_number_of_steps <= 0 or max_number_of_games <= 0 or input_size <= 0 or output_size <= 0 or threads <= 0:
            print("Error, input_size and/or output and/or max_number_of_games and/or max_number of steps size are wrong")
            exit(1)
        self.client = ezclient.Client(training_public_key,gym_game_name=gym_game_name, buffer_size = 30000, genome_input = input_size, genome_output = output_size, username=username)
        self.client_secondary = ezclient.Client(training_public_key,gym_game_name=gym_game_name, buffer_size = 30000, genome_input = input_size, genome_output = output_size, username=username)
        self.gym_game_name = gym_game_name
        self.training_public_key = training_public_key
        self.input_size = input_size
        self.output_size = output_size
        self.max_number_of_games = max_number_of_games
        self.max_number_of_steps = max_number_of_steps
        self.threads = threads
        self.remote_ip = None
        self.remote_port = None
        self.ez_server_addr = 'https://alpha-p2p.ezspark.ai/rest/'
        self.active_training = 'isTrainingActive/'
        self.environment_creation_endpoint = '/v1/envs/'
        self.next_step_endpoint = '/v1/envs/step/'
        self.first_iteration_distributed_training = True
        self.headers = {'Content-type': 'application/json'}
    
    #calling connect with try except after the except re try, but first we ask through http 
    # api if that training exists
    def connect(self, remote_ip, remote_port):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.client.connect(remote_ip,remote_port)
        
    def set_file_descriptor(self,to_secondary):
        if to_secondary:
            fd = self.client.get_file_descriptor()
            self.client_secondary.set_file_descriptor(fd)
        else:
            fd = self.client_secondary.get_file_descriptor()
            self.client.set_file_descriptor(fd)
    
    def send_http_request(self, http_request_string, complete_secondary_training_public_key):
        # the returning value
        id = None
        
        # splitting with all the 3 host sockets public keys
        first_key = complete_secondary_training_public_key[:int(len(complete_secondary_training_public_key)/3)]
        second_key = complete_secondary_training_public_key[int(len(complete_secondary_training_public_key)/3):int(2*len(complete_secondary_training_public_key)/3)]
        third_key = complete_secondary_training_public_key[int(2*len(complete_secondary_training_public_key)/3):]
        # getting the current socket file descriptor
        fd = self.client.get_file_descriptor()
        # init the secondary client
        self.client_secondary.send_http_request_setup(self.remote_ip,self.remote_port,http_request_string,complete_secondary_training_public_key,fd)
        while True:
            while(not self.client_secondary.is_disconnected()):
                self.client_secondary.trainer_direct_main_loop_http_requests()
            
            # if something went wrong
            if self.client_secondary.got_broken_pipe():
                #lets check the training is still active
                
                ret = requests.get(self.ez_server_addr+self.active_training+self.training_public_key, verify = False)
                ret = json.loads(ret.content)
                if not ret['ret']:
                    return None
                
                # try to change the target host socket
                
                temp = first_key
                if self.client_secondary.get_secondary_key() == first_key:
                    temp = second_key
                elif self.client_secondary.get_secondary_key() == second_key:
                    temp = third_key
                self.client_secondary.connect_http_handler(self.remote_ip,self.remote_port,temp,http_request_string,fd)
                continue
            try:
                id = self.client_secondary.get_string().decode('utf-8').replace("'",'"').replace('False','false').replace('True','true')
            except:
                print("Malicious host")
                exit(1)
            break
        return id
    
    def train(self):
        environment_name = None
        last = datetime.now()
        limit = 1200#20 minutes
        while(True):
            current = datetime.now()
            if (current-last).total_seconds() >= limit:
                exit(1)
            # communication protocol is acting
            while(not self.client.is_disconnected()):
                self.client.trainer_direct_main_loop()
            #we have been disconnected by the server (timeout, or we communicated something bad to host, or to the server)
            if self.client.got_broken_pipe():
                #lets check the training is still active
                
                ret = requests.get(self.ez_server_addr+self.active_training+self.training_public_key, verify = False)
                ret = json.loads(ret.content)
                if not ret['ret']:
                    return
                
                
                self.client.connect(self.remote_ip,self.remote_port)
                # adding request to server with http when was last time host connected
                # then, if the current time - last time >= limits close this process (probably the host is not hosting anymore)
                continue
            #lets check if is ok what the host has sent us
            if not self.client.is_body_ok_for_trainer_neat():
                exit(1)
            # first time receiving something
            # pay attention, if the host starts the same training closing its entire process
            # it will not have this identifier again, and this trainer will be constantly disconnected by the host
            # in that case this trainer should shut down the process and restart it
            if not self.client.identifier_is_set():
                self.client.get_body_identifier()
            # set the genomes structures
            self.client.get_genomes()
            # set the innovation number of connections
            self.client.get_global_innovation_number_connections()
            # set the innovation number of nodes
            self.client.get_global_innovation_number_nodes()
            # set the number of genomes
            self.client.get_number_of_genomes()
            # retrieve the secondary sockets public keys
            secondary_sockets= self.client.get_link().decode('utf-8')
            identifier = self.client.get_py_identifier()
            # retrieve the environment name if we didn't yet
            if environment_name == None:
                environment_name = self.client.get_environment_name().decode('utf-8')
            # python can now see the number of genomes
            n_genomes = self.client.returns_n_genomes()
            # we use the socket already initialized
            if not self.first_iteration_distributed_training:
                self.set_file_descriptor(True)
            else:
                self.first_iteration_distributed_training = False
            # creating the environments:
            res = {'full':True}
            first_time = True
            param_init = {'endpoint': self.environment_creation_endpoint, 'env_id':environment_name, 'n_instances':n_genomes, 'identifier':identifier}
            param_init_str = str(param_init)
            param_init_str = param_init_str.replace("'",'"')
            # maybe the client is already serving too  much clients
            # so we see if we can open other environments
            # if we can a res{'full':True} will be received
            # and we will wait for 1 sec to ask again
            #init the environments
            while 'full' in res and res['full']:
                if not first_time:
                    time.sleep(1)
                try:
                    res = self.send_http_request(param_init_str, secondary_sockets)
                    res = json.loads(res)
                except:
                    print("exiting")
                    exit(1)
            if 'ok' in res or res == None:
                last = datetime.now()
                self.set_file_descriptor(False)
                continue
            list_keys = list(res.keys())
            list_keys.sort()
            if n_genomes != len(list_keys):
                exit(1)
            
            #init some parameters
            list_states = []
            list_rewards = []
            steps = []
            games = []
            to_count_yet = []
            done = []
            game_done = []
            
            for i in range(n_genomes):
                list_states.append(res[list_keys[i]]['obs'])
                list_rewards.append(0)
                steps.append(0)
                games.append(0)
                to_count_yet.append(1)
                done.append(0)
                game_done.append(False)
            # training
            while True:
                list_output_to_keep = []
                for i in range(n_genomes):
                    if games[i] < self.max_number_of_games-1 or games[i] < self.max_number_of_games and steps[i] < self.max_number_of_steps-1 and not game_done[i]:
                        list_output_to_keep.append(1)
                    else:
                        list_output_to_keep.append(0)
                out = self.client.forward_genomes(list_states, self.threads, done, to_count_yet, list_keys, list_rewards, list_output_to_keep)
                param_post = {}
                list_of_environments = []
                actions = []
                j = 0
                n_done = 0
                for i in range(n_genomes):
                    if steps[i] < self.max_number_of_steps and games[i] < self.max_number_of_games:
                        steps[i]+=1
                        if game_done[i] or steps[i] == self.max_number_of_steps:
                            games[i]+=1
                            steps[i] = 0
                    if done[i] > 0:
                        to_count_yet[i] = 0
                    else:
                        if games[i] < self.max_number_of_games:
                            list_of_environments.append(list_keys[i])
                            actions.append(out[j])
                            
                        else:
                            done[i] = 1
                            to_count_yet[i] = 0
                        j+=1
                length_env_to_count = len(list_of_environments)
                if length_env_to_count == 0:
                    break
                for i in range(length_env_to_count):
                    param_post[list_of_environments[i]] = actions[i]
                    param_post['endpoint'] = self.next_step_endpoint
                    param_post_str = str(param_post)
                    param_post_str = param_post_str.replace("'",'"')
                try:
                    res = self.send_http_request(param_post_str, secondary_sockets)
                    res = json.loads(res)
                except:
                    exit(1)
                if 'ok' in res or res == None:
                    last = datetime.now()
                    self.set_file_descriptor(False)
                    continue
                list_got = list(res.keys())
                for i in range(n_genomes):
                    if list_keys[i] in list_got:
                        list_states[i] = res[list_keys[i]][0]
                        list_rewards[i] = res[list_keys[i]][1]
                        game_done[i] = res[list_keys[i]][2]
            self.client.set_values_back_in_body()
            last = datetime.now()
            self.set_file_descriptor(False)
