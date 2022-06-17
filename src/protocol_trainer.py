import ezclient
import requests
import time
import json

class Trainer:
    def __init__(self, gym_game_name, training_public_key, input_size, output_size, max_number_of_games, max_number_of_steps, threads):
        if max_number_of_steps <= 0 or max_number_of_games <= 0 or input_size <= 0 or output_size <= 0 or threads <= 0:
            print("Error, input_size and/or output and/or max_number_of_games and/or max_number of steps size are wrong")
            exit(1)
        self.client = ezclient.Client(training_public_key,gym_game_name=gym_game_name, buffer_size = 30000, genome_input = input_size, genome_output = output_size)
        self.gym_game_name = gym_game_name
        self.training_public_key = training_private_key
        self.input_size = input_size
        self.output_size = output_size
        self.max_number_of_games = max_number_of_games
        self.max_number_of_steps = max_number_of_steps
        self.threads = threads
        self.remote_ip = None
        self.remote_port = None
        self.environment_creation_endpoint = '/v1/envs/'
        self.next_step_endpoint = '/v1/envs/step/'
        self.headers = {'Content-type': 'application/json'}
        
    def connect(self, remote_ip, remote_port):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.client.connect(remote_ip,remote_port)
    
    def train(self):
        environment_name = None
        while(True):
            
            # communication protocol is acting
            while(not self.client.is_disconnected()):
                self.client.trainer_direct_main_loop()
            #we have been disconnected by the server (timeout, or we communicated something bad to host, or to the server)
            if self.client.got_broken_pipe():
                self.client.connect(self.remote_ip,self.remote_port)
                continue
            #lets check if is ok what the host has sent us
            if not self.client.is_body_ok_for_trainer_neat():
                print("malicious host, closing the connection with this training, and so the process")
                exit(1)
            # first time receiving something
            # pay attention, if the host starts the sames training closing its entire process
            # it will not have this identifier again, and this trainer will be constantly disconnected by the host
            # in that case this trainer should shut down the process and restart it
            if not self.client.identifier_is_set():
                client.get_body_identifier()
            # set the genomes structures
            self.client.get_genomes()
            # set the innovation number of connections
            self.client.get_global_innovation_number_connections()
            # set the innovation number of nodes
            self.client.get_global_innovation_number_nodes()
            # set the number of genomes
            self.client.get_number_of_genomes()
            # retrieve the url we must ask for to get the states
            base_url= self.client.get_url()
            # retrieve the environment name if we didn't yet
            if environment_name == None:
                environment_name = self.client.get_environment_name()
            # python can now see the number of genomes
            n_genomes = self.client.returns_n_genomes()
            
            
            # creating the environments:
            res = {'full':True}
            first_time = True
            param_init = {'env_id':environment_name, 'n_instances':n_genomes}
            # maybe the client is already serving too  much clients
            # so we see if we can open other environments
            # if we can a res{'full':True} will be received
            # and we will wait for 1 sec to ask again
            
            #init the environments
            while 'full' in res and res['full']:
                if not first_time:
                    time.sleep(1)
                res = requests.post(base_url+self.environment_creation_endpoint, data = json.dumps(param_init), headers=self.headers)
                res = json.loads(res.content)
            list_keys = list(res.keys())
            list_keys.sort()
            
            if n_genomes != len(list_keys):
                print("Malicious host, closing the client")
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
            train = True
            while train:
				out = self.client.forward_genomes(list_states, self.threads, done, to_count_yet, list_keys, list_rewards)
				param_post = {}
				for i in n_genomes:
					if steps[i] < self.max_number_of_steps and games[i] < self.max_number_of_games:
						steps[i]+=1
						if game_done[i]:
							games[i]+=1
					if steps[i] < self.max_number_of_steps and games[i] < self.max_number_of_games:
				len_out = len(out)
				index
                env = requests.post(self.base_url+self.next_step_endpoint, data = json.dumps(param_init), headers=headers)
                print(env)
                list_instance.append(json.loads(env.content)['instance_id'])
                state.append(json.loads(env.content)['obs'])
                rewards.append(0)
                done.append(False)
                counter.append(0)
            print("Environment created")
            flag = True
            while flag:
                start = time.time()
                print("forwarding genomes")
                done_g = []
                for i in range(n_genomes):
                    if counter[i]>1:
                        done_g.append(1)
                    else:
                        done_g.append(0)
                ret = client.forward_genomes(state, 10,done_g)
                for i in range(n_genomes):
                    if counter[i] < 1:
                        client.add_reward_to_genome(rewards[i], i)
                end = time.time()
                print("Computational time: ",end - start)
                for i in range(n_genomes):
                    if done[i]:
                        counter[i]+=1
                start = time.time()
                d = {}
                for i in range(n_genomes):
                    if counter[i] < 1:
                        if not done[i]:
                            d[list_instance[i]] = ret[i]
                            #param_action['instance_id'] = list_instance[i]
                            #param_action['action'] = ret[i]
                            #env = requests.post(step_url.format(instance_id = list_instance[i]), data = json.dumps(param_action), headers=headers)
                            #state[i] = json.loads(env.content)['observation']
                            #rewards[i] = json.loads(env.content)['reward']
                            #done[i] = json.loads(env.content)['done']
                        else:
                            rewards[i] = 0
                            done[i] = False
                            counter[i]+=1
                    else:
                        rewards[i] = 0
                env = requests.post(step_url, data = json.dumps(d), headers=headers)
                end = time.time()
                
                print("Request time: ",end - start)
                #print(env)
                #print(env.content)
                
                d = json.loads(env.content)
                for i in d.keys():
                    for j in range(len(list_instance)):
                        if i == list_instance[j]:
                            state[j] = d[i][0]
                            rewards[j] = d[i][1]
                            done[j] = d[i][2]
                            break
                flag = False
                for i in range(n_genomes):
                    if counter[i] < 1:
                        flag = True
                        break
            print("finished")
            client.set_values_back_in_body()
