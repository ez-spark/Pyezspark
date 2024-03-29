from pyezspark import gym_http_server
import gym
import ezclient
import threading
import re
import time
import requests
import json
import datetime
import os

class localTunnelRun(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.url = localtunnel.get_localtunnel_url()
    
    def get_url(self):
        return self.url
        
    def run(self):
        localtunnel.main(self.ip, self.port, self.url)

class postTunnelRun(threading.Thread):
    def __init__(self, gym_server_url,training_public_key, training_private_key, neat, gym_game_name, buffer_size, genome_input, genome_output, url_name, remote_ip, remote_port, genomes_per_client):
        threading.Thread.__init__(self)
        self.gym_server_url = gym_server_url
        self.training_public_key = training_public_key
        self.training_private_key = training_private_key
        self.neat = neat
        self.gym_game_name = gym_game_name
        self.buffer_size = buffer_size
        self.input_size = genome_input
        self.output_size = genome_output
        self.url_name = url_name
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.genomes_per_client = genomes_per_client
        self.client = ezclient.Client(self.training_public_key,training_private_key = self.training_private_key,neat_class = self.neat, gym_game_name = self.gym_game_name, buffer_size = self.buffer_size, genome_input = self.input_size, genome_output = self.output_size, url_name = self.url_name)
        self.headers = {'Content-type': 'application/json'}
    def run(self):
        self.client.connect(self.remote_ip,self.remote_port, genomes_per_client = self.genomes_per_client)
        while(True):
            # keep the communication active
            while(not self.client.is_disconnected()):
                self.client.host_direct_main_loop_http_requests()
                time.sleep(0.0001)
            # we have been disconnected by the server (bad requests or time out)
            if self.client.got_broken_pipe():
                self.client.connect(self.remote_ip, self.remote_port, genomes_per_client = self.genomes_per_client)
                continue
            # we are here, a trainer is communicating with us
            request_body = self.client.get_string().decode('utf-8')
            try:
                request_body = json.loads(request_body)
            except:
                self.client.set_body_http(0, 'null')
                continue
            if 'endpoint' not in request_body:
                self.client.set_body_http(0, 'null')
                continue
            endpoint = request_body['endpoint']
            request_body.pop('endpoint')

            if endpoint == '/v1/envs/':
                ret = gym_http_server.env_create(request_body)
            elif endpoint == '/v1/envs/step/':
                ret = gym_http_server.multi_step(request_body)
            else:
                self.client.set_body_http(0, 'null')
                continue
            self.client.set_body_http(1, ret)


class Host:
    def __init__(self, gym_game_name, alone_training_iterations, genomes_per_client, configuration_dict,
                 max_number_of_games, max_number_of_steps, training_public_key = None, training_private_key = None, socket1_training_public_key = None,
                 socket1_training_private_key = None,socket2_training_public_key = None,socket2_training_private_key = None,socket3_training_public_key = None,
                 socket3_training_private_key = None, t_val = 5, threads_for_alone_training = 4, neat_file = "neat.bin", alone_iteration_max_population = 300):
        if max_number_of_steps <= 0 or max_number_of_games <= 0 or alone_training_iterations < 0 or threads_for_alone_training < 1:
            print("Error, something among max number of steps, max number of games or alone training iterations is <= 0")
            exit(1)
        neat_as_byte = None
        if os.path.exists(neat_file):
            neat_as_byte = neat_file
        ezclient.get_randomness()
        self.t_val = t_val
        self.gym_game_name = gym_game_name
        self.input_size = configuration_dict['input_size']
        self.output_size = configuration_dict['output_size']
        self.threads_for_alone_training = threads_for_alone_training
        self.neat_file = neat_file
        self.alone_iteration_max_population = alone_iteration_max_population
        self.configuration_dict = configuration_dict
        if self.configuration_dict['minimum_reward'] == None:
            self.configuration_dict['minimum_reward'] = 0
        self.neat = ezclient.Neat(self.input_size, self.output_size,neat_as_byte = neat_as_byte, initial_population = configuration_dict['initial_population'],
                                  species_threshold = configuration_dict['species_threshold'],max_population = configuration_dict['max_population'],
                                  generations = configuration_dict['generations'], percentage_survivors_per_specie = configuration_dict['percentage_survivors_per_specie'], 
                                  connection_mutation_rate = configuration_dict['connection_mutation_rate'], new_connection_assignment_rate = configuration_dict['new_connection_assignment_rate'], 
                                  add_connection_big_specie_rate = configuration_dict['add_connection_big_specie_rate'], add_connection_small_specie_rate = configuration_dict['add_connection_small_specie_rate'], 
                                  add_node_specie_rate = configuration_dict['add_node_specie_rate'], activate_connection_rate = configuration_dict['activate_connection_rate'], 
                                  remove_connection_rate = configuration_dict['remove_connection_rate'], children = configuration_dict['children'], 
                                  crossover_rate = configuration_dict['crossover_rate'], saving = configuration_dict['saving'], limiting_species = configuration_dict['limiting_species'], limiting_threshold = configuration_dict['limiting_threshold'], same_fitness_limit = configuration_dict['same_fitness_limit'], keep_parents = configuration_dict['keep_parents'], age_significance = configuration_dict['age_significance'])
        self.alone_training_iterations = alone_training_iterations
        self.max_number_of_games = max_number_of_games
        self.max_number_of_steps = max_number_of_steps
        self.training_public_key = training_public_key
        self.training_private_key = training_private_key
        self.socket1_training_public_key = socket1_training_public_key
        self.socket1_training_private_key = socket1_training_private_key
        self.socket2_training_public_key = socket2_training_public_key
        self.socket2_training_private_key = socket2_training_private_key
        self.socket3_training_public_key = socket3_training_public_key
        self.socket3_training_private_key = socket3_training_private_key
        self.client = None
    
    def alone_training(self):
        self.neat.set_max_population(self.alone_iteration_max_population)
        list_envs = []
        list_states = []
        list_games = []
        list_steps = []
        list_keep_going = []
        for i in range(self.alone_training_iterations):
            self.neat.set_generation_iter(i)
            number_genomes = self.neat.get_number_of_genomes()
            print("number of genomes iteration number "+str(i)+": "+str(number_genomes))
            self.neat.reset_fitnesses()
            length = len(list_envs)
            for j in range(number_genomes):
                if j >= length:
                    env = gym.make(self.gym_game_name)
                    state = env.reset()
                    if type(state) == type(tuple()): 
                        state = state[0]
                    state = gym_http_server.flat_state(state)
                    reward = 0
                    done = False
                    list_envs.append(env)
                    list_states.append(state)
                    list_games.append(0)
                    list_steps.append(0)
                    list_keep_going.append(True)
                else:
                    env = list_envs[j]
                    state = env.reset()
                    if type(state) == type(tuple()): 
                        state = state[0]
                    state = gym_http_server.flat_state(state)
                    reward = 0
                    done = False
                    list_states[j] = state
                    list_games[j] = 0
                    list_steps[j] = 0
                    list_keep_going[j] = True
            
            stop_execution = False
            first_time = True
            while True in list_keep_going[:number_genomes]:
                for j in range(number_genomes):
                    list_steps[j]+=1
                start = time.time()
                out = self.neat.ff_all_genomes(list_states, self.threads_for_alone_training,number_genomes, list_genomes_to_ff = list_keep_going)
                end = time.time()
                '''
                if first_time:
                    first_time = False
                    if (end-start)/number_genomes >= 6 and number_genomes / :
                        stop_execution = True
                '''
                k = 0
                for j in range(number_genomes):
                    if list_keep_going[j]:
                        minimum = out[k][0]
                        action = 0
                        for z in range(len(out[k])):
                            if out[k][z] > minimum:
                                minimum = out[k][z]
                                action = z
                        k+=1
                        ret = list_envs[j].step(action)
                        state = ret[0]
                        reward = ret[1]
                        done = ret[2]
                        state = gym_http_server.flat_state(state)
                        list_states[j] = state
                        self.neat.increment_fitness_of_genome_ith(j,reward/self.max_number_of_games)
                        if done or list_steps[j] >= self.max_number_of_steps:
                            list_steps[j] = 0
                            list_games[j]+=1
                            state = list_envs[j].reset()
                            if type(state) == type(tuple()): 
                                state = state[0]
                            state = gym_http_server.flat_state(state)
                            list_states[j] = state
                        if list_games[j] >= self.max_number_of_games:
                            list_keep_going[j] = False
            self.neat.generation_run()
            self.neat.save_neat(self.neat_file)
            if stop_execution:
                self.alone_training_iterations = i+1
                break
        self.neat.reset_fitnesses()
    
    def distributed_training(self, remote_ip, remote_port,genomes_per_client,max_number_of_trainers, ip = '127.0.0.1', port=5000, timeout=3):
        self.neat.set_max_population(self.configuration_dict['max_population'])
        if genomes_per_client <= 0 or max_number_of_trainers <= 0:
            print("Error, genomes per client can't be <= 0, same for max number of trainers")
            exit(1)

        link = self.socket1_training_public_key+self.socket2_training_public_key+self.socket3_training_public_key

        
        # checking with http api if this training already exists , in that case wait 10 secs and re ask
        # if this training keeps existing for more than 5 minutes close the process
        
        # setting import parameters
        gym_http_server.glob_val.max_number_of_trainers = max_number_of_trainers
        gym_http_server.glob_val.max_number_genomes_per_client = genomes_per_client
        gym_http_server.glob_val.max_number_of_steps = self.max_number_of_steps
        gym_http_server.glob_val.max_number_of_games = self.max_number_of_games
        gym_http_server.glob_val.minimum_reward = self.configuration_dict['minimum_reward']
        gym_http_server.glob_val.sub_games = self.configuration_dict['sub_games']
        gym_http_server.glob_val.env_id = self.gym_game_name
        # starting the gym server on another thread
        gym_http_server.init_gym_server(self.training_private_key,ip,port)
        # starting the timeout for the environments
        gym_http_server.init_environments_timeout(self.training_private_key, timeout, self.t_val)
        
        # initializing the host client with ezspark proxy
        self.client = ezclient.Client(self.training_public_key,training_private_key = self.training_private_key,neat_class = self.neat, gym_game_name = self.gym_game_name, buffer_size = 100000, genome_input = self.input_size, genome_output = self.output_size, url_name = link)
        # connection for p2p through ezprotocol
        
        ret = {}
        
        while not 'connected' in ret or ret['connected']:
            ret = requests.get('https://alpha-p2p.ezspark.ai/rest/isTrainingConnected/'+self.training_private_key, verify=False)
            ret = json.loads(ret.content)
            if 'connected' in ret and not ret['connected']:
                break
            time.sleep(2)
        
        self.client.connect(remote_ip,remote_port, genomes_per_client = genomes_per_client)
        self.neat.set_generation_iter(self.alone_training_iterations)
        gym_http_server.glob_val.generation = self.alone_training_iterations
        generation = self.alone_training_iterations
        gym_http_server.glob_val.enter_critical_section()
        gym_http_server.glob_val.set_generation(generation)
        gym_http_server.glob_val.set_genomes(self.neat.get_number_of_genomes())
        gym_http_server.glob_val.exit_critical_section()
        
        socket1_thread = postTunnelRun('http://'+ip+':'+str(port),self.socket1_training_public_key, self.socket1_training_private_key,
                                        self.neat, self.gym_game_name, 100000, self.input_size, self.output_size, link, remote_ip, remote_port, genomes_per_client)
        socket2_thread = postTunnelRun('http://'+ip+':'+str(port),self.socket2_training_public_key, self.socket2_training_private_key,
                                        self.neat, self.gym_game_name, 100000, self.input_size, self.output_size, link, remote_ip, remote_port, genomes_per_client)
        socket3_thread = postTunnelRun('http://'+ip+':'+str(port),self.socket3_training_public_key, self.socket3_training_private_key,
                                        self.neat, self.gym_game_name, 100000, self.input_size, self.output_size, link, remote_ip, remote_port, genomes_per_client)
        
        socket1_thread.start()
        socket2_thread.start()
        socket3_thread.start()
        ss = 0
        while(True):
            # keep the communication active
            while(not self.client.is_disconnected()):
                self.client.host_direct_main_loop()
            # we have been disconnected by the server (bad requests or time out)
            if self.client.got_broken_pipe():
                ss+=1
                ret = {}
                # let's check if the problem is that someone else already started this host training
                self.client.connect(remote_ip, remote_port, genomes_per_client = genomes_per_client)
                continue
            ss = 0
            #blocking the timeouts that must talk with the p2p
            gym_http_server.glob_val.enter_critical_section()
            gym_http_server.glob_val.timeout_flag = True
            gym_http_server.glob_val.exit_critical_section()
            # we are here, a trainer is communicating with us
            is_ok = True
            current_id = None
            we_care = False
            flag2 = False
            # if we did already assigned an identifier to him:
            if self.client.trainer_has_identifier():
                # instantiate the identifier assigned
                current_id = self.client.get_identifier().decode('utf-8')
                current_id_index = self.client.get_index_value(current_id)
                gym_http_server.glob_val.enter_critical_section()
                environment_name = self.client.get_instance_name().decode('utf-8')
                
                # if environment is in reverse: we didn't finished the training for this environment name. or if the p2p id or the number of fitnesses do not match, it is malicious.
                # For the p2p id case it could more likely happen during http requests (for timeout), in those case the trainer will reset itself and restart from scratch without conencting with us
                # but if we reach this point either we were unlucky and we will close and restart the connection, or is really malicious
                if environment_name in gym_http_server.glob_val.reverse_shared_d or current_id not in gym_http_server.glob_val.id_p2p or self.client.get_number_of_fitnesses() != gym_http_server.glob_val.id_p2p[current_id]['n_genomes']:
                    is_ok = False
                    gym_http_server.glob_val.close_environments(environment_name)
                # this environment name could have been closed by timeout, ok do not disconnect lets just assign new genomes.
                # the other case is when it is malicious, well communicate genomes to him anyway
                elif environment_name not in gym_http_server.glob_val.checking_states:
                    flag2 = True
                else:#everything was fine
                    # ok assign the right fitnesses
                    d = gym_http_server.glob_val.checking_states[environment_name]['rewards']
                    l = list(d.keys())
                    l.sort()
                    l_rew = []
                    for i in l:
                        l_rew.append(d[i]/self.max_number_of_games)
                    self.client.set_fitnesses(l_rew)
                # if it is not malicious, or we could care about this
                if is_ok and not flag2:
                    #lets first surely check if we really care
                    if self.client.we_do_care_about_this_trainer():
                        we_care = True
                        try:
                            l = gym_http_server.glob_val.checking_states[environment_name]['list_to_check']
                        except:
                            l = []
                        if l == []:
                            is_ok = False
                        else:
                            gym_http_server.glob_val.exit_critical_section()# we exit here becuase it could take a little bit of time
                            is_ok = self.client.comparing_history_is_ok(l,0.00001)# comparing the history
                            gym_http_server.glob_val.enter_critical_section()
                    gym_http_server.glob_val.checking_states.pop(environment_name, None)
                gym_http_server.glob_val.exit_critical_section()
            if is_ok:# is ok as trainer, or is malicious but did not relly hurted us, or is ok but timeout timed out it
                if self.client.set_body(1, gym_http_server.glob_val.current_index, gym_http_server.glob_val.next_index, flag2):# if is true, a generation run has been completed
                    #setting some informations for the http retrivieng information host side
                    generation+=1
                    gym_http_server.glob_val.enter_critical_section()
                    gym_http_server.glob_val.set_generation(generation)
                    gym_http_server.glob_val.set_genomes(self.neat.get_number_of_genomes())
                    gym_http_server.glob_val.exit_critical_section()
                    self.neat.reset_fitnesses()
                    self.neat.save_neat(self.neat_file)
                    if current_id == None:
                        current_id = self.client.get_identifier().decode('utf-8')
                    gen_run = self.neat.get_generation_iter()
                    if gen_run >= self.configuration_dict['generations']:
                        print("end of the training")
                        exit(0)
                    gym_http_server.glob_val.enter_critical_section()
                    gym_http_server.glob_val.generation = generation
                    gym_http_server.glob_val.close_from_timeouts()
                    gym_http_server.glob_val.current_index = gym_http_server.glob_val.next_index
                    gym_http_server.glob_val.next_index = gym_http_server.glob_val.generate_indexing()
                    gym_http_server.glob_val.exit_critical_section()
                else:
                    if current_id == None:
                        current_id = self.client.get_identifier().decode('utf-8')
                    else:
                        if we_care:
                            gym_http_server.glob_val.enter_critical_section()
                            gym_http_server.glob_val.close_from_timeouts(id = current_id_index)
                            gym_http_server.glob_val.exit_critical_section()
                            
                if current_id == None:
                    current_id = self.client.get_identifier().decode('utf-8')
                current_id_index = self.client.get_index_value(current_id)
                gym_http_server.glob_val.enter_critical_section()
                gym_http_server.glob_val.id_p2p[current_id] = {'n_genomes':self.client.get_trainer_n_genomes(),'index':current_id_index}
                gym_http_server.glob_val.exit_critical_section()
            else:
                if current_id != None:
                    if current_id in gym_http_server.glob_val.id_p2p:
                        gym_http_server.glob_val.enter_critical_section()
                        gym_http_server.glob_val.id_p2p.pop(current_id, None)
                        gym_http_server.glob_val.exit_critical_section()
                self.client.set_body(0, gym_http_server.glob_val.current_index, gym_http_server.glob_val.next_index, False)# we set the body to tell the server to close this trainer
            # unlocking timeouts
            gym_http_server.glob_val.enter_critical_section()
            gym_http_server.glob_val.timeout_flag = False
            gym_http_server.glob_val.exit_critical_section()
