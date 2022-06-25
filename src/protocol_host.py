from . import gym_http_server
import ezclient

class Host:
    def __init__(self, gym_game_name, alone_training_iterations, configuration_dict,
                 max_number_of_games, max_number_of_steps, training_public_key = None, training_private_key = None):
        if max_number_of_steps <= 0 or max_number_of_games <= 0 or alone_training_iterations < 0:
            print("Error, something among max number of steps, max number of games or alone training iterations is <= 0")
            exit(1)
        ezclient.get_randomness()
        self.gym_game_name = gym_game_name
        self.input_size = configuration_dict['input_size']
        self.output_size = configuration_dict['output_size']
        self.neat = ezclient.Neat(self.input_size, self.output_size, initial_population = configuration_dict['initial_population'],
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
        self.client = None
    
    def alone_training(self):
        env = gym.make(self.gym_game_name)
        for i in range(self.alone_training_iterations):
            number_genomes = self.neat.get_number_of_genomes()
            neat.reset_fitnesses()
            for j in range(number_genomes):
                reward = 0
                done = False
                state = env.reset()
                state = env.observation_space.to_jsonable(state)
                n_games = 0
                for k in range(self.max_number_of_steps):
                    l = [state]
                    indices = [j]
                    out = self.neat.ff_ith_genomes(l,indices,1)
                    m = -1
                    ind = -1
                    for i in range(len(out)):
                        if out[i] > m:
                            m = out[i]
                            ind = i
                    [state, reward, done, info] = env.step(ind)
                    state = env.observation_space.to_jsonable(state)
                    self.neat.increment_fitness_of_genome_ith(j,reward)
                    if done:
                        n_games+=1
                        state = env.reset()
                        state = env.observation_space.to_jsonable(state)
                    if n_games >= self.max_number_of_games:
                        break
            self.neat.generation_run()
    
    def distributed_training(self, remote_ip, remote_port,genomes_per_client,max_number_of_trainers, ip = '127.0.0.1', port=5000):
        if genomes_per_client <= 0 or max_number_of_trainers <= 0:
            print("Error, genomes per client can't be <= 0, same for max number of trainers")
            exit(1)
        link = 'http://'+ip+':'+str(port) #it should be the link of localtunnel
        
        # setting import parameters
        gym_http_server.glob_val.max_number_of_trainers = max_number_of_trainers
        gym_http_server.glob_val.max_number_genomes_per_client = genomes_per_client
        gym_http_server.glob_val.max_number_of_steps = self.max_number_of_steps
        gym_http_server.glob_val.max_number_of_games = self.max_number_of_games
        # starting the gym server on another thread
        gym_http_server.init_gym_server(ip,port)
        
        # initializing the host client with ezspark proxy
        self.client = ezclient.Client(self.training_public_key,training_private_key = self.training_private_key,neat_class = self.neat, gym_game_name = self.gym_game_name, buffer_size = 30000, genome_input = self.input_size, genome_output = self.output_size, url_name = link)
        # connection for p2p through ezprotocol
        self.client.connect(remote_ip,remote_port, genomes_per_client = genomes_per_client)
        while(True):
            print('listening')
            # keep the communication active
            while(not self.client.is_disconnected()):
                self.client.host_direct_main_loop()
            # we have been disconnected by the server (bad requests or time out)
            if self.client.got_broken_pipe():
                self.client.connect(remote_ip, remote_port, genomes_per_client = genomes_per_client)
                continue
            # we are here, a trainer is communicating with us
            is_ok = True
            current_id = None
            # if we did already assigned an identifier to him:
            if self.client.trainer_has_identifier():
                print('trainer has identifier')
                current_id = self.client.get_identifier().decode('utf-8')
                # it s a trainer that has sent us stuff we are intereted in
                # lets check the history to understand if we can trust him
                environment_name = self.client.get_instance_name().decode('utf-8')
                if environment_name in gym_http_server.glob_val.reverse_shared_d or current_id not in gym_http_server.glob_val.id_p2p or self.client.get_number_of_fitnesses() != gym_http_server.glob_val.id_p2p[current_id]['n_genomes']:
                    print('something is wrong with the trainer')
                    is_ok = False
                    gym_http_server.glob_val.enter_critical_section()
                    gym_http_server.glob_val.close_environments(environment_name)
                    gym_http_server.glob_val.exit_critical_section()
                elif environment_name not in gym_http_server.glob_val.checking_states:
                    print('no such environment name')
                    is_ok = False# not such environment identifier
                else:
                    print('setting the fitnesses')
                    d = gym_http_server.glob_val.checking_states[environment_name]['rewards']
                    l = list(d.keys())
                    l.sort()
                    l_rew = []
                    for i in l:
                        l_rew.append(d[i])
                    self.client.set_fitnesses(l_rew)
                    
                if is_ok:
                    print('the trainer is ok')
                    if self.client.we_do_care_about_this_trainer():
                        print('we do care about this trainer, checking the history')
                        l = gym_http_server.glob_val.checking_states[environment_name]['list_to_check']
                        is_ok = self.client.comparing_history_is_ok(l,0.00001)# comparing the history
                    gym_http_server.glob_val.checking_states.pop(environment_name, None)
               
                    
            if is_ok:# is ok as trainer
                print('trainer is ok')
                if self.client.set_body(1, gym_http_server.glob_val.current_index, gym_http_server.glob_val.next_index):# if is true, a generation run has been completed
                    gym_http_server.glob_val.enter_critical_section()
                    gym_http_server.glob_val.current_index = gym_http_server.glob_val.next_index
                    gym_http_server.glob_val.next_index = gym_http_server.glob_val.generate_indexing()
                    gym_http_server.glob_val.exit_critical_section()
                    
                if current_id != None:
                    gym_http_server.glob_val.id_p2p[current_id] = {'n_genomes':self.client.get_trainer_n_genomes()}
                else:
                    current_id = self.client.get_identifier().decode('utf-8')
                    gym_http_server.glob_val.id_p2p[current_id] = {'n_genomes':self.client.get_trainer_n_genomes()}
            else:
                print('trainer is not ok')
                if current_id != None:
                    if current_id in gym_http_server.glob_val.id_p2p:
                        gym_http_server.glob_val.id_p2p.pop(current_id, None)
                self.client.set_body(0, gym_http_server.glob_val.current_index, gym_http_server.glob_val.next_index)# we set the body to tell the server to close this trainer
