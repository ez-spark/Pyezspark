import src.protocol_host as host

gym_game_name = 'CartPole-v0'
alone_training_iterations = 100
configuration_dict = {}
configuration_dict['input_size'] = 4
configuration_dict['output_size'] = 2
configuration_dict['initial_population'] = 100
configuration_dict['species_threshold'] = 3
configuration_dict['generations'] = 600000
configuration_dict['percentage_survivors_per_specie'] = 0.3
configuration_dict['connection_mutation_rate'] = 0.8
configuration_dict['new_connection_assignment_rate'] = 0.1
configuration_dict['add_connection_big_specie_rate'] = 0.3
configuration_dict['add_connection_small_specie_rate'] = 0.03
configuration_dict['add_node_specie_rate'] = 0.05
configuration_dict['activate_connection_rate'] = 0.25
configuration_dict['remove_connection_rate'] = 0.01
configuration_dict['children'] = 1
configuration_dict['crossover_rate'] = 0.1
configuration_dict['saving'] = 1
configuration_dict['keep_parents'] = 1
configuration_dict['limiting_species'] = 15
configuration_dict['limiting_threshold'] = 5
configuration_dict['max_population'] = 4000
configuration_dict['same_fitness_limit'] = 10
configuration_dict['age_significance'] = 0.3
max_number_of_games = 4
max_number_of_steps = 120
training_public_key = 'training_public_key1'
training_private_key = 'training_private_key1'

h = host.Host(gym_game_name, alone_training_iterations,configuration_dict,max_number_of_games,max_number_of_steps,training_public_key,training_private_key)
h.alone_training()
h.distributed_training('0.0.0.0', 8080,100,10)
