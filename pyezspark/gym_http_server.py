import threading, time, random
import uuid
import gym
import numpy as np
import six
import argparse
import sys
import json
import string
import logging
from datetime import datetime, timedelta
import requests
from itertools import chain
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)
timeout_value = 5

def flat_state(vector):
    if type(vector) == np.ndarray:
        return vector.flatten().tolist()
    elif type(vector) == list:
        v = vector
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        return v
    return None

init_all = 0

########## Globals that must be modified and shared by threads ##########
class GlobalVal:
    def __init__(self, max_number_genomes_per_client, max_number_trainers, max_number_of_steps, max_number_of_games, t_val):
        if max_number_genomes_per_client <= 0:
            print("Error, max number of genomes can't be less than 1")
            exit(1)
        if max_number_trainers <= 0:
            print("Error, max number trainers can't be less than 1")
            exit(1)
        if max_number_of_steps <= 0 or max_number_of_games <= 0:
            print("Error, max number of steps and games can't be less than 1")
            exit(1)
        self.mutex = threading.Lock()
        self.max_number_genomes_per_client = max_number_genomes_per_client#max number of genomes per clients
        self.max_number_trainers = max_number_trainers#max number of trainers that we can allow togheter
        self.max_number_of_games = max_number_of_games
        self.max_number_of_steps = max_number_of_steps
        self.minimum_reward = 0
        self.sub_games = 1
        self.env_id = None
        self.generation = 1
        self.id_p2p = {}#each p2p id is associated to n_genomes
        self.shared_d = {}
        self.timeout_d = {}#handled by other thread
        self.timeout_flag = False;
        self.training_private_key = None
        self.generation = 0
        self.total_number_of_genomes = 0
        self.t_val = t_val
        self.stored_envs = []
        self.stored_envs_flag = False
        '''
        shared_d struct:
        (key, value) = (random id,{'ids':[list_of_environment_ids], 'interactions':integer})
        '''
        self.reverse_shared_d = {}
        '''
        reverse_shared_d struct:
        (key, value) = (environment_id,[shared_d_id,n_games,n_steps,stopped_training (True/false), last call was end of game in its instance (True/false)])
        '''
        self.checking_states = {}
        '''
        checking_states struct:
        (key, value) = (environment_id[0] from the list of shared_d, {'index':0, 'associated_id':id, 'current_index':self.current_index-1,'index_got':self.current_index, 'length': n, 'list_to_check':[]})
        '''
        self.current_trainers = 0
        self.current_index = self.generate_indexing()
        self.next_index = self.generate_indexing()
    
        
    
    def set_private_key(self,private_key):
        self.training_private_key = private_key
    
    def set_generation(self,generation):
        self.generation = generation
    
    def set_genomes(self,genomes):
        self.total_number_of_genomes = genomes
        
    def generate_indexing(self):
        return 1 + int(random.uniform(0.8,1)*self.max_number_genomes_per_client/2)
    
    def enter_critical_section(self):
        self.mutex.acquire()
    
    def exit_critical_section(self):
        self.mutex.release()
    
    def generate_id(self,length):
        characters = string.ascii_letters + string.digits + string.punctuation
        id = ''.join(random.choice(characters) for i in range(length))
        return id.replace("'",",").replace('"','!')
    
    def create_environments_is_ok(self, n_instances):
        if n_instances < 1 or n_instances > self.max_number_genomes_per_client*self.sub_games:
            return False
        if self.current_trainers >= self.max_number_trainers:
            return False
        return True
    
    def check_states(self,list_of_environments_id, states, reward):
        n = len(list_of_environments_id)
        id = ''
        for i in range(n):
            if list_of_environments_id[i] not in self.reverse_shared_d:
                return False
            else:
                id = self.reverse_shared_d[list_of_environments_id[i]][0]
                break
        list_of_state_ids = list(self.checking_states.keys())
        for i in range(len(list_of_state_ids)):
            if self.checking_states[list_of_state_ids[i]]['associated_id'] == id:
                for j in range(n):
                    self.checking_states[list_of_state_ids[i]]['rewards'][self.reverse_shared_d[list_of_environments_id[j]][5]] += reward[j]
                    # if it enters in this if we are recording stuff
                    # remember we must record also the actions given back
                    # and this will always happen!
                    if self.checking_states[list_of_state_ids[i]]['index'] == self.checking_states[list_of_state_ids[i]]['current_index']:
                        l = flat_state(states[j])
                        for kk in l:
                            self.checking_states[list_of_state_ids[i]]['list_to_check'].append(kk)
                        self.checking_states[list_of_state_ids[i]]['current_index']+=self.checking_states[list_of_state_ids[i]]['index_got']
                        self.checking_states[list_of_state_ids[i]]['current_index'] = self.checking_states[list_of_state_ids[i]]['current_index']%self.checking_states[list_of_state_ids[i]]['length']
                    self.checking_states[list_of_state_ids[i]]['index']+=1
                    self.checking_states[list_of_state_ids[i]]['index'] = self.checking_states[list_of_state_ids[i]]['index']%self.checking_states[list_of_state_ids[i]]['length']
                return True
        return False
    # the history take into account also the actions given back             
    def add_actions(self, list_of_environments_id, actions):
        list_of_environments_id, actions = zip(*sorted(zip(list_of_environments_id, actions)))
        n = len(list_of_environments_id)
        id = ''
        for i in range(n):
            if list_of_environments_id[i] not in self.reverse_shared_d:
                return False
            else:
                id = self.reverse_shared_d[list_of_environments_id[i]][0]
                break
        list_of_state_ids = list(self.checking_states.keys())
        for i in range(len(list_of_state_ids)):
            if self.checking_states[list_of_state_ids[i]]['associated_id'] == id:
                for j in actions:
                    if self.checking_states[list_of_state_ids[i]]['index'] == self.checking_states[list_of_state_ids[i]]['current_index']:
                        self.checking_states[list_of_state_ids[i]]['list_to_check'].append(float(j))
                        self.checking_states[list_of_state_ids[i]]['current_index']+=self.checking_states[list_of_state_ids[i]]['index_got']
                        self.checking_states[list_of_state_ids[i]]['current_index'] = self.checking_states[list_of_state_ids[i]]['current_index']%self.checking_states[list_of_state_ids[i]]['length']
                    self.checking_states[list_of_state_ids[i]]['index']+=1
                    self.checking_states[list_of_state_ids[i]]['index'] = self.checking_states[list_of_state_ids[i]]['index']%self.checking_states[list_of_state_ids[i]]['length']
        
    def hash_enviroments(self, list_of_environments_id, states, rewards, current_id):
        '''
        it s asking us to hash these environments, this means that, first these environments
        have been just created, no other environments with same ids should exist, also,
        there should no be more or equals current trainers as max number of trainers. Also,
        the number of environments should not exceeds the number of maximum genomes per client
        '''
        if len(list_of_environments_id) > self.max_number_genomes_per_client*self.sub_games:
            return False
        if self.current_trainers >= self.max_number_trainers:
            return False
        n = len(list_of_environments_id)
        for i in range(n):
            if list_of_environments_id[i] in self.reverse_shared_d:
                return False
        self.current_trainers+=1# adding a trainer 
        id = self.generate_id(32)
        list_prim_id = []
        
        # to hash the rewards
        for i in range(int(n/self.sub_games)):
            list_prim_id.append(self.generate_id(32))
        list_prim_id.sort()
        
        #number of games each copy of the genomes must play
        list_of_iterations = [0]*self.sub_games
        count = 0
        while count != self.max_number_of_games:
            for i in range(self.sub_games):
                list_of_iterations[i]+=1
                count+=1
                if count == self.max_number_of_games:
                    break
        
        self.shared_d[id] = {'ids':list(list_of_environments_id), 'interactions':1}#to an identifier is associated the list
        self.checking_states[list_of_environments_id[0]] = {'index':0, 'associated_id':id, 'current_index':(self.current_index-1)%n,'index_got':self.current_index, 'length': n, 'list_to_check':[], 'rewards':{}}
        for i in range(n):
            self.reverse_shared_d[list_of_environments_id[i]] = [id,0,1,False,False, list_prim_id[i%(int(n/self.sub_games))], list_of_iterations[int(i/int(n/self.sub_games))]]#the reverse, id, games, steps, done or not, last time we called this it was done or not, the id of the genome associated
            self.checking_states[list_of_environments_id[0]]['rewards'][list_prim_id[i%(int(n/self.sub_games))]] = 0
        self.timeout_d[id] = {'current_id_index': self.id_p2p[current_id]['index'],'current_id':current_id, 'shared_d':id, 'checking_states':list_of_environments_id[0], 'reverse_shared_d':list(list_of_environments_id), 'last_interaction':datetime.now(), 'seconds_timeout':0.6+n*(self.generation*0.001+0.2)}
        return self.check_states(list_of_environments_id,states,rewards)
        
    # someone is asking us to make a new step, lets check if its request is fair
    def steps_check(self, list_of_environments_id):
        n = len(list_of_environments_id)
        id = None
        for i in range(n):
            if list_of_environments_id[i] not in self.reverse_shared_d:
                return False
            if id == None:
                id = self.reverse_shared_d[list_of_environments_id[i]][0]
            else:
                if self.reverse_shared_d[list_of_environments_id[i]][0] != id:
                    return False
            if self.reverse_shared_d[list_of_environments_id[i]][3]:#it s done
                return False
        if id == None:
            return False
       
        try:
            n2 = len(self.shared_d[id]['ids'])
            for i in range(n2):
                if self.shared_d[id]['ids'][i] not in list_of_environments_id:
                    return False 
        except:
            return False
        return True
    
    def get_t_val(self):
        return self.t_val
        
    def set_t_val(self, value):
        self.t_val = value
        
    def update_steps(self, list_of_environments_id, done, states, rewards):
        '''
        we must first check that these environments exist
        and that are a sublist of the list of all the environments of the identifier associated to them.
        than we must check that is asking us the environments that are not done yet with the training
        '''
        
        n = len(list_of_environments_id)
        id = None
        for i in range(n):
            if list_of_environments_id[i] not in self.reverse_shared_d:
                return False
            if id == None:
                id = self.reverse_shared_d[list_of_environments_id[i]][0]
            else:
                if self.reverse_shared_d[list_of_environments_id[i]][0] != id:
                    return False
            if self.reverse_shared_d[list_of_environments_id[i]][3]:#it s done
                return False
        if id == None:
            return False
        
        # resetting the last interaction time
        flag = False
        for i in self.timeout_d:
            for j in self.timeout_d[i]['reverse_shared_d']:
                
                if j in list_of_environments_id:
                    self.timeout_d[i]['last_interaction'] = datetime.now()
                    flag = True
                    break
            if flag:
                break
        
        # saving the correct states
        list_of_environments_id, done, states, rewards = zip(*sorted(zip(list_of_environments_id, done, states, rewards)))
        glob_val.check_states(list_of_environments_id,states,rewards)
        for i in range(n):
            self.reverse_shared_d[list_of_environments_id[i]][2]+=1
            if done[i] or self.reverse_shared_d[list_of_environments_id[i]][2] >= self.max_number_of_steps:
                self.reverse_shared_d[list_of_environments_id[i]][2] = 0
                self.reverse_shared_d[list_of_environments_id[i]][1]+=1
                self.reverse_shared_d[list_of_environments_id[i]][4] = True
            else:
                self.reverse_shared_d[list_of_environments_id[i]][4] = False
            if self.reverse_shared_d[list_of_environments_id[i]][1] >= self.reverse_shared_d[list_of_environments_id[i]][6]:
                self.reverse_shared_d[list_of_environments_id[i]][3] = True
        return True
    
    def shutdown_envs(self, list_of_environments_id):
        n = len(list_of_environments_id)
        id = None
        for i in range(n):
            if list_of_environments_id[i] not in self.reverse_shared_d:
                return None
            if id == None:
                id = self.reverse_shared_d[list_of_environments_id[i]][0]
            else:
                if self.reverse_shared_d[list_of_environments_id[i]][0] != id:
                    return None
        if id == None:
            return None
        l = []
        for i in range(n):
            if self.reverse_shared_d[list_of_environments_id[i]][3]:
                l.append(list_of_environments_id[i])
                self.shared_d[id]['ids'].remove(list_of_environments_id[i])
                envs.env_close(list_of_environments_id[i])
                self.reverse_shared_d.pop(list_of_environments_id[i], None)
        if len(self.shared_d[id]['ids']) == 0:
            self.shared_d.pop(id,None)
            self.current_trainers-=1
        return l
    
    def close_environments(self, environment):
        id = self.checking_states[environment]['associated_id']
        self.checking_states.pop(environment,None)
        l = []
        for i in self.reverse_shared_d:
            if self.reverse_shared_d[i][0] == id:
                l.append(i)
        for i in l:
            envs.env_close(i)
            self.reverse_shared_d.pop(i,None)
        self.shared_d.pop(id,None)
        self.timeout_d.pop(id,None)
        self.current_trainers-=1
    
    def close_from_timeouts(self, id = None):
        if id == None:
            l = list(self.timeout_d.keys())
            for i in l:
                if self.timeout_d[i]['reverse_shared_d'][0] in self.checking_states and i not in self.shared_d:#has been already clos only the checking states are still active
                   self.checking_states.pop(self.timeout_d[i]['reverse_shared_d'][0],None)
                   self.timeout_d.pop(i,None)
                elif self.timeout_d[i]['reverse_shared_d'][0] not in self.checking_states and i not in self.shared_d:
                    self.timeout_d.pop(i,None)
                else:
                    self.close_environments(self.timeout_d[i]['reverse_shared_d'][0])
        else:
            l = list(self.timeout_d.keys())
            for i in l:
                if self.timeout_d[i]['current_id_index'] == id:
                    if self.timeout_d[i]['reverse_shared_d'][0] in self.checking_states and i not in self.shared_d:#has been already closed only the checking states are still active
                       self.checking_states.pop(self.timeout_d[i]['reverse_shared_d'][0],None)
                       self.timeout_d.pop(i,None)
                    elif self.timeout_d[i]['reverse_shared_d'][0] not in self.checking_states and i not in self.shared_d:
                        self.timeout_d.pop(i,None)
                    else:
                        self.close_environments(self.timeout_d[i]['reverse_shared_d'][0])
                        
    def set_globals(self,max_number_genomes_per_client, max_number_trainers, max_number_of_steps, max_number_of_games, t_val):
        if max_number_genomes_per_client <= 0:
            print("Error, max number of genomes can't be less than 1")
            exit(1)
        if max_number_trainers <= 0:
            print("Error, max number trainers can't be less than 1")
            exit(1)
        if max_number_of_steps <= 0 or max_number_of_games <= 0:
            print("Error, max number of steps and games can't be less than 1")
            exit(1)
        self.mutex = threading.Lock()
        self.max_number_genomes_per_client = max_number_genomes_per_client#max number of genomes per clients
        self.max_number_trainers = max_number_trainers#max number of trainers that we can allow togheter
        self.max_number_of_games = max_number_of_games
        self.max_number_of_steps = max_number_of_steps
        self.t_val = t_val
        
        


########## Container for environments ##########
class Envs(object):
    """
    Container and manager for the environments instantiated
    on this server.

    When a new environment is created, such as with
    envs.create('CartPole-v0'), it is stored under a short
    identifier (such as '3c657dbc'). Future API calls make
    use of this instance_id to identify which environment
    should be manipulated.
    """
    def __init__(self):
        self.envs = {}
        self.id_len = 16 #increasing from 8 to 16 for security reasons

    def _lookup_env(self, instance_id):
        try:
            return self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def _remove_env(self, instance_id):
        try:
            del self.envs[instance_id]
        except KeyError:
            raise InvalidUsage('Instance_id {} unknown'.format(instance_id))

    def create(self, env_id, pre_made_env = None, seed=None):
        if pre_made_env == None:
            try:
                env = gym.make(env_id)
                if seed:
                    env.seed(seed)
            except gym.error.Error:
                raise InvalidUsage("Attempted to look up malformed environment ID '{}'".format(env_id))
        else:
            env = pre_made_env
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id
        
    def create_rough(self, env_id, seed=None):
        try:
            env = gym.make(env_id)
            if seed:
                env.seed(seed)
            return env
        except gym.error.Error:
            raise InvalidUsage("Attempted to look up malformed environment ID '{}'".format(env_id))
            return None

    def list_all(self):
        return dict([(instance_id, env.spec.id) for (instance_id, env) in self.envs.items()])

    def reset(self, instance_id):
        environment = self._lookup_env(instance_id)
        obs = environment.reset()
        return flat_state(obs)

    def step(self, instance_id, action, render):
        env = self._lookup_env(instance_id)
        if isinstance( action, six.integer_types ):
            nice_action = action
        else:
            nice_action = np.array(action)
        if render:
            env.render()
        [observation, reward, done, info] = env.step(nice_action)
        obs_jsonable = flat_state(observation)
        return [obs_jsonable, reward, done, info]

    def get_action_space_contains(self, instance_id, x):
        env = self._lookup_env(instance_id)
        return env.action_space.contains(int(x))

    def get_action_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.action_space)

    def get_action_space_sample(self, instance_id):
        env = self._lookup_env(instance_id)
        action = env.action_space.sample()
        if isinstance(action, (list, tuple)) or ('numpy' in str(type(action))):
            try:
                action = action.tolist()
            except TypeError:
                print(type(action))
                print('TypeError')
        return action

    def get_observation_space_contains(self, instance_id, j):
        env = self._lookup_env(instance_id)
        info = self._get_space_properties(env.observation_space)
        for key, value in j.items():
            # Convert both values to json for comparibility
            if json.dumps(info[key]) != json.dumps(value):
                print('Values for "{}" do not match. Passed "{}", Observed "{}".'.format(key, value, info[key]))
                return False
        return True

    def get_observation_space_info(self, instance_id):
        env = self._lookup_env(instance_id)
        return self._get_space_properties(env.observation_space)

    def _get_space_properties(self, space):
        info = {}
        info['name'] = space.__class__.__name__
        if info['name'] == 'Discrete':
            info['n'] = space.n
        elif info['name'] == 'Box':
            info['shape'] = space.shape
            # It's not JSON compliant to have Infinity, -Infinity, NaN.
            # Many newer JSON parsers allow it, but many don't. Notably python json
            # module can read and write such floats. So we only here fix "export version",
            # also make it flat.
            info['low']  = [(x if x != -np.inf else -1e100) for x in np.array(space.low ).flatten()]
            info['high'] = [(x if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]
        elif info['name'] == 'HighLow':
            info['num_rows'] = space.num_rows
            info['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in np.array(space.matrix).flatten()]
        return info

    def monitor_start(self, instance_id, directory, force, resume, video_callable):
        env = self._lookup_env(instance_id)
        if video_callable == False:
            v_c = lambda count: False
        else:
            v_c = lambda count: count % video_callable == 0
        self.envs[instance_id] = gym.wrappers.Monitor(env, directory, force=force, resume=resume, video_callable=v_c) 

    def monitor_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.close()

    def env_close(self, instance_id):
        env = self._lookup_env(instance_id)
        env.close()
        self._remove_env(instance_id)
        


########## App setup ##########
envs = Envs()
glob_val = GlobalVal(10,10,5,1,5)

########## Error handling ##########
class InvalidUsage(Exception):
    status_code = 400
    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

def get_required_param(json, param):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value=='') or (value==[]):
        logger.info("A required request parameter '{}' had value {}".format(param, value))
        raise InvalidUsage("A required request parameter '{}' was not provided".format(param))
    return value

def get_optional_param(json, param, default):
    if json is None:
        logger.info("Request is not a valid json")
        raise InvalidUsage("Request is not a valid json")
    value = json.get(param, None)
    if (value is None) or (value=='') or (value==[]):
        logger.info("An optional request parameter '{}' had value {} and was replaced with default value {}".format(param, value, default))
        value = default
    return value


def generateTable():
    timeout = '1000s'
    date = datetime.now()
    html_tag = '<html>'
    head_tag = '<head>'
    css_link = '<link href="https://fonts.googleapis.com/css?family=Lato" rel="stylesheet"><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css">'
    style = '<style> body { font-family: "lato", sans-serif; } .container { max-width: 1000px; margin-left: auto; margin-right: auto; padding-left: 10px; padding-right: 10px; } h2 { font-size: 26px; margin: 20px 0; text-align: center; } h2 small { font-size: 0.5em; } .responsive-table li { border-radius: 3px; padding: 25px 30px; display: flex; justify-content: space-between; margin-bottom: 25px; } .responsive-table .table-header { background-color: #95A5A6; font-size: 14px; text-transform: uppercase; letter-spacing: 0.03em; } .responsive-table .table-row { background-color: #ffffff; box-shadow: 0px 0px 9px 0px rgba(0, 0, 0, 0.1); } .responsive-table .col-1 { flex-basis: 30%; } .responsive-table .col-2 { flex-basis: 30%; } .responsive-table .col-3 { flex-basis: 10%; } .responsive-table .col-4 { flex-basis: 10%; } .responsive-table .col-5 { flex-basis: 10%; } .responsive-table .col-6 { flex-basis: 10%; } @media all and (max-width: 767px) { .responsive-table .table-header { display: none; } .responsive-table li { display: block; } .responsive-table .col { flex-basis: 100%; } .responsive-table .col { display: flex; padding: 10px 0; } .responsive-table .col:before { color: #6C7A89; padding-right: 10px; content: attr(data-label); flex-basis: 50%; text-align: right; } } </style> <script> window.console = window.console || function(t) {}; </script> <script> if (document.location.search.match(/type=embed/gi)) { window.parent.postMessage("resize", "*"); } </script>'
    end_head_tag = '</head>'
    body = '<body translate="no">'
    title = '<div class="container"><h2>'+glob_val.training_private_key+' Training</h2>'
    number_of_trainers = '<div><b>Number of Current Trainers: </b>'+str(glob_val.current_trainers)+'</div>'
    generation = '<div><b>Generation: </b>'+str(glob_val.generation)+'</div>'
    total_genomes = '<div><b>Number of genomes for this generation: </b>'+str(glob_val.total_number_of_genomes)+'</div>'
    table_tag = '<ul class="responsive-table">'
    table_head = '<li class="table-header"><div class="col col-1">Peer Id</div><div class="col col-2">Timeout</div><div class="col col-3">N Genomes</div><div class="col col-4">Ended Games</div><div class="col col-5">Games Playing</div><div class="col col-6">Genome index</div></li>'
    table_body = ''
    l = list(glob_val.timeout_d.keys())
    for i in l:
        if glob_val.timeout_d[i]['checking_states'] in glob_val.checking_states: 
            gym_id = i
            p2p_id = glob_val.timeout_d[i]['current_id']
            n_genomes = glob_val.checking_states[glob_val.timeout_d[i]['checking_states']]['length']
            ended_games = n_genomes
            games_playing = 0
            gym_id = gym_id.replace('<','')
            gym_id = gym_id.replace('>','')
            p2p_id = gym_id.replace('>','')
            p2p_id = gym_id.replace('<','')
            
            if i in glob_val.shared_d:
                games_playing = len(glob_val.shared_d[i]['ids'])
                ended_games = n_genomes-games_playing
            genome_index = glob_val.timeout_d[i]['current_id_index']
            
            if glob_val.timeout_d[i]['checking_states'] in glob_val.checking_states and i not in glob_val.shared_d:#has been already closed only the checking states are still active
                if glob_val.timeout_flag:
                    timeout = 'is in pause'
                else:
                    if (date-glob_val.timeout_d[i]['last_interaction']).total_seconds() >= (glob_val.current_trainers+1)*15*glob_val.timeout_d[i]['seconds_timeout']:
                        timeout = 'is expiring'
                    else:
                        timeout = str((glob_val.current_trainers+1)*15*glob_val.timeout_d[i]['seconds_timeout'] + timeout_value - ((date-glob_val.timeout_d[i]['last_interaction']).total_seconds()))+'s'
                        
            elif glob_val.timeout_d[i]['checking_states'] not in glob_val.checking_states and i not in glob_val.shared_d:
                timeout = 'is expiring'
            else:
                if (date-glob_val.timeout_d[i]['last_interaction']).total_seconds() >= (glob_val.current_trainers+1)*3*glob_val.timeout_d[i]['seconds_timeout']:
                    timeout = 'is expiring'
                else:
                    timeout = str((glob_val.current_trainers+1)*3*glob_val.timeout_d[i]['seconds_timeout'] + timeout_value - ((date-glob_val.timeout_d[i]['last_interaction']).total_seconds()))+'s'
            
            
            table_body+='<li class="table-row"><div class="col col-1" data-label="P2P id">'+str(p2p_id)+'</div><div class="col col-2" data-label="Gym Id">'+str(timeout)+'</div><div class="col col-3" data-label="N Genomes">'+str(n_genomes)+'</div><div class="col col-4" data-label="Ended Games">'+str(ended_games)+'</div><div class="col col-5" data-label="Games Playing">'+str(games_playing)+'</div><div class="col col-6" data-label="Genome index">'+str(genome_index)+'</div></li>'
    end_table_tag = '</ul></div>'
    end_body_tag = '</body>'
    end_html_tag = '</html>'
    return html_tag+head_tag+css_link+style+end_head_tag+body+title+number_of_trainers+generation+total_genomes+table_tag+table_head+table_body+end_table_tag+end_body_tag+end_html_tag


def env_create(js):
    """
    Create an instance of the specified environment

    Parameters:
        - env_id: gym environment ID string, such as 'CartPole-v0'
        - n_instances: number of instances to instantiate
        - identifier
    Returns:
        - instance_id: {obs, reward}, ...
    """
    glob_val.enter_critical_section()
    if glob_val.current_trainers >= glob_val.max_number_trainers:
        glob_val.exit_critical_section()
        ret = {}
        ret['full'] = True
        return str(ret)
    env_id = get_required_param(js, 'env_id')
    n_instances = get_required_param(js, 'n_instances')
    identifier = get_required_param(js,'identifier')
    seed = None
    if type(n_instances) != int or type(env_id) != str or not glob_val.create_environments_is_ok(n_instances) or identifier not in glob_val.id_p2p or identifier in glob_val.id_p2p and glob_val.id_p2p[identifier]['n_genomes']*glob_val.sub_games != n_instances:
        glob_val.exit_critical_section()
        ret = {}
        ret['ok'] = False
        return str(ret)
    ret = {}
    l1 = []
    l2 =  []
    l3 = []
    pre_made_envs = None
    if glob_val.stored_envs_flag:
        pre_made_envs = glob_val.stored_envs
        glob_val.stored_envs = []
        glob_val.stored_envs_flag = False
    glob_val.exit_critical_section()
    for i in range(n_instances):
        pre_made = None
        if pre_made_envs != None:
            pre_made = pre_made_envs[i]
        instance = envs.create(env_id, pre_made, seed)
        l1.append(instance)
        e = envs.reset(l1[i])
        if type(e) == type(tuple()):
            e = e[0]
        e = flat_state(e)
        reward = 0
        ret[instance] = e
        l2.append(ret[instance])
        l3.append(reward)
    if pre_made_envs != None and n_instances < len(pre_made_envs):
        for i in range(n_instances,len(pre_made_envs)):
            pre_made_envs[i].close()
        
    glob_val.enter_critical_section()
    l1, l2, l3 = zip(*sorted(zip(l1, l2, l3)))
    glob_val.hash_enviroments(l1,l2,l3, identifier)
    glob_val.shutdown_envs(l1)
    glob_val.exit_critical_section()
    return str(ret)

def multi_step(js):
    """ 
    request a step
    returning for each instance id as key a observation, reward, done and reset observation in case
    - instance1:action
    - instance2:action
    - ...
    - instancen:action
    response:
    - instance1:[obs, reward, done, rest_obs]
    - ...
    """
    
    response = {}
    keys = list(js.keys())
    keys.sort()
    glob_val.enter_critical_section()
    if not glob_val.steps_check(keys):
        glob_val.exit_critical_section()
        ret = {}
        ret['ok'] = False
        return str(ret)
    glob_val.exit_critical_section()
    try:
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        l6 = []
        for key in js:
            l5.append(key)
            l6.append(float(js[key]))
            if glob_val.reverse_shared_d[key][4]:
                obs_jsonable = envs.reset(key)
                if type(obs_jsonable) == type(tuple()):
                    obs_jsonable = obs_jsonable[0]
                reward = 0
                done = False
                
            else:
                e = envs.step(key, js[key], False)
                obs_jsonable = e[0]
                reward = e[1]
                done = e[2]
            obs_jsonable = flat_state(obs_jsonable)
            response[key] = [obs_jsonable, done]
            l1.append(key)
            l2.append(obs_jsonable)
            l3.append(reward)
            l4.append(done)
        l1, l2, l3, l4 = zip(*sorted(zip(l1, l2, l3, l4)))
    
    except:
        ret = {}
        ret['ok'] = False
        return str(ret)
       
    glob_val.enter_critical_section()
    try:
        glob_val.add_actions(l5,l6)
        glob_val.update_steps(l1,l4,l2,l3)
        glob_val.shutdown_envs(l1)
    
    except:
        ret = {}
        ret['ok'] = False
        glob_val.exit_critical_section()
        return str(ret)

    glob_val.exit_critical_section()
    return str(response)

def get_status():
    glob_val.enter_critical_section()
    s = generateTable()
    glob_val.exit_critical_section()
    return s
    
    
    



class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers()
        self.wfile.write(bytes(get_status(), encoding = 'utf-8'))

    def do_HEAD(self):
        self._set_headers()


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

class ServerRun(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
 
        # helper function to execute the threads
    def run(self):
        print('starting the gym server on '+self.ip+':'+str(self.port))
        run(addr = self.ip, port = self.port)

class timeoutRun(threading.Thread):
    def __init__(self, timeout, training_private_key):
        threading.Thread.__init__(self)
        self.timeout = timeout
        self.timeout_dict = {}
        self.training_private_key = training_private_key
        self.polling_url = 'https://alpha-p2p.ezspark.ai/rest/hostPolling/'+self.training_private_key
    def run(self):
        
        while True:
            time.sleep(self.timeout)
            
            # polling
            try:
                ret = requests.get(self.polling_url, verify=False)
            except:
                continue
            
            #pre storing some instances
            date = datetime.now()
            glob_val.enter_critical_section()
            if not glob_val.stored_envs_flag:
                glob_val.exit_critical_section()
                for i in range(glob_val.max_number_genomes_per_client*glob_val.sub_games):
                    date_now = datetime.now()
                    if (date_now-date).total_seconds() > self.timeout:
                        try:
                            ret = requests.get(self.polling_url, verify=False)
                            date = date_now
                        except:
                            print("something went wrong polling the server")
                    instance = envs.create_rough(glob_val.env_id)
                    glob_val.stored_envs.append(instance)
                glob_val.enter_critical_section()
                glob_val.stored_envs_flag = True
            
            # checking the timeouts
            date = datetime.now()
            l = list(glob_val.timeout_d.keys())
            for i in l:
                if glob_val.timeout_d[i]['checking_states'][0] in glob_val.checking_states and i not in glob_val.shared_d:#has been already closed only the checking states are still active
                    if glob_val.timeout_flag and not i in self.timeout_dict:
                        self.timeout_dict[i] = (date-glob_val.timeout_d[i]['last_interaction']).total_seconds()
                    else:
                        if i in self.timeout_dict:
                            glob_val.timeout_d[i]['last_interaction'] = date - timedelta(seconds=self.timeout_dict[i])
                        if (date-glob_val.timeout_d[i]['last_interaction']).total_seconds() >= (glob_val.current_trainers+1)*15*glob_val.timeout_d[i]['seconds_timeout'] + timeout_value:
                            glob_val.checking_states.pop(glob_val.timeout_d[i]['checking_states'],None)
                            glob_val.timeout_d.pop(i,None)
                elif glob_val.timeout_d[i]['checking_states'] not in glob_val.checking_states and i not in glob_val.shared_d:
                    glob_val.timeout_d.pop(i,None)
                else:
                    if (date-glob_val.timeout_d[i]['last_interaction']).total_seconds() >= (glob_val.current_trainers+1)*3*glob_val.timeout_d[i]['seconds_timeout'] + timeout_value:
                        glob_val.close_environments(glob_val.timeout_d[i]['checking_states'])
            if not glob_val.timeout_flag:
                self.timeout_dict = {}
            glob_val.exit_critical_section()

def init_gym_server(private_key,ip = '127.0.0.1', port = 5000):
    glob_val.set_private_key(private_key)
    starting_thread = ServerRun(ip,port)
    starting_thread.start()

def init_environments_timeout(training_private_key, timeout = 3, t_val = 5):
    timeout_value = t_val
    starting_thread = timeoutRun(timeout, training_private_key)
    starting_thread.start()
