from flask import Flask, request, jsonify
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
logger = logging.getLogger('werkzeug')
logger.setLevel(logging.ERROR)


def flat_state(vector):
    if type(vector) == np.ndarray:
        return vector.flatten()
    elif type(vector) == list:
        v = vector
        if any(isinstance(i, list) for i in vector):
            v = list(chain(*vector))
        return v
    return None

#init_all = 0

########## Globals that must be modified and shared by threads ##########
class GlobalVal:
    def __init__(self, max_number_genomes_per_client, max_number_trainers, max_number_of_steps, max_number_of_games):
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
        self.id_p2p = {}
        self.shared_d = {}
        '''
        share_d struct:
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
        (key, value) = (environment_id[0] from the list of share_d, {'index':0, 'associated_id':id, 'current_index':self.current_index-1,'index_got':self.current_index, 'length': n, 'list_to_check':[]})
        '''
        self.current_trainers = 0
        self.current_index = self.generate_indexing()
        self.next_index = self.generate_indexing()
    
    def generate_indexing(self):
        return 1 + int(random.random()*self.max_number_genomes_per_client/2)
    
    def enter_critical_section(self):
        self.mutex.acquire()
    
    def exit_critical_section(self):
        self.mutex.release()
    
    def generate_id(self,length):
        characters = string.ascii_letters + string.digits + string.punctuation
        id = ''.join(random.choice(characters) for i in range(length))
        return id
    
    def create_environments_is_ok(self, n_instances):
        if n_instances < 1 or n_instances > self.max_number_genomes_per_client:
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
                    # if it enters in this if we are recording stuff
                    # remember we must record also the actions given back
                    # and this will always happen!
                    if self.checking_states[list_of_state_ids[i]]['index'] == self.checking_states[list_of_state_ids[i]]['current_index']:
                        #print('n: '+str(n))
                        #print('index: '+str(self.checking_states[list_of_state_ids[i]]['index']))
                        #print('index got: '+str(self.checking_states[list_of_state_ids[i]]['current_index']))
                        l = flat_state(states[j])
                        for kk in l:
                            self.checking_states[list_of_state_ids[i]]['list_to_check'].append(kk)
                        #print(st)
                        self.checking_states[list_of_state_ids[i]]['list_to_check'].append(reward[j])
                        self.checking_states[list_of_state_ids[i]]['current_index']+=self.checking_states[list_of_state_ids[i]]['index_got']
                        self.checking_states[list_of_state_ids[i]]['current_index'] = self.checking_states[list_of_state_ids[i]]['current_index']%self.checking_states[list_of_state_ids[i]]['length']
                        #print('length: '+str(len(self.checking_states[list_of_state_ids[i]]['list_to_check'])))
                        
                    self.checking_states[list_of_state_ids[i]]['index']+=1
                    self.checking_states[list_of_state_ids[i]]['index'] = self.checking_states[list_of_state_ids[i]]['index']%self.checking_states[list_of_state_ids[i]]['length']
                return True
        return False
    # the history take into account also the actions given back             
    def add_actions(self, list_of_environments_id, actions):
        #return
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
        
    def hash_enviroments(self, list_of_environments_id, states, rewards):
        '''
        it s asking us to hash these environments, this means that, first these environments
        have been just created, no other environments with same ids should exist, also,
        there should no be more or equals current trainers as max number of trainers. Also,
        the number of environments should not exceeds the number of maximum genomes per client
        '''
        if len(list_of_environments_id) > self.max_number_genomes_per_client:
            return False
        if self.current_trainers >= self.max_number_trainers:
            return False
        n = len(list_of_environments_id)
        for i in range(n):
            if list_of_environments_id[i] in self.reverse_shared_d:
                return False
        self.current_trainers+=1# adding a trainer 
        id = self.generate_id(32)
        self.shared_d[id] = {'ids':list(list_of_environments_id), 'interactions':1}#to an identifier is associated the list
        for i in range(n):
            self.reverse_shared_d[list_of_environments_id[i]] = [id,0,1,False,False]#the reverse, id, games, steps, done or not, last time we called this it was done or not
            self.checking_states[list_of_environments_id[0]] = {'index':0, 'associated_id':id, 'current_index':(self.current_index-1)%n,'index_got':self.current_index, 'length': n, 'list_to_check':[]}
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
        n2 = len(self.shared_d[id]['ids'])
        for i in range(n2):
            if self.shared_d[id]['ids'][i] not in list_of_environments_id:
                return False 
        return True
    
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
            if self.reverse_shared_d[list_of_environments_id[i]][1] >= self.max_number_of_games:
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
                self.reverse_shared_d.pop(list_of_environments_id[i], None)
        if len(self.shared_d[id]['ids']) == 0:
            self.shared_d.pop(id,None)
            self.current_trainers-=1
        return l
    
    def close_environments(self, environment):
        if environment not in self.reverse_shared_d:
            return
        id = self.reverse_shared_d[environment][0]
        for i in self.reverse_shared_d:
            if self.reverse_shared_d[i][0] == id:
                self.reverse_shared_d.pop(i,None)
        self.shared_d.pop(id,None)
        self.current_trainers-=1
   
    def set_globals(self,max_number_genomes_per_client, max_number_trainers, max_number_of_steps, max_number_of_games):
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
        self.id_len = 32 #increasing from 8 to 32 for security reasons

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

    def create(self, env_id, seed=None):
        try:
            env = gym.make(env_id)
            if seed:
                env.seed(seed)
        except gym.error.Error:
            raise InvalidUsage("Attempted to look up malformed environment ID '{}'".format(env_id))

        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def list_all(self):
        return dict([(instance_id, env.spec.id) for (instance_id, env) in self.envs.items()])

    def reset(self, instance_id):
        env = self._lookup_env(instance_id)
        obs = env.reset()
        return env.observation_space.to_jsonable(obs)

    def step(self, instance_id, action, render):
        env = self._lookup_env(instance_id)
        if isinstance( action, six.integer_types ):
            nice_action = action
        else:
            nice_action = np.array(action)
        if render:
            env.render()
        [observation, reward, done, info] = env.step(nice_action)
        obs_jsonable = env.observation_space.to_jsonable(observation)
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
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
envs = Envs()
glob_val = GlobalVal(10,10,5,1)

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

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


########## API route definitions ##########
@app.route('/v1/envs/', methods=['POST'])
def env_create():
    #print('creating environments')
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
        #print('we are full')
        glob_val.exit_critical_section()
        ret = {}
        ret['full'] = True
        return jsonify(ret)
    env_id = get_required_param(request.get_json(), 'env_id')
    n_instances = get_required_param(request.get_json(), 'n_instances')
    identifier = get_required_param(request.get_json(),'identifier')
    seed = None
    if type(n_instances) != int or type(env_id) != str or not glob_val.create_environments_is_ok(n_instances) or identifier not in glob_val.id_p2p or identifier in glob_val.id_p2p and glob_val.id_p2p[identifier]['n_genomes'] != n_instances:
        #print('not ok request on creating environment')
        glob_val.exit_critical_section()
        ret = {}
        ret['ok'] = False
        return jsonify(ret)
    ret = {}
    l1 = []
    l2 =  []
    l3 = []
    for i in range(n_instances):
        instance = envs.create(env_id, seed)
        l1.append(instance)
        ret[instance] = {'obs': envs.reset(l1[i]),'reward': 0}
        l2.append(ret[instance]['obs'])
        l3.append(0)
    l1, l2, l3 = zip(*sorted(zip(l1, l2, l3)))
    glob_val.hash_enviroments(l1,l2,l3)
    glob_val.shutdown_envs(l1)
    glob_val.exit_critical_section()
    
    return jsonify(ret)

@app.route('/v1/envs/step/', methods=['POST'])#multiple step request
def multi_step():
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
    json = request.get_json()
    response = {}
    keys = list(json.keys())
    keys.sort()
    glob_val.enter_critical_section()
    if not glob_val.steps_check(keys):
        glob_val.exit_critical_section()
        ret = {}
        ret['ok'] = False
        return jsonify(ret)
    glob_val.exit_critical_section()
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    for key in json:
        l5.append(key)
        l6.append(float(json[key]))
        if glob_val.reverse_shared_d[key][4]:
            obs_jsonable = envs.reset(key)
            reward = 0
            done = False
        else:
            obs_jsonable, reward, done, info = envs.step(key, json[key], False)
        response[key] = [obs_jsonable, reward, done]
        l1.append(key)
        l2.append(obs_jsonable)
        l3.append(reward)
        l4.append(done)
    l1, l2, l3, l4 = zip(*sorted(zip(l1, l2, l3, l4)))
    glob_val.enter_critical_section()
    glob_val.add_actions(l5,l6)
    glob_val.update_steps(l1,l4,l2,l3)
    glob_val.shutdown_envs(l1)
    glob_val.exit_critical_section()
    return jsonify(response)


class ServerRun(threading.Thread):
    def __init__(self, ip, port):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
 
        # helper function to execute the threads
    def run(self):
        print('starting the gym server on '+self.ip+':'+str(self.port))
        app.run(host = self.ip, port = self.port)

def init_gym_server(ip = '127.0.0.1', port = 5000):
    starting_thread = ServerRun(ip,port)
    starting_thread.start()
