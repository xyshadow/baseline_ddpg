#!/usr/bin/env python3
from collections import deque
import random
import numpy as np
import tensorflow as tf
import gym
import gym.spaces
from gym.wrappers import Monitor
import click


## partial update of target variables
def update_variables(from_vars,to_vars,tau):
    return_ops = []
    for from_var,to_var in zip(from_vars,to_vars):
        op = to_var.assign(tf.multiply(from_var,tau) + tf.multiply(to_var,1-tau))
        return_ops.append(op)
    return return_ops

## noise for exploration
class NormalNoise:
    def __init__(self, shape, scale=0.3):
        self.shape = shape
        self.scale = scale
    def __call__(self):
        return self.scale * np.random.normal(size=self.shape)        


## simple replay buffer
class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):        
        self.buffer.append((s, a, r, t, s2))

    @property
    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        ## obtain batch
        batch = random.sample(self.buffer, min(len(self.buffer),batch_size))
        ## return each elements in a buffer as seperate lists
        return np.asarray(batch).T.tolist()

    def clear(self):
        self.buffer.clear()

## a single model combine actor and critic, this is slightly difference from othe implementations
class Model(object):
    
    def __init__(self,ob_dim,ac_dim,tau,critic_lr,actor_lr,batch_size,gamma):
        
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        ## target network update rate
        self.tau = tau
        ## batch size
        self.batch_size = batch_size
        ## discount
        self.gamma=gamma
        
        ## define graph
        self._place_holder()
        ## current network
        with tf.variable_scope('current'):
            ## current actor
            self.cur_predict_ac = self.actor(self.obs_ph,self.ac_dim)
            ## current critic using actor output, used for backprop for actor gradients
            self.cur_predict_q_internal = self.critic(self.obs_ph,self.cur_predict_ac)
            ## current critic, using input action, used for estimate Q value
            self.cur_predict_q = self.critic(self.obs_ph,self.actions_ph,reuse=True)
        ## target network
        with tf.variable_scope('target'):
            ## target actor
            self.tar_predict_ac = self.actor(self.next_obs_ph,self.ac_dim)
            ## target critic, used to estimate state values of s'
            self.tar_predict_q = self.critic(self.next_obs_ph,self.tar_predict_ac)
        ## define update target vars
        self.init_target_vars = self._update_vars('init_target_vars',1)
        self.update_target_vars = self._update_vars('update_target_vars',self.tau)
        ## loss and ops
        self._critic_loss()
        self._apply_gradient()
                    
    def _place_holder(self):
        with tf.variable_scope('place_holder'):
            ## observation
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.ob_dim], name='obs')
            ## next observation
            self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.ob_dim], name='next_obs')
            ## reward
            self.rewards_ph = tf.placeholder(dtype=tf.float32,shape=[None],name='rewards')
            ## episode terminate signal
            self.dones_ph = tf.placeholder(dtype=tf.bool,shape=[None],name='dones')
            ## actions
            self.actions_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.ac_dim], name='actions')
                
    ## actor network               
    def actor(self,obs,ac_dim,reuse=False):
        ## obs: observation
        ## ac_dim: action dimension
        with tf.variable_scope('actor',reuse=reuse):
            layer = 0
            out = obs
            layer += 1
            for (size,fanin) in zip((400,300),(self.ob_dim,400)):
                
                out = tf.layers.dense(out, 
                                      size, 
                                      kernel_initializer=tf.random_uniform_initializer(minval=-1/np.sqrt(fanin), maxval=1/np.sqrt(fanin)),
                                      name='dense{}'.format(layer))
                ## use layer_norm instead of batch norm, batch norm perform poorly when used for critic network
                ## to unify the network, switch to layer_norm in actor as well, seen performance improvement over batch norm
                out = tf.contrib.layers.layer_norm(out, center=True, scale=True)
                out = tf.nn.relu(out)
                layer += 1
            return tf.layers.dense(out, 
                                   ac_dim, 
                                   kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                   activation=tf.nn.tanh,
                                   name='dense{}'.format(layer))

    ## critic network
    def critic(self,obs,actions,reuse=False):
        ## obs: observation
        ## actions: actions
        with tf.variable_scope('critic',reuse=reuse):
            layer = 0
            out = obs
            layer += 1
            out = tf.layers.dense(out, 
                                  400,
                                  kernel_initializer=tf.random_uniform_initializer(minval=-1/np.sqrt(self.ob_dim), maxval=1/np.sqrt(self.ob_dim)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name='dense{}'.format(layer))
            out = tf.contrib.layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
            layer += 1
            ## concat out and action
            out = tf.concat([out, actions], axis=-1)

            out = tf.layers.dense(out, 
                                  300, 
                                  kernel_initializer=tf.random_uniform_initializer(minval=-1/np.sqrt(400+self.ac_dim), maxval=1/np.sqrt(400+self.ac_dim)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name='dense{}'.format(layer))
            out = tf.contrib.layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
            
            ## return a single value
            layer += 1
            out = tf.layers.dense(out, 
                                  1, 
                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-4, maxval=3e-4),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                  name='dense{}'.format(layer))
            return tf.reshape(out,[-1])
    
    ## update variable ops
    def _update_vars(self,name,tau):
        with tf.variable_scope(name):
            return update_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'current'),
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target'),
                                    tau)
                    
    ## DDPG critic loss       
    def _critic_loss(self):
        with tf.variable_scope('critic_loss'):
            self.est_values = self.rewards_ph + self.gamma * (1 - tf.cast(self.dones_ph,tf.float32)) * self.tar_predict_q
            ## compare it with Q value from current network
            critic_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.est_values,self.cur_predict_q,scope='current/critic'))
            l2_loss = tf.reduce_mean(tf.losses.get_regularization_losses(scope='current/critic'))
            self.critic_loss = critic_loss + l2_loss
   
    ## Gradients update for both critic and actor
    def _apply_gradient(self):
        ## critic
        with tf.variable_scope('critic_opt'):
            ## gradience only apply to current critic
            critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'current/critic')
            critic_grads = tf.gradients(self.critic_loss,critic_vars)
            self.critic_opt = tf.train.AdamOptimizer(self.critic_lr).apply_gradients(zip(critic_grads,critic_vars))
        ## actor
        with tf.variable_scope('actor_opt'):
            ## gradient of -predicition w.r.t actions
            ## "-" is required as actor network performs gradient ascent, which equals to minimise -loss.
            ac_grads = tf.gradients(-self.cur_predict_q_internal, self.cur_predict_ac)
            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'current/actor')
            ## gradients of actor parameters w.r.t. actor output, multipled by critic network grads w.r.t action
            grads = tf.gradients(self.cur_predict_ac,
                                 actor_vars,
                                 grad_ys=ac_grads)
            grads = list(map(lambda x: tf.div(x, self.batch_size), grads))
            self.actor_opt = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(grads,actor_vars))
    
                
class DDPG_agent(object):
    
    def __init__(self,env,model,noise,replay_buf,gamma,log_dir,ac_bound):
        self.env = env
        self.model = model
        self.noise = noise
        self.replay_buf = replay_buf
        self.gamma = gamma
        self.log_dir = log_dir
        self.ac_bound = ac_bound
        
        ## internal variables
        ## terminate indicator
        self.done = 0
        
        ## summary
        self._summary()
        
    def _summary(self):
        ## tensor board
        with tf.variable_scope('summary'):
            self.summary_writer = tf.summary.FileWriter("{0}".format(self.log_dir),tf.get_default_graph())
            all_summaries = []
            all_summaries.append(tf.summary.scalar('value_loss',self.model.critic_loss))
            all_summaries.append(tf.summary.histogram('predict_q_value',self.model.cur_predict_q))
            all_summaries.append(tf.summary.histogram('ref_values',self.model.est_values))
            all_summaries.append(tf.summary.histogram('actions',self.model.actions_ph))
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'current'):
                all_summaries.append(tf.summary.histogram(var.name,var))
            self.merged_summary = tf.summary.merge(all_summaries)
    
    ## one step in env
    def _step(self,sess,ob,test):
        ## get action from model
        action = sess.run(self.model.cur_predict_ac,feed_dict={self.model.obs_ph:ob.reshape((1,-1))})
        ## add noise
        action = action + (1 - test) * self.noise()
        return (self.env.step(action[0]*self.ac_bound)),action
    
    def train(self,sess,max_ep,batch_size,ac_bound):
        
        sess.run(tf.global_variables_initializer())
        
        ## init target network
        sess.run(self.model.init_target_vars)
        ep_count = 0
        
        ## reward tracking
        past_rewards = deque(maxlen=10)
        past_rewards.append(0)
        cur_reward = 0
        ## test run flag
        test = 0
        while ep_count < max_ep:
            ep_count += 1
            if test:
                print('test run reward {}'.format(cur_reward))
            ## add cur reward to last rewards
            past_rewards.append(cur_reward)
            ##reset env to obtain initial observation
            ob = self.env.reset()
            cur_reward = 0
            self.done = 0
            ## test without noise once every 100 eps
            if ep_count % 100 == 0:    
                test = 1
            else:
                test = 0
            while not self.done:                
                ## take one step in env
                (next_ob, reward, self.done,_),action = self._step(sess,ob,test)
                ## add reward
                cur_reward += reward
                ## store to replay buffer
                self.replay_buf.add(ob,action[0],reward,self.done,next_ob)
                ob = next_ob
                
                ## start sample if replay buf is big enough
                if self.replay_buf.size > batch_size:
                    ## sample
                    obs,actions,rewards,dones,next_obs=self.replay_buf.sample_batch(batch_size)
                    feed_dict = {self.model.obs_ph:np.asarray(obs),
                                 self.model.next_obs_ph:np.asarray(next_obs),
                                 self.model.rewards_ph:np.asarray(rewards),
                                 self.model.actions_ph:np.asarray(actions),
                                 self.model.dones_ph:np.asarray(dones)}
                    
                    ## record on tensor board and std out every 10 eps
                    if self.done and ep_count % 10 ==0:
                        max_reward = max(past_rewards)
                        adv_reward = sum(past_rewards)/len(past_rewards)
                        print('ep: {}, max reward: {}, adv reward: {}'.format(ep_count,max_reward,adv_reward))
                        _,_,summary = sess.run((self.model.critic_opt,self.model.actor_opt,self.merged_summary),feed_dict=feed_dict)
                        self.summary_writer.add_summary(summary,ep_count)
                        ## add max reward and adv reward, add test score only when test time (every 100 eps)
                        for key,value in (('max_rewards',max_reward),('adv_rewards',adv_reward),('test_score',cur_reward)):
                            if test or ((not test) and key != 'test_score'):
                                summary = tf.Summary(value=[tf.Summary.Value(tag=key,simple_value=value)])
                                self.summary_writer.add_summary(summary, ep_count)
                        ## write events to disk
                        self.summary_writer.flush()
                        
                    else:
                        sess.run((self.model.critic_opt,self.model.actor_opt),feed_dict=feed_dict)
                
                    ## update target network
                    sess.run(self.model.update_target_vars)

## main def
@click.command(help='basic DDPG agent')
@click.option('--actor_lr', type=float, help='actor network learning rate', default=0.0001)
@click.option('--critic_lr', type=float, help='critic network learning rate', default=0.001)
@click.option('--gamma', type=float, help='reward discount', default=0.99)
@click.option('--tau', type=float, help='soft target update parameter', default=0.001)
@click.option('--buffer_size', type=int, help='max size of the replay buffer', default=1000000)
@click.option('--batch_size', type=int, help='size of minibatch', default=64)
# run parameters
@click.option('--env', type=str, help='gym env', default='InvertedPendulum-v2')
@click.option('--random_seed', type=int, help='random seed', default=1234)
@click.option('--max_ep', type=int, help='max num of episodes', default=50000)
@click.option('--summary_dir', type=str, help='summary directory', default='./results/tf_ddpg')
def main(actor_lr,critic_lr,gamma,tau,buffer_size,batch_size,env,random_seed,max_ep,summary_dir):
    ## args come from argparse
    ## create single env    
    env = gym.make(env)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    env.seed(random_seed)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    ac_bound = env.action_space.high
    
    ## define a model
    model = Model(ob_dim=ob_dim,
                  ac_dim=ac_dim,
                  tau=tau,
                  critic_lr=critic_lr,
                  actor_lr=actor_lr,
                  batch_size=batch_size,
                  gamma=gamma)
    

    ## ddpg agent
    agent = DDPG_agent(env=env,
                       model=model,
                       noise=NormalNoise(ac_dim),
                       replay_buf=ReplayBuffer(buffer_size, random_seed),
                       gamma=gamma,
                       log_dir=summary_dir,
                       ac_bound=ac_bound)
    
    ## initilise
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True        
    with tf.Session(config=config) as sess:
        ## train agent
        agent.train(sess=sess,max_ep=max_ep,batch_size=batch_size,ac_bound=ac_bound)

                    
if __name__ == "__main__":
    
    main()
                       
                
                    
        
        
