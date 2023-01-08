import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import os
import random
from config import configParaser
from Algorithm.basic_module import BasicModule
from scipy import signal
from Tool import csvIO
from torch.distributions import Normal
import scipy
from mpi4py import MPI

#gpu or not
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#####################  hyper parameters  ####################

MAX_EPISODES = configParaser.MAX_EPISODES
MAX_EP_STEPS = configParaser.MAX_EP_STEPS
LR_A = configParaser.LR_A   # learning rate for actor
LR_C = configParaser.LR_C    # learning rate for critic
GAMMA = configParaser.GAMMA     # reward discount
TAU = configParaser.TAU      # soft replacement
MEMORY_CAPACITY = configParaser.MEMORY_CAPACITY
BATCH_SIZE = configParaser.BATCH_SIZE
RENDER = False
ENV_NAME = 'Pendulum-v0'

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################  DDPG  ####################################

class ANet(BasicModule):   # ae(s)=a
    def __init__(self,s_dim,a_dim,name):
        super(ANet,self).__init__()

        self.model_name = name

        log_std = -0.5 * np.ones(a_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.fc1 = nn.Linear(s_dim, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        self.out = nn.Linear(64, a_dim)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        # self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x,act = None):
        x = x.to(device)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.out(x)

        std = torch.exp(self.log_std)
        pi = Normal(x, std)
        logp_a = None
        if act is not None:
            act = act.to(device)
            logp_a = self._log_prob_from_distribution(pi, act).cpu()
        return pi, logp_a

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

class CNet(BasicModule):   # ae(s)=a
    def  __init__(self,s_dim,name):
        super(CNet,self).__init__()

        self.model_name = name

        self.fc1 = nn.Linear(s_dim,128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        self.out = nn.Linear(64, 1)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

        # self.fc1.weight.data.normal_(0,0.1)   # initialization
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        #self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s):
        x = s.to(device)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        actions_value = self.out(x)

        return actions_value.cpu()


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()

def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std

def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]

class VPGBuffer(object):
    def __init__(self, obs_dim, act_dim, size = configParaser.timeNum, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class VPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        self.mapWidth = configParaser.mapWidth
        self.mapHeight = configParaser.mapHeight
        self.mapSize = self.mapWidth*self.mapHeight

        self.buf = VPGBuffer(s_dim, a_dim, configParaser.timeNum-1, configParaser.GAMMA)

        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_target = ANet(s_dim,a_dim,'ac_t')
        self.Critic_target = CNet(s_dim,'c_t')
        self.Actor_target.to(device)
        self.Critic_target.to(device)

        self.ctrain = torch.optim.Adam(self.Critic_target.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_target.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        #self.csvTool = csvIO.CSVTool(configParaser.savePath_loss)

    def compute_loss_pi(self,data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.Actor_target(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(self,data):
        obs, ret = data['obs'], data['ret']
        return ((self.Critic_target(obs) - ret) ** 2).mean()

    def choose_action(self, s):
        with torch.no_grad():
            s = torch.unsqueeze(torch.FloatTensor(s), 0)
            s = s.to(device)
            pi = self.Actor_target(s)[0]
            a = pi.sample()
            a = torch.clamp(a, -self.a_bound, self.a_bound)
            v = self.Critic_target(s)[0]
            logp = self.Actor_target._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(),v.cpu().numpy(),logp.cpu().numpy()

    def learn(self):
        data = self.buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        self.atrain.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(self.Actor_target)  # average grads across MPI processes
        self.atrain.step()

        # Value function learning
        for i in range(80):
            self.ctrain.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.Critic_target)  # average grads across MPI processes
            self.ctrain.step()

        return loss_pi.cpu().data,loss_v.cpu().data

    def loadModel(self,modelPath):
        self.Actor_target.load(modelPath+'/ac_t')
        self.Critic_target.load(modelPath+'/c_t')

    def saveModel(self,modelPath,id):
        self.Actor_target.save(modelPath+'/ac_t'+str(id))
        self.Critic_target.save(modelPath+'/c_t'+str(id))



if __name__ == '__main__':                      #测试环境
    ###############################  training  ####################################
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    seed_torch(1)
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = VPG(a_dim, s_dim, a_bound[0])

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            #if RENDER:
            #    env.render()

            # Add exploration noise
            a, v, logp = ddpg.choose_action(s)

            s_, r, done, info = env.step(a)

            ddpg.buf.store(s, a, r, v, logp)

            s = np.reshape(s_,(3))
            ep_reward += r
            if j == MAX_EP_STEPS - 1 or done:
                ddpg.buf.finish_path(v)
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %d' % j, )
                if ep_reward > -300: RENDER = True
                break

        ddpg.learn()
    print('Running time: ', time.time() - t1)
