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
from Tool import csvIO

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
    def __init__(self,s_dim,a_dim,a_bound,name):
        super(ANet,self).__init__()

        self.model_name = name

        self.fc1 = nn.Linear(s_dim, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        self.out = nn.Linear(64, a_dim)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

        self.a_bound = a_bound
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        # self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = x.to(device)

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x * self.a_bound
        return actions_value

class CNet(BasicModule):   # ae(s)=a
    def  __init__(self,s_dim,a_dim,name):
        super(CNet,self).__init__()

        self.model_name = name

        self.fc1 = nn.Linear(s_dim+a_dim,128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        self.out = nn.Linear(64, 1)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self,s,a):
        s,a = s.to(device),a.to(device)

        x = torch.cat((s,a),1)

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        actions_value = self.out(x)

        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.mapWidth = configParaser.mapWidth
        self.mapHeight = configParaser.mapHeight
        self.mapSize = self.mapWidth*self.mapHeight
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim,a_bound,'ac_e')
        self.Actor_target = ANet(s_dim,a_dim,a_bound,'ac_t')
        self.Critic_eval = CNet(s_dim,a_dim,'c_e')
        self.Critic_target = CNet(s_dim,a_dim,'c_t')
       # if torch.cuda.device_count() > 1:
       #     print("Let's use", torch.cuda.device_count(), "GPUs!")
       #     self.Actor_eval = nn.DataParallel(self.Actor_eval)
       #     self.Actor_target = nn.DataParallel(self.Actor_target)
       #     self.Critic_eval = nn.DataParallel(self.Critic_eval)
       #     self.Critic_target = nn.DataParallel(self.Critic_target)
        self.Actor_eval.to(device)
        self.Actor_target.to(device)
        self.Critic_eval.to(device)
        self.Critic_target.to(device)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        #self.csvTool = csvIO.CSVTool(configParaser.savePath_loss)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s = s.to(device)
        self.Actor_eval.eval()
        return self.Actor_eval(s)[0].detach() # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # update at, ct with ae, ce
        self.Actor_eval.train()
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））update ae   ae（s）=a   ae（s_）=a_
        # If a is a correct behavior, then its Q should be closer to 0
        loss_a = -torch.mean(q)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # This network does not update parameters in time to predict the action in Critic's Q_target
        q_ = self.Critic_target(bs_,a_)  # This network does not update the parameters in time, which is used to give the Gradient ascent strength when the Actor updates the parameters
        q_target = br+GAMMA*q_  # q_target = minus
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) Update ce, but this ae(s) is the ba in memory, so that the Q obtained by ce is close to Q_target, making the evaluation more accurate
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        return loss_a.cpu().data,td_error.cpu().data

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def loadModel(self,modelPath):
        self.Actor_eval.load(modelPath+'/ac_e')
        self.Actor_target.load(modelPath+'/ac_t')
        self.Critic_eval.load(modelPath+'/c_e')
        self.Critic_target.load(modelPath+'/c_t')

    def saveModel(self,modelPath,id):
        self.Actor_eval.save(modelPath+'/ac_e'+str(id))
        self.Actor_target.save(modelPath+'/ac_t'+str(id))
        self.Critic_eval.save(modelPath+'/c_e'+str(id))
        self.Critic_target.save(modelPath+'/c_t'+str(id))



if __name__ == '__main__':                      #test environment
    ###############################  training  ####################################
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    seed_torch(1)
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound[0])

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            # if RENDER:
            #   env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a.cpu(), var), -a_bound, a_bound)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300: RENDER = True
                break
    print('Running time: ', time.time() - t1)
