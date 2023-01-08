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
GAMMA = configParaser.GAMMA     # reward discount
TAU = 1
ReplaceIter = 300
MEMORY_CAPACITY = configParaser.MEMORY_CAPACITY
BATCH_SIZE = configParaser.BATCH_SIZE
RENDER = False
ENV_NAME =  'CartPole-v0'

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.5
EPSILON_FINAL = 0.02
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

    def forward(self,x):
        x = x.to(device)

        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.out(x)
        return x

class DQN(object):
    def __init__(self, a_dim, s_dim, a_count):
        self.a_dim, self.s_dim, self.a_count = a_dim, s_dim, a_count
        self.mapWidth = configParaser.mapWidth
        self.mapHeight = configParaser.mapHeight
        self.mapSize = self.mapWidth*self.mapHeight
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_count,'dqn_e')
        self.Actor_target = ANet(s_dim,a_count,'dqn_t')
       # if torch.cuda.device_count() > 1:
       #     print("Let's use", torch.cuda.device_count(), "GPUs!")
       #     self.Actor_eval = nn.DataParallel(self.Actor_eval)
       #     self.Actor_target = nn.DataParallel(self.Actor_target)
       #     self.Critic_eval = nn.DataParallel(self.Critic_eval)
       #     self.Critic_target = nn.DataParallel(self.Critic_target)
        self.Actor_eval.to(device)
        self.Actor_target.to(device)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.epsilon = EPSILON_START
        self.learnCount = 0
        #self.csvTool = csvIO.CSVTool(configParaser.savePath_loss)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s = s.to(device)
        self.Actor_eval.eval()
        qvalue = self.Actor_eval(s)
        return int(torch.argmax(qvalue,1).item())

    def learn(self):
        self.learnCount += 1
        if self.learnCount>ReplaceIter:
            self.learnCount = 0
            for x in self.Actor_target.state_dict().keys():
                eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
                eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  #Update at, ct with ae, ce
        self.Actor_eval.train()
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.LongTensor(np.reshape(bt[:, self.s_dim: self.s_dim + self.a_dim],(-1,1))).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        qout = self.Actor_eval(bs)
        q_v = torch.gather(qout,1,ba)
        q_ = self.Actor_target(bs_).data.max(1)[0].unsqueeze(1)  #This network does not update the parameters in time, which is used to give the Gradient ascent strength when the Actor updates the parameters
        q_target = br + GAMMA * q_  # q_target = minus

        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        td_error.backward()
        self.atrain.step()

        return td_error.cpu().data

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def loadModel(self,modelPath):
        self.Actor_eval.load(modelPath+'/ac_e')
        self.Actor_target.load(modelPath+'/ac_t')

    def saveModel(self,modelPath,id):
        self.Actor_eval.save(modelPath+'/ac_e'+str(id))
        self.Actor_target.save(modelPath+'/ac_t'+str(id))




if __name__ == '__main__':                      #test environment
    ###############################  training  ####################################
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    seed_torch(1)
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = 1
    a_count = env.action_space.n
    ddpg = DQN(a_dim, s_dim,a_count)

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        loss = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            ddpg.epsilon = max(EPSILON_FINAL, EPSILON_START - i / EPSILON_DECAY_LAST_FRAME)
            # Add exploration noise
            if np.random.random() < ddpg.epsilon:
                a = env.action_space.sample()
            else:
                a = ddpg.choose_action(s)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                loss +=  ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % ddpg.epsilon, 'loss: %.4f'%(loss/(j+1)) )
                if ep_reward > 150: RENDER = True
                break
    print('Running time: ', time.time() - t1)
