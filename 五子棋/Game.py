import tkinter as tk
import numpy as np
import Alg
import torch

class SnakeGame:
    def __init__(self):
        self.Unit_size = 20
        self.row = 10
        self.col = 10
        self.height = self.row * self.Unit_size
        self.width = self.col * self.Unit_size
        self.now_state = np.zeros((self.row, self.col))
        self.num_actions = self.row * self.col
        self.num_states = self.row * self.col
        self.win = tk.Tk()
        self.win.title("Chess Game")
        self.canvas = tk.Canvas(self.win, width = self.width, height = self.height + 2 * self.Unit_size)
        self.canvas.pack()

        for i in range(self.col) :
            for j in range(self.row) :
                x1 = i * self.Unit_size
                y1 = j * self.Unit_size
                x2 = (i + 1) * self.Unit_size
                y2 = (j + 1) * self.Unit_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")
        
        #self.win.mainloop()

    def reset(self):
        self.Unit_size = 20
        self.row = 10
        self.col = 10
        self.height = self.row * self.Unit_size
        self.width = self.col * self.Unit_size
        self.now_state = np.zeros((self.row, self.col))
        self.num_actions = self.row * self.col
        self.num_states = self.row * self.col

        for i in range(self.col) :
            for j in range(self.row) :
                x1 = i * self.Unit_size
                y1 = j * self.Unit_size
                x2 = (i + 1) * self.Unit_size
                y2 = (j + 1) * self.Unit_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")
        
        #self.win.mainloop()

    def check_win(self):
        role = self.now_state[0][0]
        len = 0
        for i in range(self.row):
            len = 0
            for j in range(self.col):
                if self.now_state[i][j] == 0:
                    len = 0
                else:
                    if self.now_state[i][j] == role:
                        len += 1
                        role = self.now_state[i][j]
                        if len == 5:
                            return role,i,j
                    else:
                        len = 0
                        role = self.now_state[i][j]
        role = self.now_state[4][0]
        for i in range(4, self.row):
            len = 0
            for j in range(i+1):
                if self.now_state[i-j][j] == 0:
                    len = 0
                else:
                    if self.now_state[i-j][j] == role:
                        len += 1
                        role = self.now_state[i-j][j]
                        if len == 5:
                            return role,i-j,j
                    else:
                        len = 0
                        role = self.now_state[i-j][j]
        role = self.now_state[1][9]
        len = 0
        for i in range(1, 6):
            len = 0
            for j in range(9, i, -1):
                if self.now_state[i+9-j][j] == 0:
                    len = 0
                else:
                    if self.now_state[i+9-j][j] == role:
                        len += 1
                        role = self.now_state[i+9-j][j]
                        if len == 5:
                            return role,i+9-j,j
                    else:
                        len = 0
                        role = self.now_state[i+9-j][j]
        role = self.now_state[0][0]
        len = 0
        for i in range(6):
            len = 0
            for j in range(9-i+1):
                if self.now_state[i+j][j] == 0:
                    len = 0
                else:
                    if self.now_state[i+j][j] == role:
                        len += 1
                        role = self.now_state[i+j][j]
                        if len == 5:
                            return role,i+j,j
                    else:
                        len = 0
                        role = self.now_state[i+j][j]
        role = self.now_state[0][1]
        len = 0
        for j in range(1, 6):
            len = 0
            for i in range(9-j):
                if self.now_state[i][j+i] == 0:
                    len = 0
                else:
                    if self.now_state[i][j+i] == role:
                        len += 1
                        role = self.now_state[i][j+i]
                        if len == 5:
                            return role,i,j+i
                    else:
                        len = 0
                        role = self.now_state[i][j+i]
        role = self.now_state[0][0]
        len = 0
        for j in range(10):
            len = 0
            for i in range(10):
                if self.now_state[i][j] == 0:
                    len = 0
                else:
                    if self.now_state[i][j] == role:
                        len += 1
                        role = self.now_state[i][j]
                        if len == 5:
                            return role,i,j
                    else:
                        len = 0
                        role = self.now_state[i][j]
        
        return 0,0,0

    def step(self, action, role):
        done = False
        x = action // 10
        y = action % 10
        s = self.now_state
        if role == 1:
            color = "white"
        else:
            color = "black"
        self.canvas.create_rectangle(x*self.Unit_size, y*self.Unit_size, (x+1)*self.Unit_size, (y+1)*self.Unit_size, fill = color, outline = "white")
        if self.now_state[x][y] != 0:
            r = -100
            done = True
            return r, self.now_state, done
            
        self.now_state[x][y] = role
        r = 0
        ans, x, y = self.check_win()
        if x!=0 and y!=0:
            print(x,y)
        if ans == 0:
            done = False
        if ans == role:
            r = 100
            done = True
        elif ans != role and ans != 0:
            r = -100
            done = True

        return r, self.now_state, done

    def train(self):
        N_ACTIONS = self.num_actions
        N_STATES = self.num_states
        BATCH_SIZE = 32
        LR = 0.01                   # learning rate
        EPSILON = 0.9               # greedy policy
        GAMMA = 0.9                 # reward discount
        TARGET_REPLACE_ITER = 100   # target update frequency
        MEMORY_CAPACITY = 200

        dqn = []
        dqn.append(Alg.make_DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR))
        dqn.append(Alg.make_DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR))
        dqn[0].eval_net.load_state_dict(torch.load("dqn1_eval.pk1"))
        dqn[1].eval_net.load_state_dict(torch.load("dqn2_eval.pk1"))
        dqn[0].target_net.load_state_dict(torch.load("dqn1_target.pk1"))
        dqn[1].target_net.load_state_dict(torch.load("dqn2_target.pk1"))
        for i in range(1000):
            s = np.array(self.now_state).reshape(100,)
            ep_r = 0
            if i%100 == 99:
                torch.save(dqn[0].eval_net.state_dict(), "dqn1_eval.pk1")
                torch.save(dqn[1].eval_net.state_dict(), "dqn2_eval.pk1")
                torch.save(dqn[0].target_net.state_dict(), "dqn1_target.pk1")
                torch.save(dqn[1].target_net.state_dict(), "dqn2_target.pk1")
            prob = np.random.randint(0, 1)
            while True:
                a = dqn[prob].choose_action(s, EPSILON, N_ACTIONS, self.now_state)

                r, s_, done = self.step(a ,prob+1)
                self.win.update()
                s_ = np.array(s_).reshape(100,)

                dqn[prob].store_transition(s, a, r, s_, MEMORY_CAPACITY)

                ep_r += r
                if dqn[prob].memory_counter > MEMORY_CAPACITY:
                    dqn[prob].learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
                    if done:
                        print('Ep: ', i,
                            '| Ep_r: ', round(ep_r, 2))
                
                if done:
                    self.reset()
                    self.win.update()
                    break
                s = s_
                prob = 1-prob
        

env = SnakeGame()
SnakeGame.train(env)
env.win.mainloop()