from re import T
import tkinter as tk
import numpy as np
import sys
import Alg
import torch

class SnakeGame:
    def food_check(self, dir):
        x = self.snake[0][0]
        y = self.snake[0][1]
        while True:
            x += dir[0]
            y += dir[1]
            if x<0 or x>=self.row or y<0 or y>=self.row:
                break
            if self.now_state[x][y] == 2:
                return 1
        return 0
    
    def tail_check(self, dir):
        x = self.snake[0][0]
        y = self.snake[0][1]
        while True:
            x += dir[0]
            y += dir[1]
            if x<0 or x>=self.row or y<0 or y>=self.row:
                break
            if self.now_state[x][y] == 1:
                return 1
        return 0
    
    def wall_check(self, dir):
        x = self.snake[0][0]
        y = self.snake[0][1]
        d = 1
        while True:
            x += dir[0]
            y += dir[1]
            if x<0 or x>=self.row or y<0 or y>=self.row:
                break
            if self.now_state[x][y] == 3:
                return d
        return d
    
    def all_check(self):
        input_state = np.zeros((24, ))
        for i in range(len(self.dir)):
            input_state[i] = self.food_check(self.dir[i])
            input_state[i+len(self.dir)] = self.tail_check(self.dir[i])
            input_state[i+2*len(self.dir)] = self.wall_check(self.dir[i])
        return input_state

    def __init__(self):
        self.dir = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
        self.Unit_size = 20
        self.row = 8
        self.col = 8
        self.life = 300
        self.start = np.random.randint(1, self.row-1, size=2, dtype=int).tolist()
        while True:
            self.fruit = np.random.randint(0, self.row, size=2, dtype=int).tolist()
            if self.fruit != self.start and self.fruit[0] != 0 \
                and self.fruit[1] != 0 and self.fruit[0] != self.row-1 and self.fruit[1] != self.col-1:
                break
        self.height = self.row * self.Unit_size
        self.width = self.col * self.Unit_size
        self.now_action = 0
        self.now_state = np.zeros((self.row, self.col))
        self.now_state[self.start[0]][self.start[1]] = 1
        self.now_state[self.fruit[0]][self.fruit[1]] = 2
        for i in range(self.row):
            self.now_state[i][0] = 3
            self.now_state[i][self.col-1] = 3
        for i in range(self.col):
            self.now_state[0][i] = 3
            self.now_state[self.row-1][i] = 3
        self.actions = [[-1, 0], [1, 0], [0, 1], [1, 0]]
        self.snake = [self.start]
        self.num_actions = 4
        self.num_states = 24
        #self.win = tk.Tk()
        #self.win.title("Snake Game")
        #self.canvas = tk.Canvas(self.win, width = self.width, height = self.height + 2 * self.Unit_size)
        #self.canvas.pack()

        #for i in range(self.col) :
        #    for j in range(self.row) :
        #        x1 = i * self.Unit_size
        #        y1 = j * self.Unit_size
        #        x2 = (i + 1) * self.Unit_size
        #        y2 = (j + 1) * self.Unit_size
        #        if [i, j] == self.start:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "green", outline = "white")
        #        elif [i, j] == self.fruit:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "red", outline = "white")
        #        elif self.now_state[i][j] == 3:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "grey", outline = "white")
        #        else:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")


    def reset(self):
        self.row = 8
        self.col = 8
        self.life = 300
        self.start = np.random.randint(1, self.row-1, size=2, dtype=int).tolist()
        while True:
            self.fruit = np.random.randint(0, self.row, size=2, dtype=int).tolist()
            if self.fruit != self.start and self.fruit[0] != 0 \
                and self.fruit[1] != 0 and self.fruit[0] != self.row-1 and self.fruit[1] != self.col-1:
                break
        self.now_state = np.zeros((self.row, self.col))
        self.now_state[self.start[0]][self.start[1]] = 1
        self.now_state[self.fruit[0]][self.fruit[1]] = 2
        for i in range(self.row):
            self.now_state[i][0] = 3
            self.now_state[i][self.col-1] = 3
        for i in range(self.col):
            self.now_state[0][i] = 3
            self.now_state[self.row-1][i] = 3
        self.snake = [self.start]
        
        #for i in range(self.col) :
        #    for j in range(self.row) :
        #        x1 = i * self.Unit_size
        #        y1 = j * self.Unit_size
        #        x2 = (i + 1) * self.Unit_size
        #        y2 = (j + 1) * self.Unit_size
        #        if [i, j] == self.start:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "green", outline = "white")
        #        elif [i, j] == self.fruit:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "red", outline = "white")
        #        elif self.now_state[i][j] == 3:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "grey", outline = "white")
        #        else:
        #            self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")


    def step(self, action):
        tail = self.snake[-1]
        head = [self.snake[0][0] + action[0], self.snake[0][1] + action[1]]
        if head[0] >= self.row:
            head[0] = head[0] - self.row
        elif head[0] < 0:
            head[0] = self.row + head[0]
        if head[1] >= self.col:
            head[1] = head[1] - self.col
        elif head[1] < 0:
            head[1] = self.col + head[1]
        
        r = 0.1
        done = False

        s = self.now_state
        #if head[0] < 0 or head[0] >= self.row or head[1] < 0 or head[1] >= self.col:
        #    done = True
        #    r = -10

        if head == self.fruit:
            self.snake.insert(0, head)
            r = 1
            #self.canvas.create_rectangle(head[0]*self.Unit_size, head[1]*self.Unit_size, (head[0]+1)*self.Unit_size, (head[1]+1)*self.Unit_size, fill = "green", outline = "white")
            self.now_state[head[0]][head[1]] = 1
            while True:
                self.fruit = np.random.randint(0, self.row, size=2, dtype=int).tolist()
                if self.now_state[self.fruit[0]][self.fruit[1]] == 0:
                    break
            #self.canvas.create_rectangle(self.fruit[0]*self.Unit_size, self.fruit[1]*self.Unit_size, (self.fruit[0]+1)*self.Unit_size, (self.fruit[1]+1)*self.Unit_size, fill = "red", outline = "white")
            self.now_state[self.fruit[0]][self.fruit[1]] = 2
        
        elif self.now_state[head[0]][head[1]] == 1 or self.now_state[head[0]][head[1]] == 3:
            done = True
            r = -1
        
        else:
            #r = -(abs(head[0]-self.fruit[0])+abs(head[1]-self.fruit[1]))+(abs(self.snake[0][0]-self.fruit[0])+abs(self.snake[0][1]-self.fruit[1]))
            self.snake.insert(0, head)
            #self.canvas.create_rectangle(head[0]*self.Unit_size, head[1]*self.Unit_size, (head[0]+1)*self.Unit_size, (head[1]+1)*self.Unit_size, fill = "green", outline = "white")
            self.now_state[head[0]][head[1]] = 1
            self.snake.pop()
            #self.canvas.create_rectangle(tail[0]*self.Unit_size, tail[1]*self.Unit_size, (tail[0]+1)*self.Unit_size, (tail[1]+1)*self.Unit_size, fill = "silver", outline = "white")
            self.now_state[tail[0]][tail[1]] = 0
        
        return r, self.now_state, done

    def train(self):
        N_ACTIONS = self.num_actions
        N_STATES = self.num_states
        BATCH_SIZE = 32
        LR = 1e-4                   # learning rate
        EPSILON = 1              # greedy policy
        GAMMA = 0.9                 # reward discount
        TARGET_REPLACE_ITER = 100   # target update frequency
        MEMORY_CAPACITY = 2000

        dqn = Alg.make_DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)
        #dqn.eval_net.load_state_dict(torch.load("model.pk1"))
        #dqn.target_net.load_state_dict(torch.load("model.pk1"))

        for i in range(100000):
            s = self.all_check()
            ep_r = 0
            if i%100 == 0:
                torch.save(dqn.eval_net.state_dict(), "model.pk1")
            while True:
                a = dqn.choose_action(s, EPSILON, N_ACTIONS)

                r, s_, done = self.step(self.actions[a])
                s_ = self.all_check()
                #self.win.update()

                dqn.store_transition(s, a, r, s_, MEMORY_CAPACITY)

                ep_r += r
                if dqn.memory_counter > MEMORY_CAPACITY:
                    #dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
                    if done:
                        print('Ep: ', i,
                            '| Ep_r: ', round(ep_r, 2),
                            '| Len:', len(env.snake))
                
                if done:
                    self.reset()
                    #self.win.update()
                    break
                s = s_
        

env = SnakeGame()
SnakeGame.train(env)
env.win.mainloop()