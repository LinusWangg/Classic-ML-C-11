import tkinter as tk
import numpy as np
import sys
sys.path.append(r"E:\\python_Code\\Classic-ML-C-11\\贪吃蛇")
import Alg
import torch

class SnakeGame:
    def __init__(self):
        self.start = [2, 2]
        while True:
            self.fruit = np.random.randint(0, 5, size=2, dtype=int).tolist()
            if self.fruit != self.start:
                break
        self.Unit_size = 20
        self.row = 5
        self.col = 5
        self.height = self.row * self.Unit_size
        self.width = self.col * self.Unit_size
        self.now_action = 0
        self.now_state = np.zeros((self.row, self.col))
        self.now_state[self.start[0]][self.start[1]] = 1
        self.now_state[self.fruit[0]][self.fruit[1]] = 2
        self.actions = [[-1, 0], [1, 0], [0, 1], [1, 0]]
        self.snake = [self.start]
        self.num_actions = 4
        self.num_states = self.row * self.col
        self.win = tk.Tk()
        self.win.title("Snake Game")
        self.canvas = tk.Canvas(self.win, width = self.width, height = self.height + 2 * self.Unit_size)
        self.canvas.pack()

        for i in range(self.col) :
            for j in range(self.row) :
                x1 = i * self.Unit_size
                y1 = j * self.Unit_size
                x2 = (i + 1) * self.Unit_size
                y2 = (j + 1) * self.Unit_size
                if [i, j] == self.start:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "green", outline = "white")
                elif [i, j] == self.fruit:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "red", outline = "white")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")
        
        #self.win.mainloop()

    def reset(self):
        self.start = [2, 2]
        while True:
            self.fruit = np.random.randint(0, 5, size=2, dtype=int).tolist()
            if self.fruit != self.start:
                break
        self.Unit_size = 20
        self.row = 5
        self.col = 5
        self.height = self.row * self.Unit_size
        self.width = self.col * self.Unit_size
        self.now_action = 0
        self.now_state = np.zeros((self.row, self.col))
        self.now_state[self.start[0]][self.start[1]] = 1
        self.now_state[self.fruit[0]][self.fruit[1]] = 2
        self.actions = [[-1, 0], [1, 0], [0, 1], [1, 0]]
        self.snake = [self.start]
        self.num_actions = 4
        self.num_states = self.row * self.col
        

        for i in range(self.col) :
            for j in range(self.row) :
                x1 = i * self.Unit_size
                y1 = j * self.Unit_size
                x2 = (i + 1) * self.Unit_size
                y2 = (j + 1) * self.Unit_size
                if [i, j] == self.start:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "green", outline = "white")
                elif [i, j] == self.fruit:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "red", outline = "white")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill = "silver", outline = "white")
        
        #self.win.mainloop()

    def step(self, action):
        tail = self.snake[-1]
        head = [self.snake[0][0] + action[0], self.snake[0][1] + action[1]]
        if head[0] >= 5:
            head[0] = head[0] - 5
        elif head[0] < 0:
            head[0] = 5 + head[0]
        if head[1] >= 5:
            head[1] = head[1] - 5
        elif head[1] < 0:
            head[1] = 5 + head[1]
        
        r = -1
        done = False
        s = self.now_state
        #if head[0] < 0 or head[0] >= self.row or head[1] < 0 or head[1] >= self.col:
        #    done = True
        #    r = -10

        if head == self.fruit:
            self.snake.insert(0, head)
            r = 10
            self.canvas.create_rectangle(head[0]*self.Unit_size, head[1]*self.Unit_size, (head[0]+1)*self.Unit_size, (head[1]+1)*self.Unit_size, fill = "green", outline = "white")
            self.now_state[head[0]][head[1]] = 1
            while True:
                self.fruit = np.random.randint(0, 5, size=2, dtype=int).tolist()
                if self.now_state[self.fruit[0]][self.fruit[1]] == 0:
                    break
            self.canvas.create_rectangle(self.fruit[0]*self.Unit_size, self.fruit[1]*self.Unit_size, (self.fruit[0]+1)*self.Unit_size, (self.fruit[1]+1)*self.Unit_size, fill = "red", outline = "white")
            self.now_state[self.fruit[0]][self.fruit[1]] = 2
        
        elif self.now_state[head[0]][head[1]] == 1:
            done = True
            r = -10
        
        else:
            #r = -(abs(head[0]-self.fruit[0])+abs(head[1]-self.fruit[1]))+(abs(self.snake[0][0]-self.fruit[0])+abs(self.snake[0][1]-self.fruit[1]))
            self.snake.insert(0, head)
            self.canvas.create_rectangle(head[0]*self.Unit_size, head[1]*self.Unit_size, (head[0]+1)*self.Unit_size, (head[1]+1)*self.Unit_size, fill = "green", outline = "white")
            self.now_state[head[0]][head[1]] = 1
            self.snake.pop()
            self.canvas.create_rectangle(tail[0]*self.Unit_size, tail[1]*self.Unit_size, (tail[0]+1)*self.Unit_size, (tail[1]+1)*self.Unit_size, fill = "silver", outline = "white")
            self.now_state[tail[0]][tail[1]] = 0

        return r, self.now_state, done

    def train(self):
        N_ACTIONS = self.num_actions
        N_STATES = self.num_states
        BATCH_SIZE = 32
        LR = 0.01                   # learning rate
        EPSILON = 0.9               # greedy policy
        GAMMA = 0.9                 # reward discount
        TARGET_REPLACE_ITER = 100   # target update frequency
        MEMORY_CAPACITY = 2000

        dqn = Alg.make_DQN(MEMORY_CAPACITY, N_STATES, N_ACTIONS, LR)

        for i in range(1000):
            s = np.array(self.now_state).reshape(25,)
            ep_r = 0
            if i%100 == 0:
                torch.save(dqn.eval_net.state_dict(), "model.pk1")
            while True:
                a = dqn.choose_action(s, EPSILON, N_ACTIONS)

                r, s_, done = self.step(self.actions[a])
                self.win.update()
                s_ = np.array(s_).reshape(25,)

                dqn.store_transition(s, a, r, s_, MEMORY_CAPACITY)

                ep_r += r
                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn(GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, N_STATES)
                    if done:
                        print('Ep: ', i,
                            '| Ep_r: ', round(ep_r, 2))
                
                if done:
                    self.reset()
                    self.win.update()
                    break
                s = s_
        

env = SnakeGame()
SnakeGame.train(env)
env.win.mainloop()