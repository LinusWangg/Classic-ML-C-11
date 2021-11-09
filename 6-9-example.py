import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
import sys
from PyQt5.QtWidgets import *
import math
import time

class GirdWindow(QWidget):
    def __init__(self,parent=None,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)

        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.epoch = 2000
        
        font = QtGui.QFont()
        font.setFamily("Arial") #括号里可以设置成自己想要的其它字体
        font.setPointSize(1)   #括号里的数字可以设置成自己想要的字体大小

        self.action = [[0,1],[0,-1],[1,0],[-1,0]]
        self.actionnum = len(self.action)
        self.dim = [7,10]
        self.start = [3,0]
        self.end = [3,7]
        self.eps = 0.3
        self.alpha = 0.5
        self.gamma = 0.9
        self.buttons = {}

        self.windStrength = [0,0,0,-1,-1,-1,-2,-2,-1,0]

        self.Q = 0.1*np.random.randn(self.dim[0], self.dim[1], self.actionnum)
        self.Q[self.end[0]][self.end[1]] = 0

        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                self.buttons[(x,y)] = QPushButton()
                if [x,y]==self.start:
                    self.buttons[(x,y)].setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:green;")
                elif [x,y]==self.end:
                    self.buttons[(x,y)].setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:red;")
                else:
                    self.buttons[(x,y)].setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:white;")
                self.buttons[(x,y)].setFont(font)
                self.grid_layout.addWidget(self.buttons[(x,y)], x, y)
        
        train = QPushButton("train")
        train.clicked.connect(lambda:self.train())
        train.setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:white;")
        self.grid_layout.addWidget(train, 2, 12)

        generate = QPushButton("generate")
        generate.clicked.connect(lambda:self.generate())
        generate.setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:white;")
        self.grid_layout.addWidget(generate, 4, 12)

        self.grid_layout.setSpacing(0)
        self.grid_layout.setVerticalSpacing(0)
        self.grid_layout.setHorizontalSpacing(0)
        self.setWindowTitle('Basic Grid Layout')

    def eps_choose(self, dim, eps):
        prob = np.random.uniform(0.0, 1.0)
        choose_a = np.argmax(self.Q, axis=2)[dim[0]][dim[1]]
        action = [0,1,2,3]
        if prob > eps-eps/self.actionnum:
            return choose_a 
        else:
            action.remove(choose_a)
            return np.choose(np.random.randint(0,self.actionnum-2), action)
        
    def move(self, state, action):
        x = state[0]+action[0]+self.windStrength[state[1]]
        y = state[1]+action[1]
        if x<=0:
            x=0
        elif x>=self.dim[0]-1:
            x=self.dim[0]-1
        if y<=0:
            y=0
        elif y>=self.dim[1]-1:
            y=self.dim[1]-1
        return [x,y]

    def train(self):
        x_label = [i+1 for i in range(self.epoch)]
        y_label = []
        for i in range(self.epoch):
            step = 0
            if i%100 == 0:
                print(i)
            state = self.start
            action = self.eps_choose(state, self.eps)
            while True:
                if state == self.end:
                    break
                new_state = self.move(state, self.action[action])
                step+=1
                new_action = self.eps_choose(new_state, self.eps)
                self.Q[state[0]][state[1]][action]+=self.alpha*(-1+self.gamma*self.Q[new_state[0]][new_state[1]][new_action]-self.Q[state[0]][state[1]][action])
                state = new_state
                action = new_action
            y_label.append(step)
        plt.figure()
        plt.plot(x_label, y_label)
        plt.show()
    
    def generate(self):
        state_list = []
        action_list = []
        state = self.start
        while True:
            state_list.append(state)
            self.buttons[(state[0], state[1])].setStyleSheet("min-height:50px;min-width:70px; max-width:50px;max-height:70px;background-color:black;")
            if state == self.end:
                break
            action = np.argmax(self.Q, axis=2)[state[0]][state[1]]
            action_list.append(action)
            state = self.move(state, self.action[action])
        print(state_list, action_list)
        
            
            

app = QApplication(sys.argv)
windowExample = GirdWindow()
windowExample.show()
sys.exit(app.exec_())
        