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
        self.state = np.zeros((36, 36), dtype=int)
        self.state[:] = 1
        self.buttons = {}
        self.S = []
        self.A = []
        self.R = []
        
        font = QtGui.QFont()
        font.setFamily("Arial") #括号里可以设置成自己想要的其它字体
        font.setPointSize(1)   #括号里的数字可以设置成自己想要的字体大小

        self.action = [[1,1],[0,1],[-1,1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]

        self.Q = np.zeros((36, 36, 9))
        self.C = np.zeros((36, 36, 9))
        self.pi = np.random.randint(0, 9, (36, 36), dtype=int)

        for x in range(36):
            for y in range(36):
                self.buttons[(x, y)] = QPushButton(str(x)+","+str(y))
                self.buttons[(x, y)].clicked.connect(lambda:self.changeState(self.sender().text()))
                if x==0 or x==35 or y==0 or y==35:
                    self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:grey;")
                    self.state[(x, y)] = 0
                else:
                    self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:white;")
                self.buttons[(x, y)].setFont(font)
                self.grid_layout.addWidget(self.buttons[(x, y)], x, y)
        
        getgreens = QPushButton("getgreens")
        getgreens.clicked.connect(lambda:self.getStart())
        getgreens.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(getgreens, 0, 50)

        generate = QPushButton("generate")
        generate.clicked.connect(lambda:self.generateActionList(self.action, self.pi, 1, 1, 0.1))
        generate.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(generate, 2, 50)

        showGenerate = QPushButton("showGenerate")
        showGenerate.clicked.connect(lambda:self.showGenerate())
        generate.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(showGenerate, 4, 50)
        
        self.grid_layout.setSpacing(0)
        self.grid_layout.setVerticalSpacing(0)
        self.grid_layout.setHorizontalSpacing(0)
        self.setWindowTitle('Basic Grid Layout')

    def changeState(self, index):
        index = index.split(",")
        x = int(index[0])
        y = int(index[1])
        self.state[(x, y)] = (self.state[(x, y)]+1)%4
        if self.state[(x, y)] == 1:
            self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:white;")
        elif self.state[(x, y)] == 0:
            self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:grey;")
        elif self.state[(x, y)] == 2:
            self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:green;")
        elif self.state[(x, y)] == 3:
            self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:red;")

    def getStart(self):
        greens = []
        for i in range(36):
            for j in range(36):
                if self.state[i][j] == 2:
                    greens.append([i, j])
        self.greens = greens
        print(self.greens)
        return
    
    def judge(self, loc, act):
        if act[0] == 0:
            for i in range(1, act[1]+1):
                if self.state[loc[0]][loc[1]+i] == 0:
                    return 1
                if self.state[loc[0]][loc[1]+i] == 3:
                    return 2
        elif act[1] == 0:
            for i in range(1, act[0]+1):
                if self.state[loc[0]+i][loc[1]] == 0:
                    return 1
                if self.state[loc[0]+i][loc[1]] == 3:
                    return 2
        else:
            ratio = act[1] / act[0]
            for i in np.arange(0, act[0], 0.1):
                x = math.floor(loc[0]+i)
                y = math.floor(loc[1]+i*ratio)
                if self.state[x][y] == 0:
                    return 1
                if self.state[x][y] == 3:
                    return 2
        return 0

    def draw(self, loc, act):
        if act[0] == 0:
            for i in range(1, act[1]+1):
                self.buttons[(loc[0], loc[1]+i)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")
        elif act[1] == 0:
            for i in range(1, act[0]+1):
                self.buttons[(loc[0]+i, loc[1])].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")
        else:
            ratio = act[1] / act[0]
            for i in np.arange(0, act[0], 0.1):
                x = math.floor(loc[0]+i)
                y = math.floor(loc[1]+i*ratio)
                self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")
        return
    
    def generateActionList(self, action, pi, x, y, eps):
        now_state = 2
        now_xmove = 0
        now_ymove = 0
        S = []
        A = []
        R = []
        now_x = x
        now_y = y
        S.append([x, y])
        while True:
            ASt = 9
            a_usable = []
            final_a = []
            for a in action:
                if abs(a[0]+now_xmove)>5 or abs(a[1]+now_ymove)>5 or (a[0]+now_xmove==0 and a[1]+now_ymove==0):
                    ASt -= 1
                else:
                    a_usable.append(a)
            prob = np.random.uniform(0.0, 1-eps+eps*2/ASt)
            if prob < eps/ASt:
                final_a = a_usable[np.random.randint(0, len(a_usable)-1)]
            else:
                final_a = action[pi[now_x, now_y]]

            now_xmove += final_a[0]
            now_ymove += final_a[1]
            A.append(final_a)

            if abs(now_xmove)>5:
                now_xmove = 5*abs(now_xmove)/now_xmove
            if abs(now_ymove)>5:
                now_ymove = 5*abs(now_ymove)/now_ymove

            cod = self.judge([now_x, now_y], [now_xmove, now_ymove])
            now_x += now_xmove
            now_y += now_ymove
            if cod == 1:
                R.append(-1)
                if len(self.greens) == 1:
                    green = self.greens[0]
                else:
                    green = self.greens[np.random.randint(0, len(self.greens)-1)]
                now_x = green[0]
                now_y = green[1]
                S.append([now_x, now_y])
                now_xmove = 0
                now_ymove = 0
            elif cod == 2:
                R.append(0)
                self.S = S
                self.A = A
                self.R = R
                break
            elif cod == 0:
                R.append(-1)
                S.append([now_x, now_y])
        print(S, A, R)
        return
    
    def showGenerate(self):
        for s in self.S:
            self.buttons[(s[0], s[1])].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:black;")
        now_xmove = 0
        now_ymove = 0
        for i in range(0, len(self.A)):
            now_xmove += self.A[i][0]
            now_ymove += self.A[i][1]
            self.draw(self.S[i], [now_xmove, now_ymove])
        return

app = QApplication(sys.argv)
windowExample = GirdWindow()
windowExample.show()
sys.exit(app.exec_())
