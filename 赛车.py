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
        self.A_index = []
        self.A_prob = []
        self.R = []
        self.step = 0
        
        font = QtGui.QFont()
        font.setFamily("Arial") #括号里可以设置成自己想要的其它字体
        font.setPointSize(1)   #括号里的数字可以设置成自己想要的字体大小

        self.action = [[1,1],[0,1],[-1,1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]

        self.Q = np.zeros((36, 36, 9))
        self.C = np.zeros((36, 36, 9))
        self.value = np.zeros((36, 36))
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
        generate.clicked.connect(lambda:self.generateActionList(self.action, self.pi, self.greens[0][0], self.greens[0][1], 0.1))
        generate.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(generate, 2, 50)

        showGenerate = QPushButton("showGenerate")
        showGenerate.clicked.connect(lambda:self.showGenerate())
        generate.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(showGenerate, 4, 50)

        train = QPushButton("train")
        train.clicked.connect(lambda:self.train(0.9, 10000))
        train.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(train, 6, 50)

        remake = QPushButton("remake")
        remake.clicked.connect(lambda:self.remake())
        remake.setStyleSheet("min-height:30px;min-width:100px; max-width:100px;max-height:30px;background-color:white;")
        self.grid_layout.addWidget(remake, 8, 50)
        
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
        if act[0] == 0 and act[1] == 0:
            return 0
        
        if act[0] == 0:
            t1 = act[1] // abs(act[1])
            for i in range(1, abs(act[1])+1):
                if self.state[loc[0]][loc[1]+i*t1] == 0:
                    return 1
                if self.state[loc[0]][loc[1]+i*t1] == 3:
                    return 2
        
        if act[1] == 0:
            t0 = act[0] // abs(act[0])
            for i in range(1, abs(act[0])+1):
                if self.state[loc[0]+i*t0][loc[1]] == 0:
                    return 1
                if self.state[loc[0]+i*t0][loc[1]] == 3:
                    return 2
        
        if act[0] != 0 and act[1] != 0:
            ratio = abs(act[1] / act[0])
            t0 = act[0] / abs(act[0])
            t1 = act[1] / abs(act[1])
            for i in np.arange(0, abs(act[0])+0.1, 0.1):
                x = math.floor(loc[0]+i*t0)
                y = math.floor(loc[1]+i*ratio*t1)
                if self.state[x][y] == 0:
                    return 1
                if self.state[x][y] == 3:
                    return 2
        return 0

    def boundray(self, x, y):
        if x<0 or x>35 or y<0 or y>35:
            return True
        return False

    def draw(self, loc, act):
        if act[0] == 0 and act[1] == 0:
            return
        
        if act[0] == 0:
            t1 = act[1] // abs(act[1])
            for i in range(1, abs(act[1])+1):
                if self.boundray(loc[0], loc[1]+i*t1):
                    return
                self.buttons[(loc[0], loc[1]+i*t1)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")

        if act[1] == 0:
            t0 = act[0] // abs(act[0])
            for i in range(1, act[0]+1):
                if self.boundray(loc[0]+i*t0, loc[1]):
                    return
                self.buttons[(loc[0]+i*t0, loc[1])].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")
        
        if act[0] != 0 and act[1] != 0:
            ratio = abs(act[1] / act[0])
            t0 = act[0] / abs(act[0])
            t1 = act[1] / abs(act[1])
            for i in np.arange(0, abs(act[0])+0.1, 0.1):
                x = math.floor(loc[0]+i*t0)
                y = math.floor(loc[1]+i*ratio*t1)
                if self.boundray(x, y):
                    return
                self.buttons[(x, y)].setStyleSheet("min-height:30px;min-width:30px; max-width:30px;max-height:30px;background-color:purple;")

        return
    
    def remake(self):
        font = QtGui.QFont()
        font.setFamily("Arial") #括号里可以设置成自己想要的其它字体
        font.setPointSize(1)   #括号里的数字可以设置成自己想要的字体大小
        for x in range(36):
            for y in range(36):
                self.buttons[(x, y)].setText(str(x)+","+str(y))
                self.buttons[(x, y)].setFont(font)

    def showvalue(self, eps):
        font = QtGui.QFont()
        font.setFamily("Arial") #括号里可以设置成自己想要的其它字体
        font.setPointSize(18)   #括号里的数字可以设置成自己想要的字体大小
        Qmax = np.max(self.Q, axis=2)
        Qtotal = np.average(self.Q, axis=2)*9
        for x in range(36):
            for y in range(36):
                self.value[x][y] = (1-eps)*Qmax[x][y] + eps/9*(Qtotal[x][y]-Qmax[x][y])


    def generateActionList(self, action, pi, x, y, eps):
        now_state = 2
        now_xmove = 0
        now_ymove = 0
        S = []
        A = []
        R = []
        A_index = []
        A_prob = []
        now_x = x
        now_y = y
        S.append([x, y])
        while True:
            ASt = 9
            a_usable = []
            a_usable_index = []
            final_a = []
            final_a_index = 0
            for i in range(len(action)):
                if abs(action[i][0]+now_xmove)>4 or abs(action[i][1]+now_ymove)>4 or (action[i][0]+now_xmove==0 and action[i][1]+now_ymove==0):
                    ASt -= 1
                else:
                    a_usable.append(action[i])
                    a_usable_index.append(i)
            prob = np.random.uniform(0.0, 1-eps+eps*2/ASt)
            if prob < eps/ASt:
                rand_a = np.random.randint(0, len(a_usable)-1)
                final_a_index = a_usable_index[rand_a]
                final_a = a_usable[rand_a]
                A_prob.append(eps/ASt)
            else:
                final_a_index = pi[now_x, now_y]
                final_a = action[pi[now_x, now_y]]
                A_prob.append(1-eps+eps/ASt)

            now_xmove += final_a[0]
            now_ymove += final_a[1]
            A.append(final_a)
            A_index.append(final_a_index)

            if abs(now_xmove)>4:
                now_xmove = 4*abs(now_xmove)//now_xmove
            if abs(now_ymove)>4:
                now_ymove = 4*abs(now_ymove)//now_ymove

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
                self.A_prob = A_prob
                self.A_index = A_index
                break
            elif cod == 0:
                R.append(-1)
                S.append([now_x, now_y])
        #print(S, A, R)
        return

    def MC_control(self, gamma):
        G = 0
        W = 1
        T = len(self.S)-1
        for i in range(T+1):
            index = T-i
            G = gamma*G+self.R[index]
            self.C[self.S[index][0]][self.S[index][1]][self.A_index[index]]+=W
            self.Q[self.S[index][0]][self.S[index][1]][self.A_index[index]]+=(W/self.C[self.S[index][0]][self.S[index][1]][self.A_index[index]])*(G-self.Q[self.S[index][0]][self.S[index][1]][self.A_index[index]])
            self.pi[self.S[index][0]][self.S[index][1]]=np.argmax(self.Q, axis=2)[self.S[index][0]][self.S[index][1]]
            if self.A_index[index]!=self.pi[self.S[index][0]][self.S[index][1]]:
                return
            W = W*(1/self.A_prob[index])

    def train(self, gamma, epoch):
        for i in range(epoch):
            print(i)
            self.generateActionList(self.action, self.pi, self.greens[0][0], self.greens[0][1], 0.1)
            self.MC_control(gamma)
            self.showvalue(0.1)
    
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
