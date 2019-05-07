import numpy as np
from blueqat import opt

class AsyncOpt(opt.Opt):
    def run(self, shots=1, startT=5, targetT=0.02, Rtemp=0.75, verbose=False):
        """
        Run constructive RA with provided QUBO.
        Set qubo attribute in advance of calling this method.
        """
        if self.qubo != []:
            self.qi()
        J = self.reJ()
        N = len(J)

        self.E = []
        qq = []
        for i in range(shots):
            T = self.Ts
            q = np.random.choice((-1,1),N)

            EE = []
            EE.append(opt.Ei(q,self.J)+self.ep)
            while T>targetT:
                dEE = []
                x_list = []
                for x in range(N):
                    q2 = np.ones(N)*q[x]
                    q2[x] = 1
                    dE = -2*sum(q*q2*J[:,x])
                    dEE.append(dE)
                    x_list.append(x)

                for x, dE in zip(x_list, dEE):
                    if dE <= 0 or np.exp(-dE/T) > np.random.random_sample():
                        q[x] *= -1

                EE.append(opt.Ei(q,self.J)+self.ep)
                T *= Rtemp
            self.E.append(EE)
            qtemp = (np.asarray(q,int)+1)/2
            qq.append([int(s) for s in qtemp])
            if verbose == True:
                print(i,':',[int(s) for s in qtemp])
            if shots == 1:
                qq = qq[0]
        if shots == 1:
            self.E = self.E[0]
        return qq


class ConstOpt(opt.Opt):
    def const_col_run(self, unit, shots=1, targetT=0.02, Rtemp=0.75, verbose=False):
        """
        Run constructive RA with provided QUBO.
        Set qubo attribute in advance of calling this method.
        """
        if self.qubo != []:
            self.qi()
        J = self.reJ()
        N = len(J)
        numtmp = 1000

        self.E = []
        qq = []
        for i in range(shots):
            T = self.Ts
            q = np.array([-1]*N)
            bitpos = []
            for i in range(N//unit):
                j = np.random.randint(unit)
                q[unit*i + j] = 1
                bitpos.append(unit*i+j)

            EE = []
            EE.append(opt.Ei(q,self.J)+self.ep)
            while T>targetT:
                x_list = np.random.choice(range(N), numtmp)
                for x in x_list:
                    row = x // unit
                    dE1 = sum(q*J[:,bitpos[row]])
                    q2 = np.array([-1]*N)
                    q2[bitpos[row]] = 1
                    q2[x] = 1
                    dE2 = sum(q*q2*J[:,x])
                    dE = -2*(dE1+dE2)
                    
                    if dE <= 0 or np.exp(-dE/T) > np.random.random_sample():
                        q[bitpos[row]] = -1
                        q[x] = 1
                        bitpos[row] = x
                EE.append(opt.Ei(q,self.J)+self.ep)
                T *= Rtemp
            self.E.append(EE)
            qtemp = (np.asarray(q,int)+1)/2
            qq.append([int(s) for s in qtemp])
            if verbose == True:
                print(i,':',[int(s) for s in qtemp])
            if shots == 1:
                qq = qq[0]
        if shots == 1:
            self.E = self.E[0]
        return qq
