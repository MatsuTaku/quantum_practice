import numpy as np
from blueqat import opt

class ConstOpt(opt.Opt):
    def const_col_run(self, unit, shots=1, targetT=0.02, verbose=False):
        """
        Run constructive RA with provided QUBO.
        Set qubo attribute in advance of calling this method.
        """
        if self.qubo != []:
            self.qi()
        J = self.reJ()
        N = len(J)

        itetemp = max(1000, int(np.exp(1)*N))
        Rtemp = 0.75

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
                x_list = np.random.randint(0, N, itetemp)
                for x in x_list:
                    if q[x] == 1:
                        continue
                    row = x // unit
                    dE1 = sum(q*J[:,bitpos[row]])
                    q2 = np.array([-1]*N)
                    q2[bitpos[row]] = 1
                    q2[x] = 1
                    dE2 = sum(q*q2*J[:,x])
                    dE = -2*(dE1+dE2)
                    if dE < 0 or np.exp(-dE/T) > np.random.random_sample():
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
