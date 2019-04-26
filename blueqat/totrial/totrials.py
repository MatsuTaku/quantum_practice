from blueqat import *
import numpy as np
import math

# %% 1
Circuit().x[0].m[:].run(shots=1)
Circuit().h[0].run()


# %% 2
Circuit(3).x[0].cx[0,1].m[:].run(shots=1)

# %% 3
Circuit().h[0].z[0].m[:].run(shots=100)
Circuit().h[0].z[0].run()

# %% 4
Circuit().h[0].cx[0,1].m[:].run(shots=100)


# %% 5 toffori gate
Circuit().x[1:].h[0].cnot[1,0].rz(-np.pi/4)[0].cnot[2,0].rz(np.pi/4)[0].cnot[1,0].rz(-np.pi/4)[0].cnot[2,0].rz(np.pi/4)[:1].h[0].cnot[1,0].cnot[0,1].cnot[1,0].cnot[2,0].rz(-np.pi/4)[0].rz(np.pi/4)[2].cnot[2,0].m[:].run(shots=1)
Circuit().x[:2].ccx[0,1,2].m[:].run(shots=1)

# %% 6 swap gate
Circuit().x[0].cx[0,1].cx[1,0].cx[0,1].m[:].run(shots=1)

# %% 7 controlled gate
Circuit().x[0].h[1].cz[0,1].h[1].m[:].run(shots=1)
# CRz gate
Circuit().h[0].rz(math.pi/2)[0].cx[1,0].rz(-math.pi/2)[0].cx[1,0].h[0].m[:].run(shots=100)

# %% 8 adder
adder = Circuit().ccx[0,1,3].cx[1,2].cx[0,2]
# 0+0
(Circuit() + adder).m[:].run(shots=1)
# 0+1
(Circuit().x[1] + adder).m[:].run(shots=1)
# 1+0
(Circuit().x[0] + adder).m[:].run(shots=1)
# 1+1
(Circuit().x[0,1] + adder).m[:].run(shots=1)

# %% 9 adder with hadamard gate
(Circuit().h[0].h[1] + adder).m[:].run(shots=100)

# %% 10 multiplier
C1 = Circuit().ccx[0,1,2].ccx[1,3,5].ccx[0,4,6].ccx[3,4,7].ccx[5,6,8].ccx[7,8,9].cx[2,10].cx[5,11].cx[6,11].cx[7,12].cx[8,12].cx[9,13]
# 00 * 00 = 0000
(Circuit() + C1).m[:].run(shots=100)
# 01 * 01 = 0001
(Circuit().x[0,1] + C1).m[:].run(shots=100)
# 10 * 01 = 0010
(Circuit().x[3,1] + C1).m[:].run(shots=100)
# 01 * 10 = 0010
(Circuit().x[0,4] + C1).m[:].run(shots=100)
# 01 * 10 = 0010
(Circuit().x[0,4] + C1).m[:].run(shots=100)
# 10 * 10 = 0100
(Circuit().x[3,4] + C1).m[:].run(shots=100)
# 11 * 10 = 0110
(Circuit().x[0,3,4] + C1).m[:].run(shots=100)
# 10 * 11 = 0110
(Circuit().x[1,3,4] + C1).m[:].run(shots=100)
# 11 * 11 = 1001
(Circuit().x[0,1,3,4] + C1).m[:].run(shots=100)

# %% 11 GHZ
Circuit().h[:2].x[2].cx[1,2].cx[0,2].h[:].m[:].run(shots=100)


# %% 12 teleportation
a = Circuit().h[1].cx[1,2].cx[0,1].h[0].cx[1,2].cz[0,2].m[:]
a.run(shots=100)
(Circuit().h[0] + a).run(shots=100)

# %% 13 fourier transform
def rzn(circuit, rot, c, t):
    return circuit.rz(rot)[t].cx[c,t].rz(-rot)[t].cx[c,t]
BlueqatGlobalSetting.unregister_macro('rzn')
BlueqatGlobalSetting.register_macro('rzn', rzn)
# 2 qubit
Circuit().x[:].h[0].rzn(math.pi/4,1,0).h[1].run()
# 4 qubit
Circuit().x[:].h[0].rzn(math.pi/4,1,0).rzn(math.pi/8,2,0).rzn(math.pi/16,3,0).h[1].rzn(math.pi/4,2,1).rzn(math.pi/8,3,1).h[2].rzn(math.pi/4,3,2).h[3].run()
print(np.fft.fft(np.array([0,0,0,1])/2))

# %% 14 grover
mark00 = Circuit().s[:].cz[0,1].s[:]
mark01 = Circuit().s[1].cz[0,1].s[1]
mark10 = Circuit().s[0].cz[0,1].s[0]
mark11 = Circuit().cz[0,1]
aa = Circuit().h[:].x[:].cz[0,1].x[:].h[:]
(Circuit().h[:] + mark11 + aa).m[:].run(shots=100)

# %% 15 phase estimation
Circuit(2).h[:].cx[0,1].h[0].m[:].run(shots=100)


# %% vqe
angle = np.linspace(0.0,2*np.pi,21)
data = [0 for _ in range(21)]
qdata = []
for i, ang in enumerate(angle):
    c = Circuit().ry(ang)[0].z[0]
    result = c.run()
    data[i] = np.abs(result[0])*np.abs(result[0])-np.abs(result[1])*np.abs(result[1])
    qdata += [result]
%matplotlib inline
import matplotlib.pyplot as plt
plt.xlabel('Parameter value')
plt.ylabel('Expectation value')
plt.plot(angle, data)
plt.plot(angle, qdata)
plt.show()

# %% qaoa
from blueqat.pauli import qubo_bit as q
E = q(0)*q(1)
step = 2
result = vqe.Vqe(vqe.QaoaAnsatz(E, step)).run()
print(result.most_common(12))
