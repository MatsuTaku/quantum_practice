import numpy as np
from blueqat import vqe, opt
from blueqat.pauli import *
from blueqat.pauli import qubo_bit as q


# %% maxcut
hamiltonian = 0.5*Z[0]*Z[1]+0.5*Z[0]*Z[3]+0.5*Z[1]*Z[2]+0.5*Z[2]*Z[3]+0.5*Z[3]*Z[4]+0.5*Z[2]*Z[4]-3
step = 2

result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
print(result.most_common(12))


# %% and gate
hami1 = -1*q(0)+q(1)
hamiltonian = q(2) + q(0)*q(1) + q(1)*q(2) + q(0)*q(2) - q(0)*q(1)*q(2)*4
step = 4
result = vqe.Vqe(vqe.QaoaAnsatz(hami1+hamiltonian, step)).run()
print(result.most_common(12))


# %% xor get_rating_error
hami1 = -1*q(0)+q(1)
hamiltonian = q(0) + q(1) + q(2) + q(0)*q(1)*q(2)*4 - (q(0)*q(1) + q(1)*q(2) + q(0)*q(2))*2
step = 4
result = vqe.Vqe(vqe.QaoaAnsatz(hami1+hamiltonian, step)).run()
print(result.most_common(12))


# %% binary integer linear programming
E = opt.optm("(3*q0+2*q1+q2-3)^2+(5*q0+2*q1+3*q2-5)^2-2*(q0+2*q1+q2)",3)
E = opt.pauli(E)

step = 4
result = vqe.Vqe(vqe.QaoaAnsatz(E, step)).run()
print(result.most_common(5))


# %% factoring 15 = p * q = {1 + 2*q0 + 4*q1} + {1 + 2*q2}
hamiltonian = 128*q(0)*q(1)*q(2)+16*q(0)*q(1)-56*q(0)*q(2)-52*q(0)-48*q(1)*q(2)-96*q(1)-52*q(2)+196

step = 4
result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
print(result.most_common(5))
ans = result.most_common(5)[0][0]
print(1+2*ans[0]+4*ans[1], 1+2*ans[2])
