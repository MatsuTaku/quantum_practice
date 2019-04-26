# %%
from ising_solver.utils import get_molecule
from ising_solver.model import ElectronicStructureCalculator
from openfermion import *
from openfermionblueqat import UCCAnsatz
from blueqat import vqe, Circuit
import numpy as np
import time

def ground_energy_ucc(hamiltonian, steps=6):
    runner = vqe.Vqe(UCCAnsatz(hamiltonian, steps, Circuit().x[0]))
    result = runner.run()
    return runner.ansatz.get_energy(result.circuit, runner.sampler)

def ground_energy_anealing(hamiltonian, r_qubits, shots, verbose=False):
    calculator = ElectronicStructureCalculator(hamiltonian, r_qubits, verbose=verbose)
    return calculator.get_ground_state_energy(shots=shots)

def calc_ground_state_energy(element_names, r, name='', shots=100, with_ucc=False, verbose=False):
    if not name:
        name = ''.join(element_names)
    x = [x * 0.1 for x in range(3, 31)]
    e=[];a=[];fullci=[];hf=[];etime=0;atime=0
    print('Anealing shots =', shots)
    print('-'*20, 'Ground state energies of', name, '-'*20)
    print(' distance |\tFCI\t|\tAnealing(r={})\t|\tUCC\t|'.format(r))
    for i, bond_len in enumerate(x):
        m = get_molecule(element_names, bond_len)
        fullci.append(m.fci_energy)
        hf.append(m.hf_energy)

        h = bravyi_kitaev(get_fermion_operator(m.get_molecular_hamiltonian()))

        # Anealing with r
        start = time.time()
        a.append(ground_energy_anealing(h, r, shots=shots, verbose=verbose))
        atime += time.time()-start

        # UCC
        if with_ucc:
            start = time.time()
            e.append(ground_energy_ucc(h))
            etime += time.time()-start

        if with_ucc:
            print('{:8.2} \t{:.4}\t\t{:.4}\t\t{:.4}'.format(bond_len, fullci[i], e[i], a[i]))
        else:
            print('{:8.2} \t{:.4}\t\t{:.4}'.format(bond_len, fullci[i], a[i]))

    print('---Calcuration times---')
    print('anealing:', atime, ' s')
    if with_ucc:
        print('vqe-ucc:', etime, ' s')

    # %matplotlib inline
    import matplotlib.pyplot as plt
    plt.title('{} Ground State Energy'.format(name))
    plt.xlabel('Interatomic distance in Angstrom')
    plt.ylabel('Energy in Hartree')
    plt.plot(x, hf, linestyle='dashed', label='HF')
    plt.plot(x, fullci, label='FullCI')
    if with_ucc:
        plt.plot(x, e, '.', label='UCC')
    plt.plot(x, a, 'o', label='Anealing(r={})'.format(r))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # %%
    calc_ground_state_energy(['H', 'H'], 1, shots=50, verbose=False, name='H2')
    # %%
    # calc_ground_state_energy(['H', 'H'], 2, shots=100, verbose=False, name='H2')
    # %%
    # calc_ground_state_energy(['H', 'H'], 4, shots=200, verbose=True, name='H2')
    # %%
    calc_ground_state_energy(['Li', 'H'], 1, shots=500, verbose=False)
    # %%
    # calc_ground_state_energy(['Li', 'H'], 2, shots=200, verbose=True)
    # %%
    # calc_ground_state_energy(['Li', 'H'], 4, shots=100, verbose=True)
