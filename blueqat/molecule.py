from blueqat import *
from openfermion import *
from openfermionblueqat import *
import numpy as np

def get_molecule(bond_len):
    '''
    Moecule of H2
    '''
    geometry = [('H',(0.,0.,0.)),('H',(0.,0.,bond_len))]
    description = format(bond_len)
    molecule = MolecularData(geometry, 'sto-3g', 1, description=description)
    molecule.load()
    return molecule

# %%
x = [];e=[];fullci=[]
for bond_len in np.arange(0.2, 2.5, 0.1):
    m = get_molecule("{:.2}".format(bond_len))
    # ham = m.get_molecular_hamiltonian()
    # h = bravyi_kitaev(get_fermion_operator(m.get_molecular_hamiltonian()))
    # # for tup in h.terms.items():
    # #     print(tup)
    # # runner = vqe.Vqe(UCCAnsatz(h, 6, Circuit().x[0]))
    # # print(ham)
    # # print(get_fermion_operator(ham))
    runner = vqe.Vqe(UCCAnsatz(m, 6, Circuit().x[0]))
    result = runner.run()
    x.append(bond_len)
    e.append(runner.ansatz.get_energy(result.circuit, runner.sampler))
    fullci.append(m.fci_energy)

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(x, fullci)
plt.plot(x, e, 'o')
plt.show()
