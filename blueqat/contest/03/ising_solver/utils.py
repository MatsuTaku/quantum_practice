from ising_solver.config import *
from openfermion import MolecularData
from blueqat.pauli import *

def get_molecule(element_names, bond_len):
    geometry = [[element_names[0], [0, 0, 0]],
                [element_names[1], [0, 0, bond_len]]]
    basis = 'sto-3g'
    multiplicity = 1
    description = '{:.2}'.format(bond_len)
    data_directory = DATA_DIRECTORY

    molecule = MolecularData(geometry, basis, multiplicity, description=description, data_directory=data_directory)
    molecule.load()

    return molecule

def max_n(hamiltonain):
    '''
    Get max id of operator of hamiltonian of openfermion
    '''
    maxn = 0
    for qo_ops, coeff in hamiltonain.terms.items():
        for n, c in qo_ops:
            maxn = max(maxn, n)
    return maxn

def get_energy(hamiltonian, z):
    if not isinstance(hamiltonian, Expr):
        raise NotImplementedError

    energy = .0
    for term in hamiltonian:
        e = 1.0
        for n in term.n_iter():
            e *= z[n]
        energy += term.coeff * e
    return energy

def get_coeffs(hamiltonian, z):
    coeffs = []
    for term in hamiltonian:
        s = 1
        for n in term.n_iter():
            s *= z[n]
        coeffs.append(term.coeff*s)
    return coeffs
