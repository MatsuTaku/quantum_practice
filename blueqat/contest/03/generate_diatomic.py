import os
from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4

def generate_diatomic(
    element_names,
    basis='sto-3g',
    charge=0,
    multiplicity=1,
    spacings=None):

    if spacings is None:
        spacings = [0.1 * r for r in range(1, 25)]

    run_scf = 1
    run_mp2 = 1
    run_cisd = 1
    run_ccsd = 1
    run_fci = 1
    verbose = 1
    tolerate_error = 1

    for spacing in spacings:
        description = '{:.4}'.format(spacing)
        geometry = [[element_names[0], [0, 0, 0]],
                    [element_names[1], [0, 0, spacing]]]
        molecure_src = MolecularData(geometry,
                                 basis,
                                 multiplicity,
                                 charge,
                                 description,
                                 data_directory=os.path.abspath('./ising_solver/diatomic'))

        molecule = run_psi4(molecure_src,
                            run_scf=run_scf,
                            run_mp2=run_mp2,
                            run_cisd=run_cisd,
                            run_ccsd=run_ccsd,
                            run_fci=run_fci,
                            verbose=verbose,
                            tolerate_error=tolerate_error)
        molecule.save()

if __name__ == '__main__':
    # generate_diatomic(['H', 'H'], spacings=[0.7414])
    # generate_diatomic(['H', 'H'])
    # generate_diatomic(['H', 'Li'], spacings=[1.45])
    generate_diatomic(['H', 'Li'])
