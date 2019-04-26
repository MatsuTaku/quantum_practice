# %%
import numpy as np
from qiskit_chemistry import QiskitChemistry

qiskit_chemistry_dict = {
  "driver": { "name": "PYSCF" },
  "PYSCF": { "atom": "", "basis": "sto3g" },
  "operator": {
    "name": "hamiltonian",
    "qubit_mapping": "parity",
    "two_qubit_reduction": True,
    "freeze_core": True,
    "orbital_reduction": [-3, -2]
  },
  "algorithm": { "name": "VQE" },
  "optimizer": { "name": "COBYLA", "maxiter": 10000 },
  "variational_form": { "name": "UCCSD" },
  "initial_state": { "name": "HartreeFock" }
}
molecule = "H .0 .0 -{0}; Li .0 .0 {0}"

pts  = [x * 0.1  for x in range(6, 20)]
pts += [x * 0.25 for x in range(8, 16)]
pts += [4.0]
energies = np.empty(len(pts))
distances = np.empty(len(pts))
dipoles = np.empty(len(pts))

for i, d in enumerate(pts):
  qiskit_chemistry_dict["PYSCF"]["atom"] = molecule.format(d/2)
  solver = QiskitChemistry()
  result = solver.run(qiskit_chemistry_dict)
  energies[i] = result["energy"]
  dipoles[i] = result["total_dipole_moment"] / 0.393430307
  distances[i] = d

for j in range(len(distances)):
  print("{:0.2f}: Energy={:0.8f}, Dipole={:0.5f}".format(distances[j], energies[j], dipoles[j]))


# %%
%matplotlib inline
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(distances, energies, 'o-')
plt.title('LiH Ground State Energy')
plt.xlabel('Interatomic distance in Angstrom')
plt.ylabel('Energy in Hartree')

plt.figure(2)
plt.plot(distances, dipoles, 'o-')
plt.title('LiH Dipole Moment')
plt.xlabel('Interatomic distance in Angstrom')
plt.ylabel('Moment in Debye')

plt.show()
