import pdbfixer
import openmm
import torch
import os
import biotite.structure as struc

from tqdm import tqdm
from biotite.structure import AtomArray, Atom
from biotite.structure.io import save_structure
from biotite.structure.io.pdb import PDBFile

ENERGY = openmm.unit.kilocalorie_per_mole
LENGTH = openmm.unit.angstroms
torch.set_num_threads(8)

def openmm_relax(pdb_file, stiffness=10., tolerance=2.39, use_gpu=False):
    fixer = pdbfixer.PDBFixer(pdb_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    force_field = openmm.app.ForceField("amber14/protein.ff14SB.xml")
    modeller = openmm.app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(force_field)
    system = force_field.createSystem(modeller.topology)

    if stiffness > 0:
        stiffness = stiffness * ENERGY / (LENGTH**2)
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        for residue in modeller.topology.residues():
            for atom in residue.atoms():
                if atom.name in ["N", "CA", "C", "CB"]:
                    force.addParticle(
                            atom.index, modeller.positions[atom.index]
                    )
        system.addForce(force)

    tolerance = tolerance
    integrator = openmm.LangevinIntegrator(0, 0.01, 1.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")

    simulation = openmm.app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(tolerance)
    state = simulation.context.getState(getEnergy=True)
    energy = state.getKineticEnergy() + state.getPotentialEnergy()

    with open(pdb_file, "w") as f:
        openmm.app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f,
            keepIds=True
        )
    return energy

for pdb in tqdm(os.listdir("outputs/complexes/ground_truth")):
    openmm_relax(os.path.join("outputs/complexes/ground_truth", pdb))