from ase import io
from ase.parallel import paropen as open #ensures that open works in parallel environment
from ase.optimize import QuasiNewton  #geometry optimization algorithm; QuasiNewton links to BFGS line search, which is the best general-purpose optimizer, but other options are available: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
from ase.calculators.vasp import Vasp
from ase.build import bulk
from ase.visualize import view


#Iron_surface = bcc100(symbol='Fe',size=[2,2,3],a=2.87, orthogonal=True, vacuum=12.0)
#cell = Iron_surface.get_cell()

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.md import Langevin
import ase.units as units
from ase.io.trajectory import Trajectory
import numpy as np


# Set up water box at 20 deg C density
x = angleHOH * np.pi / 180 / 2
pos = [[0, 0, 0],
       [0, rOH * np.cos(x), rOH * np.sin(x)],
       [0, rOH * np.cos(x), -rOH * np.sin(x)]]
atoms = Atoms('OH2', positions=pos)

vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24))**(1 / 3.)
atoms.set_cell((vol, vol, vol))
atoms.center()

atoms = atoms.repeat((1, 1, 2))
atoms.set_pbc(True)



calcargs = dict(xc='pbesol',
        prec='Normal',
        addgrid=False,
        istart=0,
        lreal='Auto',
        kpts=[4, 4, 4],
        encut=600.,
        ediff=1E-4,
        nbands=72,
        algo='VeryFast',
        ispin=2,
#        magmom=[2.243]*12+[0]*6,
        nelm=60,
        ismear=1,
#        rwigs = [1.17]+[0.6]+[0.53],
        nwrite=1,
        sigma=0.2, #smearing width
        isif=2,
        ibrion=2,
        nsw=60)
#        npar=6)
#        outdir ='vasp.log')


calc = Vasp(**calcargs)


atoms.set_calculator(calc)

relax = QuasiNewton(atoms,logfile='water.log',trajectory='water.traj',restart='water.pckl')
relax.run(fmax=0.05)


