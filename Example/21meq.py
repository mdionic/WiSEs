from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk
from velocityverletplugin import VVIntegrator
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import os
import ommhelper as oh

class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getPeriodicBoxVectors().value_in_unit(nanometer)
        self._out.write('%g\n' %(simulation.currentStep))
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))


temperature    = 293.15*kelvin
pressure       = 1.0*bar
timestep       = 0.001*picosecond
Total_step     = 10                 
nstposition    = 1000000             
nstcheck       = 5000000             
nstlog         = 100                
nstTdrude      = 500            
nstbar         = 50

Total_step = 1000000*Total_step

simtk.openmm.app.Topology.loadBondDefinitions('./FF/residues.xml')
pdb = PDBFile('./21m.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('./FF/charge-scale0.89-pol.xml')
# add drude particles
modeller.addExtraParticles(forcefield)
print("The drude particles have been added ...")

#Creat the System
system = forcefield.createSystem(modeller.topology,nonbondedMethod=PME,nonbondedCutoff=1.2*nanometer,constraints=None,rigidWater=True)
comnbf=[f for f in system.getForces() if type(f)==CustomNonbondedForce][0]
comnbf.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
print("Cutoff Radius:",comnbf.getCutoffDistance())
print("cf Method:",comnbf.getNonbondedMethod())
donors=[]
for at in range(system.getNumParticles()):
    ParMass =system.getParticleMass(at)/dalton
    if ParMass >6.0 and ParMass < 7.0: #Li ion
        donors.append(at)
    if ParMass>0.4 and ParMass <2.0: # H atom
        donors.append(at)
print('Add TT damping between H/Li atom and Drude dipoles...')
ttforce = oh.CLPolCoulTT(system, donors,45)
print(ttforce.getEnergyFunction())

#integrator
print('Set integrator ...')
integrator = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
integrator.setMaxDrudeDistance(0.02)

#barostat
print('Set barostat ...')
barostat = MonteCarloBarostat(pressure, temperature, nstbar)
system.addForce(barostat)

platform = Platform.getPlatformByName('CUDA')
properties = {'Precision': 'mixed'}
sim = Simulation(modeller.topology, system, integrator, platform, properties)
# set positions from pdb_pos file including drude particles
sim.context.setPositions(modeller.positions)

print('Minimize...')
sim.minimizeEnergy(tolerance=0.1*kilojoule/mole, maxIterations=10000000)
state = sim.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
position = state.getPositions()
PDBFile.writeFile(sim.topology, position, open('./eq/em.pdb', 'w'))


prefix = './eq/'
pdbfile     = prefix + 'eq.pdb'
boxfile     = prefix + 'box.txt'     # the simulation box sizes
timefile    = prefix + 'time.log'    # the time left for finishing the simulation
datafile    = prefix + 'data.log'    # the simulaiton state information 
checkfile   = prefix + 'check.chk'   # the check point file


print('Equilibrating...')
sim.context.setVelocitiesToTemperature(temperature)
sim.reporters.append(PDBReporter(pdbfile, nstposition))
sim.reporters.append(ForceReporter(boxfile, nstlog))
sim.reporters.append(StateDataReporter(timefile, nstlog, progress=True, totalSteps=Total_step,remainingTime=True,separator=' '))
sim.reporters.append(StateDataReporter(datafile, nstlog, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, density=True,volume=True, separator=' '))
sim.reporters.append(CheckpointReporter(checkfile, nstcheck))

#simulation beginning..
sim.step(Total_step)
print('Simulation  Done!')
exit()

