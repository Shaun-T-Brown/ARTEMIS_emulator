import numpy as np 
from Emulator_functions import emulator
import dill

em=emulator()

em.predict('ApertureMeasurements_Mass_030kpc_PartType4',np.ones(6).reshape(1,-1)*0.5,0)
exit()
em.retrain('ApertureMeasurements_Mass_030kpc_PartType4')
em.retrain('Sat_stellar_mass_func')
