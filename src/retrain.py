import numpy as np 
from Emulator_functions import emulator
import dill

# stat = 'ApertureMeasurements_Mass_030kpc_PartType4'
# with open('./Training_data/'+stat+'003.pickle', 'rb') as j:
#     data = dill.load(j)
# print(np.min(data),np.max(data))
# exit()

em=emulator()

em.retrain('ApertureMeasurements_Mass_030kpc_PartType4')
em.retrain('Sat_stellar_mass_func')
