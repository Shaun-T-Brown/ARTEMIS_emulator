import numpy as np 
from Emulator_functions import *


em=emulator()

#em.load_stat('ApertureMeasurements_Mass_030kpc_PartType4')


data,redshift,x=em.predict('ApertureMeasurements_Mass_030kpc_PartType4',(np.ones(6)*0.5).reshape(1,-1),(0.0,4.5),return_x=True)


plt.plot(redshift,data+10,'kx')
plt.plot(redshift,data+10)

plt.show()
