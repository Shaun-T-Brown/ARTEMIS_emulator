import numpy as np 
import matplotlib.pyplot as plt 
import h5py as h5
from Emulator_functions import emulator_build


sim_directory='/cosma8/data/dp203/dc-brow5/simulations_cosma6/simulations/ARTEMIS/HYDRO/Latin_hypercube_2/'
L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
tests=np.loadtxt('./random_cube_2.txt')


param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
snap_name='029_z000p000'

em = emulator_build(param_label,L_cube,tests)
file_name,file_name_test=em.get_filename()
halos=['halo_61','halo_04','halo_32']

def most_massive_branch(loc):

    h=h5.File(loc,'r')
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])
    h.close()
    mass_branch = np.zeros(len(ID))
    ID_end_node = ID[Main_prog==-1]
        
    for i in range(len(ID_end_node)):
        
        descendent=ID_end_node[i]
        mass_tally=0
        while descendent != 0:
            if descendent==-1:
                break
            ind = ID ==descendent

            mass_tally+=mass[ind]
            mass_branch[ind] +=mass_tally
            descendent = desc_ind[ind]

    #append data to file
    try:

        g=h5.File(loc,'a')
        g.create_dataset('haloTrees/branchMass',data=mass_branch)
        g.close()
    except ValueError:
        g=h5.File(loc,'r+')
        data = g['haloTrees/branchMass']
        data[...] = mass_branch
        g.close()

    return

#loop for all haloes run
# for i in range(len(halos)):
#     for j in range(len(file_name)):

#         loc = sim_directory +'/'+halos[i]+'/'+file_name[j]+'/data/merger_trees_new/tree_029.0.hdf5'

#         print(loc)

#         most_massive_branch(loc)

#loop for all haloes run
# for i in range(len(halos)):
#     for j in range(len(file_name_test)):

#         loc = sim_directory +'/'+halos[i]+'/'+file_name_test[j]+'/data/merger_trees_new/tree_029.0.hdf5'

#         print(loc)
#         try:
#             most_massive_branch(loc)
#         except:
#             print('Failed')
#             pass


# exit()
#loop for all rerun haloes
loc = '/cosma7/data/dp004/dc-brow5/simulations/ARTEMIS/Rerun/halo_61/'
run_num = [3,4,5,6,7,9]
run_num = [1,2]

for i in range(len(run_num)):
    loc2 = loc+'Run_%d/data/merger_trees/tree_029.0.hdf5'%run_num[i]
    print(loc2)
    most_massive_branch(loc2)