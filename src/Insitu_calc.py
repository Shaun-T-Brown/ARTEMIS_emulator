import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import dill
import h5py as h5
import eagle_IO.eagle_IO as E
import os 
from matplotlib.colors import LogNorm
from Emulator_functions import emulator_build
from Emulator_functions import rescale
from Emulator_functions import main_proj_corrections
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

sim_directory='/cosma7/data/dp004/dc-brow5/simulations/ARTEMIS/Latin_hyperube_2/'
halos=['halo_04'] #'halo_61','halo_32',
L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
tests=np.loadtxt('./random_cube_2.txt')

num_snaps = 30

param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
snap_name='029_z000p000'

em = emulator_build(param_label,L_cube,tests)
file_name,file_name_test=em.get_filename()
#del file_name[0]

redshift=np.loadtxt(sim_directory+halos[0]+'/'+file_name[0]+'/redshift_list.txt')
tags=[]
for i in range(len(redshift)):
    tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))


#file_name=file_name_test+file_name

#find host projenitor
projen_sub=[]
projen_fof=[]
for k in range(len(halos)):
    projen_sub.append(np.ones((len(file_name),len(tags)),dtype=int)*-1)
    projen_fof.append(np.ones((len(file_name),len(tags)),dtype=int)*-1)
    for i in range(len(file_name)):
        #load subhalo numbers and group numbers
        projen_sub_id = main_proj_corrections(sim_directory+halos[k]+'/'+file_name[i]+'/data',num_snaps,0)

        for j in range(len(projen_sub_id)):
            if projen_sub_id[j]==-1:
                break
            group_num=E.read_array("SUBFIND",sim_directory+halos[k]+'/'+file_name[i]+'/data/',tags[-(j+1)],"Subhalo/GroupNumber",noH=False)-1
            sub_group_num=E.read_array("SUBFIND",sim_directory+halos[k]+'/'+file_name[i]+'/data/',tags[-(j+1)],"Subhalo/SubGroupNumber",noH=False)

            projen_sub[k][i,j]=sub_group_num[int(projen_sub_id[j])]
            projen_fof[k][i,j]=group_num[int(projen_sub_id[j])]

        #reverse array for consistency with ret of code
        projen_sub[k][i,:]=projen_sub[k][i,:][::-1]
        projen_fof[k][i,:]=projen_fof[k][i,:][::-1]


#loop through all files
for k in range(len(halos)):
    for i in range(len(file_name)):

        print(i,k)
        
        #read star data
        group_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-1],'PartType4/GroupNumber',noH=False)
        group_num=np.abs(group_num)-1
        subgroup_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-1],'PartType4/SubGroupNumber',noH=False)

        star_form_time=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-1],'PartType4/StellarFormationTime',noH=False)
        ID=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-1],'PartType4/ParticleIDs',noH=False)

        #sort for speed
        sort=np.argsort(ID)
        ID=ID[sort]; star_form_time=star_form_time[sort]; group_num=group_num[sort]; subgroup_num=subgroup_num[sort]
        
        #get snaphot number for formation time
        snap_form=np.empty(len(star_form_time))
        scale_factors=1/(1+redshift[:,1])
        
        #calcaulte the snapshot the star formed at (rounding up)
        star_form_snap = np.digitize(star_form_time,scale_factors)
        star_form_snap_un=np.unique(star_form_snap)

        insitu=np.ones(len(ID),dtype=int)*-1

        cut=(group_num==0) & (subgroup_num==0)
        for j in range(len(redshift)):
            
            #skip snapshot if no stars formed here
            if np.isin((29-j),star_form_snap_un)==False:
                continue
            
            print(sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-(j+1)])
            group_num2=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-(j+1)],'PartType4/GroupNumber',noH=False)
            group_num2=np.abs(group_num2)-1
            
            subgroup_num2=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-(j+1)],'PartType4/SubGroupNumber',noH=False)
            ID2=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[-(j+1)],'PartType4/ParticleIDs',noH=False)
            sort=np.argsort(ID2)
            ID2=ID2[sort]; group_num2=group_num2[sort]; subgroup_num2=subgroup_num2[sort]
            

            #first select stars from main halo and that formed at this redshift

            ID_formed = ID[(cut) & (star_form_snap==(29-j))]

            #remove IDs that are too long (deal with star particles that have been destroyed between fomraiton and snapshot)
            ID_formed = ID_formed[ID_formed<=ID2[-1]] #-1 for max as sorted

            #find subgroup number and group number for these particles

            
            cut2=np.searchsorted(ID2, ID_formed)
            #remove IDs that are too long (deal with star particles that have been destroyed between fomraiton and snapshot)
            
            sub_in = subgroup_num2[cut2]; group_in = group_num2[cut2]

            insit_form = (sub_in==projen_sub[k][i,-(j+1)]) & (group_in==projen_fof[k][i,-(j+1)])
            
            insitu[np.searchsorted(ID,ID_formed[insit_form])] = 1
            insitu[np.searchsorted(ID,ID_formed[np.invert(insit_form)])] = 0

        #create array to write to text file

        write_array=np.empty((len(ID),2),dtype=int)
        write_array[:,0]=ID
        write_array[:,1]=insitu


        np.savetxt(sim_directory+halos[k]+'/'+file_name[i]+'/processed_data/instu_star_formation.txt',write_array,fmt='%d',header='Particle ID, insitu flag (1 in situ, 0 ex-situ, -1 not calculated)')

