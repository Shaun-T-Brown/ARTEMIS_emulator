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
L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
tests=np.loadtxt('./random_cube_2.txt')

param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
snap_name='029_z000p000'

em = emulator_build(param_label,L_cube,tests)
file_name,file_name_test=em.get_filename()


def all_proj(h,num_snaps):
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])

    #start with host
    ID_current=(29)*1e12+0
    ind=ID==((29)*1e12+0)
    
    mass_proj = []
    for i in range(num_snaps):
        
        mass_proj.append(mass[np.in1d(desc_ind,ID_current)])
        
        ID_current = ID[np.in1d(desc_ind,ID_current)]
        
        if len(ID_current)==0:
            break
        if np.all(ID_current==-1):
            break
        
    return(mass_proj)

def star_form_hist(loc):
        redshift=np.loadtxt(loc+'/redshift_list.txt')
        tags=[]
        for i in range(len(redshift)):

            tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))
            
            
        #read merger tree
        
        
        h=h5.File(loc+'/data/merger_trees/tree_029.0.hdf5','r')
        
        ID=np.array(h['haloTrees/nodeIndex'])
        Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
        desc_ind=np.array(h['haloTrees/descendantIndex'])
        mass=np.array(h['haloTrees/nodeMass'])
        redshifts=np.array(h['haloTrees/redshift'])

        
        ID_initial=int(29e12)
        sub_id_mass=np.ones(30,dtype=np.int)*(-1)
        k=0
        while ID_initial!=-1:

            sub_id_mass[k]=int(ID_initial-(29-k)*1e12)

            ind = np.where(desc_ind==ID_initial)[0]
            if len(ind)==0:
                break
            
            
            mass_desc=mass[ind]
            
            print(k,'%.2e'%np.max(mass_desc))

            ID_initial=ID[ind[np.argmax(mass_desc)]]

            k+=1

        
        sub_id_mass=sub_id_mass[::-1]

        ID_initial=int(29e12)
        ID_main=[]
        sub_id=np.ones(30,dtype=np.int)*(-1)
        k=0
        while ID_initial!=-1:

            ID_main.append(ID_initial)
            sub_id[k]=int(ID_initial-(29-k)*1e12)
            ind=ID==ID_initial
            print(ind)
            prog_id=Main_prog[ind][0]

            ID_initial=prog_id
            k+=1
        sub_id=sub_id[::-1]


        stellar_mass=[]
        dm_mass=[]
        for i in tags:
            #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
            #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/Stars/Metallicity"))
            try:
                stellar_mass.append(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
                dm_mass.append(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,1]*10**10)
            except:
                stellar_mass.append([])
                dm_mass.append([])
        
        st_mass=np.ones(len(tags))*-1
        st_mass2=np.ones(len(tags))*-1
        dm_mas=np.ones(len(tags))*-1
        dm_mas2=np.ones(len(tags))*-1
        dm_mass_max=np.ones(len(tags))*-1
        st_mass_max=np.ones(len(tags))*-1
        for i in range(len(tags)):
            if sub_id[i]<=-1:
                continue
            print(sub_id[i])
            st_mass[i]=stellar_mass[i][sub_id[i]]
            dm_mas[i]=dm_mass[i][sub_id[i]]

            st_mass2[i]=stellar_mass[i][sub_id_mass[i]]
            dm_mas2[i]=dm_mass[i][sub_id_mass[i]]

            dm_mass_max[i]=np.max(dm_mass[i])
            st_mass_max[i]=stellar_mass[i][np.argmax(dm_mass[i])]

        return(st_mass,st_mass2,dm_mas,dm_mas2,dm_mass_max,st_mass_max,redshift)

def main_proj(h,num_snaps):
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])
    

    #start with host
    ind=ID==((29)*1e12+0)
    
    mass_proj = np.ones(num_snaps)*-1
    for i in range(num_snaps):
        mass_proj[i] = mass[ind]

        if Main_prog[ind]==-1:
            break
        ind = ID==Main_prog[ind]

    return(mass_proj)

def most_massive_proj(h,num_snaps):
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])

    #start with host
    ind=ID==((29)*1e12+0)
    ID_current = ((29)*1e12+0)
    mass_proj = np.ones(num_snaps)*-1
    mass_proj[0] = mass[ind]
    for i in range(num_snaps):
        #grab all projenitors
        cut = np.where(np.in1d(desc_ind,ID_current))[0]
        ID_proj = ID[cut]
        mass_proj_all = mass[cut]
        
        if len(cut)==0:
            break
        ID_current = ID_proj[np.argmax(mass_proj_all)]
        mass_proj[i+1] = np.max(mass_proj_all)

    return(mass_proj)

def most_massive_branch(h,num_snaps):
    
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])
    mass_branch = np.array(h['haloTrees/branchMass'])
    redshifts = np.array(h['haloTrees/redshift'])

    snap_num = np.array(h['haloTrees/snapshotNumber'])
    fof_id = np.array(h['haloTrees/fofIndex'])
    fof_centre = np.array(h['haloTrees/isFoFCentre'])

    #start with host
    ind=ID==((29)*1e12+0)
    ID_current = ((29)*1e12+0)
    mass_proj = np.ones(num_snaps)*-1
    mass_proj[0] = mass[ind]
    ID_branch = np.ones(num_snaps)*-1
    ID_branch[0] = ID_current
    for i in range(num_snaps):
        #grab all projenitors
        cut = np.where(np.in1d(desc_ind,ID_current))[0]
        ID_proj = ID[cut]
        mass_branch_all = mass_branch[cut]
        mass_all = mass[cut]
        
        if len(cut)==0:
            break
        ID_current = ID_proj[np.argmax(mass_branch_all)]
        mass_proj[i+1] = mass_all[np.argmax(mass_branch_all)]
        ID_branch[i+1] = ID_current

    difference=np.empty(len(mass_proj))
    mass_proj_pad = np.pad(mass_proj,1,mode='edge')
    #identify discontinuity
    for i in range(len(mass_proj)):
        difference[i] = (np.log10(mass_proj_pad[i])+np.log10(mass_proj_pad[i+2]))/2-np.log10(mass_proj_pad[i+1])

    diff_cut=0.4

    if np.sum(difference>=diff_cut)==0:
        return(mass_proj,mass_proj)

    else:

        mass_proj_correction = np.array(mass_proj)
        cut = np.where(difference>=diff_cut)[0]

        for i in range(len(cut)):

            ID_current = ID_branch[cut[i]]    
            #overwrite to be main projenitor
            ind = ID==ID_current

            central_ind = (snap_num==snap_num[ind]) & (fof_id==fof_id[ind]) & (fof_centre==1)

            mass_proj_correction[cut[i]] = mass[central_ind]
        
        return(mass_proj,mass_proj_correction)

def central_branch(h,num_snaps):
    print(h['haloTrees'].keys())
    ID=np.array(h['haloTrees/nodeIndex'])
    Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])
    mass_branch = np.array(h['haloTrees/branchMass'])
    redshifts = np.array(h['haloTrees/redshift'])

    snap_num = np.array(h['haloTrees/snapshotNumber'])
    fof_id = np.array(h['haloTrees/fofIndex'])
    fof_centre = np.array(h['haloTrees/isFoFCentre'])
    #loop through, overwritng to be main projenitor

    #start with host
    ind=ID==((29)*1e12+0)
    
    mass_proj = np.ones(num_snaps)*-1
    for i in range(num_snaps):
        mass_proj[i] = mass[ind]
        fof_id_proj = fof_id[ind]
        snap_num_proj =snap_num[ind]
        fof_centre_proj = fof_centre[ind]
        if Main_prog[ind]==-1:
            break
        
        #overwite to be central
        if fof_centre_proj==1:
            ind = ID==Main_prog[ind]

        else:
            
            ID_new = ID[(snap_num==snap_num_proj) & (fof_id_proj==fof_id) & (fof_centre==1)]
            ind = ID ==ID_new

    plt.plot(mass_proj)
    plt.show()
    exit()

halos=['halo_61','halo_32','halo_04']
redshift=np.loadtxt(sim_directory+halos[0]+'/'+file_name[0]+'/redshift_list.txt')
tags=[]
for i in range(len(redshift)):
    tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))



mass1=[]
mass2=[]
mass_massive_branch=[]
mass_massive_branch2=[]

m_dm=[]
m_star=[]
for j in range(len(halos)):
    mass2.append([])
    mass_massive_branch2.append([])
    m_dm.append([])
    m_star.append([])
    for i in range(len(file_name)):
        print(file_name[i])
        loc = sim_directory+halos[j]+'/'+file_name[i]+'/data/'
        h2=h5.File(sim_directory+halos[j]+'/'+file_name[i]+'/data/merger_trees_new/tree_029.0.hdf5')

        mass2[-1].append(main_proj(h2,30))
        mass_massive_branch2[-1].append( most_massive_branch(h2,30))

        sub_id = main_proj_corrections(sim_directory+halos[j]+'/'+file_name[i]+'/data/',30,0)

        sub_id=sub_id[::-1]

        m_dm[-1].append(np.ones(len(tags))*-1)
        m_star[-1].append(np.ones(len(tags))*-1)
        #read in masses
        for k in range(len(tags)):
            if sub_id[k]==-1:
                continue
            masses = E.read_array("SUBFIND",loc,tags[k],"Subhalo/MassType")

            m_dm[j][i][k] = masses[int(sub_id[k]),1]
            m_star[j][i][k] = masses[int(sub_id[k]),4]



fig=plt.figure()
for j in range(len(halos)):
    for i in range(len(mass_massive_branch2[j])):
        red = redshift[:,1][::-1]
        cut = red<4
        plt.plot((red[cut]+1)+5*j,mass2[j][i][cut])
plt.yscale('log')
plt.xlim(0,15)
plt.title('Main projenitor')
fig.savefig('./Figures/Merger_tree_test_most_massive_branch.png')

fig=plt.figure()
for j in range(len(halos)):
    for i in range(len(mass_massive_branch2[j])):
        red = redshift[:,1][::-1]
        cut = red<4
        plt.plot((red[cut]+1)+5*j,mass_massive_branch2[j][i][0][cut])
plt.yscale('log')
plt.xlim(0,15)
plt.title('Most massive branch')
fig.savefig('./Figures/Merger_tree_test_main_proj.png')

fig=plt.figure()
for j in range(len(halos)):
    for i in range(len(mass_massive_branch2[j])):
        red = redshift[:,1][::-1]
        cut = red<4
        plt.plot((red[cut]+1)+5*j,mass_massive_branch2[j][i][1][cut])
plt.yscale('log')
plt.xlim(0,15)
plt.title('Most massive branch correction')
fig.savefig('./Figures/Merger_tree_test_massive_branch_corrected.png')



fig=plt.figure()
for j in range(len(halos)):
    for i in range(len(mass_massive_branch2[j])):
        red = redshift[:,1]
        cut = red<4
        plt.plot((red[cut]+1)+5*j,m_dm[j][i][cut])

    break
plt.yscale('log')
plt.xlim(0,5)

fig=plt.figure()
for j in range(len(halos)):
    for i in range(len(mass_massive_branch2[j])):
        red = redshift[:,1]
        cut = red<4
        plt.plot((red[cut]+1)+5*j,m_star[j][i][cut])

    break
plt.yscale('log')
plt.xlim(0,5)


plt.show()
exit()





ID=np.array(h['haloTrees/nodeIndex'])
Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
desc_ind=np.array(h['haloTrees/descendantIndex'])
mass=np.array(h['haloTrees/nodeMass'])
redshifts=np.array(h['haloTrees/redshift'])

dm_mass=E.read_array("SUBFIND",sim_directory+file_name[3]+'/data',tags[-(5+1)],"Subhalo/MassType",noH=False)[:,1]*10**10
sub_id=np.argmax(dm_mass)
print('%.2e'%np.max(dm_mass))
ind=ID==((29-6)*1e12+sub_id)

print(ID[ind])
print(mass[ind])
print(desc_ind[ind])
print(mass[ID==desc_ind[ind]])

#walk tree 'by hand'
masses=mass[(ID>=(29-6)*1e12) & (ID<(29-5)*1e12)]
print('%.2e, %.2e'%(masses[np.argsort(-masses)][0],masses[np.argsort(-masses)][1]))

exit()
print('%.2e'%mass[ID==24e12])

ID_initial=int(29e12)
sub_id_mass=np.ones(30,dtype=np.int)*(-1)
k=0
while ID_initial!=-1:

    
    sub_id_mass[k]=int(ID_initial-(29-k)*1e12)

    ind = np.where(desc_ind==ID_initial)[0]
    if len(ind)==0:
        break
    
    
    mass_desc=mass[ind]
    #print(mass_desc)
    print(k,sub_id_mass[k],'%.2e'%np.max(mass_desc))

    ID_initial=ID[ind[np.argmax(mass_desc)]]
    print(ID_initial)
    k+=1


sub_id_mass=sub_id_mass[::-1]


exit()

end_node=np.zeros(len(ID),dtype=bool)
for i in range(len(desc_ind)):
    end_node[i] = np.sum(desc_ind==ID[i])
end_node=np.flip(end_node)

end_node_index=ID[end_node]
#walk from endnode
mass_integrated=np.zeros(len(ID))
for i in range(len(end_node_index)):

    print(i,len(end_node_index))
    ID_current=end_node_index[i]
    descendent=0
    while descendent>=0:
        #print(ID_current)
        ind=np.where(ID==ID_current)[0]
        descendent=desc_ind[ind]
        mass_integrated[ind]+=mass[ind]


        ID_current=descendent


#walk tree for host
ID_initial=int(29e12)
sub_id_mass=np.ones(30,dtype=np.int)*(-1)
k=0
while ID_initial!=-1:

    sub_id_mass[k]=int(ID_initial-(29-k)*1e12)

    ind = np.where(desc_ind==ID_initial)[0]
    if len(ind)==0:
        break
    
    
    mass_desc=mass_integrated[ind]
    
    print(k,'%.2e'%np.max(mass_desc))

    ID_initial=ID[ind[np.argmax(mass_desc)]]

    k+=1
sub_id_mass=sub_id_mass[::-1]


exit()
stellar_mass=[]
dm_mass=[]
for i in tags:
    #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
    #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/Stars/Metallicity"))
    try:
        stellar_mass.append(E.read_array("SUBFIND",sim_directory+file_name[3]+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
        dm_mass.append(E.read_array("SUBFIND",sim_directory+file_name[3]+'/data',i,"Subhalo/MassType",noH=False)[:,1]*10**10)
    except:
        stellar_mass.append([])
        dm_mass.append([])


print(len(tags))
st_mass=np.ones(len(tags))*-1
dm_mas=np.ones(len(tags))*-1
for i in range(len(tags)):
    print(i)
    if sub_id_mass[i]<=-1:
        continue

    st_mass[i]=stellar_mass[i][sub_id_mass[i]]
    dm_mas[i]=dm_mass[i][sub_id_mass[i]]


print(st_mass.shape)
print(redshift[:,1].shape)
plt.plot(redshift[:,1],st_mass)
plt.plot(redshift[:,1],dm_mas)
plt.yscale('log')
plt.xlim(0,4)
plt.ylim(10**9,6*10**11)
plt.show()
exit()
print(file_name[3])
file_name=[file_name[3]]
cmap=plt.get_cmap("tab10")

st_mass=[]
st_mass2=[]
dm_mas=[]
dm_mas2=[]
dm_mass_max=[]
st_mass_max=[]
for i in range(len(file_name)):
    out = star_form_hist(sim_directory+file_name[i])
    st_mass.append(out[0])
    st_mass2.append(out[1])
    dm_mas.append(out[2])
    dm_mas2.append(out[3])
    dm_mass_max.append(out[4])
    st_mass_max.append(out[5])
    redshift=out[6]


fig=plt.figure()
for i in range(len(st_mass)):
    plt.plot(redshift[:,1],st_mass_max[i],'--',color=cmap(i))
    plt.plot(redshift[:,1],st_mass2[i],color=cmap(i))
plt.xlim(0,4)
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('M_st')
#plt.legend(frameon=False)
fig.savefig('./Figures/Merger_tree_test_stars_all.png')

fig = plt.figure()
for i in range(len(st_mass)):
    plt.plot(redshift[:,1],dm_mass_max[i],'--',color='black',label='Most massive')
    plt.plot(redshift[:,1],dm_mas2[i],color='black',label='Dhaloes')
plt.xlim(0,4)
plt.yscale('log')
plt.xlabel('z')
plt.ylabel('M_DM')
#plt.legend(frameon=False)
fig.savefig('./Figures/Merger_tree_test_dm_all.png')
plt.show()