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

import statsmodels.api as sm
lowess = sm.nonparametric.lowess


def sph_smooth(DM_mass_bins,DM_mass,host_halo,h=0.5):
        #order by dm mass
        
        
        host_frac=np.zeros(len(DM_mass_bins))
        for i in range(len(DM_mass_bins)):
            
            dist=np.abs((DM_mass-DM_mass_bins[i])/h)
            
            weight=np.zeros(len(dist))
            weight[dist<0.5]=1-6*dist[dist<0.5]**2+6*dist[dist<0.5]
            weight[dist>0.5]=2*(1-dist[dist>0.5])**3
            host_frac[i]=np.sum(weight[dist<=1]*host_halo[dist<=1])/np.sum(weight[dist<=1])
        return(host_frac)

if __name__=='__main__':
    sim_directory='/cosma7/data/dp004/dc-brow5/simulations/ARTEMIS/Latin_hyperube_2/'
    halos=['halo_61','halo_32']
    L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
    tests=np.loadtxt('./random_cube_2.txt')

    param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
    snap_name='029_z000p000'

    em = emulator_build(param_label,L_cube,tests)
    file_name,file_name_test=em.get_filename()


    # #test if simulations are there
    # sim_fin=np.empty(len(file_name),dtype=bool)
    # for i in range(len(file_name)):
    #     sim_fin[i]=os.path.exists(sim_directory+halos[0]+'/'+file_name[i]+'/data/groups_'+snap_name)

    # L_cube=L_cube[sim_fin,:]

    # sim_fin=np.empty(len(file_name_test),dtype=bool)
    # for i in range(len(file_name_test)):
    #     sim_fin[i]=os.path.exists(sim_directory+halos[0]+'/'+file_name_test[i]+'/data/groups_'+snap_name)
        
    # tests=tests[sim_fin,:]
    
    # em = emulator_build(param_label,L_cube,tests)
    # file_name,file_name_test=em.get_filename()

    h=h5.File(sim_directory+halos[0]+'/'+file_name[0]+'/data/snapshot_029_z000p000/snap_029_z000p000.0.hdf5')
    

    
    
    
    def star_form_hist(loc):
        redshift=np.loadtxt(loc+'/redshift_list.txt')
        tags=[]
        for i in range(len(redshift)):

            tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))
            
            
        #read merger tree
        
        
        h=h5.File(loc+'/data/merger_trees/tree_029.0.hdf5')
        
        ID=np.array(h['haloTrees/nodeIndex'])
        Main_prog=np.array(h['haloTrees/mainProgenitorIndex'])
        pos=np.array(h['haloTrees/positionInCatalogue'])
        mass=np.array(h['haloTrees/nodeMass'])
        redshifts=np.array(h['haloTrees/redshift'])

        ID_initial=int(29e12)
        ID_main=[]
        sub_id=np.ones(30)*(-1)
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
        metalicity=[]
        for i in tags:
            #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
            #print(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/Stars/Metallicity"))
            try:
                stellar_mass.append(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,4]*10**10)
                dm_mass.append(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/MassType",noH=False)[:,1]*10**10)
                metalicity.append(E.read_array("SUBFIND",loc+'/data',i,"Subhalo/Stars/Metallicity",noH=False))
            except:
                stellar_mass.append([])
                dm_mass.append([])
                metalicity.append([])
        
        # #walk the tree for the host
        # ind_i=0
        # stellar_mass_hist=np.ones(len(stellar_mass))*(-1)
        # dm_mass_hist=np.ones(len(dm_mass))*(-1)
        # for i in range(len(stellar_mass)):
        #     if i==0:
        #         stellar_mass_hist[i]=stellar_mass[-(i+1)][ind_i]
        #         dm_mass_hist[i]=dm_mass[-(i+1)][ind_i]
        #         continue

        #     descendent_ids=descendent[-(i)]


        #     print(descendent_ids)
        #     index=np.where(descendent_ids==ind_i)[0]
        #     print(index)
        #     print(snap_num[-i][index])
        #     if len(index)==0:
        #         break

        #     index=index[np.argmax(dm_mass[-(i+1)][index])]

        #     print(i,index)
        #     if i==7:
        #         exit()
            
        #     stellar_mass_hist[i]=stellar_mass[-(i+1)][index]
            
        #     dm_mass_hist[i]=dm_mass[-(i+1)][index]

        #     ind_i=index
        
        stellar_mass_hist=np.ones(len(stellar_mass))*(-1)
        dm_mass_hist=np.ones(len(dm_mass))*(-1)
        met_hist=np.ones(len(dm_mass))*(-1)
        for i in range(len(stellar_mass)):
            if sub_id[i]<0:
                continue

            stellar_mass_hist[i]=stellar_mass[i][np.argmax(dm_mass[i])]
            dm_mass_hist[i]=dm_mass[i][np.argmax(dm_mass[i])]
            met_hist[i]=metalicity[i][np.argmax(dm_mass[i])]


        return(stellar_mass_hist,dm_mass_hist,redshift,met_hist)

    #file_name=[file_name[3]]

    # stellar_mass=[]
    # dm_mass=[]
    # redshift=[]
    # met=[]
    # for i in range(len(file_name)):
    #     out=star_form_hist(sim_directory+file_name[i])
    #     print(out[0],out[1],out[2],out[3])
        
    #     stellar_mass.append(out[0])
    #     dm_mass.append(out[1])
    #     redshift.append(out[2])
    #     met.append(out[3])

    # fig=plt.figure()
    # for i in range(len(stellar_mass)):
    #     #plt.plot(redshift[i][:,1]+1,dm_mass[i])
    #     plt.plot(redshift[i][:,1]+1,stellar_mass[i])
    # plt.xscale('log')
    # plt.yscale('log')
    # #plt.show()
    # #fig.savefig('./Figures/Star_formation_history.png')
    # #exit()

    
    # stellar_mass_array=np.empty((len(stellar_mass),len(stellar_mass[0])))
    # met_array=np.empty((len(stellar_mass),len(stellar_mass[0])))
    # for i in range(len(stellar_mass)):
    #     stellar_mass_array[i,:]=stellar_mass[i]
    #     met_array[i,:]=met[i]

    # part_mass=2.23*10**4
    # stellar_mass_array[stellar_mass_array<=0]=part_mass
    # stellar_mass_array=np.log10(stellar_mass_array)
    
    # description='Redshift evolution of the host stellar mass, using MassType'
    # em.train(stellar_mass_array,redshift[0][:,1],'Stellar_mass_redshift', description,replace=True,train_seperataely=True)
    

    # part_mass=2.23*10**4
    # met_array[met_array<=0]=0.0
    
    # description='Redshift evolution of the host metalicity, using MassType'
    # em.train(met_array,redshift[0][:,1],'Metalicity_redshift', description,replace=True,train_seperataely=True)

    



    # j=0
    # fig=plt.figure()
    # ax1=fig.add_subplot()

    # for j in range(len(file_name)):
    #     group_num=E.read_array("SUBFIND",sim_directory+file_name[j]+'/data',snap_name,"Subhalo/GroupNumber",noH=False)-1
    #     stellar_mass=E.read_array("SUBFIND",sim_directory+file_name[j]+'/data',snap_name,"Subhalo/MassType",noH=False)[:,4]*10**10
    #     DM_mass=E.read_array("SUBFIND",sim_directory+file_name[j]+'/data',snap_name,"Subhalo/MassType",noH=False)[:,1]*10**10

    #     HalfMassRad=E.read_array("SUBFIND",sim_directory+file_name[j]+'/data',snap_name,"Subhalo/Stars/Metallicity",noH=False)

    #     part_mass=2.23*10**4
    #     DM_mass_bins=np.logspace(np.log10(part_mass*10),9,10)
    #     # DM_mass=DM_mass[1:]
    #     # stellar_mass=stellar_mass[1:]
    #     # HalfMassRad=HalfMassRad[1:]

    #     stellar_mass=stellar_mass[DM_mass!=0]
    #     HalfMassRad=HalfMassRad[DM_mass!=0]
    #     DM_mass=DM_mass[DM_mass!=0]

    #     HalfMassRad=HalfMassRad[stellar_mass!=0]
    #     DM_mass=DM_mass[stellar_mass!=0]
    #     stellar_mass=stellar_mass[stellar_mass!=0]

    #     sort=np.argsort(stellar_mass)
    #     DM_mass=DM_mass[sort]
    #     stellar_mass=stellar_mass[sort]
    #     HalfMassRad=HalfMassRad[sort]

        
    #     half_light=sph_smooth(np.log10(DM_mass_bins),np.log10(stellar_mass),np.log10(HalfMassRad),h=0.75)

        
    #     ax1.plot(DM_mass_bins,half_light)
    #     ax1.plot(stellar_mass,np.log10(HalfMassRad),'.')
    #     ax1.set_xscale('log')
    #     #ax1.set_yscale('log')
        
        


        
    # plt.show()
    # exit()
        
        
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    # exit()


    def halo_ocup_frac(loc,sim_names,tag,m_sample):

        
        host_frac=np.empty((len(sim_names),len(m_sample)))
        for j in range(len(sim_names)):
            group_num=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/GroupNumber",noH=False)-1
            stellar_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,4]*10**10
            DM_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,1]*10**10

            DM_mass=DM_mass[1:]
            stellar_mass=stellar_mass[1:]
            stellar_mass=stellar_mass[DM_mass!=0]
            DM_mass=DM_mass[DM_mass!=0]
            sort=np.argsort(DM_mass)
            DM_mass=DM_mass[sort]
            stellar_mass=stellar_mass[sort]
            host_halo=(stellar_mass>0).astype(np.int)
            

            host_frac[j,:]=sph_smooth(np.log10(m_sample),np.log10(DM_mass),host_halo,h=0.75)

        return(host_frac)


    def halo_stellar_mass(loc,sim_names,tag,m_sample):

        
        M_st_sample=np.empty((len(sim_names),len(m_sample)))
        for j in range(len(sim_names)):
            group_num=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/GroupNumber",noH=False)-1
            stellar_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,4]*10**10
            DM_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,1]*10**10

            #cut to only include haloes with DM and stars
            stellar_mass=stellar_mass[group_num==0][1:]
            DM_mass=DM_mass[group_num==0][1:]
            DM_mass=DM_mass[stellar_mass>0]
            stellar_mass=stellar_mass[stellar_mass>0]
            stellar_mass=stellar_mass[DM_mass>0]
            DM_mass=DM_mass[DM_mass>0]

            #lowess smoothing in future
            #z=lowess(np.log10(stellar_mass),np.log10(DM_mass))
            coeff=np.polyfit(np.log10(DM_mass),np.log10(stellar_mass),1)
            
            M_st_sample[j,:]=np.log10(10**coeff[1]*m_sample**coeff[0])
            
        return(M_st_sample)

    #generate training data and train
    def satellite_count_stellar(loc,sim_names,tag):

        part_mass=2.23*10**4
        part_mass_dm=1.17*10**5
        M_bins=np.logspace(np.log10(part_mass),9.5,10)
        M_bins_dm=np.logspace(np.log10(part_mass_dm),11,10)

        count=np.zeros((len(sim_names),len(M_bins)))
        count_dm=np.zeros((len(sim_names),len(M_bins)))
        for j in range(len(sim_names)):
            print(sim_names[j])
            group_num=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/GroupNumber",noH=False)-1
            stellar_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,4]*10**10
            dm_mass=E.read_array("SUBFIND",loc+sim_names[j]+'/data',tag,"Subhalo/MassType",noH=False)[:,1]*10**10
            stellar_mass=stellar_mass[group_num==0][1:] #discount central
            dm_mass=dm_mass[group_num==0][1:]

            
            for i in range(len(M_bins)):
                count[j,i]=np.sum(stellar_mass>M_bins[i])
                count_dm[j,i]=np.sum(dm_mass>M_bins_dm[i])

        return(count,count_dm,M_bins,M_bins_dm)

    #########################################################################
    #train stellar mass function

    
    
    #########################################################################
    #train the total satellite counts, satellite galaxy stellar mass function
    counts,counts_dm,M_bins,M_bins_dm=satellite_count_stellar(sim_directory+halos[0]+'/',file_name,snap_name)
    counts_test,counts_dm_test,_,_=satellite_count_stellar(sim_directory+halos[0]+'/',file_name_test,snap_name)

    counts1,counts_dm1,M_bins,M_bins_dm=satellite_count_stellar(sim_directory+halos[1]+'/',file_name,snap_name)
    counts_test1,counts_dm_test1,_,_=satellite_count_stellar(sim_directory+halos[1]+'/',file_name_test,snap_name)
    
    counts=(counts+counts1)/2; counts_dm=(counts_dm+counts_dm1)/2
    counts_test=(counts_test+counts_test1)/2; counts_dm_test=(counts_dm_test+counts_dm_test1)/2
    
    #build emulators and test
    stat='Sat_stellar_mass_func'
    description='Cumulative satellite stellar mass function, using subfind bound stellar mass and all satellite within FOF group'
    em.train(counts,'029',M_bins,stat, description,replace=True,train_seperataely=True)
    #deg_freedom,chi22,frac_err,error=em.test(counts_test,stat)

    exit()
    
    # stat='Sat_dm_mass_func'
    # description='Cumulative satellite dm mass function, using subfind bound dm mass and all satellite within FOF group'
    # em.train(counts_dm,M_bins_dm,stat, description,replace=True,train_seperataely=True)
    # deg_freedom,chi22,frac_err,error=em.test(counts_dm_test,stat)

    # ########################################################################
    # #train halo mass stellar mass relation
    part_mass=2.23*10**4
    M_bins=np.logspace(np.log10(part_mass*10),9,10)
    M_st=halo_stellar_mass(sim_directory,file_name,snap_name,M_bins)
    M_st_test=halo_stellar_mass(sim_directory,file_name_test,snap_name,M_bins)
    stat='Stellar_mass_halo_mass'
    description='Smoothed stellar mass halo mass relation, predicting log(M_*) in unit of h^-1 M_sun'
    em.train(M_st,M_bins,stat, description,replace=True,train_seperataely=True)
    deg_freedom,chi22,frac_err,error=em.test(M_st_test,stat)
    

    # #########################################################################
    # #train the fraction of ocupied haloes
    # part_mass=1.17*10**5
    # M_bins=np.logspace(np.log10(part_mass*10),9,10)
    # host_frac=halo_ocup_frac(sim_directory,file_name,snap_name,M_bins)
    # host_frac_test=halo_ocup_frac(sim_directory,file_name_test,snap_name,M_bins)

    # #log and deal with zeros
    # #host_frac=np.log10(host_frac); host_frac_test=np.log10(host_frac_test)

    # # host_frac[np.isnan(host_frac)]=np.nanmin(host_frac)
    # # host_frac_test[np.isnan(host_frac_test)]=np.nanmin(host_frac_test)

    # stat='Gal_host_frac'
    # description='Smoothed fraction of DM haloes hosting a galaxy, (due to particle mass limit this is a galaxy with M_*>2.23*10**4 h^-1M_sun)'
    # em.train(host_frac,M_bins,stat, description,replace=True,train_seperataely=True)
    # deg_freedom,chi22,frac_err,error=em.test(host_frac_test,stat)
    

    #########################################################################
    #train the sizes of haloes
    # part_mass=2.23*10**4
    # M_bins=np.logspace(np.log10(part_mass*10),9,10)
    # host_frac=half_light_rad(sim_directory,file_name,snap_name,M_bins)
    # host_frac_test=half_light_rad(sim_directory,file_name_test,snap_name,M_bins)
    # stat='Half_stellar_mass_rad'
    # description='Logged smoothed average satellite galaxy size, as described through the half mass radius of the stars'
    # em.train(host_frac,M_bins,stat, description,replace=True,train_seperataely=True)
    # deg_freedom,chi22,frac_err,error=em.test(host_frac_test,stat)


    #########################################################################
    #train the metalicity of satellites
    part_mass=2.23*10**4
    M_bins=np.logspace(np.log10(part_mass*10),9,10)
    host_frac=matallicity(sim_directory,file_name,snap_name,M_bins)
    host_frac_test=matallicity(sim_directory,file_name_test,snap_name,M_bins)
    stat='Sat_Metallicity'
    description='Smoothed average satellite galaxy metalicity'

    em.train(host_frac,M_bins,stat, description,replace=True,train_seperataely=True)
    deg_freedom,chi22,frac_err,error=em.test(host_frac_test,stat)



    
    


