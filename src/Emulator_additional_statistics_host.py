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
import matplotlib as mpl
from scipy import stats

def universe_age(a,h,omega_m):
    omega_l = 1-omega_m

    a_integral = np.linspace(1e-4,1,2000)
    integrand = 1/(100*h*a_integral*(omega_m*a_integral**-3+omega_l)**0.5)

    t_age_tot = np.trapz(integrand,a_integral)

    a_sample = np.logspace(-2,0,100)
    t_sample = np.empty(len(a_sample))
    for i in range(len(t_sample)):
        cut = a_integral<a_sample[i]
        t_sample[i] = np.trapz(integrand[cut],a_integral[cut])

    #converstion factor to convert to years (9.78*10**11)
    t_sample*=9.78*10**11
    t_age_tot*=9.78*10**11

    t =t_age_tot - np.interp(a,a_sample,t_sample)

    return(t)


def sph_smooth(DM_mass_bins,DM_mass,mass,h=0.5):
    #order by dm mass
    
    #sort array
    sort=np.argsort(DM_mass)
    DM_mass=DM_mass[sort]
    mass=mass[sort]
    host_frac=np.zeros(len(DM_mass_bins))
    for i in range(len(DM_mass_bins)):
        select=[]
        select.append(np.searchsorted(DM_mass,np.array([DM_mass_bins[i]-h,DM_mass_bins[i]-h/2])))
        select.append(np.searchsorted(DM_mass,np.array([DM_mass_bins[i]+h/2,DM_mass_bins[i]+h])))
        select.append(np.searchsorted(DM_mass,np.array([DM_mass_bins[i]-h/2,DM_mass_bins[i]+h/2])))
        
        dist=[]
        mass_samp=[]
        for j in range(len(select)):
            dist.append(np.abs((DM_mass[select[j][0]:select[j][1]+1]-DM_mass_bins[i])/h))
            mass_samp.append(mass[select[j][0]:select[j][1]+1])
        

        weight=0.0
        for j in range(2):
            weight+=np.sum((2*(1-dist[j])**3)*mass_samp[j])
        j=2
        weight+=np.sum(1-6*dist[j]**2+6*dist[j]**3)
        
        host_frac[i]=weight
    return(host_frac)

def edge_fix(x_sample,pdf,edge,h):
    x_norm = (x_sample-edge)/h
    norm = np.ones(len(pdf))
    cut = (x_norm<=1) & (x_norm>=0.5)
    norm[cut] = 1-0.5 *(1-x_norm[cut])**4/(3/4)
    
    cut = (x_norm<0.5) & (x_norm>=0.0)
    norm[cut] = 1-(3/8-x_norm[cut]+2*x_norm[cut]**3-3/2*x_norm[cut]**4)/(3/4)

    pdf_copy = np.array(pdf)/norm
    return(pdf_copy)

sim_directory='/cosma7_old/data/dp004/dc-brow5/simulations/ARTEMIS/Latin_hyperube_2/'
halos=['halo_61','halo_32','halo_04']
L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
tests=np.loadtxt('./random_cube_2.txt')

num_snaps=30
param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
snap_name='029_z000p000'

em = emulator_build(param_label,L_cube,tests)
file_name,file_name_test=em.get_filename()
#file_name=[file_name[0]]
redshift=np.loadtxt(sim_directory+halos[0]+'/'+file_name[0]+'/redshift_list.txt')
tags=[]
for i in range(len(redshift)):
    tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))




#find host projenitor
# projen_sub=[]
# projen_fof=[]
# for k in range(len(halos)):
#     projen_sub.append(np.ones((len(file_name),len(tags)),dtype=int)*-1)
#     projen_fof.append(np.ones((len(file_name),len(tags)),dtype=int)*-1)
#     for i in range(len(file_name)):

#         #load subhalo numbers and group numbers
#         projen_sub_id = main_proj_corrections(sim_directory+halos[k]+'/'+file_name[i]+'/data',num_snaps,0)

#         for j in range(len(projen_sub_id)):
#             if projen_sub_id[j]==-1:
#                 break
#             group_num=E.read_array("SUBFIND",sim_directory+halos[k]+'/'+file_name[i]+'/data/',tags[-(j+1)],"Subhalo/GroupNumber",noH=False)-1
#             sub_group_num=E.read_array("SUBFIND",sim_directory+halos[k]+'/'+file_name[i]+'/data/',tags[-(j+1)],"Subhalo/SubGroupNumber",noH=False)

#             projen_sub[k][i,j]=sub_group_num[int(projen_sub_id[j])]
#             projen_fof[k][i,j]=group_num[int(projen_sub_id[j])]
#         projen_fof[k][i,:]=projen_fof[k][i,:][::-1]
#         projen_sub[k][i,:]=projen_sub[k][i,:][::-1]


# ###########################################################################
# #calculate moment of inertia 
# ###########################################################################
# for j in range(len(tags)):
#     proj_exist=np.zeros(len(halos))
#     for k in range(len(halos)):
#         sub_id=projen_sub[k][:,j]
#         fof_id=projen_fof[k][:,j]


#         if np.any(sub_id==-1):
#             continue

#         proj_exist[k]=1.0

#     print(np.all(proj_exist==1.0))

#     if np.all(proj_exist==1.0)==False:
#         continue
    
#     #need to also check for the existence of stars in all files
#     star=np.zeros((len(file_name),len(halos)),dtype=bool)
#     for k in range(len(halos)):
#         for i in range(len(file_name)):
            
#             h=h5.File(sim_directory+halos[k]+'/'+file_name[i]+'/data/particledata_'+tags[j]+'/eagle_subfind_particles_'+tags[j]+'.0.hdf5')
#             e = 'PartType4' in h
#             star[i,k]=e
    
#     print(star)
#     if np.any(star==False):
#         continue
    

#     print(tags[j])
#     b_a=np.empty((len(file_name),len(halos)))
#     c_a=np.empty((len(file_name),len(halos)))
    
#     for k in range(len(halos)):
#         for i in range(len(file_name)):

#             print(sim_directory+halos[k]+'/'+file_name[i])
#             print(j)
#             print(tags[j])
#             #load star particle data
            
#             pos=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/Coordinates',noH=False)
#             part_mass=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/Mass',noH=False)
#             group_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/GroupNumber',noH=False)
#             group_num=np.abs(group_num)-1
#             subgroup_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/SubGroupNumber',noH=False)

#             cop=E.read_array("SUBFIND",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'Subhalo/CentreOfPotential',noH=False)[projen_sub[k][i,j],:]
#             print(projen_fof[k][i,j],projen_sub[k][i,j])
#             cut = (group_num == projen_fof[k][i,j]) & (subgroup_num == projen_sub[k][i,j])

#             pos=pos[cut,:]
#             pos[:,0]-=cop[0]; pos[:,1]-=cop[1]; pos[:,2]-=cop[2]

#             part_mass=part_mass[cut]

#             M=np.zeros((3,3))
#             for l in range(3):
#                 for m in range(3):
#                     M[l,m]=np.sum(pos[:,l]*pos[:,m]*part_mass)

#             eig_val,_=np.linalg.eig(M)
#             eig_val=np.sort(eig_val)
            
#             b_a[i,k]=eig_val[1]/eig_val[2]
#             c_a[i,k]=eig_val[0]/eig_val[2]

#     if np.any(np.isnan(b_a)==True) or np.any(np.isnan(c_a==True)):
#         continue
    
    
#     em.train(b_a,'%03d'%j,halos,'Stellar_b_a','Moment of inertia b/a for main projenitors stellar particles',replace=True,train_seperataely=True)
#     em.train(c_a,'%03d'%j,halos,'Stellar_c_a','Moment of inertia c/a for main projenitors stellar particles',replace=True,train_seperataely=True)

# exit()



###########################################################################
#calculate the pdfs for stellar metalicities and ages
###########################################################################

h_met=0.2
h_t = 0.2
age_univ = universe_age(1e-2,0.7,0.2796)
met_bins=np.linspace(-7,0,100)
time_bins=np.linspace(0,np.log10(10),100)

num_bins = 30

j=-1
met_counts_individual=np.empty((len(file_name),len(halos),len(met_bins)))
met_counts_averaged=np.empty((len(file_name),len(met_bins)))
time_counts_individual=np.empty((len(file_name),len(halos),len(met_bins)))
time_counts_averaged=np.empty((len(file_name),len(met_bins)))

age_median = np.empty((len(file_name),num_bins-1))
age_up = np.empty((len(file_name),num_bins-1))
age_lo = np.empty((len(file_name),num_bins-1))
for i in range(len(file_name)):
    mass_all=np.array([])
    Metallicity_all=np.array([])
    star_form_time_all=np.array([])
    for k in range(len(halos)):
        print(sim_directory+halos[k]+'/'+file_name[i]+'/data')
        group_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/GroupNumber',noH=False)
        group_num=np.abs(group_num)-1
        subgroup_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/SubGroupNumber',noH=False)
        mass=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/Mass',noH=False)
        Metallicity=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/Metallicity',noH=False)
        star_form_time=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/StellarFormationTime',noH=False)
        
        #convert from scale factor to age of star particle
        star_form_time = 1/star_form_time #this is 1+z
        
        cut = (group_num == 0) & (subgroup_num == 0)

        mass=mass[cut]
        Metallicity=Metallicity[cut]
        star_form_time=star_form_time[cut]

        mass_all=np.append(mass_all,mass)
        Metallicity_all=np.append(Metallicity_all,Metallicity)
        star_form_time_all=np.append(star_form_time_all,star_form_time)

        num=sph_smooth(met_bins,np.log10(Metallicity[Metallicity>0]),mass[Metallicity>0],h=h_met)
        #normalise counts
        integral=np.trapz(num,met_bins)
        num/=integral

        met_counts_individual[i,k,:]=num

        num=sph_smooth(time_bins,np.log10(star_form_time),np.ones(len(star_form_time)),h=h_t)
        num = edge_fix(time_bins,num,0.0,h_t)

        #normalise counts
        integral=np.trapz(num,time_bins)
        num/=integral

        time_counts_individual[i,k,:]=num
    
    def median_84(x):
        perc  =np.nanpercentile(x,[50-68/2,50+68/2])   
        return(perc[1])

    def median_16(x):
        perc  =np.nanpercentile(x,[50-68/2,50+68/2])   
        return(perc[0])

    med, edges,_ = stats.binned_statistic(np.log10(Metallicity_all),np.log10(star_form_time_all),statistic=np.nanmedian,bins = np.linspace(-5,-1,num_bins))
    perc_84,edges,_ = stats.binned_statistic(np.log10(Metallicity_all),np.log10(star_form_time_all),statistic=median_84,bins = np.linspace(-5,-1,num_bins))
    perc_16,edges,_ = stats.binned_statistic(np.log10(Metallicity_all),np.log10(star_form_time_all),statistic=median_16,bins = np.linspace(-5,-1,num_bins))
    
    age_median[i,:] = med
    age_up[i,:] = perc_84
    age_lo[i,:] = perc_16
    
    num=sph_smooth(met_bins,np.log10(Metallicity_all[Metallicity_all>0]),mass_all[Metallicity_all>0],h=h_met)
    #normalise counts
    integral=np.trapz(num,met_bins)
    num/=integral

    met_counts_averaged[i,:]=num

    num=sph_smooth(time_bins,np.log10(star_form_time_all),np.ones(len(star_form_time_all)),h=h_t)
    num = edge_fix(time_bins,num,0.0,h_t)
    #normalise counts
    integral=np.trapz(num,met_bins)
    num/=integral

    time_counts_averaged[i,:]=num

age_up[np.isnan(age_up)]=0.0
age_lo[np.isnan(age_lo)]=0.0
age_median[np.isnan(age_median)]=0.0

em.train(met_counts_individual,'%03d'%29,[met_bins,halos],'Metalicity_distribution_individual','The mass weight PDFs for the metalicity of stars in the main halo, calauclated seperately for each system. Integral normalised to unity.',replace=True,train_seperataely=True)     
em.train(met_counts_averaged,'%03d'%29,met_bins,'Metalicity_distribution','The mass weight PDFs for the metalicity of stars in the main halo, averaged over all systems. Integral normalised to unity.',replace=True,train_seperataely=True)

em.train(time_counts_individual,'%03d'%29,[time_bins,halos],'Time_distribution_individual','The mass weight PDFs for the metalicity of stars in the main halo, calauclated seperately for each system. Integral normalised to unity.',replace=True,train_seperataely=True)     
em.train(time_counts_averaged,'%03d'%29,time_bins,'Time_distribution','The mass weight PDFs for the metalicity of stars in the main halo, averaged over all systems. Integral normalised to unity.',replace=True,train_seperataely=True)

m_bins = np.empty(len(edges)-1)
for i in range(len(m_bins)):
    m_bins[i] = (edges[i+1]+edges[i])/2
em.train(age_median,'%03d'%29,m_bins,'Age_met_median','The median stellat metallicity age relation.',replace=True,train_seperataely=True)     
em.train(age_up,'%03d'%29,m_bins,'Age_met_upper','The 84 percentile of the metallicity age relation.',replace=True,train_seperataely=True)
em.train(age_lo,'%03d'%29,m_bins,'Age_met_lower','The 16 percentile of the metallicity age relation.',replace=True,train_seperataely=True)
exit()



###########################################################################
#calculate insitu fractions
###########################################################################
ins=[]
for j in range(len(tags)):

    proj_exist=np.zeros(len(halos))
    for k in range(len(halos)):
        sub_id=projen_sub[k][:,j]
        fof_id=projen_fof[k][:,j]

        if np.any(sub_id==-1):
            continue

        proj_exist[k]=1.0


    if np.all(proj_exist==1.0)==False:
        continue

    star=np.zeros((len(file_name),len(halos)),dtype=bool)
    for k in range(len(halos)):
        for i in range(len(file_name)):
            
            h=h5.File(sim_directory+halos[k]+'/'+file_name[i]+'/data/particledata_'+tags[j]+'/eagle_subfind_particles_'+tags[j]+'.0.hdf5')
            e = 'PartType4' in h
            star[i,k]=e
    
    if np.any(star==False):
        continue
    

    print(tags[j])
    insit_frac=np.empty((len(file_name),len(halos)))
    for k in range(len(halos)):
        for i in range(len(file_name)):
            insit_precalc = np.loadtxt(sim_directory+halos[k]+'/'+file_name[i]+'/processed_data/instu_star_formation.txt')
            print(sim_directory+halos[k]+'/'+file_name[i]+'/data')
            print(tags[j])
            group_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/GroupNumber',noH=False)
            group_num=np.abs(group_num)-1
            subgroup_num=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/SubGroupNumber',noH=False)
            part_ID=E.read_array("PARTDATA",sim_directory+halos[k]+'/'+file_name[i]+'/data',tags[j],'PartType4/ParticleIDs',noH=False)
            cut = (group_num == projen_fof[k][i,j]) & (subgroup_num == projen_sub[k][i,j])
            insit_identification = insit_precalc[:,1][np.in1d(insit_precalc[:,0],part_ID[cut])]
            insit_frac[i,k] = np.sum(insit_identification==1)/(np.sum(insit_identification==1) + np.sum(insit_identification==0))
    
    ins.append(insit_frac)
    if np.isnan(insit_frac).any():
        continue
    

    em.train(insit_frac,'%03d'%j,halos,'Insitu_fraction','The insitu stellar fraction for the host, just looking at bound particles',replace=True,train_seperataely=True)

