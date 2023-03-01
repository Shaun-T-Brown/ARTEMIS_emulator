import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import dill
import h5py as h5
import eagle_IO.eagle_IO as E
import os 
from matplotlib.colors import LogNorm
from itertools import compress

class emulator_build:
    def __init__(self,labels,L_cube,test):
        #provide the hypercube sampling used, assume limits between 0 and 1
        self.nodes = L_cube
        self.nodes_test = test
        self.ndims = L_cube.shape[1]
        
        self.limit = np.ones((self.ndims,2))
        self.limit[:,0]=0

        self.paramter = labels

        self.training_filepath = './Training_data/'
        self.emulator_filepath = './Emulators/'


    def get_filename(self):
        nodes_eagle = rescale(self.nodes,return_eagle=True)[1]

        file_name=[]
        for i in range(len(nodes_eagle)):
            file_name.append('WDMm%.2f_fmin%.2f_fmax%.2f_starthresh%.2f_nH%.2f_reion%.2f'%(nodes_eagle[i,0],nodes_eagle[i,1],nodes_eagle[i,2],nodes_eagle[i,3],nodes_eagle[i,4],nodes_eagle[i,5]))
        
        nodes_eagle = rescale(self.nodes_test,return_eagle=True)[1]
        file_name_test=[]
        for i in range(len(nodes_eagle)):
            file_name_test.append('WDMm%.2f_fmin%.2f_fmax%.2f_starthresh%.2f_nH%.2f_reion%.2f'%(nodes_eagle[i,0],nodes_eagle[i,1],nodes_eagle[i,2],nodes_eagle[i,3],nodes_eagle[i,4],nodes_eagle[i,5]))
        return(file_name,file_name_test)

    def train(self,data,tag,x=[],statistic='',description='',replace=False,train_seperataely=False,seperate_sampling = None):

        #data can be an ND array, but first dimension must be the sampling of the 
        #L_cube

        #check that array saixe if appropriate

        if type(data)==np.ndarray: #if not numpy array assume list for seperate sampling
            shape=data.shape
            if shape[0]!=self.nodes.shape[0]:
                print('Error: first dimsion of array must match the sampling shape')
                exit()

        #deal with the list of statistics and their descriptions
        with open('statistics.txt','r') as f:
            stat = f.readlines()
        with open('statistic_descriptions.txt','r') as f:
            desc = f.readlines()

        
        statistic_exist=False
        for i in range(len(stat)):
            if stat[i].rstrip()==statistic:
                statistic_exist=True 
                stat_index=i
                break
        if statistic_exist==True and replace==False:
            print('Statistic already exists, set replace=True if you want to overwrite')
            exit() #need to ypdate to a warning

        if statistic_exist==True and replace==True:
            print('Statistic exists, overwriting!')

        #add statistic to list if it doesn't already exist
        if statistic_exist==False:
            with open('statistics.txt', 'a') as myfile:
                myfile.write(statistic+'\n')
            with open('statistic_descriptions.txt', 'a') as myfile:
                myfile.write(description+'\n')
        #update description
        if statistic_exist==True:
            desc[stat_index]=description+'\n'
            with open('statistic_descriptions.txt', 'w') as f:
                for line in desc:
                    f.write(f"{line}")
        
        if np.any(seperate_sampling==None):
            if train_seperataely==False:
                #normalise data
                norm=np.max(data)-np.min(data)
                data_min=np.min(data)
                if norm==0:
                    norm=1.0
                data=(data-data_min)/norm
                norms=[norm,data_min]

                #reshape array
                data=np.reshape(data,(shape[0],-1))

                #kernel if going to be an initial guess of width 0.5, and minimal noise 
                kernel = 0.5**2*Matern(length_scale=np.ones(self.ndims)*0.5, length_scale_bounds=(1e-2, 1e2),nu=2.5)\
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e2))

                gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100)
                gpr.fit(self.nodes, data)

            elif train_seperataely==True:
                #flatten array
                data=np.reshape(data,(shape[0],-1))

                num_stat=data.shape[1]
                gpr=[]
                norms=[]
                for k in range(num_stat):
                    dat=data[:,k]

                    norm=np.max(dat)-np.min(dat)
                    data_min=np.min(dat)
                    if norm==0:
                        norm=1.0
                    dat=(dat-data_min)/norm
                    norms.append([norm,data_min])


                    #kernel if going to be an initial guess of width 0.5, and minimal noise 
                    kernel = 0.5**2*Matern(length_scale=np.ones(self.ndims)*0.5, length_scale_bounds=(1e-2, 1e2),nu=2.5)\
                    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e2))

                    gpr.append( GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100))
                    gpr[-1].fit(self.nodes, dat)

        else:
            #assume seperate sampling for each halo. Here need tp update L_cube each time
            #assume everything as list here

            #must be trained seperately in this case 

            gpr=[]
            norms=[]
            for k in range(len(data)):

                nodes_sample = self.nodes[seperate_sampling[k],:]
                
                dat=data[k]

                #print(dat.shape)
                #print(nodes_sample.shape)
                norm=np.max(dat)-np.min(dat)
                data_min=np.min(dat)
                if norm==0:
                    norm=1.0
                dat=(dat-data_min)/norm
                norms.append([norm,data_min])


                #kernel if going to be an initial guess of width 0.5, and minimal noise 
                kernel = 0.5**2*Matern(length_scale=np.ones(self.ndims)*0.5, length_scale_bounds=(1e-2, 1e2),nu=2.5)\
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e2))

                gpr.append( GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100))
                gpr[-1].fit(nodes_sample, dat)

        #now save the data and Guassian process
        with open(self.emulator_filepath+statistic+tag+'.pickle', 'wb') as file:
            dill.dump(gpr, file)

        with open(self.emulator_filepath+statistic+tag+'_normalisation.pickle', 'wb') as file:
            dill.dump(norms, file)

        # with open(self.emulator_filepath+statistic+tag+'_shape.pickle', 'wb') as file:
        #     dill.dump(shape, file)

        with open(self.emulator_filepath+statistic+tag+'_x.pickle', 'wb') as file:
            dill.dump(x, file)

        with open(self.training_filepath+statistic+tag+'.pickle', 'wb') as file:
            dill.dump(data, file)


    def test(self,data_test,stat,tag):


        em=emulator()
        prediction,error=em.predict(stat,tag,self.nodes_test,return_std=True)

        #normalise test data
        
        deg_freedom=len(data_test)

        chi2=np.sum((data_test-prediction)**2/error**2,axis=0)

        frac_err=np.sum((data_test-prediction)**2,axis=0)**0.5/deg_freedom

        std=np.std(data_test-prediction,axis=0)
        
        #save chi2
        with open(self.training_filepath+stat+tag+'_errors.pickle', 'wb') as file:
            dill.dump([deg_freedom,chi2,frac_err,std], file)

        with open(self.training_filepath+stat+tag+'_testing_data.pickle', 'wb') as file:
            dill.dump(data_test, file)

        return(deg_freedom,chi2,frac_err,std)

    

class emulator:
    def __init__(self):
        #get file location for emulator folder
        file_path=os.path.realpath(__file__).split('/')
        self.loc=''
        for i in range(len(file_path)-1):
            self.loc=self.loc+'/'+file_path[i]
        self.loc=self.loc+'/'

        #load redshift list
        self.redshift_all = np.loadtxt(self.loc+'redshift_list.txt')

        #load data upon init
        with open(self.loc+'statistics.txt','r') as f:
            stat = f.read().splitlines() 
        
        with open(self.loc+'statistic_descriptions.txt','r') as f:
            desc = f.read().splitlines() 


        self.Guassian_proc={}
        self.normalisation={}
        self.trained_sep={}
        self.x_data={}
        self.errors={}
        self.redshifts={}
        self.snapshots={}
        self.description={}
        self.statistics=stat.copy()
        
        #load all descriptions
        for i in range(len(stat)):
            self.description[stat[i]]=desc[i]
        # for i in range(len(stat)):

        #     with open(self.loc+'Emulators/'+stat[i]+'.pickle', 'rb') as j:
        #         data = dill.load(j)
        #     with open(self.loc+'Emulators/'+stat[i]+'_normalisation.pickle', 'rb') as j:
        #         norm = dill.load(j)
        #     with open(self.loc+'Emulators/'+stat[i]+'_x.pickle', 'rb') as j:
        #         x_data = dill.load(j)

        #     #check if errors have been calculated
        #     if os.path.exists(self.loc+'Training_data/'+stat[i]+'_errors.pickle'):
        #         with open(self.loc+'Training_data/'+stat[i]+'_errors.pickle', 'rb') as j:
        #             errors = dill.load(j)
        #     else:
        #         errors = None

            
        #     self.Guassian_proc[stat[i]]=data
        #     self.normalisation[stat[i]]=norm
        #     self.x_data[stat[i]]=x_data
        #     self.errors[stat[i]]=errors
        #     self.description[stat[i]]=desc[i]

        #     if type(data) is list:
        #         self.trained_sep[stat[i]]=True
        #     else:
        #         self.trained_sep[stat[i]]=False

    def load_stat(self,stat,verbose=True):

        #first check if statistic exists
        for i in self.Guassian_proc.keys():
            if stat == i:
                if verbose==True:
                    print('Statistic already loaded')
                return(0)

        #check if statistic exists
        exist=False
        for i in range(len(self.statistics)):
            if self.statistics[i] == stat:
                exist=True 
                break 

        if exist == False:
            if verbose==True:
                print('Statistic does not exist')
            return(1)


        #check wich redshifts are present
        redshift_calc = np.zeros(len(self.redshift_all),dtype=bool)
        self.snapshots[stat] = []
        for i in range(len(self.redshift_all)):
            #print(self.loc+'Emulators/'+stat+'%03d.pickle'%i)
            redshift_calc[i] = os.path.exists(self.loc+'Emulators/'+stat+'%03d.pickle'%i)
            
            if redshift_calc[i] == True:
                self.snapshots[stat].append(i)

        self.redshifts[stat] = self.redshift_all[redshift_calc]

        
        # load data for all available redshifts
        self.Guassian_proc[stat]=[]
        self.normalisation[stat]=[]
        self.x_data[stat]=[]
        
        for i in range(len(self.snapshots[stat])):
            with open(self.loc+'Emulators/'+stat+'%03d.pickle'%self.snapshots[stat][i], 'rb') as j:
                data = dill.load(j)
            with open(self.loc+'Emulators/'+stat+'%03d_normalisation.pickle'%self.snapshots[stat][i], 'rb') as j:
                norm = dill.load(j)
            with open(self.loc+'Emulators/'+stat+'%03d_x.pickle'%self.snapshots[stat][i], 'rb') as j:
                x_data = dill.load(j)

            self.Guassian_proc[stat].append(data)
            self.normalisation[stat].append(norm)
            self.x_data[stat].append(x_data)

            #check if errors have been calculated
            self.errors[stat]=[]
            if os.path.exists(self.loc+'Training_data/'+stat+'%i_errors.pickle'%self.snapshots[stat][i]):
                with open(self.loc+'Training_data/'+stat[i]+'%i_errors.pickle'%self.snapshots[stat][i], 'rb') as j:
                    errors = dill.load(j)
            else:
                errors = None 
            self.errors[stat].append(errors)

        #set if data was trained together or seperately
        if type(self.Guassian_proc[stat][-1]) is list:
            self.trained_sep[stat]=True
        else:
            self.trained_sep[stat]=False

        return(0)

            



    def predict(self,stat,params,redshift,return_std=False,return_x=False,normalised=True,verbose=True):
        
        #load data
        load_correct=self.load_stat(stat,verbose=verbose)
        if load_correct == 1:
            if verbose==True:
                print('Statistic does not exist, not loaded correctly')
            return()

        #check redshifts to return

        #first check if redshift is iterable
        iterable=True
        try:
            some_object_iterator = iter(redshift)
        except TypeError as te:
            iterable=False

        
        if iterable == False:
            index = [np.argmin(np.abs(self.redshifts[stat][:,1]-redshift))]
            redshift_sample = [self.redshifts[stat][:,1][np.argmin(np.abs(self.redshifts[stat][:,1]-redshift))]]
            if verbose==True:
                print('Sampling redshift %.2f'%redshift_sample[0])

        if iterable == True:
            index = np.where((self.redshifts[stat][:,1]<=redshift[1]) & (self.redshifts[stat][:,1]>=redshift[0]))[0]
            redshift_sample = self.redshifts[stat][:,1][index]
            if verbose==True:
                print('Sampling between redshift %.2f and %.2f'%(np.min(redshift_sample),np.max(redshift_sample)))


        #convert if not in emulator coordinates 
        if normalised==False:
            params=rescale(params,normalised=False)

        #warn incase asking to predict outside of emulation range
        if np.min(params)<0 or np.max(params)>1:
            print('Warning: prediction out of emulation range, accuracy may be reduced')
        
        
        #get length of data that is predicted
        if self.trained_sep[stat]==False:

            stat_len_predict=self.Guassian_proc[stat][index[0]].predict(params[0,:].reshape(1,-1)).shape
            
            if len(stat_len_predict)==1:
                stat_len = 1
            else:
                stat_len = stat_len_predict[1]
        else:
            stat_len=len(self.Guassian_proc[stat][index[0]])

        #need to consider reshaping data if necesary!!!!!!!!!

        data=np.empty((len(redshift_sample),len(params),stat_len)) #currently assume single statistic
        #print(self.Guassian_proc[stat][index[0]].predict(params[0,:].reshape(1,-1)).shape)
        #print(self.trained_sep[stat])
        #print(data.shape)

        std=np.empty((len(redshift_sample),len(params),stat_len))
        for i in range(len(index)):
            #need to consider reshaping data if necesary
            if self.trained_sep[stat]==False:
                dat,sigma=self.Guassian_proc[stat][index[i]].predict(params,return_std=True)

                if len(dat.shape)==1:
                    data[i,:,0]=dat*self.normalisation[stat][index[i]][0]+self.normalisation[stat][index[i]][1]
                    std[i,:,0]=sigma*self.normalisation[stat][index[i]][0]
                else:
                    data[i,:,:]=dat*self.normalisation[stat][index[i]][0]+self.normalisation[stat][index[i]][1]
                    std[i,:,:]=sigma*self.normalisation[stat][index[i]][0]
                    

            else: 
                num_stat=len(self.Guassian_proc[stat][index[i]])
                for j in range(num_stat):
                    dat,sigma=self.Guassian_proc[stat][index[i]][j].predict(params,return_std=True)
                    data[i,:,j]=(dat*self.normalisation[stat][index[i]][j][0]+self.normalisation[stat][index[i]][j][1])
                    std[i,:,j]=(sigma*self.normalisation[stat][index[i]][j][0])

        #squeeze arrays to simplify
        data=np.squeeze(data)
        std=np.squeeze(std)
        redshift_sample=np.squeeze(redshift_sample)


        #deal with what data to return
        ret=(data,redshift_sample)
        if return_std==True:
            ret=ret+(std,)
        
        #this last step assumes 'x data' is same at all redshifts
        if return_x==True:
            ret=ret+(self.x_data[stat][0],)

        return(ret)

    def get_error(self,stat):

        return(self.errors[stat])

    def list_stats(self,stat=None):

        if stat==None:
            return(self.statistics)

        else:
            stats=[]
            for i in range(len(self.statistics)):
                if stat.casefold() in self.statistics[i].casefold():
                    stats.append(self.statistics[i])
            return(stats)

def main_proj(loc,tags):

    sub_id=np.ones(len(tags),dtype=int)*-1
    group_id=np.ones(len(tags),dtype=int)*-1
    for i in range(len(tags)):
        try:
            dm_mass=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/MassType",noH=False)[:,1]
            group_num=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/GroupNumber")-1
            sub_group_num=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/SubGroupNumber")
        except:
            continue

        sub_id[i]=int(sub_group_num[np.argmax(dm_mass)])
        group_id[i] = int(group_num[np.argmax(dm_mass)])

    group_id[-1]=0
    sub_id[-1]=0
    return(sub_id,group_id)

def main_proj_corrections(loc,num_snaps,sub_id):
    
    h=h5.File(loc+'/merger_trees_new/tree_029.0.hdf5','r')

    ID=np.array(h['haloTrees/nodeIndex'])
    desc_ind=np.array(h['haloTrees/descendantIndex'])
    mass=np.array(h['haloTrees/nodeMass'])
    mass_branch = np.array(h['haloTrees/branchMass'])
    snap_num = np.array(h['haloTrees/snapshotNumber'])
    fof_id = np.array(h['haloTrees/fofIndex'])
    fof_centre = np.array(h['haloTrees/isFoFCentre'])

    #start with host
    ind=ID==((num_snaps-1)*1e12+sub_id)
    ID_current = ((num_snaps-1)*1e12+sub_id)

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

    #now correct tree for discontinuities
    difference=np.empty(len(mass_proj))
    mass_proj_pad = np.pad(mass_proj,1,mode='edge')
    #identify discontinuity
    for i in range(len(mass_proj)):
        difference[i] = (np.log10(mass_proj_pad[i])+np.log10(mass_proj_pad[i+2]))/2-np.log10(mass_proj_pad[i+1])
    diff_cut=0.4



    cut = np.where(difference>=diff_cut)[0]

    for i in range(len(cut)):

        ID_current = ID_branch[cut[i]]    
        #overwrite to be main projenitor
        ind = ID==ID_current

        central_ind = (snap_num==snap_num[ind]) & (fof_id==fof_id[ind]) & (fof_centre==1)

        ID_branch[cut[i]] = ID[central_ind]
        mass_proj[cut[i]] = mass[central_ind]
    
    #convert from D-halos ID to sub_ids
    sub_ids=np.ones(len(ID_branch))*-1
    sub_ids[ID_branch!=-1] = np.remainder(ID_branch[ID_branch!=-1],1e12)

    return(sub_ids)



def main_proj(loc,tags):

    sub_id=np.ones(len(tags),dtype=int)*-1
    group_id=np.ones(len(tags),dtype=int)*-1
    for i in range(len(tags)):
        try:
            dm_mass=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/MassType",noH=False)[:,1]
            group_num=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/GroupNumber")-1
            sub_group_num=E.read_array("SUBFIND",loc+'/data',tags[i],"Subhalo/SubGroupNumber")
        except:
            continue

        sub_id[i]=int(sub_group_num[np.argmax(dm_mass)])
        group_id[i] = int(group_num[np.argmax(dm_mass)])

    group_id[-1]=0
    sub_id[-1]=0
    return(sub_id,group_id)

def rescale(paramters,return_eagle=False,normalised=True):
        #function to convert from the normalised coordinates to values calucalted

        param_limits=[]
        param_limits.append([0.0,0.6])
        param_limits.append([np.log10(0.5),np.log10(14)])
        param_limits.append([np.log10(0.1/3),np.log10(0.1*3)])
        param_limits.append([-0.075,4])
        param_limits.append([5.0,20.0])

        param_scaled=np.zeros(paramters.shape)

        #deal with mass sampling first, this is the most complex
        x0=0.3; m0=5.0; m_min=1.0
        a=(m_min-m0)/(1.0-x0); b=m_min-a; c=1.0/(x0*(a*x0+b))

        if normalised==True:
            #new mass sampling
            param_scaled[:,0][paramters[:,0]>=x0]=paramters[:,0][paramters[:,0]>=x0]*a+b
            param_scaled[:,0][paramters[:,0]<x0]=1/(c*paramters[:,0][paramters[:,0]<x0])


            for i in range(len(param_limits)):
                param_scaled[:,i+1]=paramters[:,i+1]*(param_limits[i][1]-param_limits[i][0])+param_limits[i][0]

            #SNII_rhogas_physdensnormfac is to be sampled loagrithmically, however parameter file is not log

            params_eagle=np.array(param_scaled)

            params_eagle[:,2]=10**param_scaled[:,2]
            params_eagle[:,1]=10**param_scaled[:,2]*param_scaled[:,1]
            params_eagle[:,3]=10**params_eagle[:,3]
            params_eagle[:,4]=10**params_eagle[:,4]

            if return_eagle==False:
                return(param_scaled)
            elif return_eagle==True:
                return(param_scaled,params_eagle)

        elif normalised==False:
            param_scaled[:,0][paramters[:,0]<=m0]=(paramters[:,0][paramters[:,0]<=m0]-b)/a
            param_scaled[:,0][paramters[:,0]>m0]=1.0/paramters[:,0][paramters[:,0]>m0]

            for i in range(len(param_limits)):
                param_scaled[:,i+1]=(paramters[:,i+1]-param_limits[i][0])/(param_limits[i][1]-param_limits[i][0])

            return(param_scaled)

def rescale2(paramters,return_eagle=False):
        return(paramters,paramters)


def get_data(sim_directory,file_name,tag,sub_id,fof_id):
    data=[]
    statistics=[]
    descriptions=[]

    #get subahlo and group number
    path=sim_directory+file_name[0]+'/data'
    print(path,tag)
    group_num = E.read_array("SUBFIND",path,tag,"Subhalo/GroupNumber",noH=False)-1
    sub_num = E.read_array("SUBFIND",path,tag,"Subhalo/SubGroupNumber",noH=False)
    
    array_loc = (group_num==fof_id[0]) & (sub_num==sub_id)
    k=0
    for i in keys:
        print(i)
        #read data to see shape etc
        
        path=sim_directory+file_name[0]+'/data'

        #this line needs updating when we are looking at more than just the host
        if prefix[k]=='FOF groups:':
            data_e=E.read_array("SUBFIND_GROUP",path,tag,"FOF/%s"%i,noH=False)[0,...]
        else:
            data_e=E.read_array("SUBFIND",path,tag,"Subhalo/%s"%i,noH=False)[0,...]
        
        

        dat_array=np.zeros((len(file_name),)+data_e.shape)
        for j in range(len(file_name)):
            path=sim_directory+file_name[j]+'/data'
            group_num = E.read_array("SUBFIND",path,tag,"Subhalo/GroupNumber",noH=False)-1
            sub_num = E.read_array("SUBFIND",path,tag,"Subhalo/SubGroupNumber",noH=False)
            
            array_loc = (group_num==fof_id[j]) & (sub_num==sub_id[j])
            
            if prefix[k]=='FOF groups:':
                dat_array[j,...] = E.read_array("SUBFIND_GROUP",path,tag,"FOF/%s"%i,noH=False)[fof_id[j],...]
            else:

                print(E.read_array("SUBFIND",path,tag,"Subhalo/%s"%i,noH=False).shape)
                print(array_loc)
                dat_array[j,...] = E.read_array("SUBFIND",path,tag,"Subhalo/%s"%i,noH=False)[array_loc,...]
        
        #deal with logging and splitting for different particle types
        log_data=False
        for j in log_phrases:
            if j.casefold() in i.casefold():
                log_data=True 
                break

        mass_data=False
        for j in mass_phrases:
            if j.casefold() in i.casefold():
                mass=True 
                break 
        #deal with edge cases for when incorectly identified statistic
        if np.sum(dat_array>0)==0:
            log_data=False; mass_data=False
        print(i,log_data)
        
        

        if mass_data==True and log_data==True:
            dat_array[dat_array==0.0]=part_mass
            dat_array=np.log10(dat_array)

        if mass_data==False and log_data==True:
            print(i)
            min_val=np.min(dat_array[dat_array>0])
            dat_array[dat_array<=0]=min_val
            dat_array=np.log10(dat_array)

        #identify if splitting as parttpye
        parttype_split=False
        if len(data_e.shape)>=1:
            if data_e.shape[0]==6:
                parttype_split=True

        #define suffix for if data is logged
        suffix=''
        if log_data==True:
            suffix=', log'
        #train emualtor
        if parttype_split==False:
            description='%s %s%s'%(prefix[k],i,suffix)
            name='%s'%(i)
            statistics.append(name.replace('/','_'))
            data.append(dat_array)
            descriptions.append(description)
            #em.train(dat_array,np.array([0]),name.replace('/','_'), description,replace=True)

        if parttype_split==True:
            
            for p in range(6):
                data_tr=dat_array[:,p]
                description='%s %s/PartType%d%s'%(prefix[k],i,p,suffix)
                name='%s/PartType%d'%(i,p)
                #em.train(data_tr,np.array([0]),name.replace('/','_'), description,replace=True)
        
                data.append(data_tr)
                statistics.append(name.replace('/','_'))
                descriptions.append(description)
        k+=1

    return(data,statistics,descriptions)

def check_sim(halos,file_name,tag):
    exist = np.zeros((len(halos),len(file_name)),dtype=bool)

    for i in range(len(halos)):
        for j in range(len(file_name)):
            exist[i,j] = os.path.exists(sim_directory+halos[i]+'/'+file_name[j]+'/data/particledata_'+tag)
    return(exist)

if __name__=='__main__':
    sim_directory='/cosma7/data/dp004/dc-brow5/simulations/ARTEMIS/Latin_hyperube_2/'
    halos=['halo_61','halo_32','halo_04']
    L_cube=np.loadtxt('./Latin_hypercube_D6_N25_strength2_v2.txt')
    tests=np.loadtxt('./random_cube_2.txt')

    param_label = ['WDM mass, m_DM', 'A, fmin=A*fmax', 'Max stellar feedack efficiency, fmax','Star formation threshold, n^*_H','Stellar feedback transition scale, n_H0', 'Reionisation redshift, z']
    snap_name='029_z000p000'
    num_snaps=30

    em = emulator_build(param_label,L_cube,tests)
    file_name,file_name_test=em.get_filename()
    
    #check snapshots
    #get and set up redshift information
    redshift=np.loadtxt(sim_directory+halos[0]+'/'+file_name[0]+'/redshift_list.txt')
    tags=[]
    for i in range(len(redshift)):
        tags.append('%03d_z%03dp%03d'%(int(redshift[i,0]),int(redshift[i,1]),int(np.round(((redshift[i,1]-np.floor(redshift[i,1]))*1000)))))
    


    
    exist = check_sim(halos,file_name,snap_name)

    #set up L_cube and filenames for each halo, accounting for simualtions ran
    file_name_i = []
    L_cube_i = []
    for i in range(len(halos)):
        file_name_i.append(list(compress(file_name, exist[i,:])))
        L_cube_i.append(L_cube[exist[i,:]])

    def traverse_datasets(hdf_file):

        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if isinstance(item, h5.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5.Group): # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)

        for path, _ in h5py_dataset_iterator(hdf_file):
            yield path

    #generate training data and train

    #get list of all subfind parameters for host
    h=h5.File(sim_directory+halos[0]+'/'+file_name[0]+'/data/groups_'+snap_name+'/eagle_subfind_tab_'+snap_name+'.0.hdf5')
    h2=h5.File(sim_directory+halos[0]+'/'+file_name[0]+'/data/snapshot_'+snap_name+'/snap_'+snap_name+'.0.hdf5')
    part_mass=h2['Header'].attrs['MassTable'][1]

    fof_keys=h['FOF'].keys()
    
    sub_keys=[]
    for dset in traverse_datasets(h['Subhalo']):
        sub_keys.append(dset[1:]) #1: to remove starting slash

    prefix=[]
    for i in range(len(fof_keys)):
        prefix.append('FOF groups:')
    for i in range(len(sub_keys)):
        prefix.append('Subhalo:')
    keys=list(fof_keys)+sub_keys

    
    prefix=[prefix[27],prefix[67]]
    keys=[keys[67]]#for appeture mass
    


    #list of phrases to log the data, mainly mass and size
    log_phrases=['mass','_m_','_r_','rad','vmax']
    mass_phrases=['mass','_m_']

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
    

    for j in range(len(tags)):
        proj_exist=np.zeros(len(halos))
        data=[]
        for k in range(len(halos)):
            sub_id=projen_sub[k][:,j]
            fof_id=projen_fof[k][:,j]

            #sub_id_test=projen_sub_test[k][:,j]
            #fof_id_test=projen_fof_test[k][:,j]
            if np.any(sub_id==-1):# or np.any(sub_id_test==-1):
                continue
            proj_exist[k]=1.0
            #get data to build emulators
            dat,stat,desc=get_data(sim_directory+halos[k]+'/',file_name_i[k],tags[j],sub_id,fof_id)
            data.append(dat)
            #data_test=get_data(file_name_test,tags[j],sub_id_test,fof_id_test)
        
        
        #skip if any of the sims don't have a projenitor at this redshift
        if np.all(proj_exist==1.0)==False:
            continue
        

        #build emulators and test
        chi2=[]
        frac_er=[]
        for i in range(len(stat)):

            #put data into correct shaped array
            data_train=[]
            for k in range(len(halos)):
                #data_train[...,k]=data[k][i]
                data_train.append(data[k][i])
            
            #identify if the haloes are sampled sperately or together
            if np.any(exist==False):
                data_train=[]
                for k in range(len(halos)):
                    #data_train[...,k]=data[k][i]
                    data_train.append(data[k][i])
                em.train(data_train,'%03d'%j,halos,stat[i],desc[i],replace=True,seperate_sampling=exist)
            else:

                
                data_train = np.zeros(data[0][i].shape+(len(halos),))
                for k in range(len(halos)):
                    data_train[...,k] = data[k][i]
                em.train(data_train,'%03d'%j,halos,stat[i],desc[i],replace=True,train_seperataely=True)
            
            
            #deg_freedom,chi22,frac_err=em.test(data_test[i],stat[i])
            #chi2.append(chi22/deg_freedom)
            #frac_er.append(frac_err)

        # for i in range(len(stat)):
        #     print(stat[i]+' %.4f, %.4f'%(chi2[i],frac_er[i]))




