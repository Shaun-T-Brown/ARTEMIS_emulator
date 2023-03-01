import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree
from scipy import optimize
from skimage import measure
import dill as pickle
from scipy.interpolate import RegularGridInterpolator
import os

class corner_plot:

    def __init__(self,n_dim,labels):

        #load interpolation tables for edge correction
        path = os.path.realpath(__file__).split('/')
        self.loc=''
        for i in range(len(path)-1):
            self.loc=self.loc+'/'+path[i]
        
        with open(self.loc+'/2d_pdf_edge_norm.pkl', 'rb') as file:
            self.precompute = pickle.load(file)

        #set up figure and axis
        self.labels=labels
        self.n_dims = n_dim
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(ncols=n_dim, nrows=n_dim, figure=self.fig,wspace=0.0, hspace=0.0)
        self.axes = []
        for i in range(n_dim):
            self.axes.append([])
            for j in range(i+1):
                self.axes[i].append(self.fig.add_subplot(self.gs[i,j]))

                #remove y ticks for 1d projection and add labels
                if i==0:
                    self.axes[i][j].set_yticks([])

                if i!=n_dim-1:
                    self.axes[i][j].set_xticks([])

                if j!=0:
                    self.axes[i][j].set_yticks([])

                if (j==0) & (i>0):
                    self.axes[i][j].set_ylabel(labels[i])

                if (i==n_dim-1):
                    self.axes[i][j].set_xlabel(labels[j])


    def tick_update(self,tick_loc,tick_label):
        for i in range(self.n_dims):
            for j in range(i+1):
                if (j==0) & (i>0):
                    self.axes[i][j].yaxis.set_ticks(tick_loc[i],tick_label[i])

                if (i==self.n_dims-1):
                    self.axes[i][j].xaxis.set_ticks(tick_loc[j],tick_label[j])

        self.limit_adjust()

        return

    def limit_adjust(self):
        for i in range(self.n_dims):
            for j in range(i+1):

                if i==j:
                    self.axes[i][j].set_xlim(self.limits[i][0],self.limits[i][1])
                    ylim_c = self.axes[i][j].get_ylim()
                    self.axes[i][j].set_ylim(0,ylim_c[1])
                else:
                    self.axes[i][j].set_xlim(self.limits[i][0],self.limits[i][1])
                    self.axes[i][j].set_ylim(self.limits[j][0],self.limits[j][1])

        return

    def truths(self,data,**kwargs):
        for i in range(self.n_dims):
            for j in range(i+1):

                if i==j:
                    ylim = self.axes[i][j].get_ylim()
                    self.axes[i][j].plot(np.ones(2)*data[i],ylim,**kwargs)

                else:
                    xlim = self.axes[i][j].get_xlim()
                    ylim = self.axes[i][j].get_ylim()

                    self.axes[i][j].plot(np.ones(2)*data[j],ylim,**kwargs)
                    self.axes[i][j].plot(xlim,np.ones(2)*data[i],**kwargs)

        return()

    def contour(self, data, percent = [0.865,0.394],alphas=[0.4,0.75],limits = None, edge_correction=False,norm_1d = False, h_1d=0.2,h_2d = 0.3,color='black',**kwargs):

        if limits == None:
            self.limits =[]
            for i in range(data.shape[1]):
                self.limits.append([np.min(data[:,i]),np.max(data[:,i])])
        else:
            self.limits=limits

        def pdf_1d(x,x_sample,h):
            pdf = np.zeros(len(x_sample))
            for i in range(len(x_sample)):
                dist = np.abs(x-x_sample[i])/h
                cut = dist<0.5
                pdf[i] += np.sum(1-6*dist[cut]**2 + 6*dist[cut]**3)
                cut = (dist>=0.5) & (dist<=1)
                pdf[i] += np.sum(2*(1-dist[cut])**3)

            return(pdf)

        def pdf_1d_norm(pdf,x_sample):
            pdf/=np.trapz(pdf,x_sample)
            return(pdf)

        def pdf_1d_correction(pdf,x_sample,edge,h):
            x_norm = 1-(x_sample+h-edge[1])/h
            norm = np.ones(len(pdf))
            cut = (x_norm<=1) & (x_norm>=0.5)
            norm[cut] = 1-0.5 *(1-x_norm[cut])**4/(3/4)
            
            cut = (x_norm<0.5) & (x_norm>=0.0)
            norm[cut] = 1-(3/8-x_norm[cut]+2*x_norm[cut]**3-3/2*x_norm[cut]**4)/(3/4)
            
            x_norm = 1+(edge[0]+x_sample-h)/h
            cut = (x_norm<=1) & (x_norm>=0.5)
            norm[cut] = 1-0.5 *(1-x_norm[cut])**4/(3/4)
            
            cut = (x_norm<0.5) & (x_norm>=0.0)
            norm[cut] = 1-(3/8-x_norm[cut]+2*x_norm[cut]**3-3/2*x_norm[cut]**4)/(3/4)

            pdf /= norm

            return(pdf)

        def pdf_2d(x,x_sample,h):
            #first normalise data to go between 0 and 1 (for sampling)
            x_sample_norm = []
            x_norm = np.array(x)
            for i in range(2):
                x_norm[:,i]-=np.min(x_sample[i])
                x_norm[:,i]*= 1/(np.max(x_sample[i])-np.min(x_sample[i]))
                x_sample_norm.append(np.array(x_sample[i]))
                x_sample_norm[i] -= np.min(x_sample[i])
                x_sample_norm[i] *= 1/(np.max(x_sample[i])-np.min(x_sample[i]))


            #set up grid of sampling point
            xx,yy = np.meshgrid( x_sample[0],x_sample[1])
            xx_norm,yy_norm = np.meshgrid( x_sample_norm[0],x_sample_norm[1])
            shape = xx.shape

            sampled_pos = np.empty((xx_norm.shape[0]*xx_norm.shape[1],2))
            sampled_pos[:,0] = xx_norm.flatten()
            sampled_pos[:,1] = yy_norm.flatten()

            #set up kd tree
            kd = KDTree(x_norm)
            query = kd.query_ball_point(sampled_pos,h)

            #loop through and create pdf
            pdf = np.zeros(len(sampled_pos))

            for i in range(len(pdf)):
                dist = ((x_norm[query[i],0] - sampled_pos[i,0])**2 + (x_norm[query[i],1] - sampled_pos[i,1])**2)**0.5/h
                
                cut = dist<0.5
                pdf[i] += np.sum(1-6*dist[cut]**2 +6*dist[cut]**3)
                cut = (dist>=0.5) & (dist<1)
                pdf[i] += np.sum(2*(1-dist[cut])**3)

            pdf=np.reshape(pdf,shape)
            return(pdf)

        def contour_find(pdf,fractions):

            def frac_calc(level,pdf_pad,frac_0):

                
                #level = level[0]
                contours = measure.find_contours(pdf_pad, level)
                tot = np.sum(pdf_pad)
                frac = 0.0
                for i in range(len(contours)):
                    mask = measure.grid_points_in_poly(pdf_pad.shape, contours[i])
                    frac += np.sum(pdf_pad[mask])

                frac/=tot
                return(frac)

            #convert to image
            
            pdf_pad = np.pad(pdf,1,'constant',constant_values=0.0)

            #normalise to make root finding easier
            pdf_pad = pdf_pad/np.max(pdf_pad)

            level = []

            level_sample = np.linspace(0,1,50) 
            frac = np.empty(len(level_sample))
            for i in range(len(level_sample)):
                frac[i] = frac_calc(level_sample[i],pdf_pad,fractions[0])

            
            for i in range(len(fractions)):
                level.append(level_sample[np.argmin((frac - fractions[i])**2)])

            #calculate the contours for the levels
            contours =[]
            for i in range(len(level)):
                cont = measure.find_contours(pdf_pad, level[i])
                
                mask = measure.grid_points_in_poly(pdf_pad.shape, cont[0])

                contours.append(cont)

            return(contours)

        def pdf_2d_correction(pdf,num_grid,h):
            
            interp = RegularGridInterpolator((self.precompute[0], self.precompute[1]), self.precompute[2])

            
            #grid assumed to be normalised
            x=np.linspace(0,1,num_grid)
            xx,yy = np.meshgrid(x,x)

            grid = np.empty((num_grid**2,2))
            grid[:,0] = xx.flatten()
            grid[:,1] = yy.flatten()
            
            dist = np.zeros(grid.shape)
            
            for i in range(2):
                cut = grid[:,i]>0.5
                dist[cut,i] = 1-grid[cut,i]
                cut = np.invert(cut)
                dist[cut,i] = grid[cut,i]

            #convert distance into interpolation coodinates
            dist/=h
            dist = -np.sort(-dist,axis=1)

            norm = np.ones(len(dist))
            #deal with corners
            cut = (dist[:,0]<=1) * (dist[:,0]>=-1) & (dist[:,1]<=1) & (dist[:,1]>=0)
            inter_dist = np.zeros((np.sum(cut),2))
            inter_dist[:,0] = dist[cut,0]
            inter_dist[:,1] = dist[cut,1]/dist[cut,0]
            inter_dist[np.isnan(inter_dist)] = 0
            norm[ cut] = interp(inter_dist)

            #deal with sides
            cut = (dist[:,0]>1) & (dist[:,1]<=1) & (dist[:,1]>=0)
            inter_dist = np.ones((np.sum(cut),2))
            inter_dist[:,1] = dist[cut,1]
            norm[ cut] = interp(inter_dist)

            norm = np.reshape(norm,(num_grid,num_grid))
            pdf_norm= np.array(pdf)
            pdf_norm *= 1/norm
            return(pdf_norm)

        def contour_norm(contour,x_sample,n):
            #here we are accounting for the padding
            dx = x_sample[0][1]-x_sample[0][0]
            dy = x_sample[1][1]-x_sample[1][0]
            xmin = x_sample[0][0]
            ymin = x_sample[1][0]

            for i in range(len(contour)):
                for j in range(len(contour[i])):

                    contour[i][j][:,0] -= 1
                    contour[i][j][:,1] -= 1

                    contour[i][j][:,0] *=dx 
                    contour[i][j][:,1] *=dy

                    contour[i][j][:,0] += xmin
                    contour[i][j][:,1] += ymin

            return(contour)


        #do diagonals first
        for i in range(self.n_dims):
            x = data[:,i]
            
            x_sample = np.linspace(self.limits[i][0],self.limits[i][1],30) 
            h = (self.limits[i][1] - self.limits[i][0])*h_1d
            pdf = pdf_1d(x,x_sample,h)
            if edge_correction==True:
                pdf = pdf_1d_correction(pdf,x_sample,self.limits[i],h)

            pdf = pdf_1d_norm(pdf,x_sample)

            if norm_1d==True:
                #renormalise so that maxima is at 1, not integral to unity
                pdf = pdf/np.max(pdf)
            self.axes[i][i].plot(x_sample,pdf,color=color,**kwargs)

        #do cross correlations
        for i in range(self.n_dims):
            for j in range(i):
                if i==j:
                    continue

                x = data[:,np.array([i,j])]
                x_sample=[]
                num_points = 50
                x_sample.append(np.linspace(self.limits[i][0],self.limits[i][1],num_points))
                x_sample.append(np.linspace(self.limits[j][0],self.limits[j][1],num_points))

                pdf = pdf_2d(x,x_sample,h_2d)
                if edge_correction==True:
                    pdf = pdf_2d_correction(pdf,num_points,h_2d)
                contour = contour_find(pdf,percent)
                #normalise contours correctly
                contour = contour_norm(contour,x_sample,num_points)

                #plot contours
                for k in range(len(contour)):
                    for l in range(len(contour[k])):
                        self.axes[i][j].plot(contour[k][l][:,0],contour[k][l][:,1],'-',color=color,**kwargs)
                        self.axes[i][j].fill(contour[k][l][:,0],contour[k][l][:,1],color=color,alpha=alphas[k])

        #adjust limits to finish plot
        self.limit_adjust()
        return

    def scatter(self,data,**kwargs):

        for i in range(self.n_dims):
            for j in range(i):
                if i==j:
                    continue

                self.axes[i][j].plot(data[:,i],data[:,j],'o',**kwargs)

        return





if __name__=='__main__':

    #calculate normalisation required for spline
    # x=np.linspace(-1,1,1000)
    # dx=x[1]-x[0]
    # xx,yy=np.meshgrid(x,x)

    # dist = (xx**2+yy**2)**0.5

    # w=np.zeros(xx.shape)
    # cut = (dist>0) & (dist<=0.5)
    # w[cut] = 1-6*dist[cut]**2+6*dist[cut]**3
    # cut = (dist>0.5) & (dist<1)
    # w[cut] = 2*(1-dist[cut])**3

    # tot = np.trapz(np.trapz(w,axis=0),axis=0)*dx*dx
    
    # #sample cuts
    # N = 100
    # x1 = np.linspace(-1,1,N)
    # alpha=np.linspace(0,1,N)
    

    # x1_m,alpha_m = np.meshgrid(x1,alpha,indexing='ij', sparse=True)

    # frac = np.zeros((len(x1),len(alpha)))
    # for i in range(len(x1)):
    #     print(i)
    #     for j in range(len(alpha)):
            
    #         cut = (xx<= x1[i]) & (yy <= x1[i]*alpha[j])

    #         w_cut = np.zeros(xx.shape)
    #         w_cut[cut] = w[cut]
            
    #         integral = np.trapz(np.trapz(w_cut,axis=0),axis=0)*dx*dx

    #         frac [i,j] = integral/tot


    
    # with open('2d_pdf_edge_norm.pkl', 'wb') as file:
    #     pickle.dump((x1,alpha,frac), file)
    # exit()


    cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    x = np.random.multivariate_normal(np.zeros(4),cov,size=10000)

    for i in range(3):
        cut = (x[:,i]>-2) & (x[:,i]<2)

        #x = x[cut,:]

    dist = (x[:,0]**2+x[:,1]**2)**0.5
    
    cp = corner_plot(4)
    cp.contour(x,limits=[[-4,4],[-4,4],[-4,4],[-4,4]])

    plt.show()