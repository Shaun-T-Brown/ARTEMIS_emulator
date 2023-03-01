import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel





#set up example data
x=np.linspace(0,2*np.pi,20)

y=np.empty((len(x),3))
y[:,0]=np.sin(x/2)
y[:,1]=np.sin((x+np.pi/2)/2)
y[:,2]=np.sin((x+np.pi)/2)


y_obs=y+np.random.normal(0,0.5,y.shape)

#train gaussian process
kernel = 1.0*RBF(length_scale=np.ones(1)*0.2, length_scale_bounds=(1e-2, 1e2))\
    + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-3, 1e2))


gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(x.reshape(-1,1), y_obs)

print(gpr.kernel_)


x_sample=np.linspace(0,2*np.pi,100)

y_sample,y_err=gpr.predict(x_sample.reshape(-1,1),return_std=True)

print(y_sample.shape)
print(y_err.shape)

plt.plot(x,y_obs,'k.')
plt.plot(x,y,'k--')
plt.plot(x_sample,y_sample)
plt.fill_between(x_sample,y_sample[:,0]-y_err[:,0],y_sample[:,0]+y_err[:,0])
plt.fill_between(x_sample,y_sample[:,1]-y_err[:,1],y_sample[:,1]+y_err[:,1])
plt.fill_between(x_sample,y_sample[:,2]-y_err[:,2],y_sample[:,2]+y_err[:,2])
