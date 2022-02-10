from stable_baselines3 import PPO

import numpy as np
import matplotlib.pyplot as plt

from human_robot_collision_RL.script.constants import *

def network_graph2(model_path,obs_len=1,d_out=2,n=10,p=1):
    modelTest = PPO.load(model_path)

    x = np.linspace(-p,p,n+1,dtype=np.float32)
    xx = [x for i in range(obs_len)]

    X = np.meshgrid(*xx)

    Z = func(xx,f=modelTest.predict,d_out=d_out)

    Zx = Z[:,:,0][0]
    Zy = Z[:,:,0][1]
    Zth = Z[:,:,0][2]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X[0],X[1],Zx, linewidth=1, antialiased=False)
    plt.show()


def func2(x,f=np.sum,d_out=1):
    dims = [xx.shape[0] for xx in x]
    z = np.zeros(tuple(dims+[d_out]),dtype=object)

    # len(dims)*prod(dims) loops
    for idx in np.ndindex(tuple(dims)):
        x_ = np.array([x[i][idx[i]] for i in range(len(dims))])
        z[idx] = f(x_)

    return z

def network_graph(f,obs_len=1,fixed_obs=0,d_out=2,n=10,p=1,view_dim=0):
    x = np.linspace(-p,p,n+1,dtype=np.float32)
    #fixed_vals = [np.array([fixed_obs]) for i in range(obs_len-2)]
    fixed_vals = [np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([2]),np.array([2])]

    x_grid = [x for i in range(2)]
    xx = x_grid+fixed_vals
    X = np.meshgrid(*x_grid,indexing='ij')

    Z = func(xx,f=f,d_out=d_out,view_dim=0)

    #norm = Normalize(Z.min(), Z.max())
    #m = plt.cm.ScalarMappable(norm=norm, cmap ='seismic');
    #m.set_array([]);
    #fcolors = m.to_rgba(Z);


    fig, ax = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    ax[0].plot_surface(X[0],X[1],Z[:,:,0], linewidth=1, antialiased=False) #facecolors=fcolors,
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_zlabel('z')
    ax[1].plot_surface(X[0],X[1],Z[:,:,1], linewidth=1, antialiased=False) #facecolors=fcolors,
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_zlabel('z')
    plt.show()


def func(x,f=np.sum,d_out=1,view_dim=0):
    dims = [xx.shape[0] for xx in x]
    z = np.zeros(tuple(dims[:2]+[d_out]),dtype=object)

    # len(dims)*prod(dims) loops
    for idx in np.ndindex(tuple(dims[:2])):
        x_ = np.array([x[i][idx[i]] for i in range(len(dims[:2]))]+[x[i][0] for i in range(len(dims[2:]))]).reshape((1,-1))
        z[idx] = f(x_)[0]

    return z


if __name__=="__main__":
    exp_num = 1
    my_model_path = PATH_SAVE+'/Experiment_'+str(exp_num)+'/models/0201_1203'
    model = PPO.load(my_model_path)
    network_graph(model.predict,8,2,3,100,1)





