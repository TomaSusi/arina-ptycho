# Python module file for running atomic phase optimization in parallel.
import numpy as np
import os,sys
sys.path.append('/home/jupyter-toma/git/stem_optimization')
import Model
#from Model import Model
#from Simulation import Simulation

def optimize_model_parallel(data, progress_bar=False, plot=False):

    image, atoms, me, method, rounds = data
    
    conv_angle = 0.034 #rad
    energy = 60000 #eV
    blur = 0.0
 
    fov = np.min([atoms.cell[0,0], atoms.cell[1,1]])/10# 

    px = np.min(image.shape)
    
    m = Model.Model(atoms, image, kernelsize=1, plot=plot)
    m.set_up_simulation(conv_angle, energy, fov, blur=blur, method=method) #, aberrations={'EHTFocus':-me['metadata']['hardware_source']['ImageRonchigram']['C10']/10})

    rounds1 = 40
    m.optimize_model(['fov','blur','translation','scale'],
                     iterations = rounds1, progress_bar=True)

    rounds2 = 40
    m.optimize_model(['fov','blur','translation','scale','positions'],
                     iterations = rounds2, progress_bar=True)

    rounds3 = rounds
    m.optimize_model(['fov','blur','translation','scale','positions','intensities'], 
                     iterations = rounds3, progress_bar=True)
    #m.normalize_intensities()
    return m
