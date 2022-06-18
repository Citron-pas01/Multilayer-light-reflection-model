import tmm_core
from numpy import pi, linspace, inf, exp, cos, average, array, vstack, imag
import pandas as pd
import numpy as np

#----------------------------------for an optimization trial---------------------------------------#

# the refractive index vs. the light wavelength files
Ag_df = pd.read_csv('Ag.csv')
Au_df = pd.read_csv('Au.csv')
SiO2_df = pd.read_csv('SiO2.csv')
TiO2_Rutile_df = pd.read_csv('TiO2_Rutile.csv')

# get the corresponding refractive index upon the light wavelength
def index(material_df,lam_vac):
    for i in range(material_df.shape[0]):            
           if lam_vac >= material_df.iloc[i,0] and lam_vac < material_df.iloc[i+1,0]:
                index = complex(material_df.iloc[i+1,1], material_df.iloc[i+1,2])
    return index


# intitialization 
n_Vac = 1.0

##############
### Step 5 ### for changing the light wavelength
##############
lowerbound = 200
upperbound = 800

## pNValue = [ 'NumUnit', 'd_SiO2', 'd_TiO2', 'd_Au', 'd_Vac1', 'd_Vac2']
############## 
### Step 3 ### for changing ### params ### of the stacking structure
############## 
pNvalue = [2, 5, 5, 5, 5, 5]
#pNvalue = [2, 5]
###############              
###  Step 4 ### for changing ### range ### of the stacking structure components
###############            
pmBounds = [[2, 20],[2, 20], [2, 20], [2, 20],[2, 20], [2, 20]]
#pmBounds = [[2, 15],[2, 20]]
##############    
### Step 2 ### for changing the stacking #### structure ####
############## 
config_Matrix = [[1, 0, 0, 0, 0, 0, 0],\
                [0, 1, 0, 0, 0, 0, 0],\
                [0, 0, 1, 0, 0, 0, 0],\
                [0, 0, 0, 1, 0, 0, 0],\
                [0, 0, 0, 0, 1, 0, 0],\
                [0, 0, 0, 0, 0, 1, 0],\
                [0, 0, 0, 0, 0, 0, 1]]



#---------------------------------the function of Stacking Reflection------------------------------#
def StackingReflection(self):
    self.decode2(self.code) # change it to decimal number   
    ref_list = []
    lam_list = []
    for lam_vac in range(lowerbound,upperbound,20):
        lam_list.append(lam_vac)
        n_Ag = index(Ag_df,lam_vac)
        n_Au = index(Au_df,lam_vac)
        n_SiO2 = index(SiO2_df,lam_vac)
        n_TiO2 = index(TiO2_Rutile_df,lam_vac)
        ##############
        ### Step 1 ### for changing the stacking structure
        ############## 
        param_Matrix = [[self.pN[1], self.pN[2], self.pN[3], self.pN[4], self.pN[3], self.pN[2], self.pN[1]],\
                        [n_SiO2, n_TiO2, n_Au, n_Vac, n_Au, n_TiO2, n_SiO2]]

        UnitSet = np.matmul(config_Matrix,np.transpose(param_Matrix))

        SideUnit = [[inf, n_Vac]]
        interUnit = [[self.pN[-1], n_Vac]]
        for rep in range(0,int(self.pN[0])):  # range(start,stop,step), without stop point
             SideUnit = np.concatenate((SideUnit,UnitSet))
             SideUnit = np.concatenate((SideUnit,interUnit))
        SideUnit[-1,0] = inf
        d_list = SideUnit[:,0].real
        n_list = SideUnit[:,1] 

        th_0 = 0 
        rr = tmm_core.unpolarized_RT(n_list, d_list, th_0, lam_vac)['reflect']   # 387 line
        ref_list.append(rr)
    self.y = np.sum(ref_list)/len(ref_list)
    return self.y

#Reflectance = StackingReflection(th_0, NumUnit,d_SiO2, d_TiO2, d_Ag, d_Au, d_Vac1,d_Vac2)

def mathfunction(self):
    self.decode2(self.code)
    self.y = np.sqrt(2000-2*self.pN[0]*self.pN[0]-6*self.pN[1]*self.pN[1] + 5*self.pN[0]*self.pN[1] + 16*self.pN[0])
    return self.y
