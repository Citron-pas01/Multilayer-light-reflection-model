import tmm_core
from numpy import pi, linspace, inf, exp, cos, average, array, vstack, imag
import pandas as pd
import numpy as np
import csv
import time

# Create a parameter matrix: thickness of (SiO2, TiO2, Ag, Au, Vacuum-inside, Internal-Vacuum, Number of unit)
class OptiSim():
    def __init__(self, wavelength, pNvalue, var1Value, var2Value, modelIndex):

        self.lowerbound = wavelength[0]
        self.upperbound = wavelength[1]
        self.d_Vac1 = pNvalue[4]
        self.d_Vac2 = pNvalue[-1]
        self.d_SiO2 = pNvalue[1]
        self.d_Ag = 15
        self.d_TiO2 = pNvalue[2]
        self.d_MgF2 = 10
        self.d_Al2O3 = 10
        self.d_Au = pNvalue[3]
        self.d_Al = 25
        self.NumUnit = pNvalue[0]
        self.lam_step = wavelength[2] #lam_step
        self.modelIndex = modelIndex

        self.h_thick1 = var1Value[1]
        self.l_thick1 = var1Value[0]
        self.thick1_step = var1Value[2]
        self.h_thick2 = var2Value[1]
        self.l_thick2 = var2Value[0]
        self.thick2_step = var2Value[2]

    def material_refraction(self):
        self.Ag_df = pd.read_csv('Ag.csv')
        #Au_df = pd.read_csv('Au.csv')
        self.Au_df = pd.read_csv('Au_UV.csv')
        self.Al_df = pd.read_csv('Al_3.csv')
        self.Al2O3_df = pd.read_csv('Al2O3.csv')
        self.MgF2_df = pd.read_csv('MgF2_UV.csv')
        self.SiO2_df = pd.read_csv('SiO2.csv')
        self.TiO2_Rutile_df = pd.read_csv('TiO2_Rutile.csv')

    def model(self):
        ##### Model 1 #####
        if self.modelIndex == 0:
            self.config_Matrix = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
            self.param_Matrix = [[self.d_Au, self.d_Vac1, self.d_Au],[self.n_Au, self.n_Vac, self.n_Au]]

        ##### Model 2 #####
        elif self.modelIndex == 1:
            self.config_Matrix = [[1, 0, 0, 0, 0, 0, 0],\
                                [0, 1, 0, 0, 0, 0, 0],\
                                [0, 0, 1, 0, 0, 0, 0],\
                                [0, 0, 0, 1, 0, 0, 0],\
                                [0, 0, 0, 0, 1, 0, 0],\
                                [0, 0, 0, 0, 0, 1, 0],\
                                [0, 0, 0, 0, 0, 0, 1]]\

            self.param_Matrix = [[self.d_SiO2, self.d_TiO2, self.d_Au, self.d_Vac1, self.d_Au, self.d_TiO2, self.d_SiO2],\
                              [self.n_SiO2, self.n_TiO2, self.n_Au, self.n_Vac, self.n_Au, self.n_TiO2, self.n_SiO2]]

        self.UnitSet = np.matmul(self.config_Matrix, np.transpose(self.param_Matrix))

    def index(self, material_df,lam_vac):
        index = 0
        for i in range(material_df.shape[0]):            
            if lam_vac >= material_df.iloc[i,0] and lam_vac < material_df.iloc[i+1,0]:
                    index = complex(material_df.iloc[i+1,1], material_df.iloc[i+1,2])
        return index

    def StackingReflectance(self, d_SiO2, d_TiO2):
        #d_Vac1 = self.d_Vac1
        #d_Vac2 = self.d_Vac2
        #d_Al2O3 = self.d_Al2O3
        self.d_SiO2 = d_SiO2
        self.d_TiO2 = d_TiO2
        #d_MgF2 = self.d_MgF2
        #d_Au = self.d_Au
        #d_Ag = self.d_Ag
        #d_Al = self.d_Al
        #NumUnit = self.NumUnit
        self.n_Vac = 1.0
        self.th_0 = 0
        ref_list = []
        tran_list = []
        nonpol_absorps = []
        ext_list = []
        for lam_vac in range(self.lowerbound,self.upperbound,self.lam_step):
            self.material_refraction()
            self.n_Ag = self.index(self.Ag_df,lam_vac)
            self.n_Au = self.index(self.Au_df,lam_vac)
            self.n_Al = self.index(self.Al_df,lam_vac)
            self.n_Al2O3 = self.index(self.Al2O3_df,lam_vac)
            self.n_MgF2 = self.index(self.MgF2_df,lam_vac)
            self.n_SiO2 = self.index(self.SiO2_df,lam_vac)
            self.n_TiO2 = self.index(self.TiO2_Rutile_df,lam_vac)

        ## generate the index tag ##
            self.model()
            SideUnit = [[inf, self.n_Vac]]
            interUnit = [[self.d_Vac2, self.n_Vac]]
        ## stacking ##
            for rep in range(0,self.NumUnit):  # range(start,stop,step), without stop point
                SideUnit = np.concatenate((SideUnit,self.UnitSet))
                SideUnit = np.concatenate((SideUnit,interUnit))
            SideUnit[-1,0] = inf
            d_list = SideUnit[:,0].real
            n_list = SideUnit[:,1]  
        
        # calculated the reflected light
            rr = tmm_core.unpolarized_RT(n_list, d_list, self.th_0, lam_vac)['reflect']   # 387 line
            ref_list.append(rr)

        # calculate the transmittion
            tt = tmm_core.unpolarized_RT(n_list, d_list, self.th_0, lam_vac)['transmit']   # 387 line
            tran_list.append(tt)
    
        # calculate the energy absorped
            nonpol_absorp = 0
            for pol in ['s', 'p']:
                coh_tmm_data = tmm_core.coh_tmm(pol, n_list, d_list, self.th_0, lam_vac)
                layer_absorp = tmm_core.absorp_in_each_layer(coh_tmm_data) # the absorped energy of each layer (line 607)
                absorp = np.sum(layer_absorp[1:len(layer_absorp)-1])   # calculate the absorped fraction
                nonpol_absorp = nonpol_absorp + absorp  # sum the absorps of "s" and "p" 
            nonpol_absorps.append(nonpol_absorp/2) # absrops at every wavelength appends
        
        # calculate the extinction
            ext = rr + nonpol_absorp/2
            ext_list.append(ext)
        self.reflect = ref_list
        self.transmit = tran_list
        self.absorp = nonpol_absorps
        self.extinct = ext_list

       
    def Optfunc(self,var_1, var_2):
        re_outs = []
        ab_outs = []
        tr_outs = []
        ext_outs = []
        final_outs = []
        for variable_1 in range(self.l_thick1,self.h_thick1,self.thick1_step):
            res = list()
            abs = list()          ##### It is very important to define the append host just outside the loop!!!
            trs = list()
            exts = list()         
            for variable_2 in range(self.l_thick2, self.h_thick2, self.thick2_step):
                self.StackingReflectance(variable_1, variable_2)
                res.append(self.reflect)
                abs.append(self.absorp)
                trs.append(self.transmit)
                exts.append(self.extinct)
                print('Look! I am running at (v_1,v_2)) = ({0}unit,{1}nm)'.format(variable_1, variable_2))            
            re_outs.append(res)
            ab_outs.append(abs)
            tr_outs.append(trs)
            ext_outs.append(exts)
        final_outs.append(re_outs)
        final_outs.append(ab_outs)
        final_outs.append(tr_outs)
        final_outs.append(ext_outs)
        return {'final_data': final_outs }


