import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.mathtext as mathtext

styles = ['-','-','-','-','-','-','-','-','--', '--','--','--','--','--', '--', '--','-.','-.','-.','-.','-.','-.',\
   '-.',':']
Colr = ['black','blue','green', 'purple', 'yellow', 'orange','magenta','red','black','blue','green', \
     'purple', 'yellow', 'orange','magenta','red','black','blue','green', 'purple']

def twoVplot_one(figNum, plC, k, vN, x, r_data, f_rv_idx, m, Output_r_idx, UnitNumber_r1, labelname, Numcol_legd):
    position = str(vN[1] + ': ' +str(UnitNumber_r1)+'nm') #
    lines = []
    fig = plt.figure(figNum)
    ax = fig.add_subplot(211)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(plC[k])             ###
    mpl.rc("font", family="Times New Roman",weight='normal')
    plt.rcParams.update({'mathtext.default':  'regular' })
    for i in range(len(Output_r_idx)):
        lines += ax.plot(x, r_data[i][f_rv_idx[m]], styles[i],color=Colr[i],\
                        label = (vN[0]+labelname[i]))
    ax.legend(fontsize='7', ncol=Numcol_legd,handleheight=0.8, labelspacing=0.03, \
                    loc='lower center',bbox_to_anchor=(0.5, -0.5), frameon=False)
    ax.set_title(plC[k]+ ' vs. ' + vN[0] + ' when ' + position) 
    fig.savefig(vN[0]+plC[k]+'Graph1.png')
    #plt.show()

def twoVplot_two(figNum,plC,k,vN, x, a_data, min_a_idx, UnitNumber_a0, labelname, Numcol_legd):
    position = str(vN[0] + ': ' + str(UnitNumber_a0) )  # + 'nm'
    lines = []
    fig = plt.figure(figNum)
    ax = fig.add_subplot(211)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(plC[k] + '') 
    mpl.rc("font", family="Times New Roman",weight='normal')
    plt.rcParams.update({'mathtext.default':  'regular' })
    for i in range(len(a_data[min_a_idx])):
        lines += ax.plot(x, a_data[min_a_idx][i], styles[i],color=Colr[i], \
                        label = (vN[1]+labelname[i]))

    ax.legend(fontsize='7', ncol=Numcol_legd,handleheight=0.8, labelspacing=0.03, \
                    loc='lower center',bbox_to_anchor=(0.5, -0.5), frameon=False)
    ax.set_title(plC[k]+ ' vs. ' + vN[1] + ' when ' + position)
    fig.savefig(vN[1]+plC[k]+'Graph2.png')
    #plt.show()

def merge_a_r(figNum, vN, l, x, r_data, a_data, f_rv_idx, f_av_idx, m, Output_r_idx, \
                    Output_a_idx, labelname, Numcol_legd):
    lines1 = []
    lines2 = []
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(211)
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Reflectance "x"')             ###
    mpl.rc("font", family="Times New Roman",weight='normal')
    plt.rcParams.update({'mathtext.default':  'regular' })
    for i in range(len(Output_r_idx)):
        lines1 += ax.plot(x, r_data[i][f_rv_idx[m]], styles[i],color=Colr[i], marker="x",\
                        label = (vN[l]+labelname[i]))
    ax2=ax.twinx()
    for i in range(len(Output_a_idx)):
        lines2 += ax2.plot(x, a_data[i][f_av_idx[m]], styles[i], color=Colr[i], marker="o",\
                        label = (vN[l]+labelname[i]))
    ax2.set_ylabel('Absorption "·"',color="blue",fontsize=14)
    ## the follow aims to 2nd variable, d_TiO2
    '''
    for i in range(len(r_data[Max_r_idx])):
        lines1 += ax.plot(x, r_data[Max_r_idx][i], styles[i],color=Colr[i], marker="x", \
                        label = (vN[l]+labelname[i]))
    ax2=ax.twinx()
    ax2.set_ylabel('Absorption "·"',color="blue",fontsize=14)
    for i in range(len(a_data[min_a_idx])):
        lines2 += ax2.plot(x, a_data[min_a_idx][i], styles[i],color=Colr[i], marker=".",\
                        label = (vN[l]+labelname[i]))
    '''
    ax.legend(fontsize='7', ncol=Numcol_legd,handleheight=0.8, labelspacing=0.03, \
                    loc='lower center',bbox_to_anchor=(0.5, -0.5), frameon=False)
    ax.set_title('Reflectance & Absorption vs. The thickness of DL layers') 
    plt.show()

def twoVPlot_rna(x, r_data, Max_r_idx, firs_rv_idx, f_rv_idx, a_data, min_a_idx, firs_av_idx, f_av_idx, \
                 t_data, min_t_idx, f_tv_idx, firs_tv_idx, Output_r_idx, Output_a_idx, Output_t_idx,\
                    unit_step,m_unit, thick_step, l_thick, labelname0, labelname1, Numcol_legd):

    #vN = ['d_Vac1','d_Vac2']
    vN = ['d_SiO2','d_TiO2']
    #vN = ['NumUnit','d_Au']
    #vN = ['d_Vac1','d_TiO2']
    plC = ['Reflectance','Absorption','Extinction'] # 'Transmission'
    UnitNumber_r0 = Max_r_idx*unit_step+m_unit        
    UnitNumber_r1 = firs_rv_idx*thick_step + l_thick
    #############################merged when optimal####################################
    position_or = str(str(UnitNumber_r0) +', '+ str(UnitNumber_r1)+'nm') #nm
    UnitNumber_a0 = min_a_idx*unit_step+m_unit         
    UnitNumber_a1 =firs_av_idx*thick_step + l_thick
    UnitNumber_t0 = min_t_idx*unit_step+m_unit 
    UnitNumber_t1 =firs_tv_idx*thick_step + l_thick
    position_oa = str(str(UnitNumber_a0)+', '+ str(UnitNumber_a1)+'nm') #nm
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(211)
    ax.plot(x, r_data[Max_r_idx][f_rv_idx[Max_r_idx]], color="red")
    ax.set_xlabel('Wavelength',fontsize=14)
    ax.set_ylabel('Reflectance',color="red",fontsize=14)
    ax2=ax.twinx()
    ax2.plot(x, a_data[min_a_idx][f_av_idx[min_a_idx]],color="blue")
    ax2.set_ylabel('Absorption', color="blue",fontsize=14)
    ax.set_title('Optimized'+ (str(vN)) + 'for R: ' + position_or + '; A: ' + position_oa)
    plt.show()
    #############################for 1st variable####################################
    print('The OPTIMAL Reflection is at {0}:{1}'.format(vN[1],UnitNumber_r1))
    twoVplot_one(2,plC,0,vN, x, r_data, f_rv_idx, Max_r_idx, Output_r_idx, UnitNumber_r1, labelname0, Numcol_legd)

    print('The OPTIMAL Absorption is at {0}:{1}'.format(vN[1],UnitNumber_a1))
    twoVplot_one(3,plC,1,vN, x, a_data, f_av_idx, min_a_idx, Output_a_idx, UnitNumber_a1, labelname0, Numcol_legd)

    print('The OPTIMAL transmission is at {0}:{1}'.format(vN[1],UnitNumber_t1))
    twoVplot_one(4,plC,2,vN, x, t_data, f_tv_idx, min_t_idx, Output_t_idx, UnitNumber_t1, labelname0, Numcol_legd)
    ############################for 2nd variable######################################
    print('The OPTIMAL Reflection is at {0}:{1}'.format(vN[0],UnitNumber_r0))
    twoVplot_two(5, plC,0,vN, x, r_data, Max_r_idx, UnitNumber_r0, labelname1, Numcol_legd)

    print('The OPTIMAL Absorption is at {0}:{1}'.format(vN[0],UnitNumber_a0))
    twoVplot_two(6, plC, 1, vN, x, a_data, min_a_idx, UnitNumber_a0, labelname1, Numcol_legd)

    print('The OPTIMAL transmission is at {0}:{1}'.format(vN[0],UnitNumber_t0))
    twoVplot_two(7, plC, 2, vN, x, t_data, min_t_idx, UnitNumber_t0, labelname1, Numcol_legd)

def surface_plot (matrix, unit_step, m_unit, thick_step, l_thick, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is rows, y is cols
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x*unit_step+m_unit, y*thick_step+l_thick, matrix, **kwargs)
    return (fig, ax, surf)

def threeDplot(allsum_re, unit_step, m_unit, thick_step, l_thick):
    allsum_re_arry = np.array(allsum_re)
    (fig, ax, surf) = surface_plot(allsum_re_arry, unit_step, m_unit, \
                                    thick_step, l_thick, cmap=plt.cm.coolwarm)

    fig.colorbar(surf)

    ax.set_xlabel('NumUnit')
    ax.set_ylabel('d_Au')
    ax.set_zlabel('R_Sum')
    plt.show()