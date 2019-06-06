#!/usr/bin/env python

# INPUT - The script expect the case directory as first argument.
# OUTPUT - The script create a Results forlder inside the case folder and ouput six files:
#    * error.txt: time and L2 error for each time step
#    * profile.txt: velocity profile of the last time step
#    * totalError.txt: error averaged for t+ in [1.5;3.5] (for convergence study)
#    * two PDF files plotting the velocity profile and the error vs time
#    * last_iteration.txt: store the last iteration treated by the script.
#    when launching the script several times for the same case while it runs,
#    the script restart from the last iteration previously treated.

# package utilises
import os # pour suivre des chemins dans un repertoire
import sys # to manage the script arguments
import numpy as np # pour faire des maths
import matplotlib.pyplot as plt # pour tracer des courbes
import matplotlib as mpl
import matplotlib.animation as ani # pour faire des animations
#ani.rcParams['animation.writer'] = 'mencoder' # pour definir le writer (si ffmpeg ne marche pas)
#import matplotlib.image as mpimg # pour lire des images
#from scipy import misc # pour avoir plus d'option de traitements d'images
plt.rc('text',usetex=True) # pour utiliser des symbols LaTeX dans les figures
plt.rc('font', family='serif')
from mpl_toolkits.mplot3d import Axes3D # pour pouvoir faire des plots en 3D
import re # pour lire des chaines de characteres
import vtk # pour lire et ecrire des fichier vtk
from vtk.util import numpy_support # pour convertir les fichiers vtk au format numpy

#%reset
plt.close('all')


loading_step=1

######################################################
### Define functions 
######################################################

# Velocity profile in fluid 0
def u1(Alpha, Omega, Lambda, Zplus):
    u1_array = np.zeros(len(Zplus))
    u1_array[:] = (Alpha**2-Zplus[:]**2)+(
            Omega*(1-Alpha)**2-Alpha**2)/(
                    Alpha+Omega*Lambda*(1-Alpha))*(Zplus[:]+Alpha)
    return u1_array

# Velocity profile in fluid 1
def u2(Alpha, Omega, Lambda, Zplus):
    u2_array = np.zeros(len(Zplus))
    u2_array = Omega*((1-Alpha)**2-Zplus[:]**2)+Omega*Lambda*(
            Omega*(1-Alpha)**2-Alpha**2)/(
                    Alpha+Omega*Lambda*(1-Alpha))*(Zplus[:]-(1-Alpha))
    return u2_array

# Velocity profile everywhere
def uplus(Alpha, Omega, Lambda, Zplus):
    u_array = np.zeros(len(Zplus))
    u_array = np.where(Zplus < 0, u1(Alpha, Omega, Lambda, Zplus),u2(Alpha, Omega, Lambda, Zplus))
    return u_array

# L2 error between theoritical and SPH profiles
def errorL2(Zplus, u_sph, u_th):
    err_num = 0.
    err_den = 0.
    for i in range(0, len(Zplus)):
        err_num += (u_sph[i,0] - u_th[i])**2 + u_sph[i,1]**2 + u_sph[i,2]**2
        err_den += u_th[i]**2
    if (err_den != 0):
        return err_num/err_den
    else :
        return 0

######################################################
### Variables d'entrees
######################################################
if len(sys.argv) > 1:
    Case = sys.argv[1]
else:
    print("You have to give me the case folder name.")
    sys.exit()

boolLight = False
if len(sys.argv) > 2:
    if str(sys.argv[2]) == "-light":
        boolLight = True
        print("Using a smaller pvd file")
    else:
        print("Using the original pvd file.")
else:
    print("Using the original pvd file.")
######################################################
### Geometrical parameters
######################################################

L = 2

######################################################
### Read summuray file to get physical/numerical parameters
######################################################
print("==========================")
print("Reading summary file")
print("==========================")

Root_case = Case

summary_file = open(os.path.join(Root_case,'summary.txt'),'r')
for fline in summary_file:
    fline = fline.strip()
    try :
        if "gravity =" in fline :
            g_str = fline.replace("gravity = (","")
            g_str = re.sub(",.+"," ",g_str)
        elif "rho0[ 0 ] = " in fline:
            rho0_str = fline.replace("rho0[ 0 ] = ","")
            rho1_str = rho0_str
        elif "rho0[ 1 ] = " in fline:
            rho1_str = fline.replace("rho0[ 1 ] = ","")
        elif "kinematicvisc[ 0 ] = " in fline:
            nu0_str = fline.replace("kinematicvisc[ 0 ] = ","")
            nu0_str = nu0_str.replace(" (m^2/s)","")
            nu1_str = nu0_str
        elif "kinematicvisc[ 1 ] = " in fline:
            nu1_str = fline.replace("kinematicvisc[ 1 ] = ","")
            nu1_str = nu1_str.replace(" (m^2/s)","")
        elif "deltap = " in fline:
            dr_str = fline.replace("deltap = ","")
    except :
        print("Error: summary file not read fully\n")
        break


g = float(g_str)
rho0 = float(rho0_str)
rho1 = float(rho1_str)
nu0 = float(nu0_str)
nu1 = float(nu1_str)
dr = float(dr_str)

######################################################
### Compute other useful quantities 
######################################################

# Kinematic viscosities
mu0 = nu0*rho0
mu1 = nu1*rho1

# Reference time and velocity
tref = L**2*rho0/mu0
uref = g*L**2/(2*nu0)
tmin = 1.5*tref 
tmax = 3.5*tref

# Dimensionless multi-fluid ratios
Omega = nu0/nu1
Lambda = rho0/rho1

print("Omega = "+str(Omega))
print("Lambda = "+str(Lambda))
print("uref = "+str(uref))
print("tref = "+str(tref))
print("tmax = "+str(tmax))

######################################################
### Colormap as defined in paraview as Blue to Red Rainbow
######################################################
cdict = {'red': ((0.000000, 0.0000000, 0.0000000),
                 (0.255776, 0.0000000, 0.0000000),
                 (0.511913, 0.0823529, 0.0823529),
                 (0.754152, 1.0000000, 1.0000000),
                 (1.000000, 1.0000000, 1.0000000)),
        'green':((0.000000, 0.0000000, 0.0000000),
                 (0.255776, 1.0000000, 1.0000000),
                 (0.511913, 1.0000000, 1.0000000),
                 (0.754152, 0.9333333, 0.9333333),
                 (1.000000, 0.0000000, 0.0000000)),
        'blue': ((0.000000, 1.0000000, 1.0000000),
                 (0.255776, 1.0000000, 1.0000000),
                 (0.511913, 0.0000000, 0.0000000),
                 (0.754152, 0.0000000, 0.0000000),
                 (1.000000, 0.0000000, 0.0000000))}

paraview_cmap = mpl.colors.LinearSegmentedColormap('paraview_cmap',cdict,256)

######################################################
### Lire le fichier de resultats
######################################################
print("==========================")
print("Reading Result files")
print("==========================")

Root = os.path.join(Case,'data')

# Creer et configurer les variables vtk
r = vtk.vtkXMLPolyDataReader()

# Read the time
t = []
file_name = []

if boolLight == True:
    VTU_str='VTUinp2.pvd'
else:
    VTU_str='VTUinp.pvd'

with open(os.path.join(Root, VTU_str),'r') as VTU_file :
    for fline in VTU_file:
        fline = fline.strip()
        try :
            if "<DataSet timestep='" in fline :
                data = fline.replace("<DataSet timestep='",'')
                data = data.replace("' group='0' name='Particles' file='",' ',)
                data = data.replace("'/>",'',)
                data = data.split(' ')
                t.append(float(data[0]))
                file_name.append(data[1])
        except :
            print("Error: VTU file not read fully\n")
            break

Nt = len(t)
t = np.array(t,dtype=float)

# Find max number of particles
Np_max = 0
Np = np.zeros(Nt,dtype=int)

for it in range(0, Nt, loading_step) :
    r.SetFileName(os.path.join(Root,file_name[it]))
    r.Update()
    data = r.GetOutput()
    #Read Number of points
    Np[it] = data.GetNumberOfPoints()
    Np_max = np.maximum(Np_max,Np[it])

###### INFORMATION #######
# print(pointData) lists the arrays etc.
#print(r.GetOutput())
# GPUSPH VTU file structure
#Number Of Arrays: 9                                                                                 
#Array 0 name = Pressure                                                                             
#Array 1 name = Density                                                                              
#Array 2 name = Mass                                                                                 
#Array 3 name = Part type                                                                            
#Array 4 name = Part flags                                                                           
#Array 5 name = Fluid number                                                                         
#Array 6 name = Part id                                                                              
#Array 7 name = CellIndex                                                                            
#Array 8 name = Velocity
#Number Of Components: 11


# Array pressure number is usually 0 so we use it as default value
pres_array_int = 0
# Look for the actual pressure array number
for fline in str(r.GetOutput()).splitlines():
    if "Array" in fline:
        if "Pressure" in fline:
            pres_array_str = fline.replace("    Array ", "")
            pres_array_str = pres_array_str.replace(" name = Pressure", "")
            pres_array_int = int(pres_array_str)


# Array velocity number is usually 8 so we use it as default value
vel_array_int = 8
# Look for the actual velocity array number
for fline in str(r.GetOutput()).splitlines():
    if "Array" in fline:
        if "Velocity" in fline:
            vel_array_str = fline.replace("    Array ", "")
            vel_array_str = vel_array_str.replace(" name = Velocity", "")
            vel_array_int = int(vel_array_str)

# Array part_type number is usually 3 so we use it as default value
part_type_array_int = 3
# Look for the actual part_typeocity array number
for fline in str(r.GetOutput()).splitlines():
    if "Array" in fline:
        if "Part type" in fline:
            part_type_array_str = fline.replace("    Array ", "")
            part_type_array_str = part_type_array_str.replace(" name = Part type", "")
            part_type_array_int = int(part_type_array_str)

# Array fluid number is usually 5 so we use it as default value
fluid_number_array_int = 5
# Look for the actual part_typeocity array number
for fline in str(r.GetOutput()).splitlines():
    if "Array" in fline:
        if "Fluid number" in fline:
            fluid_number_array_str = fline.replace("    Array ", "")
            fluid_number_array_str = fluid_number_array_str.replace(" name = Fluid number", "")
            fluid_number_array_int = int(fluid_number_array_str)

fluid_number = np.zeros([Nt,Np_max],dtype=int)
part_type = np.zeros([Nt,Np_max],dtype=int)
coord = np.zeros([Nt,Np_max,3],dtype=float)
vel = np.zeros([Nt,Np_max,3],dtype=float)
pres = np.zeros([Nt,Np_max],dtype=float)


last_iteration = -1 # initialization

Root_w = os.path.join('/home/alex/Software/gpusph/tests',Case,'Results')
if (os.path.exists(Root_w+'/last_iteration.txt')):
    w_file = open(Root_w+"/last_iteration.txt", "r") 
    last_iteration = int(w_file.read())
    w_file.close()


if(last_iteration+1 >=Nt):
    # if there is no new iteration, the last one is loaded
    loading_start = last_iteration
else:
    loading_start = last_iteration + 1

for it in range(loading_start, Nt, loading_step) :
#for it in range(1) :
    if np.mod(it,10)==0 :
        print("Time read : %f s" % t[it])
    r.SetFileName(os.path.join(Root,file_name[it]))
    r.Update()
    coord_vtk = r.GetOutput().GetPoints().GetData()
    coord[it,:Np[it],:] = numpy_support.vtk_to_numpy(coord_vtk)

    part_type_vtk = r.GetOutput().GetPointData().GetArray(part_type_array_int)
    part_type[it,:Np[it]] = numpy_support.vtk_to_numpy(part_type_vtk)

    pressure_vtk = r.GetOutput().GetPointData().GetArray(pres_array_int)
    pres[it,:Np[it]] = numpy_support.vtk_to_numpy(pressure_vtk)

    vel_vtk = r.GetOutput().GetPointData().GetArray(vel_array_int)
    vel[it,:Np[it],:] = numpy_support.vtk_to_numpy(vel_vtk)

    fluid_number_vtk = r.GetOutput().GetPointData().GetArray(fluid_number_array_int)
    fluid_number[it,:Np[it]] = numpy_support.vtk_to_numpy(fluid_number_vtk)

######################################################
### Trouver le nombre de particules de fluide
######################################################
print("==========================")
print("Extracting fluid particles")
print("==========================")

Nf = np.zeros(Nt,dtype=int)
part_type_f = np.zeros([Nt,Np_max],dtype=int)
coord_f = np.zeros([Nt,Np_max,3],dtype=float)
vel_f = np.zeros([Nt,Np_max,3],dtype=float)
vel_mag_f = np.zeros([Nt,Np_max],dtype=float)
pres_f = np.zeros([Nt,Np_max],dtype=float)
fluid_number_f = np.zeros([Nt,Np_max],dtype=int)

# Always load initial data
r.SetFileName(os.path.join(Root,file_name[0]))
r.Update()
coord_vtk = r.GetOutput().GetPoints().GetData()
coord[0,:Np[0],:] = numpy_support.vtk_to_numpy(coord_vtk)

part_type_vtk = r.GetOutput().GetPointData().GetArray(part_type_array_int)
part_type[0,:Np[0]] = numpy_support.vtk_to_numpy(part_type_vtk)

pressure_vtk = r.GetOutput().GetPointData().GetArray(pres_array_int)
pres[0,:Np[0]] = numpy_support.vtk_to_numpy(pressure_vtk)

vel_vtk = r.GetOutput().GetPointData().GetArray(vel_array_int)
vel[0,:Np[0],:] = numpy_support.vtk_to_numpy(vel_vtk)

fluid_number_vtk = r.GetOutput().GetPointData().GetArray(fluid_number_array_int)
fluid_number[0,:Np[0]] = numpy_support.vtk_to_numpy(fluid_number_vtk)

output = part_type[0][ part_type[0] == 0]
Nf[0] = len(output)
part_type_f[0,:Nf[0]] = part_type[0][ part_type[0] == 0]
fluid_number_f[0,:Nf[0]] = fluid_number[0][ part_type[0] == 0]
coord_f[0,:Nf[0]][:] = coord[0][ part_type[0] == 0][:]

# New mesh (convergence)
# Compute the theoretical interface position
#if (len(coord_f[0][ fluid_number_f[0] == 1,2]) > 0):
#    zi = np.amin(coord_f[0][ fluid_number_f[0] == 1,2]) - dr/2.
#    Alpha = zi/L
#    print("Alpha = "+str(Alpha)+"\tzi = "+str(zi))
#else:
#    zi = 0.
#    Alpha = 0.5
#    print("One-fluid simulation")

# Old mesh Chris cases
# Compute the theoretical interface position
if (len(coord_f[0][ fluid_number_f[0] == 1,2]) > 0):
    zi = np.amin(coord_f[0][ fluid_number_f[0] == 1,2]) - dr/2.
    Alpha = 1-(L/2. - zi)/L
    print("Alpha = "+str(Alpha)+"\tzi = "+str(zi))
else:
    zi = 0.
    Alpha = 0.5
    print("One-fluid simulation")

for it in range(loading_start, Nt, loading_step) :
#for it in range(1) :
    if np.mod(it,10)==0 :
        print("Particles extracted at time %f s" % t[it])
    #arg_temp = np.argsort(part_type[it,:Np[it]])
    # ne prendre que les 0 (fluid)
    #arg_temp = arg_temp[:np.searchsorted(part_type[it,arg_temp],1)]
    #Nf[it] = np.size(arg_temp)
    #part_type_f[it,:Nf[it]] = part_type[it,arg_temp]
    #coord_f[it,:Nf[it],:] = coord[it,arg_temp,:]
    #vel_f[it,:Nf[it],:] = vel[it,arg_temp,:]
    #vel_mag_f[it,:Nf[it]] = (vel[it,arg_temp,0]**2+vel[it,arg_temp,1]**2+vel[it,arg_temp,2]**2)**0.5
    #pres_f[it,:Nf[it]] = pres[it,arg_temp]
    output = part_type[it][ part_type[it] == 0]
    Nf[it] = len(output)
    part_type_f[it,:Nf[it]] = part_type[it][ part_type[it] == 0]
    fluid_number_f[it,:Nf[it]] = fluid_number[it][ part_type[it] == 0]
#    
    coord_f[it,:Nf[it]][:] = coord[it][ part_type[it] == 0][:]
#
    vel_f[it,:Nf[it],:] = vel[it][ part_type[it] == 0][:]
    pres_f[it,:Nf[it]] = pres[it][ part_type[it] == 0 ] 
    del output

#del part_type, coord, vel, pres
######################################################
### Computing L2 error 
######################################################
print("==========================")
print("Computing L2 Error")
print("==========================")

# The last iteration computed was:
error = np.zeros((Nt,2))

try:
    os.stat(Root_w)
except:
    os.mkdir(Root_w)

if (os.path.exists(Root_w+'/last_iteration.txt')):
    w_file = open(Root_w+"/last_iteration.txt", "r") 
    last_iteration = int(w_file.read())
    w_file.close()
    previous_error = np.loadtxt(Root_w+"/error.txt", skiprows=1)
    if (len(previous_error) > Nt):
        print("The previous iteration number is greater than the current one. Exiting...")
        sys.exit()
    elif (len(previous_error) != last_iteration+1):
        print("The last_iteration value is not consistent with the number of lines in error.txt. Exiting...")
        sys.exit()
    else:
        error[0:last_iteration+1,:] = previous_error
        print("Previous iterations successfully loaded!")
else:
    print("No last_iteration.txt found.")


for it in range(last_iteration+1, Nt):
    zplus_array = (coord_f[it][~np.all(coord_f[it] == 0, axis=1)][:,2] - zi)/L
    uplus_th_array = uplus(Alpha, Omega, Lambda, zplus_array)
    uplus_sph_array = vel_f[it][~np.all(coord_f[it] == 0, axis=1)]/uref
    error[it,0] = t[it]/tref
    error[it,1] = errorL2(zplus_array, uplus_sph_array, uplus_th_array)
    del zplus_array, uplus_th_array, uplus_sph_array


# Looking for index whenre 1.5 < t+ < 3.5
itTab = np.where((error[:,0] >= 1.5) & (error[:,0] <= 3.6))[0]
totError = 0
Nerror = 0

for it in itTab:
    totError += error[it,1]
    Nerror += 1

if (Nerror > 0):
    totError /= Nerror


w_file = open(Root_w+"/totalError.txt", "w")
if (len(itTab) == 0):
    print('Not in the converged area yet.\n')
    w_file.write('Not in the converged area yet.\n')
    w_file.write('t+ = ' + str(t[Nt-1]/tref) + '\ttotalError = '+str(totError))
elif t[itTab[len(itTab)-1]]/tref < 3.5:
    print('Still in the converged area. You should wait longer.\n')
    w_file.write('Still in the converged area. You should wait longer.\n')
    w_file.write('t+ = ' + str(t[itTab[len(itTab)-1]]/tref) + '\ttotalError = '+str(totError))
else:
    print('Converged. You can stop the simulation.\n')
    w_file.write('Converged. You can stop the simulation.\n')
    w_file.write('t+ = ' + str(t[len(itTab)-1]/tref) + '\ttotalError = '+str(totError))
w_file.close 


w_file = open(Root_w+"/last_iteration.txt", "w")
w_file.write(str(Nt-1))
w_file.close()
np.savetxt(Root_w+"/error.txt",error, header="Time\tE2", comments='')

plt.semilogy(error[:,0], error[:,1])
#plt.show()
fig_name = Root_w+"/Error-BiFluidPoiseuilleFlow-"+str(it)+".pdf"
plt.savefig(fig_name, transparent=True)
plt.clf()

print("==========================")


######################################################
### Plot the last profile and save it 
######################################################
print("==========================")
print("Saving velocity profile images")

coord_f_t=coord[Nt-1][~np.all(coord_f[Nt-1] == 0, axis=1)] # remove the zeros
vel_f_t  =vel[Nt-1][~np.all(coord_f[Nt-1] == 0, axis=1)]

coord_f_tt=coord_f_t[ coord_f_t[:,0] <= dr ]
vel_f_tt=vel_f_t[ coord_f_t[:,0] <= dr ]
del coord_f_t, vel_f_t

coord_f_t=coord_f_tt[  coord_f_tt[:,0] > -dr/2. ]
vel_f_t=vel_f_tt[  coord_f_tt[:,0] > -dr/2. ]
del coord_f_tt, vel_f_tt

coord_f_tt=coord_f_t[  coord_f_t[:,1] <= dr ]
vel_f_tt=vel_f_t[  coord_f_t[:,1] <= dr ]
del coord_f_t, vel_f_t

coord_f_t=coord_f_tt[ coord_f_tt[:,1] > -dr/2. ]
vel_f_t=vel_f_tt[ coord_f_tt[:,1] > -dr/2. ]
del coord_f_tt, vel_f_tt

zplus_array = (coord_f_t[:,2] - zi)/L
del coord_f_t
uplus_sph_array = vel_f_t/uref
del vel_f_t

uplus_th_array = uplus(Alpha, Omega, Lambda, zplus_array)

# Recenter the profile in 0 after getting the analytical solution
zplus_array = zplus_array + zi/L

np.savetxt(Root_w+"/profile.txt", np.column_stack((zplus_array, uplus_sph_array[:,0], uplus_th_array)), header="zstar\tusphx_star\tuth_star", comments='')

PLOT=True
if PLOT==True:
    xfigmin = 0. # min expected velocity
    xfigmax = 1.
    yfigmin = -Alpha*L
    yfigmax = (1.-Alpha)*L
    f_size_x = 1.
    f_size_y = 1.
    #
    figw, figh = plt.figaspect(float(f_size_y/f_size_x))
    #
    fig = plt.figure('Velocity profile at time t/T %f'% (t[it]/tref),
        figsize=(figw,figh), dpi=80, facecolor='w', edgecolor='k')
    #ax = fig.add_axes([1, 1, 1, 1])
    #
    #plt.axis((xfigmin,xfigmax,yfigmin,yfigmax))
    #
    #plt.tight_layout()
    #plt.axis('off')
    plt.plot(zplus_array, uplus_th_array, linestyle="None", marker=".",label=r'Th.')
    plt.plot(zplus_array, uplus_sph_array[:,0], linestyle="None", marker="+", label=r'SPH')
    plt.legend()
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'$u^{+}$')
    fig_name = Root_w+"/BiFluidPoiseuilleFlow-"+str(it)+".pdf"
    plt.savefig(fig_name, transparent=True)

######################################################
### End of file
######################################################




# OTHER WORKING EXAMPLE
# https://stackoverflow.com/questions/23138112/vtk-to-maplotlib-using-numpy
#import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
#import numpy as np
#import vtk
#from vtk.util.numpy_support import vtk_to_numpy
#
## load a vtk file as input
#reader = vtk.vtkXMLUnstructuredGridReader()
#reader.SetFileName("my_input_data.vtk")
#reader.Update()
#
## Get the coordinates of nodes in the mesh
#nodes_vtk_array= reader.GetOutput().GetPoints().GetData()
#
##The "Temperature" field is the third scalar in my vtk file
#temperature_vtk_array = reader.GetOutput().GetPointData().GetArray(3)
#
##Get the coordinates of the nodes and their temperatures
#nodes_nummpy_array = vtk_to_numpy(nodes_vtk_array)
#x,y,z= nodes_nummpy_array[:,0] , nodes_nummpy_array[:,1] , nodes_nummpy_array[:,2]
#
#temperature_numpy_array = vtk_to_numpy(temperature_vtk_array)
#T = temperature_numpy_array
#
##Draw contours
#npts = 100
#xmin, xmax = min(x), max(x)
#ymin, ymax = min(y), max(y)
#
## define grid
#xi = np.linspace(xmin, xmax, npts)
#yi = np.linspace(ymin, ymax, npts)
## grid the data
#Ti = griddata((x, y), T, (xi[None,:], yi[:,None]), method='cubic')  
#
### CONTOUR: draws the boundaries of the isosurfaces
#CS = plt.contour(xi,yi,Ti,10,linewidths=3,cmap=cm.jet) 
#
### CONTOUR ANNOTATION: puts a value label
#plt.clabel(CS, inline=1,inline_spacing= 3, fontsize=12, colors='k', use_clabeltext=1)
#
#plt.colorbar() 
#plt.show() 
