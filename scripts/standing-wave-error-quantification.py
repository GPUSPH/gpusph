#!/bin/python3
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

# this file is for evaluating results obtained from standing wave simulations, based on their printed energy behavior.
# it performs some postprocessing on the kinetic energy (extracts locations and values of maxima and minima, renormalizes
# the signal and subtracts the moving average) and subsequently fits an exponential function to the values of the minima and maxima
# in the least-squares sense in order to characterize the decay.

# usage: python3 errorquantification.py path/to/resultsfolder/ [plot]

# path/to/resultsfolder/ should then contain the data folder.
# two evalforms are chosable, either a least-squares fitting 'fitting', or a moving average 'averagedamplitude'.
# min_t determines the time at which the evaluation starts

evalform = 'fitting'
min_t = 3.0

case = sys.argv[1]
df = pd.read_csv(case + 'data/energy.txt', sep = '\t')
s = df['kinetic0'].to_numpy()
#s = savgol_filter(s, 9, 3) #for high frequency oscillations in signal
t = df['time'].to_numpy()
q_u = np.zeros(s.shape)
q_l = np.zeros(s.shape)
mean = np.zeros(s.shape)
pp_s = np.zeros(s.shape)
moving_avg = np.convolve(s, np.ones((40,))/40, mode = 'same')

#Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

u_x = [0,]
u_y = [s[0],]

l_x = [0,]
l_y = [s[0],]

#Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

for k in range(1,len(s)-1):
    if (np.sign(s[k]-s[k-1])==1) and (np.sign(s[k]-s[k+1])==1 and t[k] > min_t):
        u_x.append(k)
        u_y.append(s[k])

    if (np.sign(s[k]-s[k-1])==-1) and ((np.sign(s[k]-s[k+1]))==-1 and t[k] > min_t):
        l_x.append(k)
        l_y.append(s[k])

l_y[0] = 0

#Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

u_x.append(len(s)-1)
u_y.append(s[-1])

l_x.append(len(s)-1)
l_y.append(s[-1])

#print(u_x)
#print(l_x)

#Evaluate each model over the domain of (s)
for k in range(0,len(s)):
    mean[k] = 0.5 * (q_u[k] + q_l[k])
    pp_s[k] = (s[k] - moving_avg[k])/s[0]

if (evalform == 'averagedamplitude'):
    # compute decay coefficients
    i = 1 #first maximum
    j = 5 #number of periods to be evaluated
    ampl = 0

    print('average amplitude between the ' + str(i) + 'th maximum and the following ' + str(j) + ' periods')
    for k in range(j):
        ampl = ampl + pp_s[u_x[i + k]] - pp_s[l_x[i + 1 + k]]

    ampl = ampl/j

    print(ampl)

if (evalform == 'fitting'):
    i = 1 #first maximum
    j = 35 #last maximum 35
    values = []
    xvalues = []
    for k in range(j-i):
        #print(k)
        xvalues.append(0.5*(df['time'].iloc[u_x[i + k]] + df['time'].iloc[l_x[i + k]]))
        #maxvalues.append(pp_s[maxvalues_x[k]])
        values.append(0.5*(abs(pp_s[u_x[i + k]]) + abs(pp_s[l_x[i + k]])))
        #print(maxvalues[k])
    #linear regression
    #coeff, intercept, r_value, p_value, std_err = stats.linregress(df['time'].iloc[maxvalues_x], maxvalues)
    #exponential fitting, resulting from linear regression from log(y)
    coeff, intercept = np.polyfit(xvalues, np.log(values), 1)
    print('exponential coefficient:')
    print(coeff)
    print('intercept:')
    print(np.exp(intercept))

#Plot everything for debugging purposes, also commented options for visualization
if(sys.argv[-1] == 'plot'):
    my_dpi = 96
    #plt.figure(figsize=(16/9 * 900/my_dpi, 900/my_dpi), dpi=my_dpi)
    #plt.style.use("ggplot")
    fig, axs = plt.subplots(2)
    fig.set_size_inches(16/9 * 900/my_dpi, 900/my_dpi)
    #plt.rc('text', usetex=True)
    #plt.rc('font', family = 'serif')
    axs[0].plot(df['time'], s, label = 'signal')
    axs[0].plot(df['time'], moving_avg, 'k', label = 'moving avg')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(df['time'], pp_s, 'm', label = 'postprocessed signal')
    if (evalform == 'fitting'):
        #axs[1].plot(df['time'].iloc[maxvalues_x], intercept+coeff*df['time'].iloc[maxvalues_x], label = 'fitted curve')
        t_cont = np.linspace(xvalues[0], xvalues[-1], 1000)
        axs[1].plot(t_cont, np.exp(intercept)*np.exp(coeff*t_cont), label = 'fitted curve')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('xx-small')
    axs[1].legend(prop = fontP)
    axs[1].grid()
    plt.xlabel('time')
    plt.ylabel('kinetic energy')
    #plt.rc('font', family = 'serif')
    plt.show()
    import tikzplotlib
    #tikzplotlib.save('standard_exponential.tex', axis_height = '\\ahdp', axis_width = '\\awdp')


