# Need to parse .csv log files, process them, and plot results
# in an organize fashion.
# Thesis work
# Michael C. Murphy
# Code started: March 1, 2016

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log
from statistics import variance
import time

def main():
    print( "Detected", len(sys.argv), "arguments")

    # Here is the array of box widths
    d = [3, 5, 10, 20, 30, 50, 100, 150, 300]

    F = {}  # Data Dictionary using filenames as keys
    filelist = sys.argv[1:]  # hold a list of files used
    for filename in filelist:
        N = {} # Data Dictonary holding Noise data
        file = open(filename, 'r')
        for line in file:
            if line[0] != '#':
                line = line.strip()
                data = line.split(',')
                # Noise % is the key, the array of counted boxes is the data
                N[ data[1] ] = data[4:13]
        F[filename] = N

    # Steal image file name from the path:
    image_name = sys.argv[1].split("/")
    image_name = image_name[-1].split("_")
    image_name = image_name[0]

    # Debug print the size of our data structure
    print(">> Size of F:",len(F))

    # Now that we've read in the data, it is time to process it.
    R = {}  # Data Dictionary holding aggregate results
    for filekey in F:
        for noisekey in F[filekey]:
            # F[filekey][noisekey] holds one array of 9 boxcounts
            # Need to calculate an arithmetic mean of the dimension (dim_mean)
            
            log_bc = []
            # store the log(boxcounts)
            for bc in F[filekey][noisekey]:
                bc = int(bc)
                if( bc == 0 ):
                    log_bc.append( log(1) )
                else:
                    log_bc.append( log(bc) )

            slopes = []
            # Calculate Dimension slopes by taking delta(log(boxcounts) / delta(log(1/d))
            for i in range(len(d)-1):
                slopes.append( (log_bc[i] - log_bc[i+1]) / (log(1/d[i]) - log(1/d[i+1])) )

            dim_mean = np.mean(slopes)
            
            # Need to calculate the variance of the dimensional slopes (dsl_var)
            dsl_var = np.var(slopes, ddof=1)   #ddof = 1 matches old-style variance calculations

            # Add each dim/var calculation to an array and store in results.
            if noisekey in R:
                R[noisekey].append( (dim_mean,dsl_var) )
            else:
                R[noisekey] = [ (dim_mean,dsl_var) ]

    # Now all the results should be processed;  currently R is an array of tuples (Dim, Var)
    print("Size of R:", len(R), "# results:",len(R['0.2']))
    #print(">>",R['0.2'])

    # separate the values and run statistics on them.
    aggregated_results = []   # store final data to print here
    for noisekey in R:
        dim_array = []
        dsl_array = []
        # separate the data
        for dataset in R[noisekey]:  # dataset is 100 pairs of (Dim, Var)
            dim_array.append(dataset[0])
            dsl_array.append(dataset[1])
        
        # calculate statistics
        average_dimensional_estimate = np.mean(dim_array)
        average_slope_variance_estimate = np.mean(dsl_array)
        dim_standard_error = np.std(dim_array,ddof=1)/len(R[noisekey])
        dsl_standard_error = np.std(dsl_array,ddof=1)/len(R[noisekey])
        # add to aggregated results
        aggregated_results.append( (float(noisekey), average_dimensional_estimate, dim_standard_error, average_slope_variance_estimate, dsl_standard_error) )

    aggregated_results.sort()
    
    #for item in aggregated_results:
    #    print(">> ", item)
        
    # TODO: Need to save this data to a file
    aggregate_datalog( aggregated_results, filelist, image_name )
        
    # Attempt to plot:
    nx = []
    dimy = []
    vary = []
    dimerr = []
    varerr = []
    for item in aggregated_results:
        nx.append( item[0] )
        dimy.append( item[1] )
        dimerr.append( item[2] )
        vary.append( item[3] )
        varerr.append( item[4] )

    fig, axarr = plt.subplots(2, sharex=True)
    fig.suptitle("Box Count Algorithm on " + image_name + " (Uniform Noise, 100 Seeds)", fontsize=14, fontweight='bold')
    axarr[0].set_ylabel("dimension")
    axarr[0].set_xscale('log')
    axarr[0].set_title("Mean Fractal Dimension vs. Noise %")
    axarr[0].set_ylim(0,2)
    axarr[0].errorbar(nx,dimy,dimerr,fmt='r')
    axarr[0].plot(nx,dimy, label="dimension")

    axarr[1].set_ylabel("slope variance")
    axarr[1].set_xlabel("noise %")
    axarr[1].set_xscale('log')
    axarr[1].set_title("Mean Slope Variance vs. Noise %")
    axarr[1].set_ylim(0,1)
    axarr[1].errorbar(nx,vary,varerr,fmt='r')
    axarr[1].plot(nx,vary, label="variance")

    plt.savefig("figures/" + image_name + "_uniform_D_V_vs_N.png")
    #plt.show()
        

# results_array - an array of tuples of the form (noise, dimension, dimension std err, variance, variance std err)
# filelist - a list of filenames that were included in the aggregated data
# Version: V1.00 - initial log format
def aggregate_datalog( results_array, filelist, image_name ):
    LOG_VERSION = "1.00"
    pathname = "logfiles/"
    filename = pathname + image_name + "_" + str(len(filelist)) + "_aggregateLog.csv"

    log = open(filename, 'w')
    # Write the header
    log.write("# Aggregate Log Version: " + LOG_VERSION + ",,,,\n")
    log.write("# Log Name: " + filename + ",,,,\n")
    log.write("# Noise%, Dimension, Dim Std Err, Variance, Var Std Err\n")

    # Write the data
    for data in results_array:
        log.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "\n")

    # Write a list of the files included in this data aggregation
    log.write("# The data files below are included in this aggregation:,,,,\n")
    for file in filelist:
        log.write("# " + file + ",,,,\n")

    print(">> Created aggregate datalog:", filename)
    log.close()

    
    
    







main()
'''

# example data
#x = np.arange(0, 10, 1)
noise = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 50])
dim = np.array([1.1, 1.15, 1.3, 1.6, 1.85, 1.95, 2])
var = np.array([0.1, 0.3, 0.2, 0.1, 0.01, 0.001, 0])



# example variable error bar values
stderr = np.array([.01,.1,.1,.05,.02,.01,0])
#xerr = 0.1 + yerr

# First illustrate basic byplot interface, using defaults where possible
#fig = plt.figure()  # what does this do?
#fig.suptitle("Main Title", fontsize=14, fontweight='bold')


fig, axarr = plt.subplots(2, sharex=True)
fig.suptitle("Dimension and Slope Variance vs. Noise %", fontsize=14, fontweight='bold')
axarr[0].set_ylabel("dimension")
axarr[0].set_xscale('log')
axarr[0].errorbar(noise,dim,stderr,fmt='ro')
axarr[0].plot(noise,dim, label="dimension")

axarr[1].set_ylabel("slope variance")
axarr[1].set_xlabel("noise %")
axarr[1].set_xscale('log')
axarr[1].errorbar(noise,var,stderr,fmt='ro')
axarr[1].plot(noise,var, label="variance")

'''

'''
# create subplot (so that axis can be manipulated
ax = fig.add_subplot(111)
ax.set_title("subtitle")
ax.set_xlabel("noise axis")
ax.set_ylabel("dimension")
ax.set_xscale('log')            # set x-axis to log scale

#plt.errorbar(x, dim, yerr=yerr, fmt='ro')  # error bar plot, formatting
ax.errorbar(noise,dim, stderr, fmt='ro')
ax.errorbar(noise,var, stderr, fmt='ro')
ax.plot(noise,dim)                          # normal plot, formatting
ax.plot(noise,var)
#plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
'''

'''
plt.savefig("thesisPlotsExcample.png")

plt.show()
'''




