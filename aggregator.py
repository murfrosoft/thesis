# Need to parse .csv log files, process them, and plot results
# in an organize fashion.
# TODO:  Add detailed usage to header
# TODO:  Add acceleration calculations for key exploration  Log V1.01
# TODO:  Add other key exploration calculations      Log V1.02
#
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
    # USAGE:
    # python3 aggregator.py (-u or -g) /path/to/datalogs/*
    print( "Detected", len(sys.argv), "arguments")

    if( len(sys.argv) < 3 ):
        print("Insufficient arguments for aggregator.py")
        print("USAGE: python3 aggregator.py (-u or -g) /path/to/datalogs/*")
        return

    noise_type = sys.argv[1][1]

    # Here is the array of box widths
    d = [3, 5, 10, 20, 30, 50, 100, 150, 300]

    F = {}  # Data Dictionary using filenames as keys
    filelist = sys.argv[2:]  # hold a list of files used
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

    # Grab image file name from the path:
    image_name = sys.argv[2].split("/")
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

            # FRACTAL DIMENSION = average of slope estimates
            dim_mean = np.mean(slopes)
            
            # KEY1: Slope Variance -- Need to calculate the variance of the dimensional slopes (dsl_var)
            dsl_var = np.var(slopes, ddof=1)   #ddof = 1 matches old-style variance calculations

            # KEY2: Sum( Abs( Diff ) ) SAD
            SAD = 0
            for i in range(len(slopes) - 1):
                SAD += np.fabs( slopes[i]-slopes[i+1] )

            # KEY3: Sum( Abs( Central Difference ) ) SAC
            CentralDiffs = []
            for i in range(len(slopes) - 2):
                CentralDiffs.append( slopes[i]/2 - slopes[i+2]/2)

            SAC = 0
            for cd in CentralDiffs:
                SAC += np.fabs( cd )

            # KEY4: Avg Difference from mean squared - ADMS
            DMSsums = 0
            for slope in slopes:
                DMSsums += (slope - dim_mean)**2
            ADMS = DMSsums / len(slopes)              

            # Add each dim/var calculation to an array and store in results.
            if noisekey in R:
                R[noisekey].append( (dim_mean, dsl_var, SAD, SAC, ADMS) )
            else:
                R[noisekey] = [ (dim_mean, dsl_var, SAD, SAC, ADMS) ]

    # Now all the results should be processed;  currently R is an array of tuples (Dim, Var)
    print("Size of R:", len(R)) #, "# results:",len(R['0.2']))
    #print(">>",R['0.2'])

    # separate the values and run statistics on them.
    aggregated_results = []   # store final data to print here
    for noisekey in R:
        dim_array = []
        dsl_array = []
        SAD_array = []
        SAC_array = []
        ADMS_array = []
        # separate the data
        for dataset in R[noisekey]:  # dataset is 100 pairs of (Dim, Var)
            dim_array.append(dataset[0])
            dsl_array.append(dataset[1])
            SAD_array.append(dataset[2])
            SAC_array.append(dataset[3])
            ADMS_array.append(dataset[4])
        
        # calculate statistics
        avg_dim = np.mean(dim_array)
        avg_var = np.mean(dsl_array)
        avg_sad = np.mean(SAD_array)
        avg_sac = np.mean(SAC_array)
        avg_adms = np.mean(ADMS_array)
        dim_ste = np.std(dim_array,ddof=1)/len(R[noisekey])
        var_ste = np.std(dsl_array,ddof=1)/len(R[noisekey])
        sad_ste = np.std(SAD_array,ddof=1)/len(R[noisekey])
        sac_ste = np.std(SAC_array,ddof=1)/len(R[noisekey])
        adms_ste = np.std(ADMS_array,ddof=1)/len(R[noisekey])
        # add to aggregated results
        aggregated_results.append( (float(noisekey), avg_dim,dim_ste, avg_var,var_ste, avg_sad,sad_ste, avg_sac,sac_ste, avg_adms,adms_ste ) )

    aggregated_results.sort()
    
    #for item in aggregated_results:
    #    print(">> ", item)
        
    # TODO: Need to save this data to a file
    aggregate_datalog( aggregated_results, filelist, image_name , noise_type)

    # Plot aggregated data
    plot_agg( aggregated_results, image_name, len(F), noise_type)


def plot_agg( aggregated_results, image_name, seed_count, noise_type):
    # Attempt to plot:
    nx = []
    dimy = []
    vary = []
    sady = []
    sacy = []
    admsy = []
    dimerr = []
    varerr = []
    saderr = []
    sacerr = []
    admserr = []
    for item in aggregated_results:
        nx.append( item[0] )
        dimy.append( item[1] )
        dimerr.append( item[2] )
        vary.append( item[3] )
        varerr.append( item[4] )
        sady.append( item[5] )
        saderr.append( item[6] )
        sacy.append( item[7] )
        sacerr.append( item[8] )
        admsy.append( item[9] )
        admserr.append( item[10] )

    # Create a figure with 5 subplots
    fig, axarr = plt.subplots(5, sharex=True)
    if( noise_type == 'u'):
        fig.suptitle("Key Comparisons on " + image_name + " (Uniform Noise, " + str(seed_count) + " Seeds)", fontsize=14, fontweight='bold')
    else:
        fig.suptitle("Key Comparisons on " + image_name + " (Gaussian Noise, " + str(seed_count) + " Seeds)", fontsize=14, fontweight='bold')
    fig.set_size_inches(8,12,forward=True)  # try to set size of plot??
        
    axarr[0].set_ylabel("Dimension")
    axarr[0].set_xscale('log')
    axarr[0].set_title("Mean Fractal Dimension vs. Noise %")
    axarr[0].set_ylim(0,2)
    axarr[0].errorbar(nx,dimy,dimerr,fmt='r')
    axarr[0].plot(nx,dimy, label="dimension")

    axarr[1].set_ylabel("Slope Variance")
    #axarr[1].set_xlabel("noise %")
    axarr[1].set_xscale('log')
    axarr[1].set_title("Mean Slope Variance vs. Noise %")
    axarr[1].set_ylim(0,1)
    axarr[1].errorbar(nx,vary,varerr,fmt='r')
    axarr[1].plot(nx,vary, label="variance")
    
    axarr[2].set_ylabel("Mean Difference Squared")
    #axarr[2].set_xlabel("noise %")
    axarr[2].set_xscale('log')
    axarr[2].set_title("Mean Avg Diff Mean Squared vs. Noise %")
    axarr[2].set_ylim(0,1)
    axarr[2].errorbar(nx,admsy,admserr,fmt='r')
    axarr[2].plot(nx,admsy, label="adms")
    
    axarr[3].set_ylabel("Sum abs(Diff)")
    #axarr[3].set_xlabel("noise %")
    axarr[3].set_xscale('log')
    axarr[3].set_title("Mean Sum of Diffs vs. Noise %")
    axarr[3].set_ylim(0,3)
    axarr[3].errorbar(nx,sady,saderr,fmt='r')
    axarr[3].plot(nx,sady, label="sad")

    axarr[4].set_ylabel("Sum abs(Cent. Diff)")
    axarr[4].set_xlabel("noise %")
    axarr[4].set_xscale('log')
    axarr[4].set_title("Mean Sum of Central Diffs vs. Noise %")
    axarr[4].set_ylim(0,3)
    axarr[4].errorbar(nx,sacy,sacerr,fmt='r')
    axarr[4].plot(nx,sacy, label="sac")
    
    if( noise_type == 'u'):
        plt.savefig("figures/" + image_name + "_uniform_keyComparisonsFig.png")
    else:
        plt.savefig("figures/" + image_name + "_gaussian_keyComparisonsFig.png")
    
    plt.show()
        

# results_array - an array of tuples of the form (noise, dimension, dimension std err, variance, variance std err)
# filelist - a list of filenames that were included in the aggregated data
# Version: V1.00 - initial log format
# Version: V1.01 - simpler naming convention and new keys added
def aggregate_datalog( results_array, filelist, image_name, noise_type ):
    LOG_VERSION = "1.01"
    pathname = "logfiles/"
    filename = pathname + image_name + "_" + noise_type + str(len(filelist)) + "_aglog.csv"

    log = open(filename, 'w')
    # Write the header
    log.write("# Aggregate Log Version: " + LOG_VERSION + ",,,,,,,,,,\n")
    log.write("# Log Name: " + filename + ",,,,,,,,,,\n")
    log.write("# Noise%, Dimension, Dim Std Err, Variance, Var Std Err, Sum(Abs(Diff), SAD Std Err, Sum(Abs(CenDiff)), SAC Std Err, Avg Diff Mean Squared, ADMS Ste\n")

    # Write the data
    for data in results_array:
        log.write(str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "," + str(data[6]) + ",")
        log.write(str(data[7]) + "," + str(data[8]) + "," + str(data[9]) + "," + str(data[10]) + "\n")

    # Write a list of the files included in this data aggregation
    log.write("# The data files below are included in this aggregation:,,,,,,,,,,\n")
    for file in filelist:
        log.write("# " + file + ",,,,,,,,,,\n")

    print(">> Created aggregate datalog:", filename)
    log.close()


main()




