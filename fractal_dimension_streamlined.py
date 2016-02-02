# Base code for calculating the fractal dimension of a digital image.
# Thesis
# Michael C. Murphy
# Code started: November, 2016

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import log
from statistics import variance
import time
from random import randint


# Create an 8 digit job code
def jobcode():
    s = ''
    while len(s) < 8:
        n = randint(48,122)
        if chr(n).isalnum():
            s += chr(n)
    return s
    

# Create a time stamp for data logging
def timestamp():
    # Get current time
    t = time.localtime()
    stamp = str(t.tm_mday) + str(t.tm_mon) + str(t.tm_year)
    stamp += '_' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec)
    return stamp

# Create (append to) a log file
def datalog( D, filename ):
    ''' log data stored in dictionary D into file named filename '''
    exists = False
    try:
        log = open(filename, 'r')
        exists = True
        log.close()
    except:
        pass

    log = open(filename, 'a')
    if not exists:
        log.write("Timestamp,")
        log.write("Jobcode,")
        log.write("Imagename,")
        log.write("Noise,")
        log.write('G' + str(len(D['results'])) + ',')
        for i in range(len(D['results'])):
            log.write(str(D['results'][i][1]) + ',')
        log.write('\n')
    log.write(timestamp() + ',')
    log.write(D['jobcode'] + ',')
    log.write(D['image'] + ',')
    log.write(str(D['noise']) + ',')
    log.write('R' + str(len(D['results'])) + ',')
    for i in range(len(D['results'])):
        log.write(str(D['results'][i][0]) + ',')
    log.write('\n')
    log.close()
    return



# Small helper function: imports the png image file,
# strips the alpha channel, and returns a numpy array
# containing the red-green-blue channel for each pixel
def get_rgb_from(filename):
    img3D = mpimg.imread(filename, )            # read png img file
    img3D = np.delete(img3D,np.s_[3:],axis=2)   # strips alpha channel
    # condense into true 2D array
    img = np.empty( [len(img3D),len(img3D[0])],dtype=int )
    for row in range( len(img) ):
        for col in range( len(img[0]) ):
            if img3D[row][col].sum() == 0:
                img[row][col] = 0
            else:
                img[row][col] = 1
    return img

# add Noise (-1's in our case) to our image array)
def addNoise( inputArray, threshold, maxThreshold ):
    # 1st create a random array from 0 to maxThreshold value that is the
    # same size as our input array
    noise = np.random.randint(maxThreshold, size=(len(inputArray),len(inputArray[0])))

    # next, set the noise level to -1 for noisy image (would reduce a signal of 1 to noise of 0)
    # based on the threshold value
    noise = np.where( noise < threshold, -1, 0 )

    # Add Noise
    inputArray = inputArray + noise

    # Re-normalize
    inputArray = np.where( inputArray > 0, 1, 0 )
    
    return inputArray

# count number of black pixels ([0,0,0])
def countSignal( img ):
    return len(img)*len(img[0])-img.sum()
    '''
    pixels = 0
    for row in range( len(img) ):
        for col in range( len(img[0]) ):
            if img[row][col] == 0:
                pixels += 1
    return pixels
    '''

# test a small box of the image --
# return True of the box contains a black pixel ([0,0,0])
def testBox( img, x, y, d ):
    height = len(img)
    width  = len(img[0])

    for h in range(y, y+d):
        if img[h][x:x+d].sum() < d:
            return True
        #for w in range(x, x+d):
        #    if( h < height ) and (w < width):
        #        if img[h][w] == 0:
        #            return True
    return False

# pass the img array and a box width
# returns number of boxes counted at particular width
def boxCount( img, d ):
    #print("> Testing box size", d)
    height = len(img)
    width  = len(img[0])

    # verify d size is smaller than image dimensions
    if( d > min(height, width) ):
        print("[ERROR] boxCount box width exceeds image dimensions")
        return(0,0)
    counted = 0
    for y in range(0, height, d):
        for x in range(0, width, d):
            if( testBox(img, x, y, d) ):
                counted += 1
    return (counted, d)



# Convert an array of tuples of the form (Boxes Counted, Box Width) into a
# Fractal Dimension Estimate
# Pack data into a dictionary that can store multiple results data
# D['key'] => value
# D['dimension'] => Fractal Dimension
# D['results']   => List of tuples ( Boxes Counted, Box Width )
# D['l_results'] => List of tuples ( log(Boxes Counted), log(1/Box Width) )
# D['slopes']    => Array of log/log slopes

def bcaConvertResults( results ):

    D = {}
    D['results'] = results
    
    log_results = []
    
    for result in results:
        if( result[0] == 0 ):
            log_results.append( (log(1), log(1/result[1])) )
        else:
            log_results.append( (log(result[0]), log(1/result[1])) )

    D['l_results'] = log_results

    # calculate Dimensional estimate (using average)
    slopes = []
    for i in range(1,len(log_results)):
        slopes.append( (log_results[i][0]-log_results[i-1][0])/(log_results[i][1]-log_results[i-1][1]) )

    D['slopes'] = slopes
    D['variance'] = variance(slopes)
    # the average of the slopes gives us a fractal dimension estimate
    avg = 0
    for s in slopes:
        avg += s
    avg /= (len(slopes))

    D['dimension'] = avg

    return D

# Box Count Algorithm takes an img array (3D array containing pixel data in the
# form [r g b] each ranging from 0-1, arranged in a 2D matrix.
# [ [ [1,1,1], [1,1,1], ...
# [ [ [1,0,1], ...
# [ [ ...
# and takes a grid containing all the box widths we will test against
def boxCountAlgorithm( img, grid ):
    results = []        # (boxes counted, d)

    for d in grid:
        r = boxCount(img,d)
        results.append(r)

    D = bcaConvertResults( results )
    
    return D

# Print nicely formatted timestamp
def timeit( seconds ):
    m = int(seconds//60)
    s = int(seconds - m*60)
    ms = int( (seconds - m*60 - s)*100 )
    return format(m, '02d') + ':' + format(s, '02d') + '.' + format(ms, '03d')
    

# Create a pre-formated 2x2 plot of the output data.
# a, b, c, and d are arrays of tuples containing (Dimension, Noise%) data
def plot2x2( a, b, c, d ):
    noise = []
    for tup in a:
        noise.append(tup[1])
    ay = []
    by = []
    cy = []
    dy = []
    for i in range(len(a)):
        ay.append(a[i][0])
        by.append(b[i][0])
        cy.append(c[i][0])
        dy.append(d[i][0])

    fig = plt.figure(1)
    fig.suptitle('Fractal Dimension Results', fontsize = 20)

    ax = fig.add_subplot(221)
    ax.set_title('Circle')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Dimension')
    plt.plot(noise,ay,linestyle='-', linewidth=1.0)

    ax = fig.add_subplot(222)
    ax.set_title('KochSnowflake')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Dimension')
    plt.plot(noise,by,linestyle='-', linewidth=1.0)

    ax = fig.add_subplot(223)
    ax.set_title('Canopy')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Dimension')
    plt.plot(noise,cy,linestyle='-', linewidth=1.0)

    ax = fig.add_subplot(224)
    ax.set_title('Checkers')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Dimension')
    plt.plot(noise,dy,linestyle='-', linewidth=1.0)
    
    plt.show()

# Create a pre-formated 2x2 plot of the output data.
# a, b, c, and d are arrays of tuples containing (Dimension, Noise%) data
def plot2( a, b ):
    noise = []
    for tup in a:
        noise.append(tup[1])
    ay = []
    by = []

    for i in range(len(a)):
        ay.append(a[i][0])
        by.append(b[i][0])

    plt.figure(num=1, figsize=(12,8))
    fig = plt.figure(1)
    fig.suptitle('Fractal Dimension Results', fontsize = 20)
    

    ax = fig.add_subplot(2,1,1)
    ax.set_title('Fractal Dimension vs. Noise')
    #ax.set_xlabel('Noise')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('Dimension')
    plt.plot(noise,ay,linestyle='-', linewidth=1.0)

    bx = fig.add_subplot(2,1,2, sharex=ax)
    bx.set_title('Variance vs. Noise')
    bx.set_xlabel('Noise')
    bx.set_ylabel('Variance')
    plt.plot(noise,by,linestyle='-', linewidth=1.0)
    #plt.plot(noise,ay,linestyle='-', linewidth=2.0)

    
    plt.show()


''' START CODE BELOW '''
start_time = time.time()

job = jobcode()

# Test new dimensional function

# Now let's import an image
# Image is 1800 pixels tall and 2100 pixels wide.  We will use box sizes of
# 2, 5, 10, 20, 30, 50, 100, 150, 300
# NOTE: 9 iterations of 2100x1800 pixels means we could check up to
#       34 million pixels per image

'''
a = [(1, 0), (1.2, 1), (1.3, 2), (1.4, 3)]
b = [(1.2, 0), (1.3, 1), (1.4, 2), (1.5, 3)]
c = [(1.6, 0), (1.7, 1), (1.8, 2), (1.9, 3)]
d = [(2, 0), (1.9, 1), (2, 2), (2, 3)]

plot6330( a, b, c, d )

input("stopped")
'''
# Test setup:  box width list and list of images
grid = [3, 5, 10, 20, 30, 50, 100, 150, 300]
#image_library = ["circle.png","kochSnowflake.png","canopy.png","checkers.png"]
image_library = ["test_images/Larger/kochSnowflake.png"]
dim_test = [] # Holds the results of each images noise vs. dimension test
var_test = []

for image in image_library:
    # Open the image and convert to 2D nparray
    img = get_rgb_from(image)
    print("Image imported")
    print("---", timeit(time.time()-start_time), "---")
    print("Array is",len(img),"x",len(img[0]))

    height = len(img)
    width = len(img[0])

    dim_data = []  # Prepare to append array of tuples containing (Dimension, Noise%)
    var_data = []
    for noise in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,70,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1700,1900]:
        print("Adding Noise", noise, "%")
        newimg = addNoise( img, noise, 2000)
        c = countSignal( newimg )
        print("Signal Counted:",c)
        print("---", timeit(time.time()-start_time), "---")
        D = boxCountAlgorithm(newimg, grid)
        # Log the results
        D['image'] = image
        D['jobcode'] = job
        D['noise'] = noise
        datalog(D, 'long_noise_snowflake.log')

        print("Fractal Dimension:", D['dimension'])
        print("---", timeit(time.time()-start_time), "---")
        dim_data.append( (D['dimension'], noise) )
        var_data.append( (D['variance'], noise) )
    dim_test.append(dim_data)

    var_test.append(var_data)

print("--- Finished in: ", timeit(time.time()-start_time), "---")
# When we are done processing, let's plot the results:
plot2( dim_test[0], var_test[0] )







        
        
