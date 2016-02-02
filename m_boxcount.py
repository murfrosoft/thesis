# This module will perform the box count algorithm on a BMP image object

from m_filter import intensity   # should i define this locally?
from math import log
from statistics import variance
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# getID() written for Master's Thesis work
# by Michael C. Murphy
# on June 8, 2015
# ------------------------------------------------
def getID():
    """ Use local machine datetime to create a 12 digit ID
    Returns a 12 digit ID as a string """
    return datetime.strftime(datetime.now(), '%y%m%d%H%M%S')



#############################################################
# paintbox(x,y,size) - colors a box section that is counted
# x -> starting x coordinate
# y -> starting y coordinate
# size -> size of the box to paint
#############################################################
def paintBox(bmp,x,y,gridsize):
    ''' alters global g_pix pixel data
    (x = starting x coordinate, y = starting y coordinate, size = boxsize)
    '''
    # paint a region red that detects a crossing
    for h in range(y,y+gridsize):
        for w in range(x,x+gridsize):
            if( h < bmp.height ) and ( w < bmp.width ):
                bmp.pixels[h][w] = (0,255,0)


# ------------------------------------------------
# testBox() written for Master's Thesis work
# by Michael C. Murphy
# on May 12, 2015
# ------------------------------------------------
def testBox(bmp,x,y,gridsize,paint=False):
    """ bmp -> BMP image object
        x -> starting x coordinate
        y -> starting y coordinate
        gridsize -> size of box to text
    """
    
    for h in range(y,y+gridsize):
        for w in range(x,x+gridsize):
            if( h < bmp.height ) and ( w < bmp.width ):
                # if intensity is above some defined threshold
                if( intensity(bmp.pixels[h][w]) == 0 ):
                    if( paint == True ):
                        paintBox(bmp,x,y,gridsize)
                    return True
    return False



# ------------------------------------------------
# boxCount() written for Master's Thesis work
# by Michael C. Murphy
# on May 12, 2015
# ------------------------------------------------
def boxCount( bmp, gridsize, start_xy = (0,0), end_xy = "default" ):
    """ bmp -> BMP image object
        gridsize -> width of box we are counting
        start_xy -> subimage starting coordinate (default bottom corner)
        end_xy -> subimage ending coordinate (default do full image)

        returns: tuple containing (#counted, gridsize)
    """
    paint = False

    ''' debug outputs '''
    #print(">Detected image width:",bmp.width)
    #print(">Detected image height:",bmp.height)
    #print(">Box size:",gridsize)
    
    # Verify start_xy is within image range
    if( start_xy[0] >= bmp.width or start_xy[1] >= bmp.height):
        print("[E01] Error, start_xy coordinate out of image range.")
        return(0,0)
    # Verify end_xy is within image range AND greater than start_xy
    if( end_xy == "default" ):
        end_xy = (bmp.width, bmp.height)
    if( end_xy[0] > bmp.width ):
        print("[E10] Error: end_x over width range.")
        return(0,0)
    elif( end_xy[1] > bmp.height ):
        print("[E11] Error: end_y over height range.")
        return(0,0)
    elif( end_xy[0] <= start_xy[0] ):
        print("[E12] Error: end_x less than start_x.")
        return(0,0)
    elif( end_xy[1] <= start_xy[1] ):
        print("[E13] Error: end_y less than start_y.")
        return(0,0)
    # Verify grid size parameter
    if( gridsize > min(bmp.width, bmp.height) ):
        print("[E20] Error: grid size larger than image dimensions.")
        return(0,0)

    # We should have valid parameters here
    #print("> Parameters:", start_xy, end_xy, gridsize)

    # Let's do the algorithm
    #total = 0
    counted = 0
    for y in range(start_xy[1], end_xy[1], gridsize):
        for x in range(start_xy[0], end_xy[0], gridsize):
            #total += 1
            if( testBox(bmp,x,y,gridsize,paint) ):
                counted += 1
    #print(" - total",total)
    if( paint == True ):
        bmp.save("_p_g" + str(gridsize) + "_c" + str(counted) + "_" + bmp.filename)
    return (counted,gridsize)


def bca( bmp, grid, start_xy = (0,0), end_xy = "default" ,plot=False):
    """ box-count algorithm: computes the estimated Dimension and Variance of
        a sub-section of an image.
        bmp => image class
        grid => an array of grid sizes (e.g. [100,50,25,20,10,5,2])
        start_xy => start coordinate (if applying to subimage)
        end_xy => end coordinate (if applying to subimage)
        returns: Returns a dictonary of interesting values:
        d[avg_dimension]
        d[slope_variance] ...
    """
    results = []        # save (counted, gridsize)
    log_results = []    # save (log(counted), log(1/gridsize))

    # for each grid test case, count the number of boxes and append to results
    for g in grid:
        r = boxCount(bmp,g)
        results.append(r)
        # protect against log(0) case:
        if( r[0] == 0 ):
            log_results.append( (log(1),log(1/r[1])) )
        else:
            log_results.append( (log(r[0]),log(1/r[1])) )

    # calculate Dimensional estimate (using average)
    slopes = []
    for i in range(1,len(log_results)):
        slopes.append( (log_results[i][0]-log_results[i-1][0])/(log_results[i][1]-log_results[i-1][1]) )

    avg = 0
    for s in slopes:
        avg += s
    avg /= (len(slopes))
    var = variance(slopes)

    # Create a 'unique' timestamp ID for output file
    runID = getID()
    
    # Messages to display after algorithm completes:
    print(" > BCA on", bmp.filename,"(ID "+runID+") - D("+str(round(avg,5))+") V("+str(round(var,5))+")")   
    #print(results)
    #print(log_results)
    #print(slopes)
    

    '''
    # Save output to file
    output = "BCA on '"+bmp.filename+"' (ID: "+runID+")\n"
    output += "Size\tCount\n"
    for r in results:
        output += str(r[1])+"\t"+str(r[0])+"\n"
    output += "Dimension: "+str(avg)+"\n"
    output += "Variance: "+str(var)+"\n"
    ezSave(output,runID+"_"+bmp.filename[0:-4]+".txt",OUTPUT_FILE_PATH)
    '''
    #return (avg,var)
    d={}
    d['image'] = bmp.filename
    d['grid'] = grid
    d['avgD'] = avg
    d['var'] = var
    d['results'] = log_results
    d['slopes'] = slopes
    return d


    
