#!/usr/bin/env python

# My libraries:
from m_fileio import create, append_to
from m_bitmapClass import BitmapFileHeader, DIBHeader, BMP
from m_filter import attenuate, addNoise
from datetime import datetime


from m_filter import filterBNW
from m_filter import filterEdgeDetect2, filterThreshold
from m_filter import intensity
from math import log, sqrt
from random import random
from statistics import variance
import time

# Adding some project setup parameters to aid in file management
# Test Images to be processed are here:
TEST_IMAGE_FILE_PATH = "../../thesis/test_images/"
# Output files will be saved here:
OUTPUT_FILE_PATH = "../../thesis/output_files/"


# Globals --- YUCK :(
# TEST PARAMETERS:
#SUBWINDOW_SIZE = 100  #was 100
SCAN_RATE = 2  # scan rate in pixels ... how many pixels to jump between subsections
TC1 = (100, [50,25,10,5,2], "TC1")     # (Subwindow_size, Box-count widths)
TC2 = (100, [50,25,10,5,2,1], "TC2")   # (Subwindow_size, Box-count widths)
TC3 = (50, [25,10,5,2], "TC3")         # (Subwindow_size, Box-count widths)
TC4 = (50, [25,10,5,2,1], "TC4")       # (Subwindow_size, Box-count widths)
TESTCASE = TC3
SUBWINDOW_SIZE = TESTCASE[0]
BOXWIDTHS = TESTCASE[1]
TESTCASE_DESCRIPTION = TESTCASE[2] + "_" + str(SCAN_RATE)


COLOR_WHEEL = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255), (0,255,255)]
g_CWINDEX = 0
g_imageID = 10001

WHITE = [255,255,255]
BLACK = [0,0,0]

g_img_header = []
g_pix = []
g_height = 0
g_width = 0
g_padding = 0

# ------------------------------------------------
# boxCount() written for Master's Thesis work
# by Michael C. Murphy
# on May 12, 2015
# ------------------------------------------------
def getID():
    """ Use local machine datetime to create a 12 digit ID
    Returns a 12 digit ID as a string """
    return datetime.strftime(datetime.now(), '%y%m%d%H%M%S')


# THIS FUNCTION SHOULD BE DEPRECIATED !!!       
def createPixelMatrix( imgdata, width, height):
    """ Create a global array of pixel data from the 'guts' of a .bmp file

    Inputs:
    -- imgdata: The .bmp file raw pixel data array
    -- width: width of image data (stripped from .bmp header)
    -- height: height of image data (stripped from .bmp header)

    Returns nothing, but populates the global array g_pix

    Pixel data is stored as an 3-member array: [b,g,r] 
    """
    global g_pix
    
    padding = width%4
    # clear old data (needed for repeat runs)
    g_pix = []

    # populate matrix with RGB data
    for h in range(height):
        tmp = []
        for w in range(width):
            rgb = [imgdata[3*w+h*(width*3+padding)+0], \
                   imgdata[3*w+h*(width*3+padding)+1], \
                   imgdata[3*w+h*(width*3+padding)+2]]
            tmp.append(rgb)
        g_pix.append(tmp)


    
# THIS FUNCTION SHOULD BE DEPRECIATED !!!  
def import_image( filename , debug = False):
    global g_img_header, g_height, g_width, g_padding
    
    with open(filename, 'rb') as f:
        data = bytearray(f.read())
    # debug: print the file data
    '''
    for i in range(len(data)):
        print(i, "-", data[i])
    '''
    f.close()

    # Make Bitmap File Header
    BFH = BitmapFileHeader(filename, data[0:14])
    # not all Bitmaps are the same!  Need to pull the DIB header from
    # index 14 to BFH.startaddr
    DIB = DIBHeader(data[14:54])
    # Save Bitmap File Header
    g_img_header = data[0:BFH.startaddr]
    if debug:
        print(BFH)
        print(DIB)

    g_height = DIB.height
    g_width = DIB.width
    g_padding = DIB.padding

    createPixelMatrix(data[BFH.startaddr:BFH.filesize],DIB.width, DIB.height)



# THIS FUNCTION SHOULD BE DEPRECIATED !!! 
def save_image( filename ):
    global g_pix
    # Save original image header (we are only manipulating pixels)
    data = g_img_header
    
    # Create byte array from global pixel (g_pix) data
    for h in range(g_height):
        for w in range(g_width):
            for d in range(3):
                data.append(g_pix[h][w][d])
        for d in range(g_padding):
            data.append(0x00)

    # Write file
    with open(filename, 'wb') as f:
        f.write(data)

    print("Saved to [" + filename + "], file size", len(data), "bytes")

def d2g( file_prefix, folder = ""):
    """ Explore the gradient of fdm / fdvarm
        gradient = sqrt( delta(x)^2 + delta(y)^2 )
    """

    """ First read fdm file from /fdvm/ folder """
    # This will determine size of image file to convert
    file = open(folder + "fdvm/" + file_prefix + "_fdm.txt", "r")
    raw = file.read()
    file.close()
    
    data = []
    dataRow = []
    dataText = ''
    for c in raw:
        if c == ',':
            dataRow.append(float(dataText))
            dataText = ''
        elif c == '\n':
            data.append(dataRow)
            dataRow = []
        else:
            dataText += c

    d_width = len(data[0])
    d_height = len(data)

    """ Now create gradient file """
    maximum = 0
    #s = ""
    for x in range(d_height-1):
        for y in range(d_width-1):
            data[x][y] = sqrt( (data[x][y]-data[x+1][y])**2 + (data[x][y]-data[x][y+1])**2 )
            #s += str(data[x][y]) + ","
            if( data[x][y] > maximum ):
                maximum = data[x][y]
        #s += "\n"

    """ Now normalize gradient data and add 1 (put in range between 1 and 2)"""    
    s = ""
    for x in range(d_height):
        for y in range(d_width):
            data[x][y] = 1 + (data[x][y]/maximum)
            s += str(data[x][y]) + ","
        s += "\n"    

    file = open("gra_" + file_prefix + ".txt", "w")
    file.write(s)
    file.close()
    print(">> gradient processed? (maximum:",maximum,") >","gra_" + file_prefix + ".txt")

    # NOW BITMAPIFY
    print("> Expecting an image of size",len(data[0]),"x",len(data))

    imgFileName = folder + "o" + str(len(data[0])) + "x" + str(len(data)) + ".bmp"
    print("> Opening",imgFileName)
    bmp = BMP(imgFileName)

    """ Convert FDM data to Image Structure """
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w] = fd_to_color(data[h][w])

    bmp.save(folder + "grad_" + file_prefix + "_fdm.bmp")


def d2b( file_prefix, folder = ""):
    """ Convert data files to image files (file_prefix, folder = "")
        file_prefix => gives the root name of the test case and image
        i.e. for TC2_T01_fdvarm.txt, file_prefix = "TC2_T01"
        folder => folder if different than root"""

    """ First read fdm file from /fdvm/ folder """
    # This will determine size of image file to convert
    file = open(folder + "fdvm/" + file_prefix + "_fdm.txt", "r")
    raw = file.read()
    file.close()
    
    data = []
    dataRow = []
    dataText = ''
    for c in raw:
        if c == ',':
            dataRow.append(float(dataText))
            dataText = ''
        elif c == '\n':
            data.append(dataRow)
            dataRow = []
        else:
            dataText += c

    print("> Expecting an image of size",len(data[0]),"x",len(data))

    imgFileName = folder + "o" + str(len(data[0])) + "x" + str(len(data)) + ".bmp"
    print("> Opening",imgFileName)
    bmp = BMP(imgFileName)

    """ Convert FDM data to Image Structure """
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w] = fd_to_color(data[h][w])

    bmp.save(folder + "fdvm/" + file_prefix + "_fdm.bmp")

    """ Second read fdm file from /fdvarm/ folder """
    # This will determine size of image file to convert
    file = open(folder + "fdvarm/" + file_prefix + "_fdvarm.txt", "r")
    raw = file.read()
    file.close()
    
    data = []
    dataRow = []
    dataText = ''
    for c in raw:
        if c == ',':
            # Variance requires some scaling to match color conversion range
            dataRow.append(float(dataText)*4 + 1)
            dataText = ''
        elif c == '\n':
            data.append(dataRow)
            dataRow = []
        else:
            dataText += c

    print("> Expecting an image of size",len(data[0]),"x",len(data))

    imgFileName = folder + "o" + str(len(data[0])) + "x" + str(len(data)) + ".bmp"
    print("> Opening",imgFileName)
    bmp = BMP(imgFileName)

    """ Convert FDM data to Image Structure """
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w] = fd_to_color(data[h][w])

    bmp.save(folder + "fdvarm/" + file_prefix + "_fdvarm.bmp")
                                     
    return


def save_to( filename, fdvm ):
    # save fdvm data structure to save file
    s = ''
    for line in fdvm:
        for fd in line:
            s += str(fd) + ","
        s += "\n"
    
    file = open(filename, "w")
    file.write(s)
    file.close()
    print(filename + " was saved.")
    return


'''
# MOVED TO m_filter.py #######################
def intensity( rgb ):
    """ Returns the average value of the rgb pixel data.

    Inputs:
    -- rgb: rgb pixel array in the form [b,g,r]

    Returns: average of the three values
    """
    return int( (rgb[0] + rgb[1] + rgb[2])/3 )
'''

def filter_bnw():
    """ Converts global pixel color data to grayscale.

    Inputs: None

    Returns: None

    This function modifies the data store in the global pixel array.
    """
    global g_pix

    for h in range(g_height):
        for w in range(g_width):
            i = intensity(g_pix[h][w])
            g_pix[h][w] = [i,i,i]


#############################################################
# paintbox(x,y,size) - colors a box section that is counted
# x -> starting x coordinate
# y -> starting y coordinate
# size -> size of the box to paint
#############################################################
def paintbox(x,y,size):
    ''' alters global g_pix pixel data
    (x = starting x coordinate, y = starting y coordinate, size = boxsize)
    '''
    global g_pix
    
    # paint a region red that detects a crossing
    for h in range(x,x+size):
        for w in range(y,y+size):
            if( h < g_height ) and ( w < g_width ):
                g_pix[h][w]=COLOR_WHEEL[g_CWINDEX]


#############################################################
# testbox(x,y,size) - test a box to determine a 'black' pixel
#                     exists
# x -> starting x coordinate
# y -> starting y coordinate
# size -> size of the box to paint
#############################################################               
def testbox(x,y,size):
    ''' tests global g_pix pixel data for threshold limit
    (x = starting x coordinate, y = starting y coordinate, size = boxsize)
    '''
    global g_pix
    
    for h in range(x,x+size):
        for w in range(y,y+size):
            if( h < g_height ) and ( w < g_width ):
                # if intensity is above some defined threshold
                if( g_pix[h][w][0] == 0 ):
                    paintbox(x,y,size)
                    return True
    return False


    




# This boxcount method returns the box-count for a
# subsection of the processed image, starting at (startx,starty)
def boxcount(size, startx, starty):
    # Return a tuple containing (boxes counted / total boxes)
    counted = 0
    total = 0
    if( startx+SUBWINDOW_SIZE > g_width or starty+SUBWINDOW_SIZE > g_height ):
        print("ERROR: boxcount starting coordinates outside allowed range!")
        return(0,0)
    # TODO: add check to ensure startx,starty + size are inside image dimensions!
    for i in range(starty,starty+SUBWINDOW_SIZE,size):   # range(0,g_height,size)
        for j in range(startx,startx+SUBWINDOW_SIZE,size):  # range(0,g_width,size)
            total += 1
            if( testbox(i,j,size) ):
                counted += 1
    # not sure what to do with the case where we count 0 boxes, because
    # it hurts the log calculation later on... let's try this:
    if counted == 0:
        counted = 1
    return (counted,size)

# test the box-count of a subsection of an image
# filename -> name of image file (save in common directory)
# size -> pixel dimension of counting box
# startx -> starting x coordinate of subimage to test
# starty -> starting y coordinate of subimage to test
# save -> If save == True, then a copy of the processed image will be saved 
def testBoxcount(filename, size, startx, starty, save=False):
    global g_imageID
    
    import_image(filename)
    r = boxcount(size, startx, starty)
    if( save == True ):
        save_image(str(g_imageID) + "-result_" + filename + "(" +str(r[0]) + "-" + str(r[1]) + ").bmp")
        g_imageID += 1
    return r




def fd_to_color( fd ):
    # take in a fractal dimension between .875 to 2+
    # and convert it to an RGB color code
    rgb = [0,0,0]
    # Blue Curve:
    if fd < 0.875:
        rgb[0] = 0
    elif fd < 1.125:
        rgb[0] = round(255*(fd-.875)/(1.125-.875))
    elif fd < 1.375:
        rgb[0] = 255
    elif fd < 1.625:
        rgb[0] = round(255*(fd-1.625)/(1.375-1.625))
    else:
        rgb[0] = 0
    # Green Curve:
    if fd < 1.125:
        rgb[1] = 0
    elif fd < 1.375:
        rgb[1] = round(255*(fd-1.125)/(1.375-1.125))
    elif fd < 1.625:
        rgb[1] = 255
    elif fd < 1.875:
        rgb[1] = round(255*(fd-1.875)/(1.625-1.875))
    else:
        rgb[1] = 0
    # Red Curve:
    if fd < 1.375:
        rgb[2] = 0
    elif fd < 1.625:
        rgb[2] = round(255*(fd-1.375)/(1.625-1.375))
    elif fd < 1.875:
        rgb[2] = 255
    elif fd < 2.125:
        rgb[2] = round(255*(fd-2.125)/(1.875-2.125))
    else:
        rgb[2] = 0
    return rgb

def postProcessOutput(filename, data):
    # data should be a 2D matrix array of the same dimension as output file to overwrite...
    global g_pix
    
    import_image(filename,True)

    for h in range(g_height):
        for w in range(g_width):
            g_pix[h][w] = fd_to_color(data[h][w])

    keycode = int(10000*random())

    savefile = "x_" + str(keycode) + "_" + filename
    save_image(savefile)
    print("Output FDVM saved as " + savefile)
    return


    




'''
# MOVED TO m_filter.py ###############
def imageFilterColorBias(filename,bias):
    global g_pix
    
    import_image(filename)

    for h in range(g_height):
        for w in range(g_width):
            if( abs(g_pix[h][w][0]-g_pix[h][w][1]) > bias  or
                abs(g_pix[h][w][0]-g_pix[h][w][2]) > bias  or
                abs(g_pix[h][w][1]-g_pix[h][w][2]) > bias ):
                g_pix[h][w] = [255,255,255]

    savefile = "cb_" + filename
    save_image(savefile)
    return savefile
'''
'''
# OLD AN NOT USEFUL....
def imageFilterEdgeDetect1(filename,threshold):
    global g_pix

    import_image(filename)

    for h in range(g_height):
        for w in range(g_width-1):
            if( intensity(g_pix[h][w]) > threshold + intensity(g_pix[h][w+1] ) or
                intensity(g_pix[h][w]) < intensity(g_pix[h][w+1]) - threshold ):
                g_pix[h][w] = [0,0,0]
            else:
                g_pix[h][w] = [255,255,255]

    savefile = "ed1_" + filename
    save_image(savefile)
    return savefile
'''
'''
# Moved to m_filter.py ##########################33
def imageFilterEdgeDetect2(filename,threshold):
    """

    """
    global g_pix

    import_image(filename)

    for h in range(g_height-1):
        for w in range(g_width-1):
            ii = intensity(g_pix[h][w])
            if( ii > threshold + intensity(g_pix[h][w+1]) or
                ii < intensity(g_pix[h][w+1]) - threshold or
                ii > threshold + intensity(g_pix[h+1][w]) or
                ii < intensity(g_pix[h+1][w]) - threshold ):
                g_pix[h][w] = [0,0,0]
            else:
                g_pix[h][w] = [255,255,255]

    savefile = "ed2_" + str(threshold) + "_" + filename
    save_image(savefile)
    return savefile
'''

#This maintains color data
def imageFilterEdgeDetect2c(filename,threshold):
    global g_pix

    import_image(filename)

    for h in range(g_height-1):
        for w in range(g_width-1):
            ii = intensity(g_pix[h][w])
            if( ii > threshold + intensity(g_pix[h][w+1]) or
                ii < intensity(g_pix[h][w+1]) - threshold or
                ii > threshold + intensity(g_pix[h+1][w]) or
                ii < intensity(g_pix[h+1][w]) - threshold ):
                g_pix[h][w] = [0,0,0]
            #else:
            #    g_pix[h][w] = [255,255,255]

    savefile = "ed2c_" + filename
    save_image(savefile)
    return savefile

def imageFilterDespec(filename):
    global g_pix

    import_image(filename)
    count = 0
    for h in range(g_height):
        for w in range(g_width):
            if g_pix[h][w] == BLACK:  #only check if this pixel is black
                if h != 0 and h!= g_height-1 and w != 0 and w != g_width-1:
                    if( g_pix[h-1][w] == WHITE and g_pix[h+1][w] == WHITE and
                        g_pix[h][w-1] == WHITE and g_pix[h][w+1] == WHITE):
                        g_pix[h][w] = WHITE
    
    savefile = "ds_" + filename
    save_image(savefile)
    return savefile

##################################################
# Convert loaded image to stark Black/White image
##################################################
def imageFilterBNW(filename, bias):
    ''' Filter image to stark Black/White. Bias value is threshold level '''
    global g_pix
    
    import_image(filename)

    for h in range(g_height):
        for w in range(g_width):
            if( intensity(g_pix[h][w]) > bias ):
                g_pix[h][w] = [255,255,255]
            else:
                g_pix[h][w] = [0,0,0]

    savefile = "bnw_" + filename
    save_image(savefile)
    return savefile










#############################################################
# testbox(x,y,size) - test a box to determine a 'black' pixel
#                     exists
# x -> starting x coordinate
# y -> starting y coordinate
# size -> size of the box to paint
#############################################################               
def objTestbox(bmp,x,y,size):
    ''' tests global g_pix pixel data for threshold limit
    (x = starting x coordinate, y = starting y coordinate, size = boxsize)
    '''
    global g_pix
    
    for h in range(x,x+size):
        for w in range(y,y+size):
            if( h < bmp.height ) and ( w < bmp.width ):
                # if intensity is above some defined threshold
                if( bmp.pixels[h][w][0] == 0 ):
                    #paintbox(x,y,size)
                    return True
    return False


# NOTE:  This box-count algorithm only counts in a subwindow size.  Need to (re)make
# box-counting algorithm that is generic
def objBoxcount( bmp, size, startx, starty ):
    """ bmp -> BMP object, size -> box-width, startx/starty: pixel coordinates """
    # Return a tuple containing (boxes counted / total boxes)
    counted = 0
    total = 0
    if( startx+SUBWINDOW_SIZE > bmp.width or starty+SUBWINDOW_SIZE > bmp.height ):
        print("ERROR: boxcount starting coordinates outside allowed range!")
        return(0,0)
    # TODO: add check to ensure startx,starty + size are inside image dimensions!
    for i in range(starty,starty+SUBWINDOW_SIZE,size):   # range(0,g_height,size)
        for j in range(startx,startx+SUBWINDOW_SIZE,size):  # range(0,g_width,size)
            total += 1
            if( objTestbox(bmp,i,j,size) ):
                counted += 1
    # not sure what to do with the case where we count 0 boxes, because
    # it hurts the log calculation later on... let's try this:
    if counted == 0:
        counted = 1
    return (counted,size)


#############################################################
# testbox(x,y,size) - test a box to determine a 'black' pixel
#                     exists
# x -> starting x coordinate
# y -> starting y coordinate
# size -> size of the box to paint
#############################################################






           


# Fractal Dimension Varation Map (Using BMP class)
# Parameters:  filename => filename of image to process
#              bnw => flag to process black and white filter
#              cb  => flag to process color bias filter
def objFDVM(filename,folder = ""): 
    """ Print out Test Parameters """
    testcases = BOXWIDTHS
    print("# Testing file:",filename)
    print("# Using Test Case:", TESTCASE_DESCRIPTION)
    print("# Box-count Window Size:",SUBWINDOW_SIZE)
    print("# Box-count Scan Rate:",SCAN_RATE)
    print("# Box widths of:",BOXWIDTHS)

    """ Create a bitmap object from image file """
    bmp = BMP(folder+filename)

    # Calculate number of processes expected
    out_width = int((bmp.width-SUBWINDOW_SIZE)/SCAN_RATE + 1)
    out_height = int((bmp.height-SUBWINDOW_SIZE)/SCAN_RATE + 1)
    print(">> Expect a " + str(out_width) + " x " + str(out_height) + " output file.")
    total = out_width * out_height
    current = 1
    
    #g_CWINDEX = 0

    # define box count width cases in pixels
    #testcases = BoxWidths #[50,25,10,5,2]
    FDVMap = []
    FDVarMap = []

    for scanheight in range(0, bmp.height-SUBWINDOW_SIZE+1,SCAN_RATE):
        FDVMrow = []
        FDVarMrow = []
        for scanwidth in range(0,bmp.width-SUBWINDOW_SIZE+1,SCAN_RATE):
            # The following code calculates the fractal dimension of ONE
            # subsection of the image
            results = []
            """ For each test case, count boxes """
            for t in testcases:
                """ objBoxcount returns (count, size) of boxcount test """
                results.append(objBoxcount( bmp, t, scanwidth, scanheight))
                #results.append(testBoxcount(filename,t,scanwidth,scanheight,False)) # True => save results

            log_results = []
            for r in results:
                log_results.append( (log(r[0],2),log(1/r[1],2) ) )

            # calculate slopes (Dimension!)
            slopes = []
            for i in range(1,len(log_results)):
                slopes.append( (log_results[i][0]-log_results[i-1][0])/(log_results[i][1]-log_results[i-1][1]) )

            avg = 0
            for s in slopes:
                avg += s
            avg /= (len(slopes))

            #print(">Results:",results)
            #print(">LOG:",log_results)
            #print(">Slopes:",slopes)
            #print(">Variance:",str(variance(slopes)))
            if( current%100 == 0 ):
                print(str(current) + "/" + str(total) +" - " + str(round((current/total)*100,1)) + "% DIMENSION =",avg )
            current += 1
            
            FDVMrow.append(avg)
            FDVarMrow.append(variance(slopes))

            # Rotate Color Wheel (not currently using...)
            #g_CWINDEX += 1
            #if g_CWINDEX >= len(COLOR_WHEEL) :
            #    g_CWINDEX = 0
                
        FDVMap.append(FDVMrow)
        FDVarMap.append(FDVarMrow)

    save_to(folder + "fdvm/" + TESTCASE_DESCRIPTION + "_" + filename[0:-4] + "_fdm.txt",FDVMap)
    save_to(folder + "fdvarm/" + TESTCASE_DESCRIPTION + "_" + filename[0:-4] + "_fdvarm.txt", FDVarMap)

    return 



def GO(filename, folder = ""):
    """ Wrapper function for FDVM algorithm """

    """ Run FDVM Algorithm """
    fdvm = objFDVM(filename, folder)

    return


'''
# This boxcount method works well for processing an entire image
# (expects a width and height of image to be multiple of 100)
'''
def boxcount_old(size):
    # if size == 100, one box,
    # if size == 10, 100 boxes, etc...
    # Return a tuple containing (boxes counted / total boxes)
    counted = 0
    total = 0
    for i in range(0,g_height,size):
        for j in range(0,g_width,size):
            total += 1
            if( testbox(i,j,size) ):
                counted += 1
    return (counted,size)


''' Older algorithm '''
def testBoxCount_old(filename,size):
    import_image(filename)
    r = boxcount_old(size)
    #save_image("result_" + filename + "(" +str(r[0]) + "-" + str(r[1]) + ").bmp")
    return r




    
def fracDim(filename,testcases=[25,20,10,5,2]):
    # Save results to text file for easy transfer to Emily
    testfile = filename
    
    ''' (filename, colorbias filter (60), black&white filter (160) ) '''
    #if cb > 0:
    #    filename = imageFilterColorBias(filename,cb)  # 60 default
    #if bnw > 0:
    filename = imageFilterBNW(filename,120)   # 160 default

    t = time.time()
    
    results = []
    for t in testcases:
        results.append(testBoxCount_old(filename,t))

    log_results = []
    for r in results:
        log_results.append( (log(r[0],2),log(1/r[1],2) ) )

    # calculate slopes (Dimension!)
    slopes = []
    for i in range(1,len(log_results)):
        slopes.append( (log_results[i][0]-log_results[i-1][0])/(log_results[i][1]-log_results[i-1][1]) )

    avg = 0
    for s in slopes:
        avg += s
    avg /= (len(slopes))



    print(results)
#    print(log_results)
#    print(slopes)
    print("DIMENSION =",avg, "in " + str(time.time()-t))


def avgVariance( testfile, folder ):
    # Load FDVarM data #################################
    file = open(folder+testfile, "r")
    raw = file.read()
    file.close()

    dataArray = []
    dataValue = ''
    count = 0
    # read in raw data into matrix structure
    for c in raw:
        if c == ',':
            #print(">>",dataValue)
            dataArray.append(float(dataValue))
            dataValue = ''
            count += 1
        elif c == '\n':
            pass
        else:
            dataValue += c

    print("> Read",count,"values, length=", len(dataArray))
    a_sum = 0
    for a in dataArray:
        a_sum += a
    print("> Avg Variance =",a_sum/len(dataArray))
    # Difference in Variance
    diff_sum = 0
    prev = 0
    for a in dataArray:
        diff_sum += abs(a - prev)   # take absolute difference
        prev = a
    print("> Diff Variance =",diff_sum/len(dataArray))
    
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
    total = 0
    counted = 0
    for y in range(start_xy[1], end_xy[1], gridsize):
        for x in range(start_xy[0], end_xy[0], gridsize):
            total += 1
            if( testBox(bmp,x,y,gridsize,paint) ):
                counted += 1
    print(" - total",total)
    if( paint == True ):
        bmp.save("_p_g" + str(gridsize) + "_c" + str(counted) + "_" + bmp.filename)
    return (counted,gridsize)

def bca( bmp, grid, start_xy = (0,0), end_xy = "default" ):
    """ box-count algorithm: computes the estimated Dimension and Variance of
        a sub-section of an image.
        bmp => image class
        grid => an array of grid sizes (e.g. [100,50,25,20,10,5,2])
        start_xy => start coordinate (if applying to subimage)
        end_xy => end coordinate (if applying to subimage)
    """
    results = []        # save (counted, gridsize)
    log_results = []    # save (log(counted), log(1/gridsize))
    for g in grid:
        r = boxCount(bmp,g)
        results.append(r)
        log_results.append( (log(r[0]),log(1/r[1])) )

    # calculate Dimensional estimate (using average)
    slopes = []
    for i in range(1,len(log_results)):
        slopes.append( (log_results[i][0]-log_results[i-1][0])/(log_results[i][1]-log_results[i-1][1]) )

    avg = 0
    for s in slopes:
        avg += s
    avg /= (len(slopes))

    print( "~D=",avg, "; V=",variance(slopes))
    
    print(results)
    print(log_results)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    image_name = "circle.bmp"
    testimage = TEST_IMAGE_FILE_PATH + image_name

    # Create BMP object of image file    
    bmp = BMP(testimage)

    # Setup test parameters
    grid = [100,50,25,20,10,5,2]

    # Perform Box Count Algorithm
    bca(bmp, grid)
    
    '''
    for i in range(10):
        bmp = addNoise(bmp,1000,"n+" + str((i+1)*1000) + "_circle.bmp")
        
    '''

    
    
   

