#!/usr/bin/ python3.4
#!/usr/bin/env python

# Here are some filters that operate on a BMP Class object
# Written by MCM on March 20, 2015

from random import random
WHITE = [255,255,255]
BLACK = [0,0,0]


def intensity( rgb ):
    """ Returns the average value of the rgb pixel data.

    Inputs:
    -- rgb: rgb pixel array in the form [b,g,r]

    Returns: average of the three values
    """
    return int( (rgb[0] + rgb[1] + rgb[2])/3 )

def luminosity( rgb ):
    return int( 0.21*rgb[2] + 0.72*rgb[1] + 0.07*rgb[0] )

def RGBtoHSL( rgb ):
    """ return a triple tuple containing HSL values given
        a triple tuple of rgb data ( blue, green, red ) format
    """
    # R' = R/255 (G' = G/255, B' = B/255)
    Rp = rgb[2]/255
    Gp = rgb[1]/255
    Bp = rgb[0]/255
    Cmax = max(Rp,Gp,Bp)
    Cmin = min(Rp,Gp,Bp)
    Delta = Cmax - Cmin
    if Delta == 0:
        Hue = 0
    elif Cmax == Rp:
        Hue = 60*(((Gp-Bp)/Delta)%6)
    elif Cmax == Gp:
        Hue = 60*((Bp-Rp)/Delta + 2)
    else:
        Hue = 60*((Rp-Gp)/Delta + 4)

    Lit = (Cmax+Cmin)/2

    if Delta == 0:
        Sat = 0
    else:
        Sat = Delta/(1-abs(2*Lit-1))
    #print("H:",Hue,"S:",Sat,"L:",Lit)
    return (Hue,Sat,Lit)

# NOTE:  Not sure if this filter is working...
def filterToHue( bmp, savefile = '' ):
    """ convert rgb to hue grayscale """
    for h in range(bmp.height):
        for w in range(bmp.width):
            HSL = RGBtoHSL( bmp.pixels[h][w] )
            hue = int(255*HSL[0]//360)    # convert to 0-255 range
            bmp.pixels[h][w] = (hue,hue,hue)
    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

# NOTE:  Not sure if this filter is working...
def filterToSat( bmp, savefile = '' ):
    """ convert rgb to hue grayscale """
    for h in range(bmp.height):
        for w in range(bmp.width):
            HSL = RGBtoHSL( bmp.pixels[h][w] )
            sat = int(255*HSL[1])    # convert to 0-255 range
            bmp.pixels[h][w] = (sat,sat,sat)
    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

# NOTE:  Not sure if this filter is working...
def filterToLight( bmp, savefile = '' ):
    """ convert rgb to hue grayscale """
    for h in range(bmp.height):
        for w in range(bmp.width):
            HSL = RGBtoHSL( bmp.pixels[h][w] )
            lit = int(255*HSL[2])    # convert to 0-255 range
            bmp.pixels[h][w] = (lit,lit,lit)
    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

# filter image using BMP class
def filterToRed( bmp, savefile = '' ):
    ''' bmp -> BMP object, savefile -> save a copy here
    Only pass red pixel data '''
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w][0] = bmp.pixels[h][w][2]
            bmp.pixels[h][w][1] = bmp.pixels[h][w][2]
    if( savefile != '' ):
        bmp.save(savefile)
    return bmp


# filter image using BMP class
def filterToGreen( bmp, savefile = '' ):
    ''' bmp -> BMP object, savefile -> save a copy here
    Only pass green pixel data '''
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w][0] = bmp.pixels[h][w][1]
            bmp.pixels[h][w][2] = bmp.pixels[h][w][1]
    if( savefile != '' ):
        bmp.save("a_green.bmp")
    return bmp


# filter image using BMP class
def filterToBlue( bmp, savefile = '' ):
    ''' bmp -> BMP object, savefile -> save a copy here
    Only pass blue pixel data '''
    for h in range(bmp.height):
        for w in range(bmp.width):
            bmp.pixels[h][w][1] = bmp.pixels[h][w][0]
            bmp.pixels[h][w][2] = bmp.pixels[h][w][0]
    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

def addNoise( bmp, noise, savefile = '' ):
    """ First, calculate the number of non-black pixels in image.
    Second, add the number of black pixels by 'noise' amount.
    -- bmp: BMP object
    -- noise: adjacent intensity comparison
    -- savefile: (optional) filename to save output
    """
    w_count = 0       # count number of non-black pixels
    non_signal = []   # holds coordinates of w_counts
    for h in range(bmp.height):
        for w in range(bmp.width):
            # Add coordinates to non_signal
            if( intensity(bmp.pixels[h][w]) != 0 ):
                w_count += 1
                non_signal.append((h,w))

    #print("> Counted", w_count, "non-signal pixels; length signal",len(non_signal))

    # If noise is greater than number of white pixels, then set noise ceiling
    if( noise > w_count ):
        noise = w_count

    # Now add noise
    noise_index = []    # holds random indicies for non_signal manipulation
    
    # we will fill up noise_index with random indicies
    while( len(noise_index) < noise ):
        # randomize based on size of non-black pixels
        add = int(random()*w_count)
        if( add not in noise_index ):
            noise_index.append(add)

    for n in noise_index:
        # --------- pixel height  ----  pixel width ------------
        bmp.pixels[non_signal[n][0]][non_signal[n][1]] = (0,0,0)

    # Save image to a new file if prompted
    if( savefile != '' ):
        bmp.save(savefile)
        
    return bmp

def attenuate( bmp, threshold, savefile = '' ):
    """ First, calculate the number of black pixels in image.
    Second, reduce the number of black pixels by 'threshold' amount.
    -- bmp: BMP object
    -- threshold: adjacent intensity comparison
    -- savefile: (optional) filename to save output
    """
    b_count = 0   # count the number of signal
    signal = []   # store the coordinates of signal
    # Count the amount of signal (black pixels) in image
    for h in range(bmp.height):
        for w in range(bmp.width):
            if (intensity(bmp.pixels[h][w]) == 0 ):
                b_count += 1
                signal.append((h,w))
    print("> Counted", b_count, "black pixels; length signal",len(signal))
    
    # Verify threshold <= black pixels
    if (b_count < threshold):
        print("> Threshold greater than signal.")
        return -1
    
    # Threshold is <= signal: time to remove signal
    atten_to = b_count - threshold  # attenuate signal to this count
    while( b_count > atten_to ):
        # Scan through image looking for black pixels
        for s in signal:
            # if black-pixel, randomly decide if it should be attenuated
            if( intensity(bmp.pixels[s[0]][s[1]]) == 0 ):
                if( int(random()*b_count) == 0 ):
                    bmp.pixels[s[0]][s[1]] = (255,255,255)
                    b_count -= 1
                    if( b_count <= atten_to ):
                        return bmp
                    
        
    

def filterEdgeDetect2( bmp, threshold, savefile = '' ):
    """ Edge detection by comparison to 2 neighbors
    Inputs
    -- bmp: BMP object
    -- threshold: adjacent intensity comparison
    -- savefile: (optional) filename to save output
    """
    b_count = 0
    w_count = 0
    for h in range(bmp.height-1):
        for w in range(bmp.width-1):
            ii = intensity(bmp.pixels[h][w])
            """ Compare to North and Right neighbor """
            if( ii > threshold + intensity(bmp.pixels[h][w+1]) or
                ii < intensity(bmp.pixels[h][w+1]) - threshold or
                ii > threshold + intensity(bmp.pixels[h+1][w]) or
                ii < intensity(bmp.pixels[h+1][w]) - threshold ):
                bmp.pixels[h][w] = BLACK
                b_count += 1
            else:
                bmp.pixels[h][w] = WHITE
                w_count += 1

    """ Test statistics about filter results """
    if( w_count == 0 ):
        w_count = 1  # avoid divide-by-zero
    print(">> Ratio of Black to White is",b_count,"/",w_count,":",str(round(b_count*100/w_count,1)) + "%")

    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

def filterThreshold( bmp, threshold, savefile = '' ):
    """ Threshold filter: if intensity of pixel is less
        than threshold, pixel is white, else black.
    Inputs
    -- bmp: BMP object
    -- threshold: intensity comparison
    -- savefile: (optional) filename to save output
    """
    b_count = 0
    w_count = 0
    for h in range(bmp.height):
        for w in range(bmp.width):
            ii = intensity(bmp.pixels[h][w])
            """ Compare to North and Right neighbor """
            if( ii > threshold ):
                bmp.pixels[h][w] = WHITE
                w_count += 1
            else:
                bmp.pixels[h][w] = BLACK
                b_count += 1

    """ Test statistics about filter results """
    if( w_count == 0 ):
        w_count = 1  # avoid divide-by-zero
    print(">> Ratio of Black to White is",b_count,"/",w_count,":",str(round(b_count*100/w_count,1)) + "%")


    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

def filterColorBias( bmp, bias, savefile = '' ):
    """

    """
    for h in range(bmp.height):
        for w in range(bmp.width):
            if( abs(bmp.pixels[h][w][0]-bmp.pixels[h][w][1]) > bias  or
                abs(bmp.pixels[h][w][0]-bmp.pixels[h][w][2]) > bias  or
                abs(bmp.pixels[h][w][1]-bmp.pixels[h][w][2]) > bias ):
                bmp.pixels[h][w] = WHITE

    if( savefile != '' ):
        bmp.save(savefile)
    return bmp

def filterBNW(bmp, threshold, savefile = '' ):
    """ Filter image to stark Black/White by comparing
        pixel intensity to the threshold
    """
    for h in range(bmp.height):
        for w in range(bmp.width):
            if( intensity(bmp.pixels[h][w]) > threshold ):
                bmp.pixels[h][w] = WHITE
            else:
                bmp.pixels[h][w] = BLACK

    if( savefile != '' ):
        bmp.save(savefile)
    return bmp


