#!/usr/bin/ python3.4
#!/usr/bin/env python
import copy

# Let's make a REAL BMP class
class BMP(object):
    def __init__(self, filename):
        ''' pass in a filename of bitmap image and strip down to components '''
        self.filename = filename
        self.header = []            # save header for ease of saving modified images
        self.pixels = []            # save pixel information here
        self.height = 0             # image height
        self.width  = 0             # image width
        self.padding = 0            # BMP padding data

        ''' open file and read raw data '''
        with open(self.filename, 'rb') as f:
            data = bytearray(f.read())
        f.close()

        ''' create Bitmap File Header '''
        BFH = BitmapFileHeader(self.filename, data[0:14])
        # not all Bitmaps are the same!  Need to pull the DIB header from
        # index 14 to BFH.startaddr
        DIB = DIBHeader(data[14:54])

        ''' create file header '''
        self.header = data[0:BFH.startaddr]

        ''' save height/width/padding details '''
        self.height = DIB.height
        self.width = DIB.width
        self.padding = DIB.padding

        """ Create an array of pixel data from the 'guts' of a .bmp file 
            Pixel data is stored as a 3-member array: [b,g,r] 
        """  
        padding = self.width%4   # explain this?
        print("padding:",padding)

        """ select image data subset """
        imageData = data[BFH.startaddr:BFH.filesize]
        
        """ Populate pixel matrix with RGB data """
        for h in range(self.height):
            tmp = []
            for w in range(self.width):
                rgb = [imageData[3*w+h*(self.width*3+padding)+0], \
                       imageData[3*w+h*(self.width*3+padding)+1], \
                       imageData[3*w+h*(self.width*3+padding)+2]]
                tmp.append(rgb)
            self.pixels.append(tmp)

    def __str__(self):
        s = "Filename: " + self.filename + " is "
        s += str(self.width) + "x" + str(self.height) + " pad[" + str(self.padding) + "]\n"
        s += "Pixel Count: " + str(len(self.pixels)*len(self.pixels[0])) + "\n"
        return(s)

    def save( self, savefilename ):
        # Save original image header (we are only manipulating pixels)
        data = copy.copy(self.header)
        
        # Create byte array from pixel data
        for h in range(self.height):
            for w in range(self.width):
                for d in range(3):
                    data.append(self.pixels[h][w][d])
            for d in range(self.padding):
                data.append(0x00)

        # Write file
        with open(savefilename, 'wb') as f:
            f.write(data)

        print("Saved to [" + savefilename + "], file size", len(data), "bytes")





class BitmapFileHeader(object):
    def __init__(self, name, bfhArray):
        self.filename = name
        self.type = bfhArray[0:2]
        self.filesize = int.from_bytes(bfhArray[2:6], byteorder='little')
        self.startaddr = int.from_bytes(bfhArray[10:14], byteorder='little')
        if( self.type[0] == 66 and self.type[1] == 77 ):
            self.valid = "yes"
        else:
            self.valid = "no"

    def __str__(self):
        s = "BFH:\t"
        s += "name: [" + self.filename + "]"
        s += "\n\ttype: [" + str(self.type) + "]"
        s += "\n\tsize: [" + str(self.filesize) + "]"
        s += "\n\taddr: [" + str(self.startaddr) + "]"
        s += "\tvalid: [" + self.valid + "]"
        return(s)

class DIBHeader(object):
    def __init__(self, dibArray):
        self.size = int.from_bytes(dibArray[0:4], byteorder='little')
        self.width = int.from_bytes(dibArray[4:8], byteorder='little')
        self.height = int.from_bytes(dibArray[8:12], byteorder='little')
        self.colorPlanes = int.from_bytes(dibArray[12:14], byteorder='little')
        self.bitsPerPixel = int.from_bytes(dibArray[14:16], byteorder='little')
        self.compression = int.from_bytes(dibArray[16:20], byteorder='little')
        self.imgsize = int.from_bytes(dibArray[20:24], byteorder='little')
        self.hRez = int.from_bytes(dibArray[24:28], byteorder='little')
        self.vRez = int.from_bytes(dibArray[28:32], byteorder='little')
        self.colorPalette = int.from_bytes(dibArray[32:36], byteorder='little')
        self.numImptClrs = int.from_bytes(dibArray[36:40], byteorder='little')
        self.padding = self.width%4

    def __str__(self):
        s = "DIB:\t"
        s += "WxH: [" + str(self.width) + "x" + str(self.height) + "]"
        s += "\timgSz: [" + str(self.imgsize) + "]"
        s += "\tsize: [" + str(self.size) + "]"
        s += "\n\t#planes-" + str(self.colorPlanes) + " "
        s += "bpPix-" + str(self.bitsPerPixel) + " "
        s += "compr-" + str(self.compression) + " "
        s += "\n\thRez-" + str(self.hRez) + " "
        s += "vRez-" + str(self.vRez) + " "
        s += "clrPal-" + str(self.colorPalette) + " "
        s += "imptclrs-" + str(self.numImptClrs)
        s += "\n\tpadding: [" + str(self.padding) + "]"
        return(s)
