from m_filter import filterEdgeDetect2
from m_bitmapClass import *

image = "test_images/fern.bmp"
image_name = image.split('/')

for threshold in range(5,80,5):
    bmp = BMP(image)
    bmp = filterEdgeDetect2(bmp,threshold,str(threshold)+image_name[1])
