#!/usr/bin/env python

INPUT_FILEPATH = "../../thesis/testsuite/"
OUTPUT_FILEPATH = "../../thesis/testsuite/"


# Bitmap class
from m_bitmapClass import BMP

# Filter library
from m_filter import countSignal    # Counts black pixels in image
from m_filter import attenuate      # Removes black signal from image

# Output results helper function
def print_results(result ):
    if( result ):
        print("> PASS\n")
        return 0
    else:
        print("> FAIL\n")
        return 1

total_tests = 0
total_fails = 0

# TEST 001 --------------------------------------
print("> Testcase #001 (import Bitmap Image)")
total_tests += 1
result = True
bmp = BMP("norway.bmp",INPUT_FILEPATH)
if( bmp.width != 500 ):
    result = False
    print("(!) Width Dimension Error")
if( bmp.height != 400 ):
    result = False
    print("(!) Height Dimension Error")
total_fails += print_results(result)

# TEST 002 --------------------------------------
print("> Testcase #002 (test countSignal filter)")
total_tests += 1
result = True
bmp = BMP("norway.bmp",INPUT_FILEPATH)
c = countSignal(bmp)
if( c != 5114 ):
    result = False
    print("(!) Error in countSignal result:",c)
total_fails += print_results(result)

# TEST 003 --------------------------------------
print("> Testcase #003 (test attenuate filter)")
total_tests += 1
result = True
bmp = BMP("norway.bmp",INPUT_FILEPATH)
c1 = countSignal(bmp)
bmp = attenuate(bmp,1000)
c2 = countSignal(bmp)
if( (c1-c2) != 1000 ):
    result = False
    print("(!) Error in attenuate filter:",c1,c2)
bmp = attenuate(bmp,1000)
c3 = countSignal(bmp)
if( (c2-c3) != 1000 ):
    result = False
    print("(!) Error in attenuate filter:",c2,c3)
print(c1,c2,c3)
total_fails += print_results(result)













# Final Results:
print("> Total Tests Run:",total_tests)
print("> # Failed:",total_fails)



