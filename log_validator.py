# Check a Uniform or Gaussian datalog file to insure it has all data.
# (Prompted because I moved half-finished data logs to repository because I
#  thought they were complete )
#
# Thesis work
# Michael C. Murphy
# Code started: May 14, 2016

import sys

def main():
    # USAGE:
    # python3 log_validator.py (-u or -g) path/to/datalogs/*

    # Read in each argument file and count the number of non-comment lines.
    # Should be 47 for Uniform, Y for Gaussian

    if( len(sys.argv) < 3):
        print("Insufficient arguments for log_validator.py")
        print("USAGE: python3 log_validator.py (-u or -g) path/to/datalogs/*")
        return

    noise_type = sys.argv[1][1]  # Should be 'u' or 'g'

    filelist = sys.argv[2:]
    badfiles = 0
    for filename in filelist:
        file = open(filename, 'r')

        count = 0
        for line in file:
            # commented lines are preceeded with '#'...
            if line[0] != '#':
                count += 1

        if noise_type == 'u':
            if count != 47:
                print(">>",filename,"counted (" + str(count) + "/47)")
                badfiles += 1
        elif noise_type == 'g':
            if count != 30:
                print(">>",filename,"counted (" + str(count) + "/30)")
                badfiles += 1

    if( badfiles == 0 ):
        print(">> All",str(len(filelist)),"Log Files Look Good!")
    else:
        print(">> Detected",badfiles,"bad files.")
        


main()

