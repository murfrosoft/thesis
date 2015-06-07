#!/usr/bin/ python3.4
#!/usr/bin/env python

# Here are some file I/O functions to help support other applications
# Written by MCM on February 22, 2015

def create( filename ):
    ''' filename to create '''
    file = open(filename, "w")
    file.close()
    return

def append_to(filename, data):
    ''' filename = file to append to, data = string '''
    
    # Open and read file data
    file = open(filename, "r")
    raw = file.read()
    file.close()

    # Re-open and append file data
    file = open(filename, "w")
    raw += data
    file.write(raw)
    file.close()
    print("> Data appended to " + filename)
    return
