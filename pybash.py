# create the bash script that will run all my datalog aggregation

file = open("aggregate.bash", 'w')
file.write("!#bin/bash\n\n")

# Template is: python3 aggregator.py path/to/logfiles/*

header = "python3 aggregator.py logfiles/"

folderset = ["blank","canopy","checkers","circle","fifty50","koch3lines","kochSnowflake","line"]

for folder in folderset:
    file.write(header + folder + "/*\n")

# Add fallleaf and owl edge detect set
for threshold in range(5,80,5):
    file.write(header + str(threshold) + "fallleaf/*\n")
    file.write(header + "owl_edge/" + str(threshold) + "owl/*\n")

# Add owl threshold set
for threshold in range(10,210,10):
    file.write(header + "owl_thresh/" + str(threshold) + "_thr_owl/*\n")

file.close()

