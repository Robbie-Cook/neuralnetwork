infile = "irisoutput.txt"

file = open(infile)

line = file.readline()
while line:
    line = line.replace("\n", "")
    splitline = line.split()
    print(list(map(lambda x: float(x), splitline)), end=",\n")

    # print(splitline)
    line = file.readline()
