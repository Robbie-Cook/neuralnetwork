import neuralnetwork as nn

success = 0
failure = 0
maxEpochs = 2000
repeats = 1
errorThreshold = 0.02

for i in range(0, repeats):
    nn.initalise()
    j = 0
    for j in range(0, maxEpochs):
        error = nn.runEpoch()
        if error < errorThreshold:
            break
    if nn.runEpoch(teach=False) < errorThreshold:
        success += 1
    else:
        failure += 1
    print("Iteration {} completed: {} epochs taken".format(i+1, j+1))

print("{} successful, {} failed: {}% success".format(success, failure, (success/(success+failure))*100))
# nn.testNetwork()