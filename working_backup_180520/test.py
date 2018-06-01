import neuralnetwork as nn

success = 0
failure = 0
maxEpochs = 3000
errorThreshold = 0.02

for i in range(0, 2000):
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

print("{} successful, {} failed: {}".format(success, failure, (success/(success+failure))*100))
nn.testNetwork()