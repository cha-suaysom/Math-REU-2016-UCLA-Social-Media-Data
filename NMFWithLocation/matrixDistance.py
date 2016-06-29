import pickle
import numpy as np
import scipy.sparse as sps
print("Here0")
KnownLocationNMFMatrix = pickle.load(open('KnownLocationMatrix.pkl', 'rb'))
print("Here1")
(W,H) = pickle.load(open('location_NMF_100_topics_barc_WH_with_alpha.pkl', 'rb'))
print("Here2")
W = sps.csr_matrix(W)
print("Here3")
H = sps.csr_matrix(H)
unknownIndex = pickle.load(open('unknownIndex.pkl', 'rb'))

rows = 100
cols = 100
def oldAndNewLocation(KnownLocationNMFMatrix, W, H):
    predictedLocationMatrix = W*H #Instead of W tried all the unknown rows
    							  #Then tried the 10,000 columns of H
    print("Finish")
    #Obtain the row*col rightmost data to be LocationMatrix
    knownLocationMatrix = sps.vstack(KnownLocationNMFMatrix[i] for i in unknownIndex)
    print(knownLocationMatrix.shape)
    print("Finish1")
    
    #Obtain the row*col rightmost data to be predictedLocationMatrix
    length = knownLocationMatrix.shape[0]
    predictedLocationMatrix = predictedLocationMatrix.toarray()
    predictedLocation = []
    for i in unknownIndex:
        predictedLocation += [predictedLocationMatrix[i][(-rows*cols):]]
    #predictedLocationMatrix = sps.vstack(predictedLocationMatrix[i][(-rows*cols+1):] for i in unknownIndex)
    predictedLocation = sps.csr_matrix(predictedLocation)
    print(predictedLocation.shape)
    print("Finish2")
    return (knownLocationMatrix, predictedLocation)
print("Here4")

knownLocationMatrix, predictedLocation = oldAndNewLocation(KnownLocationNMFMatrix, W, H)



def testAccuracy(knownLocationMatrix, predictedLocationMatrix):
    j = 0
    k = 0
    for i in list(range(0,knownLocationMatrix.shape[0])):
        #print("Actual")
        #print(sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist())
        #print("Predicted")
        A = predictedLocation[i].toarray()[0].tolist()
        leftRange = sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist()[0]
        rightRange = sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist()[-1]

        #print(A.index(max(A)))
        print(sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist())
        print(str(A.index(max(A))))
        if ((A.index(max(A)) <= rightRange) and (A.index(max(A)) >= leftRange)):
            j += 1
        if (A.index(max(A)) in sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist()):
            k += 1
    print("Accuracy")
    print(j)
    print("Super Accuracy")
    print(k)
testAccuracy(knownLocationMatrix, predictedLocation)