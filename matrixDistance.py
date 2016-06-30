import pickle
import numpy as np
import scipy.sparse as sps
import math
FRACTION = 0.5
(W,H) = pickle.load(open('location_NMF_100_topics_barc_WH.pkl', 'rb'))
W = sps.csr_matrix(W)
H = sps.csr_matrix(H)
L_full = pickle.load(open('Location_matrix_full.pkl', 'rb'))
length = L_full.shape[0]
print(length)
#L_known = L_full[:int(FRACTION*length),:]
#print(L_known.shape)
L_known = L_full[(int(FRACTION*length)):,:]
L_known = L_known.toarray()
print(L_known.shape)

ROWS = 100
COLS = 100

predictedLocationMatrix = W*H

L_predict = predictedLocationMatrix[int(FRACTION*length):, -(ROWS*COLS):]
#SHUFFLE L_PREDICT
import sklearn.utils
L_predict = sklearn.utils.shuffle(L_predict)
L_predict = L_predict.toarray()
print(L_predict.shape)

# for i in list(range(0,L_predict.shape[0])):
#     #print("Actual")
#     #print(sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist())
#     #print("Predicted")
#     predictedCoordinateList = L_full[i].toarray()[0]#.tolist()
#     if (i <= 5):
#         print(predictedCoordinateList)

#     actualCoordinateList = sps.csr_matrix.nonzero(L_predict[i])[1].tolist()
#     if (i<=5):
#         print(actualCoordinateList)

#     #print(A.index(max(A)))
#     #print(actualCoordinateList)
#     k = 0
#     l = 0 
#     n = 0
    #maxPredictedCoordinate = predictedCoordinateList.index(max(predictedCoordinateList))
maxPredictedCoordinate = np.argmax(L_predict, axis = 1)
print(maxPredictedCoordinate[:20])
maxKnownCoordinate = np.argmax(L_known, axis = 1)
print(maxKnownCoordinate[:20])

def wrapper(indexArray,row,col):
    yCoordinate = indexArray//row
    xCoordinate = indexArray % col
    return (xCoordinate, yCoordinate)


xPredict, yPredict = wrapper(maxPredictedCoordinate, ROWS, COLS)
xKnown, yKnown = wrapper(maxKnownCoordinate, ROWS, COLS)
xDiff = xPredict - xKnown
yDiff = yPredict - yKnown
distance = (xDiff**2 + yDiff**2)
DistanceList = distance[:].tolist()
DistanceList.sort()
print(DistanceList[:1200])
#random shuffle L_predict

#(xCoordinate, yCoordinate) = wrapper(maxPredictedCoordinate)

# print(str(maxPredictedCoordinate))
# if (maxPredictedCoordinate in actualCoordinateList):
#     k += 1
# # if (distanceFromCenter(actualCoordinateList,predictedCoordinateList) < 3):
# #     l += 1
# if (actualCoordinateList == []):
#     n += 1
# print(k)
# print(l)
# print(n)
# rows = 100
# cols = 100
# def oldAndNewLocation(KnownLocationNMFMatrix, W, H):
#     predictedLocationMatrix = W*H #Instead of W tried all the unknown rows
#     							  #Then tried the 10,000 columns of H
#     print("Finish")
#     #Obtain the row*col rightmost data to be LocationMatrix
#     knownLocationMatrix = sps.vstack(KnownLocationNMFMatrix[i] for i in unknownIndex)
#     print(knownLocationMatrix.shape)
#     print("Finish1")
    
#     #Obtain the row*col rightmost data to be predictedLocationMatrix
#     length = knownLocationMatrix.shape[0]
#     predictedLocationMatrix = predictedLocationMatrix.toarray()
#     predictedLocation = []
#     for i in unknownIndex:
#         predictedLocation += [predictedLocationMatrix[i][(-rows*cols):]]
#     #predictedLocationMatrix = sps.vstack(predictedLocationMatrix[i][(-rows*cols+1):] for i in unknownIndex)
#     predictedLocation = sps.csr_matrix(predictedLocation)
#     print(predictedLocation.shape)
#     print("Finish2")
#     return (knownLocationMatrix, predictedLocation)
# print("Here4")

# knownLocationMatrix, predictedLocation = oldAndNewLocation(KnownLocationNMFMatrix, W, H)


# #Change index to (x,y) coordinate


# def distanceFromCenter(actualCoordinateList, predictedCoordinateList):
#     center = actualCoordinateList[4] #MAGIC NUMBER HERE!!!
#     centerCoordinate = wrapper(center,rows,cols)
#     predict = predictedCoordinateList.index(max(predictedCoordinateList))
#     predictCoordinate = wrapper(predict,rows,cols)
#     return float(math.sqrt( (predictCoordinate[1]-centerCoordinate[1])**2+ (predictCoordinate[0]-centerCoordinate[0])**2))

# def testAccuracy(knownLocationMatrix, predictedLocationMatrix):
#     j = 0
#     k = 0
#     l = 0
#     n = 0
#     for i in list(range(0,knownLocationMatrix.shape[0])):
#         #print("Actual")
#         #print(sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist())
#         #print("Predicted")
#         predictedCoordinateList = predictedLocation[i].toarray()[0].tolist()
#         actualCoordinateList = sps.csr_matrix.nonzero(knownLocationMatrix[i])[1].tolist()
        

#         #print(A.index(max(A)))
#         print(actualCoordinateList)
#         maxPredictedCoordinate = predictedCoordinateList.index(max(predictedCoordinateList))
#         print(str(maxPredictedCoordinate))
#         if (maxPredictedCoordinate in actualCoordinateList):
#             k += 1
#         if (distanceFromCenter(actualCoordinateList,predictedCoordinateList) < 3):
#             l += 1
#         if (actualCoordinateList == []):
#             n += 1
#     print("1 Grid away")
#     print(k*1.0/knownLocationMatrix.shape[0])
#     print("2-3 Grid away")
#     print(l*1.0/knownLocationMatrix.shape[0])
#     print("Empty nonzero list")
#     print(n)
#     print("-1")
#     #print(predictedLocation[-1].toarray()[0].tolist())
#     print("-2")
#     #print(predictedLocation[-2].toarray()[0].tolist())
# testAccuracy(knownLocationMatrix, predictedLocation)

