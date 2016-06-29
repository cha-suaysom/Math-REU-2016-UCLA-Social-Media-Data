import math
import pickle

#Change index to (x,y) coordinate
def wrapper(index,row,col):
    xCoordinate = math.floor(index*1.0/col)
    yCoordinate = index % row
    return (xCoordinate, yCoordinate)

rows =100
cols =100
def distanceFromCenter(actualCoordinateList, predictedCoordinateList):
    center = actualCoordinateList[4] #MAGIC NUMBER HERE!!!
    centerCoordinate = wrapper(center,rows,cols)
    predict = predictedCoordinateList.index(max(predictedCoordinateList))
    predictCoordinate = wrapper(predict,rows,cols)
    return float(math.sqrt( (predictCoordinate[1]-centerCoordinate[1])**2+ (predictCoordinate[0]-centerCoordinate[0])**2))
