Location_Matrix = __import__("Location Matrix")
#from Location_Matrix import LocationMatrix
from Location_Calculation import LocationCalculation
Predict_Location = __import__("Predict Location")
#from Predict_Location import predictLocation
import numpy as np


# for a in range(2):
#     Location_Matrix.LocationMatrix(alpha = 0.1*a, training_fraction = 0.10)
#     LocationCalculation(alpha = 0.1*a, training_fraction= 0.10)
#     A = Predict_Location.predictLocation(alpha = 0.1*a, training_fraction = 0.10)
#     print(len(A.index))
#     try:
#         overall = overall.append(A)
#     except:
#         overall = A
# print(len(overall.index))
Location_Matrix.LocationMatrix()
LocationCalculation()
A = Predict_Location.predictLocation()



