from sklearn.linear_model import ElasticNetCV
from usa import csv_matrix, gen_slic
from evaluation import precision_value
from recommendation import recommendation_slim
import numpy as np
import multiprocessing
import ctypes
import sys
import pdb
from sklearn.preprocessing import Binarizer
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import datetime

print ('>>> Start: %s' % datetime.datetime.now())

train_file,test_file=sys.argv[1:]
xt=csv_matrix(train_file)
xt[xt>0]=1
shared_array_base = multiprocessing.Array(ctypes.c_double, xt.shape[1]**2)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(xt.shape[1], xt.shape[1])
def qark(params, S=shared_array):
    from_j = params[0]
    to_j = params[1]
    M = params[2]
    model = params[3]
    counter = 0

    for j in range(int(from_j), int(to_j)):
        counter += 1
        if counter % 10 == 0:
            print ('Range %s -> %s: %2.2f%%' % (from_j, to_j,
                            (counter/float(to_j - from_j)) * 100))
        mlinej = M[:, j].copy()

        # We need to remove the column j before training
        M[:, j] = 0

        model.fit(M, mlinej.toarray().ravel())

        # We need to reinstate the matrix
        M[:, j] = mlinej

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w<0] = 0

        for el in w.nonzero()[0]:
            S[(el, j)] = w[el]



def trainingslim(xt, l1_reg=0.001, l2_reg=0.0001):
   
     model = ElasticNetCV(
              cv=10,fit_intercept=True,alphas=[0.14],l1_ratio=[0.795],max_iter=100 )

     num_coloum  = xt.shape[1]
     ranges = gen_slic(num_coloum)
     separated_tasks = []

     for from_j, to_j in ranges:
        separated_tasks.append([from_j, to_j, xt, model])

     pool = multiprocessing.Pool()
     pool.map(qark, separated_tasks)
     pool.close()
     pool.join()
    
    
    

     return shared_array


S = trainingslim(xt)

kgi=xt*S
""" 
print("inverse are:")
print(kgi)
"""
 
jjj=sqrt(mean_squared_error(xt.toarray(),kgi))
print('RMSE value are:')
print(jjj)



recommendations = recommendation_slim(xt, S)
precision_value(recommendations, test_file)
print ('>>> End: %s' % datetime.datetime.now())
