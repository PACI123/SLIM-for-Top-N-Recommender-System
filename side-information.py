from sklearn.linear_model import ElasticNetCV
import numpy as np
from recommendation import recommendation_slim
from usa import (csv_matrix, gen_slic,
                  make_compatible, normalize_values, save_matrix)
from evaluation import precision_value
import multiprocessing
import ctypes
import sys
from scipy.sparse import vstack
import argparse
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

print ('>>> Start: %s' % datetime.datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Matrix file to train the model',
                                        required=True)
parser.add_argument('--test', help='Matrix file to test the model',
                                        required=True)
parser.add_argument('--side_information',
                                        help='Side information to improve learning', required=True)
parser.add_argument('--beta', type=float, help=('Parameter that gives weight to the '
                                              'side_information matrix'))
parser.add_argument('--output', help=('File to put the results'), required=True)
parser.add_argument('--normalize', type=int, help=('Parameter that defines if data'
                    'in side information matrix will be normalized'),
                    required=False)
parser.add_argument('--fold', type=int, help=('Parameter only to mark the fold '
                    'that is being calculated'),
                    required=False)
args = parser.parse_args()

train_file = args.train
user_sideinformation_file = args.side_information
output = args.output
test_file = args.test
beta = args.beta or 0.011
normalize = args.normalize
fold = args.fold or 0
A = csv_matrix(train_file)
A[A>0]=1

SS = csv_matrix(user_sideinformation_file)

if normalize:
    SS = normalize_values(SS)

A, SS = make_compatible(A, SS)


shared_array_base = multiprocessing.Array(ctypes.c_double, A.shape[1]**2)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(A.shape[1], A.shape[1])


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

      
        M[:, j] = 0

        model.fit(M, mlinej.toarray().ravel())

       
        M[:, j] = mlinej

        w = model.coef_
        

      
        w[w<0] = 0
        

        for el in w.nonzero()[0]:
            S[(el, j)] = w[el]
    
   
 
        

    return S 

def  trainingslim(A,SS,beta=3):
   
    
    
    
     model =  ElasticNetCV(
        cv=10,fit_intercept=True,alphas=[0.14],l1_ratio=[0.815],max_iter=100
         )

   
     Balpha = 3 * SS
     Mline = vstack((A, Balpha),format='lil')

   
     num_coloum = Mline.shape[1]
     ranges = gen_slic(num_coloum)
     separated_tasks = []

     for from_j, to_j in ranges:
         separated_tasks.append([from_j, to_j, Mline, model])

     pool = multiprocessing.Pool()
     pool.map(qark, separated_tasks)
     pool.close()
     pool.join()
        
       

     return shared_array


S =  trainingslim(A, SS, beta=3)


Balpha = 3 * SS
Mline = vstack((A, Balpha),format='lil')

kd=Mline*S
rad=sqrt(mean_squared_error(Mline.toarray(),kd))
print('rmse are')
print(rad)



recommendations = recommendation_slim(A, S)

precisions = precision_value(recommendations, test_file)

print ('>>> End: %s' % datetime.datetime.now())


