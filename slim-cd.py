import pdb
from sklearn.linear_model import ElasticNetCV
from usa import csv_matrix
from evaluation import precision_value
from recommendation import recommendation_slim
import numpy as np
from scipy.sparse import lil_matrix
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix



def trainingslim(A, l1_reg=0.001, l2_reg=0.0001):
    
    

    model = ElasticNetCV(
              cv=10,fit_intercept=True,alphas=[0.14],l1_ratio=[0.795],max_iter=100 )


    
    r, t = A.shape

    
    S = lil_matrix((t, t))

    for j in range(t):
        if j % 50 == 0:
            print( '-> %2.2f%%' % ((j/float(t)) * 100))

        aj = A[:, j].copy()
        
        A[:, j] = 0

        model.fit(A, aj.toarray().ravel())
        
        A[:, j] = aj

        w = model.coef_
        
        
		
        w[w<0] = 0
        kg=w.nonzero()[0]
        

        for el in w.nonzero()[0]:
            S[(el, j)] = w[el]

    return S


def main(train_file, test_file):
    A = csv_matrix(train_file)

    S = trainingslim(A)
    kgi=A*S
    jjj=sqrt(mean_squared_error(A.toarray(),kgi.toarray()))
    print('rmse are:')
    print(jjj)

    
    recommendations = recommendation_slim(A, S)
    dam= recommendations
    gg=str(dam)
    fr=open("tfr.txt","w")
    fr.write(gg)
    fr.close()
    precision_value(recommendations, test_file)
    

if __name__ == '__main__':
    main('trained.csv',
         'tested.csv')

