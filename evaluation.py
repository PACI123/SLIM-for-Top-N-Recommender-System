import csv
from decimal import Decimal

PRECISION_Value_AT = 10

def precision_value(recommendations, test_file):
    
    users_items = {}
    with open(test_file,'r') as test:
         csv_reader=csv.reader(test)
         for lin in csv_reader:
              u, i, v = lin
              u, i = int(u), int(i)
            
              v = 1
              if u in users_items:
                 users_items[u].add(i)
              else:
                  users_items[u] = set([i])

    precisions = []
  
    total_users =Decimal( len(recommendations.keys()))
    for at in range(1, PRECISION_Value_AT+1):
        mean = 0
        kjj=0
        for u in recommendations.keys():
            
            try:
                relevant_item = users_items[u]
        
                retrieve_item = recommendations[u][:at]
                precision = len(relevant_item & set(retrieve_item))/Decimal(len(retrieve_item))
                gf=len(relevant_item & set(retrieve_item))/Decimal(len(relevant_item))
                kjj+=gf
                mean += precision
            except KeyError:
                None

        
        print('Average Precision @%s: %s' % (at, (mean/total_users)))
        
        
        precisions.append([at, (mean/total_users)])
        
            
    return precisions




   
