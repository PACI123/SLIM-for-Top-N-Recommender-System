import pdb


def recommendation_slim(A, S):
   
    # Saving the memory
    S= A * S
    A_predict = S

    recommendation_item = {}
    m, n = A.shape

  
    for u in range(1, m):
        for i in range(1, n):
            v = A_predict[(u,i)]
            if v > 0:
               
                if A[(u, i)] == 0:
                    if u not in recommendation_item:
                        recommendation_item[u] = [(i, v)]
                    else:
                        recommendation_item[u].append((i, v))

   
    for u in recommendation_item.keys():
        recommendation_item[u].sort(reverse=False, key=lambda x: x[1])

    for u in recommendation_item:
        for i, t in enumerate(recommendation_item[u]):
            recommendation_item[u][i] = t[0]

    return recommendation_item


