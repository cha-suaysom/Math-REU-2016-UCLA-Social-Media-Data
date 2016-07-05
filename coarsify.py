# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:37:57 2016

@author: Puck Rombach

The script "coarsify" takes as input an axb matrix M (in sparse, csr format), 
of which each column i corresponds to the set of tweets present in location i 
on a rowsnrxcolsnr map.
Therefore, we should have b=rowsnrxcolsnr. It also takes as input "minweight", 
which is the total weight at which to collapse columns.

The columns of M are collapsed as follows: the map is split in half through its 
largest dimsion (a "landscape" map is split in half using a vertical cut, a 
"portrait" map using a horizontal cut). Then, each half is either merged into 
one block (if the total weight is at most minweight), or cut again, and we 
proceed recursively.

The output is a new, smaller matrix Ms, and a dictionary D. D encodes which 
columns of M have been merged. For example, if columns i and j were merged, 
then D[i]=[i,j].


"""


def listcoord(rowsnr,colsnr):
    import math
    if rowsnr>colsnr:
        rowsnr1=math.floor(rowsnr/2)
        rowsnr2=math.ceil(rowsnr/2)
        colsnr1=colsnr
        colsnr2=colsnr
        l1=list(range(0,rowsnr1*colsnr))
    else:
        colsnr1=math.floor(colsnr/2)
        colsnr2=math.ceil(colsnr/2)
        rowsnr1=colsnr
        rowsnr2=colsnr
        l1=[]
        [[l1.append(j) for j in range(i*colsnr,i*colsnr+math.floor(colsnr/2))] for i in range(rowsnr)]    
    l2=list(set(range(rowsnr*colsnr))-set(l1))
    return (l1,l2,rowsnr1,rowsnr2,colsnr1,colsnr2)



def merge(M,rowsnr,colsnr,minweight,l,D):
    if rowsnr==1 and colsnr==1:
        D[l[0]]=l[0]
    else:
        if M[:,l].sum()<=minweight:
            D[l[0]]=l[:]
        else:
            (l1,l2,rowsnr1,rowsnr2,colsnr1,colsnr2)=listcoord(rowsnr,colsnr)
            D=merge(M,rowsnr1,colsnr1,minweight,l[l1],D)
            D=merge(M,rowsnr2,colsnr2,minweight,l[l2],D)
    return D


def coarsify(M,rowsnr,colsnr,minweight):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    l=np.array(range(rowsnr*colsnr))
    D={}
    D=merge(M,rowsnr,colsnr,minweight,l,D)
    k=list(D.keys())
    Ms=csr_matrix(np.zeros([M.shape[0],1]))
    for i in range(len(k)):
        Ms=csr_matrix(hstack([Ms,M[:,D[k[i]]].sum(axis=1)]))
    Ms=Ms[:,1:]
    return (Ms,D)
           




    