import numpy as np
import math
from scipy.special import comb

from sklearn import cluster
from sklearn import metrics
from sklearn import neighbors

# return nmi,f1; n_cluster = num of classes 
def evaluate_cluster(feats,labels,n_clusters):

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(feats)
    centers = kmeans.cluster_centers_

    ### k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers,range(len(centers)))

    idx_in_centers = neigh.predict(feats)
    num = len(feats)
    d = np.zeros(num)
    for i in range(num):
        d[i] = np.linalg.norm(feats[i,:] - centers[idx_in_centers[i],:])  

    labels_pred = np.zeros(num)
    for i in range(n_clusters):
        index = np.where(idx_in_centers == i)[0];
        ind = np.argmin(d[index]);
        cid = index[ind];
        labels_pred[index] = cid;


    nmi,f1 =  compute_clutering_metric(labels, labels_pred)
    return nmi,f1


def evaluate_recall(features,labels):
   
    class_ids = labels
    dims = features.shape

    D2 = distance_matrix(features)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    diagn = np.diag([float('inf') for i in range(0, D.shape[0])])
    D = D + diagn
    recall = []
    for K in [1,5]:
        recall.append(compute_recall_at_K(D, K, class_ids, num))

    return recall

def evaluate_recall_asym(features_gallery,labels_gallery,features_query,labels_query):
   
    
    dims = features_query.shape

    D2 = distance_matrix_asym(features_query,features_gallery)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    recall = []
    for K in [1,10,20,30,40]:
        recall.append(compute_recall_at_K_asym(D, K, labels_gallery, labels_query, num))

    return recall


  
def compute_clutering_metric(idx,item_ids):

    N = len(idx);

    centers = np.unique(idx);
    num_cluster = len(centers);

    # count the number of objects in each cluster
    count_cluster = np.zeros((num_cluster));
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0]);


    # build a mapping from item_id to item index
    keys = np.unique(item_ids);
    num_item = len(keys);
    values = range(num_item);
    item_map = dict();
    for i in range(len(keys)):
        item_map.update([(keys[i],values[i])])



    # count the number of objects of each item
    count_item = np.zeros(num_item);
    for i in range(N):
        index = item_map[item_ids[i]];
        count_item[index] = count_item[index] + 1;

    # compute purity
    purity = 0;
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0];
        member_ids = item_ids[member];
        
        count = np.zeros(num_item);
        for j in range(len(member)):
            index = item_map[member_ids[j]];
            count[index] = count[index] + 1;
        purity = purity + max(count);

    purity = purity / N;

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item));
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0];
        index_item = item_map[item_ids[i]];
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1;


    # mutual information
    I = 0;
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]));
                I = I + s;


    # entropy
    H_cluster = 0;
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N));
        H_cluster = H_cluster + s;

    H_item = 0;
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N));
        H_item = H_item + s;

    NMI = 2 * I / (H_cluster + H_item);

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0;
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2);

    # compute True Positive (TP)
    tp = 0;
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0];
        member_ids = item_ids[member];
        
        count = np.zeros(num_item);
        for j in range(len(member)):
            index = item_map[member_ids[j]];
            count[index] = count[index] + 1;
        
        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2);

    # False Positive (FP)
    fp = tp_fp - tp;

    # compute False Negative (FN)
    count = 0;
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2);

    fn = count - tp;

    # compute True Negative (TN)
    tn = N*(N-1)/2 - tp - fp - fn;

    # compute RI
    RI = (tp + tn) / (tp + fp + fn + tn);

    # compute F measure
    P = tp / (tp + fp);
    R = tp / (tp + fn);
    beta = 1;
    F = (beta*beta + 1) * P * R / (beta*beta * P + R);

    return NMI,F




def distance_matrix(X):
    X = np.matrix(X)
    m = X.shape[0]
    t = np.matrix(np.ones([m, 1]))
    x = np.matrix(np.empty([m, 1]))
    for i in range(0, m):
        n = np.linalg.norm(X[i, :])
        x[i] = n * n
    D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)
    return D


def distance_matrix_asym(A, B):
    A = np.matrix(A)
    B = np.matrix(B)
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA())#**0.5
    return np.matrix(ED)


def compute_recall_at_K(D, K, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids[i] for i in knn_inds]

        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct)/float(num)

    return recall

def compute_recall_at_K_asym(D, K, class_ids_gallery, class_ids_query, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids_query[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids_gallery[i] for i in knn_inds]
        

        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct)/float(num)

    return recall



