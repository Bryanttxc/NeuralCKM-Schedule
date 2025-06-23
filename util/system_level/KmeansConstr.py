import os
import numpy as np
import argparse
import time
from scipy.io import loadmat, savemat 

from k_means_constrained import KMeansConstrained

def constrain_Kmeans(ue_list, num_lab, max_num_sublab, folder_path, numSlot):
    
    result = []
    cls_time = []
    for num_ue_idx in range(1,len(ue_list)+1):
        for lab in range(1, num_lab+1):
            for sub_lab in range(1, max_num_sublab+1):
                
                    start_time = time.time()
                    mat = os.path.join(folder_path, f'SE_matrix_lab_{lab}_{sub_lab}_numUE_{ue_list[num_ue_idx-1]}.mat')
                    
                    if os.path.exists(mat):
                        in_file = loadmat(mat)
                        X = in_file['SE_matrix']

                        numUE = X.shape[0]
                        numIRS = X.shape[1] - 1

                        # X = np.array([[3.4, 2.5, 2], [4.6, 1.4, 1], [2.5, 4.2, 1.5],
                                    #    [3.3, 3.5, 2.8], [2.6, 2.2, 2], [5, 3.8, 3]])
                        clf = KMeansConstrained(
                            n_clusters=numIRS,
                            size_min=numSlot,
                            size_max=np.ceil(numUE/numIRS),
                            random_state=0
                        )
                        res = clf.fit_predict(X)
                        tmp_time = time.time() - start_time
                        result.append(res)
                        cls_time.append(tmp_time)
                    else:
                        break

    savemat(os.path.join(folder_path, 'cluster_label.mat'), {'cluster_label': result, 'cluster_time': cls_time})

if __name__ == '__main__':

    # create parser
    parser = argparse.ArgumentParser('Config KmeansConstr')
    parser.add_argument('--ue_list', type=int, nargs='+', required=True, help='List of UE')
    parser.add_argument('--folder_path', type=str, required=True, help='Folder path')
    parser.add_argument('--num_slot', type=int, required=True, help='Number of slots')
    parser.add_argument('--num_lab', type=int, required=True, help='Number of labs')
    parser.add_argument('--max_num_sublab', type=int, required=True, help='Max number of sublabs')

    # parse
    args = parser.parse_args()

    ue_list = args.ue_list
    folder_path = args.folder_path
    numSlot = args.num_slot
    num_lab = args.num_lab
    max_num_sublab = args.max_num_sublab
    constrain_Kmeans(ue_list, num_lab, max_num_sublab, folder_path, numSlot)
