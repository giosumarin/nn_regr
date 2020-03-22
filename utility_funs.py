import numpy as np


def make_structured_input_for_root_NN(bin_data, labels, split, dim_set):
    position_labels = np.copy(labels)
    size_batch = dim_set // split
    remain = dim_set % split #ceil((dim_set / split - size_batch) * split)
    for i in range(0, split, 1):
        position_labels[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))]=i
    
    position_labels = np.reshape(position_labels, (-1,1))
    splitted_labels = [np.reshape(labels[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))]/dim_set, (-1,1)) for i in range(split)]
    splitted_bin_data = [bin_data[(i*size_batch):(((i+1)*size_batch if i < (split-1) else (i+1)*size_batch+remain))] for i in range(split)] 
    
    perm_splitted_bin_data = []
    perm_splitted_labels = []
    for i in range(split):
        #np.random.RandomState(42)
        p_split = np.random.RandomState(seed=0).permutation(len(splitted_bin_data[i]))
        
        perm_splitted_labels.append(splitted_labels[i][p_split])
        perm_splitted_bin_data.append(splitted_bin_data[i][p_split])
    
    

    #perm_splitted_bin_data = [splitted_bin_data[i][p_split] for i in range(split)]
    p = np.random.RandomState(seed=0).permutation(dim_set)
    perm_position_labels = position_labels[p]
    #perm_splitted_labels = [splitted_labels[i][p_split] for i in range(split)]


    return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels
    
    
def make_labels_for_class(position_labels):
    n_rows = position_labels.shape[0]
    result = np.zeros((n_rows,2))
    for i in range(n_rows):
        result[i,position_labels[i]] = 1
    return result

def scaler(original_label):
    return (original_label - np.min(original_label))/(np.max(original_label) - np.min(original_label))

def descaler(original_label, predicted_scaled):
    return predicted_scaled * (np.max(original_label) - np.min(original_label)) + np.min(original_label)

def number_to_tex(num):
    exp=0
    while(num<1):
        num *= 10
        exp-=1
    if exp == 0:
        return "{}".format(round(num,2))
    else:
        return "{}\\times 10^{{{}}}".format(round(num,2), exp)

def remove_column_zero(bin_data):
    i = 0
    while True:
        if not np.any(bin_data[:,i]):
            i += 1
        else:
            break
    return bin_data[:,i:bin_data.shape[1]], i

def reshape_nozero(w, index_nonzero):
    weights = w[0][0]
    bias = w[0][1]
    if len(w) == 1:
        return [[weights[index_nonzero:weights.shape[0],:], bias]]
    else:
        return [[weights[index_nonzero:weights.shape[0],:], bias]]+list(w[1:])
