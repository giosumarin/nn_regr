import pickle
import gzip
import numpy as np
import sys
import h5py
import time 
from math import floor, sqrt, ceil
from sklearn.preprocessing import MinMaxScaler


from NN_no_hidden import NN as NN1
from NN_no_hidden import pruning_module as pr1
from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from itertools import product

def make_structured_input_for_root_NN(bin_data, labels, split, dim_set):
    position_labels = np.copy(labels)
    size_batch = dim_set // split
    remain = ceil((dim_set / split - size_batch) * split)
    for i in range(0, split, 1):
        position_labels[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain))]=i
    position_labels = np.reshape(position_labels, (-1,1))
    splitted_labels = [np.reshape(labels[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain))]/dim_set, (-1,1)) for i in range(split)]
    splitted_bin_data = [bin_data[(i*size_batch):((i+1)*(size_batch if i < (split-1) else size_batch+remain)),:] for i in range(split)] 

    p = np.random.RandomState(seed=42).permutation(dim_set)
    p_split = np.random.RandomState(seed=42).permutation(dim_set//split)

    perm_splitted_bin_data = [splitted_bin_data[i][p_split] for i in range(split)]
    perm_position_labels = position_labels[p]
    perm_splitted_labels = [splitted_labels[i][p_split] for i in range(split)]


    return bin_data[p], perm_splitted_bin_data, perm_position_labels, perm_splitted_labels
    
    
def make_labels_for_class(position_labels):
    n_rows = position_labels.shape[0]
    result = np.zeros((n_rows,2))
    for i in range(n_rows):
        result[i,position_labels[i]] = 1
    return result


N_FEATURES = 64
N_CLASSES = 1

for size in [32, 64, 128]:
    np.random.RandomState(0)
    weights1 = np.random.randn(N_FEATURES, size).astype(np.float32) * sqrt(1/(N_FEATURES+size))
    bias1 = np.ones((1, size)).astype(np.float32)*0.001

    weights2 = np.random.randn(size, N_CLASSES).astype(np.float32) * sqrt(1/(N_FEATURES+size))
    bias2 = np.ones((1, N_CLASSES)).astype(np.float32)*0.001
    wh= [[weights1, bias1], [weights2, bias2]]

    for i in [3,7]:
            with open("to_tex_h0.txt", "a+") as tex:
                    tex.write("\nfile {} size_hidden {}\n".format(i, size))
            for spl in [2,4,8,16]:
                with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
                    data = f.get('Sb') 
                    bin_data = np.array(data, dtype=np.bool)
                    bin_data = np.transpose(bin_data)
                    bin_data = np.flip(bin_data,axis=1)
                    dim_set = len(bin_data)
                
                split = spl
                labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
                bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)


                max_errs = []
                for s in range(split):
                    ww = np.copy(wh)
                    scaler = MinMaxScaler()
                    transformed_lab = scaler.fit_transform(splitted_labels[s])

                    # nn = NN.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.01, mu=0., output_classes=1, lambd=0, minibatch=32, disableLog=True)
                    nn = NN.NN(training=[splitted_bin_data[s], transformed_lab], testing=[[0],[0]], lr=0.01, mu=0., output_classes=1, lambd=0, minibatch=32, disableLog=True)

                    nn.addLayers([size],['leakyrelu','leakyrelu'], ww)
                    nn.set_patience(10)
                    now= time.time()
                    loss = nn.train(stop_function=3, num_epochs=20000)
                    difference = round(time.time() - now, 3)
                    #pr = np.floor(nn.predict(splitted_bin_data[s]) * dim_set)
                    pr = np.floor(scaler.inverse_transform(nn.predict(splitted_bin_data[s])) * dim_set)
                    lab = splitted_labels[s] * dim_set
                    max_err = np.max(np.abs(pr-lab)).astype("int64")
                    max_errs.append(max_err)
                    
                    
                    print("1 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
                    .format(i, spl, dim_set/split, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                
                
                with open("to_tex_h0.txt", "a+") as tex:
                    tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, list(max_errs), max(max_errs), round(nn.get_memory_usage(dim_set)*spl,5)))
                print("-*-*"*35)





# for size1,size2 in product([8, 16, 32, 64],[8, 16, 32, 64]):
#     if size1 >= size2:
#         # weights1 = np.random.randn(N_FEATURES, size).astype(np.float32) * sqrt(2/N_FEATURES)
#         # bias1 = np.ones((1, size)).astype(np.float32)*0.001

#         # weights2 = np.random.randn(size, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#         # bias2 = np.ones((1, N_CLASSES)).astype(np.float32)*0.001

#         # wh= [[weights1, bias1], [weights2, bias2]]

#         weights1 = np.random.randn(N_FEATURES, size1).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias1 = np.ones((1, size1)).astype(np.float32)*0.001

#         weights2 = np.random.randn(size1, size2).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias2 = np.ones((1, size2)).astype(np.float32)*0.001

#         weights3 = np.random.randn(size2, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias3 = np.ones((1, N_CLASSES)).astype(np.float32)*0.001

#         whh= [[weights1, bias1], [weights2, bias2], [weights3, bias3]]



#         for i in [3,7,10]:
#             with open("to_tex_h1.txt", "a+") as tex:
#                     tex.write("\nfile {} size_hidden {}\n".format(i, [size1, size2]))
#             for spl in [2,4,8,16]:
#                 with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
#                     data = f.get('Sb') 
#                     bin_data = np.array(data, dtype=np.bool)
#                     bin_data = np.transpose(bin_data)
#                     bin_data = np.flip(bin_data,axis=1)
#                     dim_set = len(bin_data)
                
#                 split = spl
#                 labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
#                 bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)


#                 max_errs = []
#                 for s in range(split):
#                     ww = np.copy(whh)
#                     scaler = MinMaxScaler()
#                     transformed_lab = scaler.fit_transform(splitted_labels[s])

#                     nn = NN.NN(training=[splitted_bin_data[s], transformed_lab], testing=[[0],[0]], lr=0.05, mu=0.9, output_classes=1, lambd=0, minibatch=32, disableLog=True)
#                     nn.addLayers([size1, size2],['leakyrelu','leakyrelu','leakyrelu'], ww)
#                     nn.set_patience(5)
#                     now= time.time()
#                     loss = nn.train(stop_function=3, num_epochs=20000)
#                     difference = round(time.time() - now, 3)
#                     pr = np.floor(scaler.inverse_transform(nn.predict(splitted_bin_data[s])) * dim_set)
#                     lab = splitted_labels[s] * dim_set
#                     max_err = np.max(np.abs(pr-lab)).astype("int64")
#                     max_errs.append(max_err)
                    
                    
#                     print("1 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
#                     .format(i, spl, dim_set/split, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                
                
#                 with open("to_tex_h1.txt", "a+") as tex:
#                     tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, list(max_errs), max(max_errs), round(nn.get_memory_usage(dim_set)*spl,5)))
#                 print("-*-*"*35)

# for size1,size2 in product([32, 64, 128],[8, 16, 32]):
#     if size1 >= size2:
#         # weights1 = np.random.randn(N_FEATURES, size).astype(np.float32) * sqrt(2/N_FEATURES)
#         # bias1 = np.ones((1, size)).astype(np.float32)*0.001

#         # weights2 = np.random.randn(size, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#         # bias2 = np.ones((1, N_CLASSES)).astype(np.float32)*0.001

#         # wh= [[weights1, bias1], [weights2, bias2]]

#         weights1 = np.random.randn(N_FEATURES, size1).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias1 = np.ones((1, size1)).astype(np.float32)*0.001

#         weights2 = np.random.randn(size1, size2).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias2 = np.ones((1, size2)).astype(np.float32)*0.001

#         weights3 = np.random.randn(size2, N_CLASSES).astype(np.float32) * sqrt(2/N_FEATURES)
#         bias3 = np.ones((1, N_CLASSES)).astype(np.float32)*0.001

#         whh= [[weights1, bias1], [weights2, bias2], [weights3, bias3]]



#         for i in [3,7,10]:
#             with open("to_tex_h1.txt", "a+") as tex:
#                     tex.write("\nfile {} size_hidden {}\n".format(i, [size1, size2]))
#             for spl in [2,4,8,16]:
#                 with h5py.File('Resource2/file'+str(i)+'uniform_bin.sorted.mat','r') as f:
#                     data = f.get('Sb') 
#                     bin_data = np.array(data, dtype=np.bool)
#                     bin_data = np.transpose(bin_data)
#                     bin_data = np.flip(bin_data,axis=1)
#                     dim_set = len(bin_data)
                
#                 split = spl
#                 labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
#                 bin_data, splitted_bin_data, position_labels, splitted_labels = make_structured_input_for_root_NN(bin_data, labels, split, dim_set)


#                 max_errs = []
#                 for s in range(split):
#                     ww = np.copy(whh)
#                     nn = NN.NN(training=[splitted_bin_data[s], splitted_labels[s]], testing=[[0],[0]], lr=0.05, mu=0.9, output_classes=1, lambd=0, minibatch=32, disableLog=True)
#                     nn.addLayers([size1, size2],['leakyrelu','leakyrelu','leakyrelu'], ww)
#                     nn.set_patience(5)
#                     now= time.time()
#                     loss = nn.train(stop_function=3, num_epochs=20000)
#                     difference = round(time.time() - now, 3)
#                     pr = np.floor(nn.predict(splitted_bin_data[s]) * dim_set)
#                     lab = splitted_labels[s] * dim_set
#                     max_err = np.max(np.abs(pr-lab)).astype("int64")
#                     max_errs.append(max_err)
                    
                    
#                     print("1 hidden --> file {}, split={}, dim={}: epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
#                     .format(i, spl, dim_set/split, nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                
                
#                 with open("to_tex_h1.txt", "a+") as tex:
#                     tex.write("${}$ & ${}$ & ${}$ & ${}$ \\\ \n".format(spl, list(max_errs), max(max_errs), round(nn.get_memory_usage(dim_set)*spl,5)))
#                 print("-*-*"*35)
