import pickle
import gzip
import numpy as np
from numpy.random import seed
import sys
import h5py
import time 
from math import floor, sqrt, ceil
from itertools import chain


from NN_pr import NN
from NN_pr import pruning_module as pruning
from NN_pr import WS_module as ws
from NN_pr import combined_module as combined

from utility_funs import make_structured_input_for_root_NN, make_labels_for_class, number_to_tex, remove_column_zero, reshape_nozero

#numero di bit di ogni esempio del dataset
N_FEATURES = 64
#numero di classi di output della rete
N_CLASSES = 1

neurons = 32
weights1 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(N_FEATURES, neurons))#.astype(np.float32)
bias1 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, neurons))#.astype(np.float32)

weights2 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(neurons, N_CLASSES))#.astype(np.float32)
bias2 = np.random.RandomState(seed=0).normal(loc=0., scale = 0.05 ,size=(1, N_CLASSES))#.astype(np.float32)
w= [[weights1, bias1],[weights2, bias2]]

#carico dati e converto in numpy
num_file=3
with h5py.File('Resource2/file'+str(num_file)+'uniform_bin.sorted.mat','r') as f:
    data = f.get('Sb') 
    bin_data = np.array(data, dtype=np.bool)
    bin_data = np.transpose(bin_data)
    bin_data = np.flip(bin_data,axis=1)
    dim_set = len(bin_data)
    
#cerco indice della prima colonna non zero e rimuovo colonne inutilizzate
bin_data, first_nonzero = remove_column_zero(bin_data)     
labels = np.linspace(1, len(bin_data), num=len(bin_data), dtype=np.float64)
labels /= dim_set              
labels = np.reshape(labels, (-1, 1))

#permuto dati e labels con seed
p = np.random.RandomState(seed=42).permutation(dim_set)
bin_data = bin_data[p]
labels = labels[p]


                     
ww = np.copy(w)
#configuro la rete con i dati e parametri (testing non usato, valuto sul train)
#per vedere i log mettere a False il parametro disableLog e greare una cartella "log" nella directory di questo file
nn = NN.NN(training=[bin_data, labels], testing=[[0],[0]], lr=3e-2, mu=0.9, output_classes=N_CLASSES, lambd=0, minibatch=64, disableLog=True)
#configuro il modello della rete ([neuroni_strato_1, neuroni_strato_2], [funz_attivazione_strato_1, funz_attivazione_strato_2, funz_attivazione_strato_output])     
nn.addLayers([neurons], ["relu", "relu"], reshape_nozero(ww, first_nonzero))
#setto la patience (numero di epoche in cui il valore della loss non diminuisce prima di stoppare il train)
nn.set_patience(10)

now=time.time()
loss = nn.train(stop_function=3, num_epochs=1000)
difference = round(time.time() - now, 5)

predict = nn.predict(bin_data)
pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
max_err = np.max(np.abs(pr-labels*dim_set)).astype("int64")

#print("epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
#.format(nn.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn.get_memory_usage(dim_set),5)))
                    
print("-*-*"*35)

#salvo i pesi di questa configurazione (scegliere una directory/nome_file nel primo argomento della open)
with open("MATRICI_PESI/pesi", "wb") as f:
    pickle.dump(nn.getWeight(), f)


#carico pesi per eseguire compressioni
with open("MATRICI_PESI/pesi", "rb") as f:
    w_loaded = pickle.load(f)

#compressioni: 0 --> pruning, 1 --> weightsharing, 2 --> pruning+ws (con salvataggio dell
compression = 2

if compression == 0:
    nn_compress = pruning.NN_pruned(training=[bin_data, labels], testing=[[0],[0]], lr=1e-2, mu=0.9, output_classes=N_CLASSES, lambd=0, minibatch=64, disableLog=True)
    
    #primi due parametri di set_layers come la rete base, come terzo parametro passo i pesi salvati e caricati al passo precedente
    nn_compress.addLayers([neurons], ["relu", "relu"], w_loaded)
    now = time.time()

    #setto il pruning (% di connessioni tagliate)
    nn_compress.set_pruned_layers(pruning = 60, weights=w_loaded)
    

if compression == 1:
    nn_compress = ws.NN_WS(training=[bin_data, labels], testing=[[0],[0]], lr=1e-2, mu=0.9, output_classes=N_CLASSES, lambd=0, minibatch=64, disableLog=True)
    
    #primi due parametri di set_layers come la rete base, come terzo parametro passo i pesi salvati e caricati al passo precedente
    nn_compress.addLayers([neurons], ["relu", "relu"], w_loaded)
    now = time.time()

    #setto la lista delle dimensioni dei centroidi (1 elemento per ogni matrice dei pesi)
    nn_compress.set_ws(cluster = [50, 30], weights=w_loaded)
    
    
    
if compression == 2:
    nn_compress = combined.NN_combined(training=[bin_data, labels], testing=[[0],[0]], lr=1e-2, mu=0.9, output_classes=N_CLASSES, lambd=0, minibatch=64, disableLog=True)
    
    #primi due parametri di set_layers come la rete base, come terzo parametro passo i pesi salvati e caricati al passo precedente
    nn_compress.addLayers([neurons], ["relu", "relu"], w_loaded)
    now = time.time()

    #setto il pruning (% di connessioni tagliate) e il numero dei centroidi (nell'esempio una lista con 2 valori ovvero per i pesi tra input e hidden e per i pesi tra hidden e output)
    nn_compress.set_combined_compression(pruning = 60, cluster = [500, 20], weights=w_loaded)
    
loss = nn_compress.train(stop_function=3, num_epochs=1000)
difference = round(time.time() - now, 5)


predict = nn_compress.predict(bin_data)
pr = np.ceil((np.multiply(predict,predict>0)) * dim_set)
max_err = np.max(np.abs(pr-labels*dim_set)).astype("int64")

#nel caso del combined non vediamo compressione dalla print perchè teniamo le matrici espanse
#print("epoch: {} -- maxerr={} -- %err={} -- meanErr={} -- time={}s -- spaceOVH={}"
#.format(nn_compress.epoch, max_err, round(max_err/(dim_set)*100,3), round(loss, 5), difference, round(nn_compress.get_memory_usage(),5)))
print("-*-*"*35)


if compression == 2:
    #salvataggio delle matrici con pruning e weightsharing espanse nel seguente formato --> [[matrice numpy pesi tra input e hidden, vettore numpy bias], [matrice numpy pesi tra hidden e output, vettore numpy bias]]
    sparse_ws_layers = [[(combined.idx_matrix_to_matrix(nn_compress.idx_layers[i][0], nn_compress.centers[i], nn_compress.layers_shape[i])*nn_compress.mask[i]), nn_compress.idx_layers[i][1]] for i in range(nn_compress.nHidden+1)]
    
    _, freq = np.unique(sparse_ws_layers[0][0], return_counts=True)
    print(len(freq))
    
    with open("MATRICI_PESI/pesi_sparsi_ws_espansi", "wb") as f:
        pickle.dump(sparse_ws_layers, f)
