from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from os.path import dirname, join as pjoin
from scipy import io as sio
from GrandeNN import *
import numpy as np
import os
import torch 

def main():

    ## Chiediamo un basso batch size perché non serve avere una stima dell errore su un grande campione, non ha un particolare
    # avere una media dell'errore su tante partite.
    ## La loss è L1 perche con numeri minori di uno penalizza di piu di MSE.
    ## Un numero troppo alto di dati costringe ad usare meno epoche (minor tempo di calcolo...), 
    # in questo modo pero non riesce a capire bene il pattern perché i dati vengono analizzati meno volte
    ## Gli esiti sono stati riscalati perche, distinguendo bene i due esiti, si riescono ad identificare meglio gli errori, che sono
    # piu grandi
    ## Aumentare e diminuire i neuroni sul (primo) layer sembra diminuire la accuracy
    ## Duplicare il (primo) layer sembra non creare particolari cambiamenti

    input_size = 10
    output_size = 1

    mat_fname = pjoin(dirname(os.getcwd()), 'MyProjects', 'MATLAB', 'GrandePartita', 'GrandiPartite_train_validation.mat')
    raw_data = sio.loadmat(file_name=mat_fname)
    np_matches = raw_data['match']
    vsplit_np_matches = np.vsplit(np_matches, [10, 11, 13])
    guerrieri, esiti, aspettative = vsplit_np_matches[0], vsplit_np_matches[1], vsplit_np_matches[2]

    train_variable = 2*esiti - 1 # between -1 and 1

    inputs = torch.Tensor(guerrieri.transpose())
    labels = torch.Tensor(train_variable.transpose())
    data_train, data_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.3)
    dataset = TensorDataset(data_train, labels_train)
    loader_train = DataLoader(dataset, batch_size=20, shuffle=True)

    net = GrandeNN(input_size,output_size)

    n_epochs = 150
    for epoch in range(n_epochs):
        for data in loader_train:
            input_data, label = data
            net.train_step(input_data, label)


    out = net.forward(data_test)
    predictions = torch.sign(out).detach().numpy()
    
    labels_test = labels_test.detach().numpy()

    print( "Mean: {} ; StD: {}".format(predictions.mean(), predictions.std()) )
    print( "Accuracy: {}".format(accuracy_score(y_pred=predictions, y_true=labels_test).round(2)) )
    print( "Balanced accuracy: {}".format(balanced_accuracy_score(y_pred=predictions, y_true=labels_test).round(2)) )


    cfg_fname = pjoin(dirname(os.getcwd()), 'MyProjects', 'PYTHON', 'GrandePartita','GrandiPartite_test_cfg.pth')
    torch.save(net.state_dict(), cfg_fname)

    


if __name__ == "__main__": main()