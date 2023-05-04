from os.path import dirname, join as pjoin
from scipy import io as sio
from GrandeNN import *
import os
import numpy as np
import torch

def main():

    input_size = 10
    output_size = 1

    mat_fname = pjoin(dirname(os.getcwd()), 'MyProjects', 'MATLAB', 'GrandePartita','GrandiPartite_check.mat')
    raw_data = sio.loadmat(file_name=mat_fname)
    guerrieri = raw_data['match_check']
    inputs = torch.t(torch.Tensor(guerrieri))

    net_fname = pjoin(dirname(os.getcwd()), 'MyProjects', 'PYTHON','GrandePartita', 'GrandiPartite_test_cfg.pth')

    net = GrandeNN(10,1)
    cfg_dict = torch.load(net_fname)
    net.load_state_dict(cfg_dict)
    net.eval()

    out = net.forward(inputs)

    ## Print check
    
    print("")

    out_equilibrated_matches = out[0:99]
    print("Equilibrated matches:")
    #print(out_equilibrated_matches.detach().numpy().flatten())
    print("Average confidency: {}".format(out_equilibrated_matches.abs().mean()))
    print("")

    out_first_team_stronger = out[100:149]
    print("First-team-stronger matches:")
    #print(out_first_team_stronger.detach().numpy().flatten())
    print("Average confidency: {}".format(out_first_team_stronger.abs().mean()))
    print("")

    out_second_team_stronger = out[150:199]
    print("Second-team-stronger matches:")
    #print(out_second_team_stronger.detach().numpy().flatten())
    print("Average confidency: {}".format(out_second_team_stronger.abs().mean()))
    print("")

if __name__ == "__main__": main()