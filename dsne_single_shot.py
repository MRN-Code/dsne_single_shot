#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:59:22 2017

@author: dsaha
"""
import numpy as np
import json
import argparse

from remote import remote_site
from local import local_site

if __name__ == "__main__":



    ''' Call remote and local site and finally collect low dimensional output
    
     Args(Passing arguments):
     remote_output:
        remote_output (dictionary):  {
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0
        "shared_Y" = "Y_values.txt"
        } 
        
        computation_phase (string): field specifying which part (local/remote) of the 
        decentralized computation is going to be performed:    
        
     Returns:
        LY: Contains low dimensional Y values of remote and local site together. In the top, remote site Y values and then local site Y is placed              

     '''




    parser = argparse.ArgumentParser(
        description='''read in coinstac args for remote computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')

    sharedData = ''' {
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0
    } '''

    args = parser.parse_args(['--run', sharedData])
    remote_output = remote_site(args.run, computation_phase='remote')


    local_output = local_site(remote_output, computation_phase='local')

    #Receive local site data
    LY = np.loadtxt(local_output["local"])
