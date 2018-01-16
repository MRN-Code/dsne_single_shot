"""
Created on Mon Jan 152017

@author: Deb
"""
import numpy as np
import json
import argparse

from remote import remote_site
from local import local_site

if __name__ == "__main__":



    ''' Call remote and local site and finally collect low dimensional output
    
     remote_output (dictionary): {
        "shared_X" (str): file path to remote site data,
        "shared_Label" (str): file path to remote site labels
        "no_dims" (int): Final plotting dimensions,
        "initial_dims" (int): number of dimensions that PCA should produce
        "perplexity" (int): initial guess for nearest neighbor
        "shared_Y" (str):  the low-dimensional remote site data
        }
                
     computation_phase (string): field specifying which part (local/remote) of the 
        decentralized computation is going to be performed.            

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
