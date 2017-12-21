#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:48:24 2017

@author: gazula
"""

import numpy as np
from tsneFunctions import normalize_columns, tsne


def remote_site(args, computation_phase):


    ''' It will receive parameters from dsne_single_shot. After receiving parameters it will compute tsne on high dimensional remoter data and pass low dimensional values of remote site data

     Args:
        args (dictionary):  {
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0
        }
        computation_phase (string): remote

        normalize_columns:
        Shared data is normalized through this function

     Returns:
        Return args will contain previous args value in addition of Y[low dimensional Y values] values of shared_Y.
        args(dictionary):  {
            "shared_X": "Shared_Mnist_X.txt",
            "shared_Label": "Shared_Label.txt",
            "no_dims": 2,
            "initial_dims": 50,
            "perplexity": 20.0,
            "shared_Y" : "Y_values.txt";
        }
    '''








    shared_X = np.loadtxt(args["shared_X"])
    #    sharedLabel = np.loadtxt(args.run["shared_Label"])
    no_dims = args["no_dims"]
    initial_dims = args["initial_dims"]
    perplexity = args["perplexity"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    # shared data computation in tsne
    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase=computation_phase)

    with open("Y_values.txt", "w") as f:
        for i in range(0, len(shared_Y)):
            f.write(str(shared_Y[i][0]) + '\t')
            f.write(str(shared_Y[i][1]) + '\n')

    args["shared_Y"] = "Y_values.txt"


    return args
