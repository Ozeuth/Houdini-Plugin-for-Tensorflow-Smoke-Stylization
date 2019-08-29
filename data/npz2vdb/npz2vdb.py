import argparse
import os
import numpy as np
from manta import *

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, default='')
args = parser.parse_args()

def npz2vdb(file_path):
    # load density
    with np.load(file_path) as data:
        d = data['x']

    res_x = d.shape[2]
    res_y = d.shape[1]
    res_z = d.shape[0]
    # print(res_x, res_y, res_z)
    vdb_path = str(file_path[:-3] + 'vdb')

    # to vdb
    # print(vdb_path)
    gs = vec3(res_x, res_y, res_z)
    s = Solver(name='main', gridSize=gs, dim=3)
    density = s.create(RealGrid)
    copyArrayToGridReal(d, density)
    density.save(vdb_path)

npz2vdb(args.src_path)