"""
    slice the body to a single layer
"""
from elementsBody import *
import numpy as np
from collections import OrderedDict


def slice(obj, job):
    if not isinstance(obj, ElementsBody):
        raise ValueError('input argument should be of class Elementsbody for function "slice"')
    eLen = obj.get_eLen()
    eps = 1.e-4  # a value nearly 0
    dm = 2 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! modify later
    
    def getCenter(ele):
        nodes = []
        for node in ele:
            nodes.append(obj.nodes[int(node)])
        nodes = np.array(nodes)
        center = [nodes[:, i].sum() / len(nodes[:, i]) for i in range(len(nodes[0]))]
        return center
    
    print('\n len(obj.elements) =', len(obj.elements))
    elesNew = []
    for ele in obj.elements:
        cen = getCenter(ele)
        if cen[dm] < 0.5:
            elesNew.append(ele)
    print('\n len(elesNew) =', len(elesNew))

    # reset the node number
    nodesNew = OrderedDict()
    for ele_ in elesNew:
        ele = ele_.tolist()
        for node in ele:
            nodesNew[node] = obj.nodes[node]
            if nodesNew[node][dm] > eLen * 1.e-4:  # slice direction, coor from 0 -> 1
                nodesNew[node][dm] = 1.
    old2new = {}  # match old node number to new node number
    for i, node in enumerate(nodesNew):
        old2new[node] = i + 1
    
    print(
        '\033[31m'
        'nodesNew = '
        '\033[0m'
    )
    print(nodesNew)
    
    # write a new file for the slice
    with open("outputData/{}_slice.inp".format(job), 'w') as output, open('inputData/{}.inp'.format(job), 'r') as input:
        clone, readNod, readEle = True, False, False
        for line in input:
            if clone:
                output.write(line)
            if (not clone) and readEle and '*' in line:
                clone = True
                output.write(line)
            if '*Node' in line or '*NODE' in line or '*node' in line:
                readNod = True
                clone = False
                for node in nodesNew:
                    output.write('    {}, {}, {}, {} \n'.format(
                        old2new[node], nodesNew[node][0], nodesNew[node][1], nodesNew[node][2]
                    ))
            elif '*Element' in line or '*ELEMENT' in line or '*element' in line:
                readEle = True
                output.write(line)
                for iele, ele in enumerate(elesNew):
                    output.write('    {},    '.format(iele + 1))
                    for node in ele:
                        output.write('{}, '.format(
                            old2new[int(node)]
                        ))
                    output.write('\n')
            

if __name__ == '__main__':

    job = 'ellip4_24'
    
    nodes, eles = readInp('inputData/{}.inp'.format(job))
    obj = ElementsBody(nodes, eles)

    slice(obj, job)