"""
    slice the body to a single layer
"""
from elementsBody import *
import numpy as np
from collections import OrderedDict


def DeleteObjectAllProperties(objectInstance):
    if not objectInstance:
        return
    listPro =[key for key in objectInstance.__dict__.keys()]
    for key in listPro:
        objectInstance.__delattr__(key)


def write_nodes_elements(file, obj):
    nodes, eles = obj.nodes, obj.elements
    file.write('*Node\n')
    for node in nodes:
        file.write('    {}, {}, {}, {} \n'.format(
            node, nodes[node][0], nodes[node][1], nodes[node][2]
        ))
    file.write('*Element, type=C3D8R\n')
    for iele, ele in enumerate(eles):
        file.write('    {},    '.format(iele + 1))
        for node in ele:
            file.write('{}, '.format(
                int(node)
            ))
        file.write('\n')
    file.write('*Nset, nset=Set-1, generate\n')
    file.write('    1, {}, 1 \n'.format(len(nodes)))
    file.write('*Elset, elset=Set-1, generate\n')
    file.write('    1, {}, 1 \n'.format(len(eles)))


def write_instanceSection(file, obj, dm, instance):
    """
        have a relationship with function "writePBC"
    """
    obj.getEdgeVertexForPBC()
    # section for 4 edges
    lines = {
        0: [
            [obj.v_x0y0z0, obj.v_x1y0z0], 
            [obj.v_x0y0z1, obj.v_x1y0z1], 
            [obj.v_x0y1z0, obj.v_x1y1z0], 
            [obj.v_x0y1z1, obj.v_x1y1z1], 
        ],
        1: [
            [obj.v_x0y0z0, obj.v_x0y1z0], 
            [obj.v_x0y0z1, obj.v_x0y1z1], 
            [obj.v_x1y0z0, obj.v_x1y1z0], 
            [obj.v_x1y0z1, obj.v_x1y1z1], 
        ],
        2: [
            [obj.v_x0y0z0, obj.v_x0y0z1], 
            [obj.v_x0y1z0, obj.v_x0y1z1], 
            [obj.v_x1y0z0, obj.v_x1y0z1],  
            [obj.v_x1y1z0, obj.v_x1y1z1], 
        ],
    }
    plates = {
        0: [obj.v_x0y0z0, obj.v_x0y0z1, obj.v_x0y1z0, obj.v_x0y1z1],
        1: [obj.v_x0y0z0, obj.v_x0y0z1, obj.v_x1y0z0, obj.v_x1y0z1],
        2: [obj.v_x0y0z0, obj.v_x0y1z0, obj.v_x1y0z0, obj.v_x1y1z0],
    }
    name = ["corner_00", "corner_01", "corner_10", 'corner_11']

    file.write('**\n')
    for iline, line in enumerate(lines[dm]):
        file.write('*Nset, nset={}, instance={}\n'.format(name[iline], instance))
        file.write('    {}, {}\n'.format(line[0], line[1]))
    file.write('*Nset, nset={}, instance={}\n'.format('corner_plate', instance))
    file.write('    {}, {}, {}, {}\n'.format(plates[dm][0], plates[dm][1], plates[dm][2], plates[dm][3])) 


def writePBC(file, obj, dm):
    """
        have a relationship with function "write_instanceSection"
    """
    obj.getEdgeVertexForPBC()
    names = ["corner_00", "corner_01", "corner_10", 'corner_11']
    length = max(
        obj.nodes[obj.v_x1y0z0][0] - obj.nodes[obj.v_x0y0z0][0], 
        obj.nodes[obj.v_x0y1z0][1] - obj.nodes[obj.v_x0y0z0][1], 
        obj.nodes[obj.v_x0y0z1][2] - obj.nodes[obj.v_x0y0z0][2], 
    )
    strain = length * 0.1
    dof = {0: [2, 3], 1: [3, 1], 2: [1, 2]}
    dof = dof[dm]  # degree of freedom
    load = [
        [-strain, -strain], 
        [ strain, -strain], 
        [-strain,  strain], 
        [ strain,  strain], 
    ]
    file.write('**\n')
    file.write('** Name: loading Type: Displacement/Rotation\n')
    file.write('*Boundary\n')
    for iName, name in enumerate(names):
        for i_dim, dim in enumerate(dof):
            file.write('{}, {}, {}, {}\n'.format(name, dim, dim, load[iName][i_dim]))
    file.write('** Name: fix Type: Displacement/Rotation\n')
    file.write('*Boundary\n')
    file.write('corner_plate, {}, {}\n'.format(dm + 1, dm + 1))


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
    
    """
        # print(
        #     '\033[1;31;40m'
        #     'nodesNew = '
        #     '\033[0m'
        # )
        # print(nodesNew)
    """

    # reset nodes and elements for this object
    obj.__delattr__('elements')
    elements = tch.tensor([], dtype=tch.int)
    for ele in elesNew:
        tmp = [old2new[int(i)] for i in ele]
        elements = tch.cat([elements, tch.tensor(tmp)])
    elements = elements.reshape((-1, 8))

    obj.__delattr__('nodes')
    nodes = OrderedDict()
    for node in nodesNew:
        nodes[old2new[node]] = nodesNew[node]

    DeleteObjectAllProperties(obj)
    obj.__init__(nodes=nodes, elements=elements)
    
    # obj.nodesNew, obj.elesNew, obj.old2new, obj.eLen = nodesNew, elesNew, old2new, eLen


    file_ = "outputData/{}_slice.inp".format(job)
    functions = {
        "write_nodes_elements": write_nodes_elements, 
        "write_instanceSection": write_instanceSection,
        "writePBC": writePBC,
    }

    # get instance name
    instance = 'Part-1'
    with open('inputData/{}.inp'.format(job), 'r') as file:
        for line in file:
            if '*Instance' in line and 'name=' in line:
                instance = line.split(',')
                instance = instance[1].split('=')
                instance = instance[-1]
                print('instance =', instance)
                break
    with open(file_, 'w') as output, open('inputData/{}.inp'.format(job), 'r') as input: 
        status = "clone"
        for line in input:
            if status not in functions:
                output.write(line)
                if '*Instance' in line and 'name' in line and 'part' in line:
                    status = "write_nodes_elements"
                    functions[status](output, obj)
                elif '*End Instance' in line:
                    status = "write_instanceSection"
                    functions[status](output, obj, dm, instance)
                elif '**' in line and 'BOUNDARY' in line and 'CONDITIONS' in line:
                    status = "writePBC"
                    functions[status](output, obj, dm)
            
            # change status?
            if status == "write_nodes_elements" and "Section" in line:
                status = "clone"
                output.write(line)
            elif status == "write_instanceSection" and "*End Assembly" in line:
                status = "clone"
                output.write(line)
            elif status == "writePBC" and "OUTPUT REQUESTS" in line:
                status = "clone"
                output.write(line)


if __name__ == '__main__':

    job = 'ellip4_24'
    file_ = 'inputData/{}.inp'.format(job)

    nodes, elements = readInp('inputData/{}.inp'.format(job))
    obj = ElementsBody(nodes, elements)
    nodes, elements = [], []

    print('len(obj.nodes) = {}, len(obj.elements) = {}'.format(len(obj.nodes), len(obj.elements)))
    slice(obj, job)
    print('len(obj.nodes) = {}, len(obj.elements) = {}'.format(len(obj.nodes), len(obj.elements)))

