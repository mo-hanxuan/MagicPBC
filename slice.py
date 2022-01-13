"""
    slice the body to a single layer
"""
from elementsBody import *
import numpy as np
from collections import OrderedDict

from getPeriodicBoundaryCondition import write_PBC_Nset, write_PBC_equation, adjustCoordinatesForPBC


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
        have a relationship with function "writeBoundaryCondition"
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


def writeBoundaryCondition(file, obj, dm):
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


def slice(obj, inpFile):

    if not isinstance(inpFile, str):
        raise ValueError('inp file name should be of type string with its path. ')
    job = inpFile.split("/")[-1][:-4] if "/" in inpFile else inpFile.split("\\")[-1][:-4]

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
        #     '\033[31m'
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

    writePBC = input('\033[33m{}\033[0m'.format(
        'do you want to write the periodic boundary condition (PBC) for this file? (y/n): '
    ))
    while writePBC not in ['y', 'n']:
        writePBC = input('please insert "y" or "n": ')
    if writePBC == 'y':
        obj.getFaceForPBC()
        adjustCoor = input('do you want to adjust the coordinates for PBC? \033[33m{}\033[0m'.format('(y/n): '))
        while adjustCoor not in ['y', 'n']:
            adjustCoor = input('\033[33m{}\033[0m'.format('please insert "y" or "n": '))
        if adjustCoor == 'y':
            adjustCoordinatesForPBC(obj)
        del obj.faceMatch
        obj.getFaceForPBC()


    if writePBC == 'n':
        file_ = "outputData/{}_slice.inp".format(job)
    elif writePBC == 'y':
        file_ = "outputData/{}_slice_PBC.inp".format(job)
    
    functions = {
        "nodes_elements": write_nodes_elements, 
        "instanceSection": write_instanceSection,
        "boundaryCondition": writeBoundaryCondition,
    }

    # get instance name
    instance = 'Part-1'
    with open(inpFile, 'r') as file:
        for line in file:
            if '*Instance' in line and 'name=' in line:
                instance = line.split(',')
                instance = instance[1].split('=')
                instance = instance[-1]
                print('instance =', instance)
                break
    # write the new .inp file
    with open(file_, 'w') as newFile, open(inpFile, 'r') as oldFile: 
        status = "clone"
        for line in oldFile:

            if writePBC == 'y':
                if "Section:" in line and "**" in line:
                    write_PBC_Nset(newFile, obj)
                elif '*End Assembly' in line:
                    write_PBC_equation(newFile, obj, instance)

            if status not in functions:
                newFile.write(line)  # write the line from old file to new file
                if '*Instance' in line and 'name' in line and 'part' in line:
                    status = "nodes_elements"
                    functions[status](newFile, obj)
                elif '*End Instance' in line:
                    status = "instanceSection"
                    functions[status](newFile, obj, dm, instance)
                elif '**' in line and 'BOUNDARY' in line and 'CONDITIONS' in line:
                    status = "boundaryCondition"
                    functions[status](newFile, obj, dm)
            
            # change status?
            if status == "nodes_elements" and "Section" in line:
                status = "clone"
                newFile.write(line)
            elif status == "instanceSection" and "*End Assembly" in line:
                status = "clone"
                newFile.write(line)
            elif status == "boundaryCondition" and "OUTPUT REQUESTS" in line:
                status = "clone"
                newFile.write(line)
    print("\033[40;36;1m {} {} \033[40;33;1m {} \033[0m".format(
        "file", "./" + file_, "has been written. "
    ))


if __name__ == '__main__':

    inpFile = input('please insert the \033[33m{}\033[0m name (include the path): '.format('original .inp file'))

    obj = ElementsBody(*readInp(inpFile))

    print('len(obj.nodes) = {}, len(obj.elements) = {}'.format(len(obj.nodes), len(obj.elements)))
    slice(obj, inpFile)
    print('len(obj.nodes) = {}, len(obj.elements) = {}'.format(len(obj.nodes), len(obj.elements)))

