"""
    生成周期性边界条件，只用 BFS算法，比如，左右两面成周期性边界条件的话
    左面BFS, 右面根据左面 BFS 的矢量方向来遍历节点，左右两边同时经过的节点即可作为一对 pair

    好处：时间复杂度是 n, n 是左面节点的数量
    否则:
        普通算法1及缺点：根据所有距离矩阵来计算，时间复杂度会是 n**2
        普通算法2及缺点：面节点先排序再对应 （根据坐标排序），时间复杂度是 nlogn, 且排序会受到浮点数舍入误差的影响


    重要功能！！！ (已实现)
        对于PBC, 在左右两面的节点拼接带有误差的时候，
        带有节点坐标调整功能, 使得左右两面的节点几乎可以完全匹配
        见函数 "adjustCoordinatesForPBC()"
"""

import torch as tch
import numpy as np
from elementsBody import *


def write_PBC_equation(file, obj, instance):
    """
        write the PBC for the 8 outer vertex, and 12 edges, and 6 faces, with three steps:
            1. make the 8 outer vertexes to form a parallel hexahedron (平行六面体))
            2. make 12 edges to satisfy PBC
            3. make the inside nodes of face-pair to coincide
    """
    if not hasattr(obj, 'v_x0y0z0'):
        obj.getEdgeVertexForPBC()
    
    ##  1.1 make the y0face to be parallogram
    file.write('************************** make the y0face to be parallogram \n')
    for dm in [1, 2, 3]:
        file.write('*Equation\n4 \n')
        file.write('{}.N{}, {},  1 \n'.format(instance, obj.v_x1y0z0, dm))
        file.write('{}.N{}, {}, -1 \n'.format(instance, obj.v_x0y0z0, dm))
        file.write('{}.N{}, {}, -1 \n'.format(instance, obj.v_x1y0z1, dm))
        file.write('{}.N{}, {},  1 \n'.format(instance, obj.v_x0y0z1, dm))

    ##   1.2 make vertexes of ylines to form parallel hexahedron
    file.write('************************** make vertexes of 4 ylines to coincide \n')
    for yline in obj.ylines[1:]:
        for dm in [1, 2, 3]:
            file.write('*Equation\n4 \n')
            file.write('{}.N{}, {},  1 \n'.format(instance, yline['end'], dm))
            file.write('{}.N{}, {}, -1 \n'.format(instance, yline['beg'], dm))
            file.write('{}.N{}, {}, -1 \n'.format(instance, obj.ylines[0]['end'], dm))
            file.write('{}.N{}, {},  1 \n'.format(instance, obj.ylines[0]['beg'], dm))
    
    #  2. make all outer edges to coincide
    file.write('************************** make all outer edges to coincide \n')
    xyzEdges = [obj.xlines, obj.ylines, obj.zlines]
    for edges in xyzEdges:
        for edge in edges[1:]:
            for node in range(len(edge['inside'])):
                for dm in [1, 2, 3]:
                    file.write('*Equation\n4 \n')
                    file.write('{}.N{}, {},  1 \n'.format(instance, edge['inside'][node], dm))
                    file.write('{}.N{}, {}, -1 \n'.format(instance, edge['beg'], dm))
                    file.write('{}.N{}, {}, -1 \n'.format(instance, edges[0]['inside'][node], dm))
                    file.write('{}.N{}, {},  1 \n'.format(instance, edges[0]['beg'], dm))

    #  3. make all corresponding face-pairs to coincide
    file.write('************************** make all corresponding face-pairs to coincide \n')
    edgeNodes = set()
    for edges in [obj.xlines, obj.ylines, obj.zlines]:
        for edge in edges:
            edgeNodes |= ({edge['beg']} | {edge['end']} | set(edge['inside']))
    for iface, face in enumerate(obj.faceMatch):
        for node in face:
            for dm in [1, 2, 3]:
                if node not in edgeNodes:
                    file.write('*Equation\n4 \n')
                    file.write('{}.N{}, {},  1 \n'.format(instance, node, dm))
                    file.write('{}.N{}, {}, -1 \n'.format(instance, obj.baseNodes[iface][0], dm))
                    file.write('{}.N{}, {}, -1 \n'.format(instance, face[node], dm))
                    file.write('{}.N{}, {},  1 \n'.format(instance, obj.baseNodes[iface][1], dm))


def write_PBC_Nset(file, obj):
    if not hasattr(obj, 'faceNode'):
        obj.getFaceNode()
    for node in obj.faceNode:
        file.write('*Nset, nset=N{} \n'.format(node))
        file.write('{}, \n'.format(node))


def write_nodes(file, obj):
    nodes = obj.nodes
    for node in nodes:
        file.write('    {}, {}, {}, {} \n'.format(
            node, nodes[node][0], nodes[node][1], nodes[node][2]
        ))


def adjustCoordinatesForPBC(obj):
    """
        adjust the nodal coordiantes for periodic boundary condition (PBC)
        make the nodes at face-pair to be strictly coincide at initial state
    """
    if not hasattr(obj, 'v_x0y0z0'):
        obj.getEdgeVertexForPBC()
    
    makenp = False
    for node in obj.nodes:
        if type(obj.nodes[node]) == type([]):
            makenp = True
        break
    if makenp:
        for node in obj.nodes:
            obj.nodes[node] = np.array(obj.nodes[node])

    ##  1.1 make the y0face to be parallogram
    obj.nodes[obj.v_x1y0z0] = \
        obj.nodes[obj.v_x0y0z0] + \
        (obj.nodes[obj.v_x1y0z1] - obj.nodes[obj.v_x0y0z1])

    ##   1.2 make vertexes of ylines to form parallel hexahedron
    for yline in obj.ylines[1:]:
        obj.nodes[yline['end']] = \
            obj.nodes[yline['beg']] + \
            obj.nodes[obj.ylines[0]['end']] - obj.nodes[obj.ylines[0]['beg']]
    
    #  2. make all outer edges to coincide
    xyzEdges = [obj.xlines, obj.ylines, obj.zlines]
    for edges in xyzEdges:
        for edge in edges[1:]:
            for node in range(len(edge['inside'])):
                obj.nodes[edge['inside'][node]] = \
                    obj.nodes[edge['beg']] + \
                    obj.nodes[edges[0]['inside'][node]] - obj.nodes[edges[0]['beg']]

    #  3. make all corresponding face-pairs to coincide
    edgeNodes = set()
    for edges in [obj.xlines, obj.ylines, obj.zlines]:
        for edge in edges:
            edgeNodes |= ({edge['beg']} | {edge['end']} | set(edge['inside']))
    for iface, face in enumerate(obj.faceMatch):
        for node in face:
            if node not in edgeNodes:
                obj.nodes[node] = \
                    obj.nodes[obj.baseNodes[iface][0]] + \
                    obj.nodes[face[node]] - obj.nodes[obj.baseNodes[iface][1]]
    obj.nodesAdjusted = True


if __name__ == "__main__":
    # get the inp file and the object
    job = 'ellip4_24_slice'
    nodes, elements = readInp('inputData/{}.inp'.format(job))
    obj = ElementsBody(nodes, elements)

    obj.getFaceForPBC()
    # print('obj.faceMatch =')
    # for pair in obj.faceMatch:
    #     print(pair)
    #     print('')

    adjustCoor = input('do you want to adjust the coordinates for PBC? \033[33m{}\033[0m'.format('(y/n): '))
    while adjustCoor not in ['y', 'n']:
        adjustCoor = input('\033[33m{}\033[0m'.format('please insert "y" or "n": '))
    if adjustCoor == 'y':
        adjustCoordinatesForPBC(obj)
    del obj.faceMatch
    obj.getFaceForPBC()
    

    # find the instance name
    instance = 'Part-1'
    with open('inputData/{}.inp'.format(job), 'r') as file:
        for line in file:
            if '*Instance' in line and 'name=' in line:
                instance = line.split(',')
                instance = instance[1].split('=')
                instance = instance[-1]
                print('instance =', instance)
                break
    
    writeInp = input(
        'ok to write the .inp file with PBC inside the file ? \033[36m{}\033[0m'.format('(y/n): ')
    )
    while writeInp not in ['y', 'n']:
        writeInp = input('\033[31m{}\033[0m'.format(
            'please insert "y" or "n": '
        ))
    if writeInp == 'y':
        with open('outputData/{}_PBC.inp'.format(job), 'w') as newFile, open('inputData/{}.inp'.format(job), 'r') as oldFile:
            clone = True
            for line in oldFile:
                if "Section:" in line and "**" in line:
                    write_PBC_Nset(newFile, obj)
                elif '*End Assembly' in line:
                    write_PBC_equation(newFile, obj, instance)
                    
                if clone == False and '*' in line:
                    clone = True
                if clone:
                    newFile.write(line)  # write the line from old file to new file

                if "*Node" in line or "*NODE" in line or "*node" in line:
                    if hasattr(obj, 'nodesAdjusted'):
                        clone = False
                        write_nodes(newFile, obj) 
    else:
        # write the Nset
        with open('outputData/Nset_{}.txt'.format(job), 'w') as file:
            for node in obj.faceNode:
                file.write('*Nset, nset=N{} \n'.format(node))
                file.write('{}, \n'.format(node))
        # write the equation for PBC
        with open('outputData/Equation_{}.txt'.format(job), 'w') as file:
            write_PBC_equation(file, obj, instance)
        
    