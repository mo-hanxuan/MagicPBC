"""
    generate periodic boundary condition (PBC). 

    Two methods to detect and partition the surface-nodes:
        1. graph-method: (recommended, can deal with arbitrary deformed shape):
            use dictionary-data-structure to map facet-nodes to element-number, 
            where the surface-facet is shared by only one element. 
            Construct the node-linking graph of surface, and the node-linking graph of the outlines. 
            Using outlines as boundaries, 
            partition the graph into different faces (left-, right-, down-, up-, back-, front- surfaces) by union-find algorithm.
        2. method of xMin, xMax, yMin, yMax, zMin, zMax: 
            detect the surface simply by coordinates of all nodes. 
            This method can only be applied to the object with cuboid shape.

    Two methods match nodes on opposites of the surface:
        1. BFS method to match the nodes (time complexity of O(V + E), V and E are number of nodes and edges respectively): 
            Matching nodes during traversing of surface-node-graphs of opposite faces. 
            Given a matched node-pair, use similar vectors (pointed from current node to neighbors) to match their neighbors. 
        2. nearest-coordinates method:  Could be very slow when there are many many nodes on a surface (with time complexity of O(V^2)).
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
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
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


def write_PBC_equation_byGraph(file, obj, instance):
    """
        use graph-method to get the PBC info, 
        write the PBC for the 8 outer vertex, and 12 edges, and 6 faces, with three steps:
            1. make the 8 outer vertexes to form a parallel hexahedron (平行六面体))
            2. make 12 edges to satisfy PBC
            3. make the inside nodes of face-pair to coincide

            the node-number of megaElement 
            (composed of vertex of outer surface) is shown as follows,  
                      v3------v7
                     /|      /|
                    v0------v4|
                    | |     | |
                    | v2----|-v6
             y ^    |/      |/
               |    v1------v5
               --->
              /    x
             z       
    """
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
    obj.getFaceForPBC_byGraph()
    obj.getEdgeForPBC_byGraph()
    
    ##  1.1 make the y0face to be parallogram
    file.write('************************** make the y0face to be parallogram \n')
    for dm in [1, 2, 3]:
        file.write('*Equation\n4 \n')
        file.write('{}.N{}, {},  1 \n'.format(instance, obj.megaElement[6], dm))
        file.write('{}.N{}, {}, -1 \n'.format(instance, obj.megaElement[2], dm))
        file.write('{}.N{}, {}, -1 \n'.format(instance, obj.megaElement[5], dm))
        file.write('{}.N{}, {},  1 \n'.format(instance, obj.megaElement[1], dm))

    ##   1.2 make vertexes of ylines to form parallel hexahedron
    file.write('************************** make vertexes of 4 ylines to coincide \n')
    for i, j in [[7, 6], [3, 2], [0, 1]]:
        for dm in [1, 2, 3]:
            file.write('*Equation\n4 \n')
            file.write('{}.N{}, {},  1 \n'.format(instance, obj.megaElement[i], dm))
            file.write('{}.N{}, {}, -1 \n'.format(instance, obj.megaElement[j], dm))
            file.write('{}.N{}, {}, -1 \n'.format(instance, obj.megaElement[4], dm))
            file.write('{}.N{}, {},  1 \n'.format(instance, obj.megaElement[5], dm))
    
    #  2. make all outer edges to coincide
    file.write('************************** make all outer edges to coincide \n')
    edgeId = [
        [[0, 4], [3, 7], [2, 6], [1, 5]],  # xEdges
        [[1, 0], [5, 4], [6, 7], [2, 3]],  # yEdges
        [[2, 1], [6, 5], [7, 4], [3, 0]]   # zEdges
    ]
    for edges in edgeId:  # edges = xEdges or yEdges or zEdges
        edge0 = (obj.megaElement[edges[0][0]], obj.megaElement[edges[0][1]])
        if edge0 in obj.outlines:
            for edge in edges[1:]:
                edge1 = (obj.megaElement[edge[0]], obj.megaElement[edge[1]])
                for node in range(len(obj.outlines[edge0])):
                    for dm in [1, 2, 3]:
                        file.write('*Equation\n4 \n')
                        file.write('{}.N{}, {},  1 \n'.format(instance, obj.outlines[edge1][node], dm))
                        file.write('{}.N{}, {}, -1 \n'.format(instance, edge1[0], dm))
                        file.write('{}.N{}, {}, -1 \n'.format(instance, obj.outlines[edge0][node], dm))
                        file.write('{}.N{}, {},  1 \n'.format(instance, edge0[0], dm))

    #  3. make all corresponding face-pairs to coincide
    file.write('************************** make all corresponding face-pairs to coincide \n')
    for twoFacets in obj.faceMatch:
        faceMatch = obj.faceMatch[twoFacets]
        for node in faceMatch:
            for dm in [1, 2, 3]:
                file.write('*Equation\n4 \n')
                file.write('{}.N{}, {},  1 \n'.format(instance, node, dm))
                file.write('{}.N{}, {}, -1 \n'.format(instance, twoFacets[0], dm))
                file.write('{}.N{}, {}, -1 \n'.format(instance, faceMatch[node], dm))
                file.write('{}.N{}, {},  1 \n'.format(instance, twoFacets[4], dm))


def write_PBC_Nset(file, obj):
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
    if not hasattr(obj, 'faceNode'):
        obj.getFaceNode()
    for node in obj.getFaceNode():
        file.write('*Nset, nset=N{} \n'.format(node))
        file.write('{}, \n'.format(node))


def write_nodes(file, obj):
    nodes = obj.nodes
    for node in nodes:
        file.write('    {}, {}, {}, {} \n'.format(
            node, nodes[node][0], nodes[node][1], nodes[node][2]
        ))


def adjustCoordinatesForPBC_byGraph(obj):
    """
        use graph method to get the node-relation, 
        adjust the nodal coordiantes for periodic boundary condition (PBC)
        make the nodes at face-pair to be strictly coincide at initial state
    """
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
    obj.getFaceForPBC_byGraph()
    obj.getEdgeForPBC_byGraph()
    
    makenp = False
    for node in obj.nodes:
        if type(obj.nodes[node]) == type([]):
            makenp = True
        break
    if makenp:
        for node in obj.nodes:
            obj.nodes[node] = np.array(obj.nodes[node])

    ##  1.1 make the y0face to be parallogram
    obj.nodes[obj.megaElement[6]] = \
        obj.nodes[obj.megaElement[2]] + \
        (obj.nodes[obj.megaElement[5]] - obj.nodes[obj.megaElement[1]])

    ##   1.2 make vertexes of ylines to form parallel hexahedron
    for i, j in [[7, 6], [3, 2], [0, 1]]:
        obj.nodes[obj.megaElement[i]] = \
            obj.nodes[obj.megaElement[j]] + \
                obj.nodes[obj.megaElement[4]] - obj.nodes[obj.megaElement[5]]
    
    #  2. make all outer edges to coincide
    edgeId = [
        [[0, 4], [3, 7], [2, 6], [1, 5]],  # xEdges
        [[1, 0], [5, 4], [6, 7], [2, 3]],  # yEdges
        [[2, 1], [6, 5], [7, 4], [3, 0]]   # zEdges
    ]
    for edges in edgeId:  # edges = xEdges or yEdges or zEdges
        edge0 = (obj.megaElement[edges[0][0]], obj.megaElement[edges[0][1]])
        if edge0 in obj.outlines:
            for edge in edges[1:]:
                edge1 = (obj.megaElement[edge[0]], obj.megaElement[edge[1]])
                for node in range(len(obj.outlines[edge0])):
                    obj.nodes[obj.outlines[edge1][node]] = \
                        obj.nodes[edge1[0]] + \
                            obj.nodes[obj.outlines[edge0][node]] - obj.nodes[edge0[0]]

    #  3. make all corresponding face-pairs to coincide
    for twoFacets in obj.faceMatch:
        faceMatch = obj.faceMatch[twoFacets]
        for node in faceMatch:
            obj.nodes[faceMatch[node]] = \
                obj.nodes[twoFacets[4]] + \
                    obj.nodes[node] - obj.nodes[twoFacets[0]]
    obj.nodesAdjusted = True


def adjustCoordinatesForPBC(obj):
    """
        adjust the nodal coordiantes for periodic boundary condition (PBC)
        make the nodes at face-pair to be strictly coincide at initial state
    """
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
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

    testState = False

    # get the inp file and the object
    inpFile = input("\033[0;33;40m{}\033[0m".format("please insert the .inp file name (include the path): "))
    job = inpFile.split("/")[-1].split(".inp")[0] if "/" in inpFile else inpFile.split("\\")[-1].split(".inp")[0]
    path = inpFile.split(job + ".inp")[0]
    obj = ElementsBody(*readInp(inpFile))

    key = input("\033[35;1m{}\033[0m".format(
        "which method do you want to use? \n"
        "1: graph-method (recomended); \n"
        "2: xMin, xMax, yMin, yMax, zMin, zMax;  \n(insert 1 or 2): "
    ))
    if key == "1":
        getFaceForPBC = obj.getFaceForPBC_byGraph
        writeEquations = write_PBC_equation_byGraph
        adjustCoordinate = adjustCoordinatesForPBC_byGraph
    elif key == "2":
        getFaceForPBC = obj.getFaceForPBC
        writeEquations = write_PBC_equation
        adjustCoordinate = adjustCoordinatesForPBC

    getFaceForPBC()

    adjustCoor = input("do you want to adjust the coordinates for PBC? "
                       "(not recommended)\n\033[33m{}\033[0m".format('(y/n): '))
    while adjustCoor not in ['y', 'n']:
        adjustCoor = input('\033[33m{}\033[0m'.format('please insert "y" or "n": '))
    if adjustCoor == 'y':
        adjustCoordinate(obj)
    if testState:
        del obj.faceMatch
        getFaceForPBC()
    
    # find the instance name
    instance = 'Part-1'
    with open(inpFile, 'r') as file:
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
        newFileName = path + job + "_PBC.inp"
        with open(newFileName, 'w') as newFile, open(inpFile, 'r') as oldFile:
            clone = True
            for line in oldFile:
                if "Section:" in line and "**" in line:
                    write_PBC_Nset(newFile, obj)
                elif '*End Assembly' in line:
                    writeEquations(newFile, obj, instance)
                    
                if clone == False and '*' in line:
                    clone = True
                if clone:
                    newFile.write(line)  # write the line from old file to new file

                if "*Node\n" in line:
                    if hasattr(obj, 'nodesAdjusted'):
                        clone = False
                        print("\033[35;1m{}\033[0m".format("write new nodes for obj"))
                        write_nodes(newFile, obj) 
        print("\033[40;36;1m {} {} \033[35;1m {} \033[0m".format(
            "file", newFileName, "has been written. "
        ))
    elif input(
            "\033[32;1m write nset- and equations- files for PBC? (y/n): \033[0m"
        ) in ["y", ""]:
            # write the Nset
            with open(path + '{}_nset.txt'.format(job), 'w') as file:
                for node in obj.getFaceNode():
                    file.write('*Nset, nset=N{} \n'.format(node))
                    file.write('{}, \n'.format(node))
            print("\033[40;36;1m {} {} \033[35;1m {} \033[0m".format(
                "file", path + '{}_nset.txt'.format(job), "has been written. "
            ))
            # write the equation for PBC
            with open(path + '{}_equation.txt'.format(job), 'w') as file:
                writeEquations(file, obj, instance)
            print("\033[40;36;1m {} {} \033[35;1m {} \033[0m".format(
                "file", path + '{}_equation.txt'.format(job), "has been written. "
            ))
        
    