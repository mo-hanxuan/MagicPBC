"""
    get the body that composed by C3D8 elements

    attention: node index start from 1, not 0!
    (注意！ 本程序的节点编号从 1 开始， 与 inp 中的节点编号相同)

    加速方法：
        self.eleNeighbor 用 字典
        eleFacet 用 字典

"""
from typing import ValuesView
import torch as tch
import threading, time

from torch._C import Value
from progressBar import *

from functools import cmp_to_key


def readInp(fileName='donut.inp'):
    """
    read the inp file, returns:
        nodes: the coordinates of all nodes
        elements: corresponding node numbers of all elements
    """

    nodes = {}
    cout = False
    with open(fileName, 'r') as file:
        for line in file:
            if '*' in line:
                if cout:
                    break
            
            if cout:
                data = line.split(',')
                data = list(map(float, data))
                nodes[int(data[0])] = data[1:]
            
            if '*Node' in line or '*NODE' in line or '*node' in line:
                cout = True

    elements = tch.tensor([], dtype=tch.int)
    cout = False
    text = []
    with open(fileName, 'r') as file:
        for line in file:
            if '*' in line:
                if cout:
                    break
            
            if cout:
                data = line.split(',')
                tex = []
                for x in data:
                    if x != '\n':
                        tex.append(x)
                text.extend(tex)
            
            if '*ELEMENT' in line or '*Element' in line or '*element' in line:
                cout = True
    data = list(map(int, text))
    elements = tch.tensor(data)
    elements = elements.reshape((-1, 9))
    # print('elements.size() =', elements.size())
    elements = elements[:, 1:]

    return nodes, elements


def shapeFunc(nc, phi, type='C3D8'):
    """
    nc: natural coordinates
    phi: the field values at nodes of this element
    return:
        val: the interpolation value at the point with corresponding position
    """
    if type == 'C3D8':
        ncNode = tch.tensor([[-1, -1,  1], 
                             [ 1, -1,  1], 
                             [ 1, -1, -1], 
                             [-1, -1, -1], 
                             [-1,  1,  1], 
                             [ 1,  1,  1], 
                             [ 1,  1, -1], 
                             [-1,  1, -1]], dtype=tch.int)
        val = 0.125
        for dm in range(len(ncNode[0, :])):
            val *= (1. + ncNode[:, dm] * nc[dm])
        val = val * phi
        return val.sum()
    


class ElementsBody(object):
    def __init__(self, nodes, elements, name='elesBody1'):
        """
        nodes[] are coordinates of all nodes
        elements[] are the corresponding node number for each element 
        """
        for node in nodes:
            if (not tch.is_tensor(nodes[node])) and type(nodes[node]) != type([]):
                print('type(nodes[node]) =', type(nodes[node]))
                raise ValueError('item in nodes should be a torch tensor or a list')
            break
        if not tch.is_tensor(elements):
            raise ValueError('elements should be a torch tensor')
        for node in nodes:
            if tch.is_tensor(nodes[node]):
                print('nodes[node].size() =', nodes[node].size())
                if nodes[node].size()[0] != 3:
                    raise ValueError('nodes coordinates should 3d')
            else:
                # print('len(nodes[node]) =', len(nodes[node]))
                if len(nodes[node]) != 3:
                    raise ValueError('nodes coordinates should 3d')
            break
        if len(elements.size()) != 2:
            raise ValueError('len(elements.size()) should be 2 !')
        if elements.max() != max(nodes):
            print('elements.max() - 1 =', elements.max() - 1)
            print('max(nodes) =', max(nodes))
            raise ValueError('maximum element nodes number not consistent with nodes number')
        
        self.nodes = nodes
        self.elements = elements
        self.name = name

        self.nod_ele = None
        self.eleNeighbor = None
        self.allFacets = None  # all element facets of this body

        # node number starts from 1
        # element number starts from 0

        """
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
    
    def eles(self):
        # element nodes coordinates: self.eles
        print('now, we plug in the node coordinates for all elements')
        eles = tch.tensor([])
        for i, ele in enumerate(self.elements):
            
            if i % 100 == 0:
                percent = i / self.elements.size()[0] * 100.
                progressBar_percentage(percent)

            for j in range(len(ele)):
                eles = tch.cat((eles, self.nodes[ele[j] - 1, :]))
        
        eles = eles.reshape((-1, 8, 3))
        self.eles = eles
        return self.eles


    def get_nod_ele(self):  # node number -> element number
        if not self.nod_ele:
            nod_ele = {i:set() for i in self.nodes}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    nod_ele[int(node)].add(iele)
            self.nod_ele = nod_ele

        return self.nod_ele


    def get_eleNeighbor(self):  # get the neighbor elements of the given element
        if not self.nod_ele:
            self.get_nod_ele()
        if not self.eleNeighbor:
            neighbor = {i: set() for i in range(len(self.elements))}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    for eNei in self.nod_ele[int(node)]:
                        if eNei != iele:
                            neighbor[iele].add(eNei)
            self.eleNeighbor = neighbor
        return self.eleNeighbor
    
    
    def get_allFacets(self):  # get the element facets
        if not self.eleNeighbor:
            self.get_eleNeighbor()

        if not self.allFacets:
            
            # facet normal points to the positive direction of natural coordinates
            facets = tch.tensor([[0, 1, 2, 3],  # x
                                 [4, 5, 6, 7], 
                                
                                 [1, 5, 6, 2],  # y
                                 [0, 4, 7, 3],
                                
                                 [3, 2, 6, 7],  # z
                                 [0, 1, 5, 4]], dtype=tch.int)
            
            allFacets = {'node':[], 'ele':[]}
            eleFacet = {
                i: [[], [], []] for i in range(len(self.elements))
            }

            print('now, generate all the element facets')
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for ifacet, facet in enumerate(facets):  # 6 facets
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    # see if the face the same with the previous face
                    flag = True
                    for eNei in self.eleNeighbor[iele]:
                        if eNei < iele:
                            for facet2 in facets:
                                f2 = []
                                for node in facet2:
                                    nod = self.elements[eNei, node]
                                    f2.append(int(nod))
                                if set(f2) == set(f):
                                    flag = False
                                    break
                            if flag == False:
                                break
                    if flag:
                        allFacets['node'].append(f)
                        allFacets['ele'].append([iele, ])
                        eleFacet[iele][int(ifacet / 2)].append(len(allFacets['node']) - 1)

                        # find another element that shares this same face
                        flag = True
                        for eNei in self.eleNeighbor[iele]:
                            if eNei > iele:
                                for jfacet, facet in enumerate(facets):
                                    f2 = []
                                    for node in facet:
                                        nod = self.elements[eNei, node]
                                        f2.append(int(nod))
                                    if set(f2) == set(f):
                                        allFacets['ele'][-1].append(eNei)
                                        eleFacet[eNei][int(jfacet / 2)].append(len(allFacets['node']) - 1)
                                        flag = False
                                        break
                                if flag == False:
                                    break
            print('')  # break line for progress bar
            self.allFacets = allFacets
            self.eleFacet = eleFacet
        
        return self.allFacets   


    def getVolumes(self):
        """
            compute the volume of each element
        """
        ncNode = tch.tensor([
            [-1,  1,  1], 
            [-1, -1,  1], 
            [-1, -1, -1], 
            [-1,  1, -1], 
            [ 1,  1,  1], 
            [ 1, -1,  1], 
            [ 1, -1, -1], 
            [ 1,  1, -1],
        ], dtype=tch.int)

        volumes = []
        print('\n now we begin to cpmpute the volume of each element')
        for iele, ele in enumerate(self.elements):
            if iele % 100 == 0:
                progressBar_percentage((iele / len(self.elements)) * 100.)
            
            eleCoor = tch.tensor([])
            for node in ele:
                eleCoor = tch.cat((eleCoor, tch.tensor(self.nodes[int(node)])), dim=0)
            eleCoor = eleCoor.reshape((-1, 3))

            jacobi = tch.tensor([])
            ksi = tch.tensor([0., 0., 0.], requires_grad=True)
            coo = tch.tensor([0., 0., 0.])

            for dm in range(3):
                tem = 0.125  # shape function
                for dm1 in range(len(ncNode[0, :])):
                    tem1 = ncNode[:, dm1] * ksi[dm1]
                    tem1 = 1. + tem1
                    tem *= tem1
                coo[dm] = (tem * eleCoor[:, dm]).sum()
                tuple_ = tch.autograd.grad(coo[dm], ksi, retain_graph=True)
                jacobi = tch.cat((jacobi, list(tuple_)[0]))
            
            jacobi = jacobi.reshape((-1, 3))
            # if iele < 5:
            #     print('jacobi =\n', jacobi)
            #     print('tch.det(jacobi) =', tch.det(jacobi))
            volumes.append((tch.det(jacobi) * 8.).tolist())
        
        print('\n')  # line break for the progress bar
        self.volumes = tch.tensor(volumes)
        return self.volumes
    
    
    def get_eLen(self):
        """
            get the characteristic element length
        """
        if not hasattr(self, 'eLen'):
            ### first, get the average element volume
            if not hasattr(self, 'volumes'):
                self.getVolumes()
            aveVol = self.volumes.sum() / len(self.volumes)
            self.eLen = aveVol ** (1./3.)
        return self.eLen
    

    def getFaceNode(self):
        if not hasattr(self, 'allFacets'):
            self.get_allFacets()
        elif self.allFacets == None:
            self.get_allFacets()
        faceNode = set()
        for facet in range(len(self.allFacets['ele'])):
            if len(self.allFacets['ele'][facet]) == 1:
                faceNode |= set(self.allFacets['node'][facet])
        self.faceNode = faceNode

        # facets = tch.tensor([
        #     [0, 1, 2, 3],  # x
        #     [4, 5, 6, 7], 
                                
        #     [1, 5, 6, 2],  # y
        #     [0, 4, 7, 3],
                                
        #     [3, 2, 6, 7],  # z
        #     [0, 1, 5, 4]
        # ], dtype=tch.int)
        # faceNode = set()
        # for iele, ele in enumerate(self.elements):
        #     if iele % 100 == 0:
        #         percentage = iele / len(self.elements) * 100.
        #         progressBar_percentage(percentage)
            

        return faceNode
    

    def getXYZface(self):
        """
            get the X, Y, Z surfaces for PBC (periodic boundary condition)
            some future improments:
                for parallelogram (平行四边形) or parallel hexahedron(平行六面体)
                we can use face normal to define:
                    x0Face, x1Face, y0Face, y1Face, z0Face, z1Face
        """
        x0Face, x1Face, y0Face, y1Face, z0Face, z1Face = set(), set(), set(), set(), set(), set()
        if not hasattr(self, 'faceNode'):
            self.getFaceNode()
        if not hasattr(self, 'eLen'):
            self.get_eLen()
        
        xMin_key = min(self.nodes, key=lambda x: self.nodes[x][0])
        if xMin_key not in self.faceNode:
            print('xMin_key =', xMin_key)
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        xMin = self.nodes[xMin_key][0]

        xMax_key = max(self.nodes, key=lambda x: self.nodes[x][0])
        if xMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        xMax = self.nodes[xMax_key][0]

        yMin_key = min(self.nodes, key=lambda x: self.nodes[x][1])
        if yMin_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        yMin = self.nodes[yMin_key][1]

        yMax_key = max(self.nodes, key=lambda x: self.nodes[x][1])
        if yMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        yMax = self.nodes[yMax_key][1]

        zMin_key = min(self.nodes, key=lambda x: self.nodes[x][2])
        if zMin_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        zMin = self.nodes[zMin_key][2]

        zMax_key = max(self.nodes, key=lambda x: self.nodes[x][2])
        if zMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        zMax = self.nodes[zMax_key][2]

        print('xMin = {}, xMax = {}, yMin = {}, yMax = {}, zMin = {}, zMax = {}'.format(
            xMin, xMax, yMin, yMax, zMin, zMax
        ))

        eLen = self.eLen
        for node in self.faceNode:
            if abs(self.nodes[node][0] - xMin) < eLen * 1.e-4:
                x0Face.add(node)
            if abs(self.nodes[node][0] - xMax) < eLen * 1.e-4:
                x1Face.add(node)
            if abs(self.nodes[node][1] - yMin) < eLen * 1.e-4:
                y0Face.add(node)
            if abs(self.nodes[node][1] - yMax) < eLen * 1.e-4:
                y1Face.add(node)
            if abs(self.nodes[node][2] - zMin) < eLen * 1.e-4:
                z0Face.add(node)
            if abs(self.nodes[node][2] - zMax) < eLen * 1.e-4:
                z1Face.add(node)
        self.x0Face, self.x1Face, \
        self.y0Face, self.y1Face, \
        self.z0Face, self.z1Face \
        = \
            x0Face, x1Face, \
            y0Face, y1Face, \
            z0Face, z1Face
        
        print(
            'len(x0Face) = {}, len(x1Face) = {}, \n'
            'len(y0Face) = {}, len(y1Face) = {}, \n'
            'len(z0Face) = {}, len(z1Face) = {},'.format(
                len(x0Face), len(x1Face), len(y0Face), len(y1Face), len(z0Face), len(z1Face),
            )
        )
    

    def getEdgeVertexForPBC(self):
        """
            get the 12 edges and 8 vertexes for PBC (periodic boundary condition)
        """
        if not hasattr(self, 'x0Face'):
            self.getXYZface()
        faces = [
            [self.x0Face, self.x1Face],
            [self.y0Face, self.y1Face],
            [self.z0Face, self.z1Face],
        ]
        permu = [[1, 2], [2, 0], [0, 1]]
        
        xlines, ylines, zlines = [], [], []
        lines = [xlines, ylines, zlines]

        for dm in range(len(faces)):
            edge1 = faces[permu[dm][0]][0] & faces[permu[dm][1]][0]
            edge2 = faces[permu[dm][0]][0] & faces[permu[dm][1]][1]
            edge3 = faces[permu[dm][0]][1] & faces[permu[dm][1]][0]
            edge4 = faces[permu[dm][0]][1] & faces[permu[dm][1]][1]
            lines[dm].extend([edge1, edge2, edge3, edge4])
        
        # get the outer vertexes
        vertexes = set()
        for dm in range(len(lines)):
            for edge1 in lines[dm]:
                for dm2 in permu[dm]:
                    for edge2 in lines[dm2]:
                        vertexes |= (edge1 & edge2)
        self.vertexes = vertexes

        # distinguish the vertexes (very important)
        v_x0y0z0 = list(self.x0Face & self.y0Face & self.z0Face)[0]
        v_x0y0z1 = list(self.x0Face & self.y0Face & self.z1Face)[0]
        v_x0y1z0 = list(self.x0Face & self.y1Face & self.z0Face)[0]
        v_x0y1z1 = list(self.x0Face & self.y1Face & self.z1Face)[0]     
        v_x1y0z0 = list(self.x1Face & self.y0Face & self.z0Face)[0]
        v_x1y0z1 = list(self.x1Face & self.y0Face & self.z1Face)[0]
        v_x1y1z0 = list(self.x1Face & self.y1Face & self.z0Face)[0]
        v_x1y1z1 = list(self.x1Face & self.y1Face & self.z1Face)[0]
        print(
            'v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1 =',
            v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1
        )
        
        # seperate the lines by beg node, end node, and inside nodes
        x0Set = {v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1}
        x1Set = {v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1}
        for i_line, line in enumerate(xlines):
            beg = list(x0Set & line)[0]
            end = list(x1Set & line)[0]
            inside = line - {beg} - {end}
            xlines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][0])}
        
        y0Set = {v_x1y0z0, v_x0y0z0, v_x1y0z1, v_x0y0z1}
        y1Set = {v_x1y1z0, v_x0y1z0, v_x1y1z1, v_x0y1z1}
        for i_line, line in enumerate(ylines):
            beg = list(y0Set & line)[0]
            end = list(y1Set & line)[0]
            inside = line - {beg} - {end}
            ylines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][1])}
        
        z0Set = {v_x0y0z0, v_x0y1z0, v_x1y0z0, v_x1y1z0}
        z1Set = {v_x0y0z1, v_x0y1z1, v_x1y0z1, v_x1y1z1}
        for i_line, line in enumerate(zlines):
            beg = list(z0Set & line)[0]
            end = list(z1Set & line)[0]
            inside = line - {beg} - {end}
            zlines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][2])}
        
        print('\033[1;36;40m' 'xlines =' '\033[0m')
        for edge in xlines:
            print(edge)
        print('\033[1;36;40m' 'ylines =' '\033[0m')
        for edge in ylines:
            print(edge)
        print('\033[1;36;40m' 'zlines =' '\033[0m')
        for edge in zlines:
            print(edge)
        
        self.v_x0y0z0, self.v_x0y0z1, self.v_x0y1z0, self.v_x0y1z1, self.v_x1y0z0, self.v_x1y0z1, self.v_x1y1z0, self.v_x1y1z1 = \
            v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1
        self.xlines, self.ylines, self.zlines = xlines, ylines, zlines
        return


    def getNodeGraph(self):
        """
            get the node link of the  body
            every node links to other nodes, as a graph
        """
        if not hasattr(self, 'graph'):
            links = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [3, 7], [2, 6], [1, 5]
            ]
            graph = {i: set() for i in self.nodes}
            for ele_ in self.elements:
                ele = ele_.tolist()
                for link in links:
                    graph[ele[link[0]]].add(ele[link[1]])
                    graph[ele[link[1]]].add(ele[link[0]])
            self.graph = graph
        return self.graph
    

    def getFaceForPBC(self):
        """
            get the face-pairs for periodic boundary condition (PBC)
            return:
                faceMatch
        """
        eps = 1.e-3  # a value nearly zero
        tolerance = eps ** 2 * 1000.  # the tolerance for vector difference

        def compare(a, b):
            """
                first compare by idx1,
                if equal, then compare by idx2
            """
            if abs(self.nodes[a][idx1] - self.nodes[b][idx1]) > eLen * eps:
                if self.nodes[a][idx1] > self.nodes[b][idx1]:
                    return 1
                else:
                    return -1
            else:
                if abs(self.nodes[a][idx2] - self.nodes[b][idx2]) > eLen * eps:
                    if self.nodes[a][idx2] > self.nodes[b][idx2]:
                        return 1
                    else:
                        return -1
                else:
                    return 0

        if not hasattr(self, 'x0Face'):
            self.getXYZface()
        if not hasattr(self, 'graph'):
            self.getNodeGraph()
        if not hasattr(self, 'eLen'):
            self.get_eLen()
        faceMatch, baseNodes = [], []
        pairs = [
            [self.x0Face, self.x1Face],
            [self.y0Face, self.y1Face],
            [self.z0Face, self.z1Face]
        ]
        permu = [[1, 2], [2, 0], [0, 1]]
        for idx, pair in enumerate(pairs):
            faceMatch.append({})
            f0, f1 = pair[0], pair[1]
            # find the node with minimum coordinates
            idx1, idx2 = permu[idx][0], permu[idx][1]
            eLen = self.eLen
            n0 = min(f0, key=cmp_to_key(compare))
            n1 = min(f1, key=cmp_to_key(compare))
            baseNodes.append([n0, n1])
            
            faceMatch[-1][n0] = n1
            print('\033[1;31;40m' 'n0 = {}, n1 = {}' '\033[0m'.format(n0, n1))
            
            ## start from n0, n1; and traverse other nodes by BFS
            visited0 = {i: False for i in f0}
            print('len(f0) = {}, len(f1) = {}'.format(len(f0), len(f1)))
            if len(f0) != len(f1):
                raise ValueError(
                    '\033[1;31;40m' 
                    'nodes quantity does not coincide for opposite faces, f0 ({}) nodes != f1 ({}) nodes' 
                    '\033[0m'.format(len(f0), len(f1))
                )
            lis0, lis1 = [n0], [n1]
            visited0[n0] = True
            while lis0:
                lisNew0, lisNew1 = [], []
                for i_node0, node0 in enumerate(lis0):
                    node1 = lis1[i_node0]

                    if len(self.graph[node0]) != len(self.graph[node1]):
                        print('\033[1;31;40m''len(self.graph[node0]) = {} \nlen(self.graph[node1]) = {}''\033[0m'.format(
                            len(self.graph[node0]), len(self.graph[node1])
                        ))

                    vec0s = {}
                    for nex0 in self.graph[node0]:
                        if nex0 in f0:
                            if not visited0[nex0]:
                                visited0[nex0] = True
                                lisNew0.append(nex0)
                                vec0s[nex0] = tch.tensor(self.nodes[nex0]) - tch.tensor(self.nodes[node0])
                    
                    # from another face (f1), find the most similar vec
                    vec1s = {}
                    for nex1 in self.graph[node1]:
                        if nex1 in f1:
                            vec1s[nex1] = tch.tensor(self.nodes[nex1]) - tch.tensor(self.nodes[node1])
                    
                    # link nex0 to nex1
                    for nex0 in vec0s:
                        partner = min(vec1s, key=lambda x: ((vec0s[nex0] - vec1s[x])**2).sum())
                        ## test whether nex0 and partner coincide with each other
                        relativeError = (((vec0s[nex0] - vec1s[partner]) / eLen) ** 2).sum()         
                        if relativeError < tolerance:
                            faceMatch[-1][nex0] = partner
                        else:
                            print(
                                '\033[1;31;40m'
                                'node0 = {}, nex0 = {}, node1 = {}, nex1 = {}, \n''vec0 = {}, vec1 = {}'
                                '\033[0m'.format(
                                    node0, nex0, node1, nex1, vec0s[nex0], vec1s[partner],
                                )
                            )
                            print(
                                '\033[1;33;40m''warning! relativeError ({:5f}) > tolerance ({}) '
                                'between vector({} --> {}) and vector({} --> {})'
                                '\033[0m'.format(
                                relativeError, tolerance, node0, nex0, node1, nex1
                            ))
                            omit = input('\033[1;36;40m' 'do you want to continue? (y/n): ' '\033[0m')
                            if omit == 'y' or omit == '':
                                remainTol = input('\033[1;36;40m''remain the current tolerance? (y/n): ' '\033[0m')
                                if remainTol == 'n':
                                    tolerance = float(input('\033[1;36;40m''reset tolerance = ' '\033[0m'))
                                faceMatch[-1][nex0] = partner
                            else:
                                raise ValueError('relativeError > tolerance, try to enlarge tolerence instead')
                for nex0 in lisNew0:
                    lisNew1.append(faceMatch[-1][nex0])
                lis0, lis1 = lisNew0, lisNew1
        
        self.faceMatch, self.baseNodes = faceMatch, baseNodes
        return self.faceMatch


if __name__ == '__main__':

    # fileName = 'donut.inp'
    fileName = 'tilt45.inp'

    nodes, elements = readInp(fileName=fileName)
    
    print('elemens.size() =', elements.size())
    print('nodes.size() =', nodes.size())

    print('elements[599, :] =\n', elements[599, :])

    elesBody1 = ElementsBody(nodes, elements)

    # ==============================================================
    # calssify the element facet into x-type, y-type, and z-type
    # ==============================================================
    print('\n --- now get all facets with their type ---')

    tBeg = time.time()
    elesBody1.get_allFacets()  # get the facets information
    tEnd = time.time()
    print('\n facets done, consuming time is {} \n ---'.format(tEnd - tBeg))
