"""
    get the body that composed by C3D8 elements

    attention: node index start from 1, not 0!
    (注意！ 本程序的节点编号从 1 开始， 与 inp 中的节点编号相同)

    hashNodes algorithm:
        very fast to identify all facets,
        facets stored by a dict(), 
        (see 'facetDic')
        key of dict:
            sorted tuple of node numbers of a facet
    
    author: mo-hanxuan
"""
from typing import ValuesView
from attr import has
import torch as tch
import threading, time

from torch._C import Value
from progressBar import *

from functools import cmp_to_key
from collections import deque
import numpy as np
import copy


def leftIdx(lis, idx):
    left = idx - 1
    if left < 0:
        left = len(lis) - 1
    return left


def rightIdx(lis, idx):
    right = idx + 1
    if right > len(lis) - 1:
        right = 0
    return right


def sortByClockwise(lis):
    """
        sort a list by clockwise or counterClockwise order
        applied for the facet nodes list

        start from the smallest nodes,
        if original direction's next node < reverse direction's next node
            use original direction
        else:
            use reverse direction
    """
    ### find the smallest node's idx
    start = lis.index(min(lis))
    res = [lis[start]]
    if lis[rightIdx(lis, start)] < lis[leftIdx(lis, start)]:
        cur = start
        for i in range(len(lis) - 1):
            res.append(lis[rightIdx(lis, cur)])
            cur = rightIdx(lis, cur)
    else:
        cur = start
        for i in range(len(lis) - 1):
            res.append(lis[leftIdx(lis, cur)]) 
            cur = leftIdx(lis, cur)
    return res


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
                data = line[:-1].rstrip().rstrip(',')
                data = data.split(',')
                tex = []
                for x in data:
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
                eles = tch.cat((eles, tch.tensor(self.nodes[ele[j]])))
        
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
    
    
    def get_facetDic(self):  # get the element facets
        """
            'HashNodes algorithm': (invented by MoHanxuan)
                a very fast algorithm for facet generating!
                use hash dict to store all facets
                nodes number is the key to identify a facet
                e.g.:
                    facet with nodes [33, 12, 5, 7] has key '5,7,12,33'
                    , where 4 nodes are transverted to 
                    a key of tuple with sorted sequence

                see variable 'facetDic'
        """

        if not self.eleNeighbor:
            self.get_eleNeighbor()

        if not hasattr(self, "facetDic"):
            
            # facet normal points to the positive direction of natural coordinates
            facets = tch.tensor([[0, 1, 2, 3],  # x
                                 [4, 5, 6, 7], 
                                
                                 [1, 5, 6, 2],  # y
                                 [0, 4, 7, 3],
                                
                                 [3, 2, 6, 7],  # z
                                 [0, 1, 5, 4]], dtype=tch.int)
            
            eleFacet = {
                i: [[], [], []] for i in range(len(self.elements))
            }

            print('now, generate all the element facets')
            timeBeg = time.time()
            facetDic = {}
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for ifacet, facet in enumerate(facets):  # 6 facets
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    tmp = tuple(sortByClockwise(f))
                    if tmp in facetDic:
                        facetDic[tmp].append(iele)
                    else:
                        facetDic[tmp] = [iele]
                    eleFacet[iele][ifacet // 2].append(tmp)
            
            print('')  # break line for progress bar
            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'time for facetDic is', time.time() - timeBeg, "seconds"
            ))
            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(facetDic), 'facets'
            ))
            self.facetDic = facetDic
            self.eleFacet = eleFacet
        
        return self.facetDic  


    def get_edgeDic(self):  # get the element edges
        """
            'HashNodes algorithm': (invented by MoHanxuan)
                a very fast algorithm for edge generating!
                use hash dict to store all edges
                nodes number is the key to identify a edge
                e.g.:
                    edge with nodes [33, 12, 5, 7] has key '5,7,12,33'
                    , where 4 nodes are transverted to 
                    a key of tuple with sorted sequence

                see variable 'edgeDic'
        """

        if not self.eleNeighbor:
            self.get_eleNeighbor()

        if not hasattr(self, "edgeDic"):
            
            # edge normal points to the positive direction of natural coordinates
            edges = tch.tensor([
                [0, 4], [3, 7], [2, 6], [1, 5],
                [1, 0], [5, 4], [6, 7], [2, 3], 
                [2, 1], [6, 5], [7, 4], [3, 0]
            ], dtype=tch.int)

            print('now, generate all the element edges')
            timeBeg = time.time()
            edgeDic = {}
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for iedge, edge in enumerate(edges):  # 12 edges
                    f = [int(ele[node]) for node in edge]
                    tmp = tuple(sorted(f))
                    if tmp in edgeDic:
                        edgeDic[tmp].append(iele)
                    else:
                        edgeDic[tmp] = [iele]
            
            print('')  # break line for progress bar
            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'time for edgeDic is', time.time() - timeBeg, "seconds"
            ))
            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(edgeDic), 'edges'
            ))
            self.edgeDic = edgeDic
        
        return self.edgeDic  


    def get_surfaceGraph(self):
        if not hasattr(self, "surfaceGraph"):
            surfaceGraph = {}
            self.get_facetDic()
            for facet in self.facetDic:
                if len(self.facetDic[facet]) == 1:  # facet maps to only one element
                    for idx, node in enumerate(facet):
                        if node in surfaceGraph:
                            surfaceGraph[node].add(facet[(idx + 1) % len(facet)])
                            surfaceGraph[node].add(facet[(idx - 1) % len(facet)])
                        else:
                            surfaceGraph[node] = {
                                facet[(idx + 1) % len(facet)], 
                                facet[(idx - 1) % len(facet)]
                            }
            self.surfaceGraph = surfaceGraph
        return self.surfaceGraph
    

    def get_outlineGraph(self, ):
        """
            outlines are the outer edges of this body
        """
        if not hasattr(self, "outlineGraph"):
            outlineGraph = {}
            self.get_edgeDic()
            for edge in self.edgeDic:
                if len(self.edgeDic[edge]) == 1:  # edge maps to only one element
                    if edge[0] in outlineGraph:
                        outlineGraph[edge[0]].add(edge[1])
                    else:
                        outlineGraph[edge[0]] = {edge[1], }
                    if edge[1] in outlineGraph:
                        outlineGraph[edge[1]].add(edge[0])
                    else:
                        outlineGraph[edge[1]] = {edge[0], }
            self.outlineGraph = outlineGraph
        return self.outlineGraph
    

    def get_vertexGraph(self, ):
        if not hasattr(self, "vertexGraph"):
            vertexGraph = {}
            self.get_outlineGraph()
            for node in self.outlineGraph:
                if len(self.outlineGraph[node]) == 3:
                    vertexGraph[node] = set()
            ### use breadth first search (BFS) to get the link relation of vertexes
            for source in vertexGraph:
                visited = {node: False for node in self.outlineGraph}
                que = deque([source])
                visited[source] = True
                while que:
                    beg = que.popleft()
                    for nex in self.outlineGraph[beg]:
                        if not visited[nex]:
                            visited[nex] = True
                            if nex in vertexGraph:
                                vertexGraph[source].add(nex)
                                vertexGraph[nex].add(source)
                            else:
                                que.append(nex)  # didn't find vertex yet, continue to search
            self.vertexGraph = vertexGraph
        return self.vertexGraph
    

    def get_vertexCircles(self, ):
        """
            find the 6 outer faces out the whole body, 
            for each face, the vertexes form a circle, 
            
            thus, we need to find the circles (with 4 nodes) in vertexGraph,
            use depth first search (DFS) to find the circle with max-depth of 4
        """
        if not hasattr(self, "vertexCircles"):
            circles = set()
            self.get_vertexGraph()
            source = list(self.vertexGraph)[0]

            def dfs(path):  # path has a dumb node at the first
                for nex in self.vertexGraph[path[-1]]:
                    if nex != path[-2]:
                        if nex == path[1]:
                            circles.add(tuple(sortByClockwise(path[1:])))
                        elif len(path) < 5:
                            path.append(nex)
                            dfs(path)
                            path.pop()
            
            dfs([None, source])
            visitedNodes = set()
            for circle in circles:
                visitedNodes |= set(circle)
            remainNode = set(self.vertexGraph) - visitedNodes
            if len(remainNode) != 1:
                print("\033[31;1m remainNodes = {}\033[0m".format(remainNode))
                print("\033[31;1m circles = {}\033[0m".format(circles))
                raise ValueError("error, len(remainNode) != 1")
            remainNode = list(remainNode)[0]
            dfs([None, remainNode])
            self.vertexCircles = circles

            ### get the megaElement by the outer vertexes
            megaElement = []
            circle1 = list(circles)[0]
            for circle in circles:
                if len(set(circle) & set(circle1)) == 0:
                    circle2 = circle
                    megaElement.extend(circle1)
                    for i in range(4):
                        node = megaElement[i]
                        for nex in self.vertexGraph[node]:
                            if nex in circle2:
                                megaElement.append(nex)
                                break
                    break
            self.megaElement = megaElement

        return self.vertexCircles
    

    def get_surfaceSets(self):
        if not hasattr(self, "surfaceSets"):
            self.get_surfaceGraph()
            self.get_outlineGraph()
            self.get_vertexGraph()

            parent = {node: node for node in set(self.surfaceGraph) - set(self.outlineGraph)}
            treeSize = {node: 0 for node in parent}
            
            # union-find set
            def find(node):  # find the root node
                if parent[node] != node:
                    parent[node] = find(parent[node])  # flattening
                return parent[node]
            
            def union(n1, n2):  
                root1 = find(n1)
                root2 = find(n2)
                if root1 != root2:  # merge them when they are not the same root
                    if treeSize[root1] < treeSize[root2]:
                        root1, root2 = root2, root1
                    parent[root2] = root1
                    treeSize[root1] += treeSize[root2]
            
            for node in parent:
                for nex in self.surfaceGraph[node]:
                    if nex not in self.outlineGraph:
                        union(node, nex)
            
            print("\033[32;1m surfaceSet size = {}\033[0m".format(
                sum(parent[node] == node for node in parent)
            ))

            surfaceSets = {}
            for node in parent:
                root = find(node)
                if root not in surfaceSets:
                    surfaceSets[root] = {node, }
                else:
                    surfaceSets[root].add(node)
            self.surfaceSets = surfaceSets

            ### find the corresponding vertexes of each surafce-set
            surfaceVertexes = {node: set() for node in self.surfaceSets}
            for source in self.surfaceSets:
                ### breadth first search (BFS)
                visited = {node: False for node in parent}
                que = deque([source])
                while que:
                    node = que.popleft()
                    for nex in self.surfaceGraph[node]:
                        if nex not in self.outlineGraph:
                            if not visited[nex]:
                                que.append(nex)
                                visited[nex] = True
                        else:  # hit the outline, go one step to find the vertex
                            for nex2 in self.outlineGraph[nex]:
                                if nex2 in self.vertexGraph:
                                    surfaceVertexes[source].add(nex2)
            self.mapSurfaceNodeToVertexes = surfaceVertexes
        return self.surfaceSets


    def getVolumes(self, eleNum='all'):
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

        if eleNum == 'all':
            print('\n now we begin to cpmpute the volume of each element')
            for iele, ele in enumerate(self.elements):
                if iele % 100 == 0:
                    progressBar_percentage((iele / len(self.elements)) * 100.)
                
                eleCoor = tch.tensor([])
                for node in ele:
                    node = node.tolist()
                    eleCoor = tch.cat((eleCoor, tch.tensor(self.nodes[node])), dim=0)
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
        else:
            iele = int(eleNum)
            ele = self.elements[iele]
            eleCoor = tch.tensor([])
            for node in ele:
                node = node.tolist()
                eleCoor = tch.cat((eleCoor, tch.tensor(self.nodes[node])), dim=0)
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
            return (tch.det(jacobi) * 8.).tolist()
    
    
    def get_eLen(self):
        """
            get the characteristic element length
        """
        if not hasattr(self, 'eLen'):
            ### first, get the average element volume
            if not hasattr(self, 'volumes'):
                vol = self.getVolumes(eleNum=0)
            print('\033[31m{}\033[0m \033[33m{}\033[0m'.format('volume (ele No.0) =', vol))
            self.eLen = vol ** (1./3.)
            print('\033[31m{}\033[0m \033[33m{}\033[0m'.format('characteristic element length (No.0) =', self.eLen))
        return self.eLen
    

    def getFaceNode(self):
        if not hasattr(self, 'facetDic'):
            self.get_facetDic()
        elif self.facetDic == None:
            self.get_facetDic()
        faceNode = set()
        for facet in self.facetDic:
            faceNode |= set(facet)
        self.faceNode = faceNode
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

        eps = 1.e-3
        eLen = self.eLen
        for node in self.faceNode:
            if abs(self.nodes[node][0] - xMin) < eLen * eps:
                x0Face.add(node)
            if abs(self.nodes[node][0] - xMax) < eLen * eps:
                x1Face.add(node)
            if abs(self.nodes[node][1] - yMin) < eLen * eps:
                y0Face.add(node)
            if abs(self.nodes[node][1] - yMax) < eLen * eps:
                y1Face.add(node)
            if abs(self.nodes[node][2] - zMin) < eLen * eps:
                z0Face.add(node)
            if abs(self.nodes[node][2] - zMax) < eLen * eps:
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
        
        print('\033[36m' 'xlines =' '\033[0m')
        for edge in xlines:
            print(edge)
        print('\033[36m' 'ylines =' '\033[0m')
        for edge in ylines:
            print(edge)
        print('\033[36m' 'zlines =' '\033[0m')
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
    

    def getFaceForPBC_byGraph(self):
        """
            get the face-pairs for periodic boundary condition (PBC),
            using graph theory (by BFS, DFS, union-find etc.)
            return:
                faceMatch: dict, 
                    key: (*facet1, *facet2)
                    value: dict, map node-of-facet1 to node-of-facet2
        """
        if not hasattr(self, "faceMatch"):
            for node in self.nodes:
                self.nodes[node] = np.array(self.nodes[node])
            self.get_surfaceSets()
            self.get_vertexCircles()
            eLen = self.get_eLen()
            tolerance = 3.e-4

            method = input("\033[35;1m which method do you want to use to match nodes between two faces?\n "
                           "    1: nearest-coordinates. \n"
                           "    2: breadth-first-search (traverse the nodes by simular path)"
                           "insert 1 or 2: \n\033[0m")
            while method not in ['1', '2']:
                method = input("please insert 1 or 2: ")

            megaFacetsIdx = [
                [0, 1, 2, 3],  # x
                [4, 5, 6, 7], 
            
                [1, 5, 6, 2],  # y
                [0, 4, 7, 3],
            
                [3, 2, 6, 7],  # z
                [0, 1, 5, 4]
            ]
            megaFacets = [
                [self.megaElement[idx] for idx in facet] for facet in megaFacetsIdx
            ]
            print("\033[32;1m megaFacets = {}\033[0m".format(megaFacets))
            visitedFacet = {node: False for node in self.surfaceSets}
            faceMatch = {}
            for node1 in self.surfaceSets:
                if not visitedFacet[node1]:
                    ### lhs is represented by node1 and facet1, rhs is represented by node2 and facet2
                    visitedFacet[node1] = True
                    facet1 = self.mapSurfaceNodeToVertexes[node1]
                    for facetId, facet in enumerate(megaFacets):
                        if set(facet1) == set(facet):
                            facet1 = facet
                            facet2 = megaFacets[facetId // 2 * 2 + (not facetId % 2)]
                            self.mapSurfaceNodeToVertexes[node1] = facet1
                            break
                    node2 = None
                    for n in self.mapSurfaceNodeToVertexes:
                        if set(self.mapSurfaceNodeToVertexes[n]) == set(facet2):
                            node2 = n
                            self.mapSurfaceNodeToVertexes[n] = facet2
                            break
                    visitedFacet[node2] = True

                    if len(self.surfaceSets[node1]) != len(self.surfaceSets[node2]):
                        raise ValueError(
                            '\033[31m' 
                            ' nodes quantity does not coincide for opposite faces, f0 ({}) nodes != f1 ({}) nodes ' 
                            '\033[0m'.format(len(self.surfaceSets[node1]), len(self.surfaceSets[node2]))
                        )

                    ### lhs starts from source1, rhs starts from source2
                    source1 = None
                    for nex in self.surfaceGraph[facet1[0]]:
                        if nex not in self.vertexGraph:
                            for nex2 in self.surfaceGraph[nex]:
                                if nex2 in self.surfaceSets[node1]:
                                    source1 = nex2
                                    break
                            if source1:
                                break
                    source2 = None
                    for nex in self.surfaceGraph[facet2[0]]:
                        if nex not in self.vertexGraph:
                            for nex2 in self.surfaceGraph[nex]:
                                if nex2 in self.surfaceSets[node2]:
                                    source2 = nex2
                                    break
                            if source2:
                                break
                    if source1 and source2:
                        key = (*facet1, *facet2)
                        faceMatch[key] = {source1: source2, }

                        if method == "1":  # match nodes by nearst coordinates
                            for nodeA in self.surfaceSets[node1]:
                                nodeB = min(
                                    self.surfaceSets[node2], 
                                    key=lambda node: 
                                        np.linalg.norm(
                                            (self.nodes[node] - self.nodes[facet2[0]]) - \
                                            (self.nodes[nodeA] - self.nodes[facet1[0]]))
                                )
                                faceMatch[key][nodeA] = nodeB
                        
                        elif method == "2":  # match nodes by traversing the nodes wit simular path (by breadth-first-search)
                            visited = {node: False for node in self.surfaceSets[node1]}
                            que1, que2 = deque([source1]), deque([source2])
                            visited[source1] = True
                            while que1:
                                cur1, cur2 = que1.popleft(), que2.popleft()

                                vec1s = {}
                                for nex1 in self.surfaceGraph[cur1]:
                                    if nex1 not in self.outlineGraph:
                                        if not visited[nex1]:
                                            visited[nex1] = True
                                            que1.append(nex1)
                                            vec1s[nex1] = self.nodes[nex1] - self.nodes[cur1]
                                    
                                # from another face (f2), find the most similar vec
                                vec2s = {}
                                for nex2 in self.surfaceGraph[cur2]:
                                    if nex2 not in self.outlineGraph:
                                        vec2s[nex2] = self.nodes[nex2] - self.nodes[cur2]
                                
                                # link nex1 to nex2
                                for nex1 in vec1s:
                                    partner = min(vec2s, key=lambda x: ((vec1s[nex1] - vec2s[x])**2).sum())
                                    ## test whether nex1 and partner coincide with each other
                                    relativeError = (((vec1s[nex1] - vec2s[partner]) / eLen) ** 2).sum()     
                                    if relativeError < tolerance:
                                        faceMatch[key][nex1] = partner
                                        que2.append(partner)
                                    else:
                                        print(
                                            '\033[31m'
                                            'node0 = {}, nex1 = {}, node1 = {}, nex2 = {}, \n''vec0 = {}, vec1 = {}'
                                            '\033[0m'.format(
                                                cur1, nex1, cur2, nex2, vec1s[nex1], vec2s[partner],
                                            )
                                        )
                                        print(
                                            '\033[33m''warning! relativeError ({:5f}) > tolerance ({}) '
                                            'between vector({} --> {}) and vector({} --> {})'
                                            '\033[0m'.format(
                                            relativeError, tolerance, cur1, nex1, cur2, nex2
                                        ))
                                        omit = input('\033[36m' 'do you want to continue? (y/n): ' '\033[0m')
                                        if omit == 'y' or omit == '':
                                            remainTol = input('\033[36m''remain the current tolerance? (y/n): ' '\033[0m')
                                            if remainTol == 'n':
                                                tolerance = float(input('\033[36m''reset tolerance = ' '\033[0m'))
                                            faceMatch[key][nex1] = partner
                                            que2.append(partner)
                                        else:
                                            raise ValueError('relativeError > tolerance, try to enlarge tolerence instead')
            self.faceMatch = faceMatch
        return self.faceMatch
    

    def getEdgeForPBC_byGraph(self):
        """
            get the edge-pairs for periodic boundary condition (PBC)
            return:
                edgeMatch
        """
        if not hasattr(self, "outlines"):
            self.get_outlineGraph()
            self.get_vertexCircles()
            ### get the 12 edges 
            edgeIdx = [
                [0, 4], [3, 7], [2, 6], [1, 5],
                [1, 0], [5, 4], [6, 7], [2, 3], 
                [2, 1], [6, 5], [7, 4], [3, 0]
            ]
            edges = [
                [self.megaElement[i] for i in edge] for edge in edgeIdx
            ]
            paths = []
            for edge in edges:
                vec = np.array(self.nodes[edge[1]]) - np.array(self.nodes[edge[0]])
                vec = vec / np.linalg.norm(vec)
                path = [edge[0]]
                while path[-1] != edge[1]:
                    nexs = set(self.outlineGraph[path[-1]]) - set([path[-1]])
                    vecs = {nex: np.array(self.nodes[nex]) - np.array(self.nodes[path[-1]]) for nex in nexs}
                    for nex in vecs:
                        vecs[nex] = vecs[nex] / np.linalg.norm(vecs[nex])
                    nex = max(vecs, key=lambda node: vecs[node] @ vec)  # find the vector with least angle between the outer edge
                    path.append(nex)
                    if nex in self.vertexGraph:
                        paths.append(path)
                        break
            i = 0
            while i < len(paths):
                if len(paths[i]) == 2:
                    del paths[i]
                else:
                    i += 1
            outlines = {}
            for path in paths:
                outlines[(path[0], path[-1])] = path[1:-1]
                outlines[(path[-1], path[0])] = path[-2:0:-1]
            
            ### test whether the outlines has same number of nodes
            for i in range(3):  # 3 directions of 3D
                hasEdges = [(
                    edges[i * 4 + j][0], 
                    edges[i * 4 + j][1]
                ) in outlines for j in range(4)]  # 4 edges of this direction
                if not (all(hasEdges) or (not any(hasEdges))):
                    raise ValueError("error, body doesn't have 4 edges for x-, y-, or z-direction")
                else:
                    xedges = [(edges[i * 4 + j][0], edges[i * 4 + j][1]) for j in range(4)]
                    for edgeKey in xedges[1:]:
                        if len(outlines[edgeKey]) != len(outlines[xedges[0]]):
                            raise ValueError("error, node numbers on edges don't coincide!")
            self.outlines = outlines
        return self.outlines


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
            print('\033[31m' 'n0 = {}, n1 = {}' '\033[0m'.format(n0, n1))
            
            ## start from n0, n1; and traverse other nodes by BFS
            visited0 = {i: False for i in f0}
            print('len(f0) = {}, len(f1) = {}'.format(len(f0), len(f1)))
            if len(f0) != len(f1):
                raise ValueError(
                    '\033[31m' 
                    ' nodes quantity does not coincide for opposite faces, f0 ({}) nodes != f1 ({}) nodes ' 
                    '\033[0m'.format(len(f0), len(f1))
                )
            lis0, lis1 = [n0], [n1]
            visited0[n0] = True
            while lis0:
                lisNew0, lisNew1 = [], []
                for i_node0, node0 in enumerate(lis0):
                    node1 = lis1[i_node0]

                    if len(self.graph[node0]) != len(self.graph[node1]):
                        print('\033[31m''len(self.graph[node0]) = {} \nlen(self.graph[node1]) = {}''\033[0m'.format(
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
                                '\033[31m'
                                'node0 = {}, nex0 = {}, node1 = {}, nex1 = {}, \n''vec0 = {}, vec1 = {}'
                                '\033[0m'.format(
                                    node0, nex0, node1, nex1, vec0s[nex0], vec1s[partner],
                                )
                            )
                            print(
                                '\033[33m''warning! relativeError ({:5f}) > tolerance ({}) '
                                'between vector({} --> {}) and vector({} --> {})'
                                '\033[0m'.format(
                                relativeError, tolerance, node0, nex0, node1, nex1
                            ))
                            omit = input('\033[36m' 'do you want to continue? (y/n): ' '\033[0m')
                            if omit == 'y' or omit == '':
                                remainTol = input('\033[36m''remain the current tolerance? (y/n): ' '\033[0m')
                                if remainTol == 'n':
                                    tolerance = float(input('\033[36m''reset tolerance = ' '\033[0m'))
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
    fileName = input("\033[35;1m please input the file name (include path), fileName = \033[0m")

    nodes, elements = readInp(fileName=fileName)
    
    print('elemens.size() =', elements.size())
    print('len(nodes) =', len(nodes))

    print('elements[599, :] =\n', elements[599, :])

    body = ElementsBody(nodes, elements)

    # ==============================================================
    # calssify the element facet into x-type, y-type, and z-type
    # ==============================================================
    print('\n --- now get all facets with their type ---')

    # tBeg = time.time()
    # body.get_facetDic()  # get the facets information
    # tEnd = time.time()
    # print('\n facets done, consuming time is {} \n ---'.format(tEnd - tBeg))
    print("\033[35;1m body.vertexGraph = \033[40;33;1m{}\033[0m".format(body.get_vertexGraph()))
    print("\033[35;1m body.vertexCircles = \033[40;33m{}\033[0m".format(body.get_vertexCircles()))
    body.get_surfaceSets()
    body.getFaceForPBC_byGraph()
    body.getEdgeForPBC_byGraph()
