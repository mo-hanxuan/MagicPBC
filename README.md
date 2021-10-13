# MagicPBC
make periodic boundary condition (PBC) for the grid of finite element analysis 
(given the .inp file from Abaqus and modify the file to contain PBC) 

## hightlights
+ newly invented algorithm for nodes matching between face-pair, 
    dramatically acclerates the nodes matching process
    (namely, BFS-match algorithm, time complexity reduses from n^2 to n)
+ more robust algorithm to identify the outer face of a grid
    (not sensity to the rotation and deformation of the grid)
+ mutch more robust, not sensitive to relative error between pair faces.
    (i.e., not sensitive to relative error of node coordinates between pair faces, 
    since nodes matching is augmented by topological information of grid, 
    unlike traditional method which dones not consider the connection (as a graph) of nodes on a face)
+ automaticall adjust nodes position between two corresponding faces, 
    so that each pair of faces can match excactly at initially state

## usage
+ give the .inp file into the 'inputData' folder
+ run 'getPeriodicBoundaryCondition.py' to get a result .inp file at the 'outputData' folder
