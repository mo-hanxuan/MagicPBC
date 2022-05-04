"""
    change a body's nodes-coordinates, 
    test whether the PBC-generate method can deal with a deformed body, 
    where the method of (xMax, xMin, xMax, xMin, xMax, xMin) doesn't apply,
    where graph-method is needed.
"""
import numpy as np
import sys
sys.path.append("..")
from elementsBody import *
from getPeriodicBoundaryCondition import write_nodes


def deformBody(obj, ):
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
    
    node0 = list(obj.nodes)[0]
    if not isinstance(obj.nodes[node0], np.ndarray):
        for node in obj.nodes:
            obj.nodes[node] = np.array(obj.nodes[node])
    
    dfgrad = np.array([
        [1., -0.3, 0.], 
        [0., 1., 0.], 
        [0., 0., 1.]
    ])

    for node in obj.nodes:
        obj.nodes[node] = dfgrad @ obj.nodes[node]
    

if __name__ == "__main__":

    ### get the inp file and the object
    inpFile = input("\033[0;33;40m{}\033[0m".format("please insert the .inp file name (include the path): "))
    job = inpFile.split("/")[-1].split(".inp")[0] if "/" in inpFile else inpFile.split("\\")[-1].split(".inp")[0]
    path = inpFile.split(job + ".inp")[0]
    obj = ElementsBody(*readInp(inpFile))

    deformBody(obj)
    writeInp = input(
        'ok to write the .inp file with deformed body? \033[36m{}\033[0m'.format('(y/n): ')
    )
    if writeInp == 'y':
        newFileName = path + job + "_deformed.inp"
        with open(newFileName, 'w') as newFile, open(inpFile, 'r') as oldFile:
            clone = True
            for line in oldFile:
                    
                if clone == False and '*' in line:
                    clone = True
                if clone:
                    newFile.write(line)  # write the line from old file to new file

                if "*Node\n" in line:
                    clone = False
                    write_nodes(newFile, obj) 
        print("\033[40;36;1m {} {} \033[35;1m {} \033[0m".format(
            "file", newFileName, "has been written. "
        ))