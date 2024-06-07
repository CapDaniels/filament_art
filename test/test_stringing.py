import numpy as np

class GSketch:
    def __init__(name):
        pass

class G:
    def __init__(self, x=None, y=None, z=None, e=None):
        if not (x or y or z):
            raise ValueError("Cannot create G command with empty coordinates")
        self.x = x
        self.y = y
        self.z = z
        self.e = e
    
class G1(G):
    def __init__(self, x=None, y=None, z=None, e=None):
        super().__int__(x,y,z,e)
    
    def 