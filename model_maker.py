import numpy as np
from stl import mesh, Mode
import numpy.typing as npt

from mpl_toolkits import mplot3d
from matplotlib import pyplot
from typing import Callable

def normalize(v):
    v = np.array(v)
    return v / np.sqrt(np.sum(np.square(v), axis=-1))

def L2_norm(v1: npt.NDArray | tuple[float, float] | tuple[float, float, float], v2: npt.NDArray | tuple[float, float] | tuple[float, float, float]):
    return np.sqrt(np.sum(np.square(np.array(v2) - np.array(v1)), axis=-1))

class Stl_maker:
    def __init__(self, contour_fnc:Callable[[float], tuple[float, float]] | Callable[[float], tuple[float, float, float]] | Callable[[npt.NDArray|float], npt.NDArray], width, height, n_segments=500) -> None:
        self.path_func = contour_fnc
        if L2_norm(self.path_func(0.), self.path_func(1.)) > 1e-5:
            raise ValueError("contour must be closed!")
        self.width, self.height = width, height
        self.vertices = None
        self.faces = None
        self.model = None
        self.path_points = None
        self.n_segments = n_segments
  

# def path_function(s):
#     # Define your path function here
#     # return np.array([s, np.sin(s), 0])  # Example: A sine wave path
#     return np.array([np.cos(s*2*np.pi), np.sin(s*2*np.pi), 0])  # Example: A sine wave path

    def _generate_path_points(self):
        s_values = np.linspace(0, 1, self.n_segments)
        if len(self.path_func(0)) == 3:
            self.path_points = np.array([self.path_func(s) for s in s_values])
        elif len(self.path_func(0)) == 2:
            self.path_points = np.array([[*self.path_func(s), self.height/2] for s in s_values])
        else:
            raise ValueError(f"Encountered invalid return shape of contour: {len(self.path_func(0)) = }!")


    def _create_rectangular_cross_section(self):
        # rect must be defined perpendicular to x direction
        hw = self.width / 2
        hh = self.height / 2
        return np.array([
            [0, -hw, -hh],
            [0, -hw, hh],
            [0, hw, hh],
            [0, hw, -hh]
        ])

    @staticmethod
    def _rotate_cs(v, n):
        """rotate vectors so that they have n as a normal
        this function assumes the x direction as the start of the transformation"""
        ex = np.array([1.0,0,0])
        ez = np.array([0,0,1.0])
        # angle in xy plane with respect to the x unit vector
        phi = np.arccos(np.dot(ex[:-1], normalize(n[:-1])))
        theta =  np.arccos(np.dot(ez, normalize(n))) - np.pi/2
        if n[1] < 0:
            phi = -phi
        # this removes cases, where the x and y component is zero
        # not really nessecary in the foreseen usecase, but better save than sorry
        phi = np.where(np.isnan(phi), 0, phi)

        # rotate the cross section
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta),0,np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0],[0,0,1]])
        v = Rz @ Ry @ v.T
        return v.T

    def extrude_path(self):
        cross_section = self._create_rectangular_cross_section()
        if self.path_points is None:
            self._generate_path_points()
        if self.path_points is None:
            # this line is mainly here to let the linter chill...
            raise ValueError("Something went wrong while creating path points!")
        n_points = len(self.path_points)
        vertices = []
        faces = []
        
        for i, (x, y, z) in enumerate(self.path_points):
            path_dir_vec = normalize(self.path_points[(i+1)%n_points] - self.path_points[(i-1)%n_points])
            cs_rotated = self._rotate_cs(cross_section, path_dir_vec)

            cs_translated = cs_rotated + np.array([x, y, z])
            vertices.append(cs_translated)
            if i > 0:
                prev_index = (i - 1) * 4
                curr_index = i * 4
                # Create faces between current and previous cross sections
                faces.extend([
                    [prev_index, prev_index+1, curr_index+1],
                    [prev_index, curr_index+1, curr_index],
                    [prev_index+1, prev_index+2, curr_index+2],
                    [prev_index+1, curr_index+2, curr_index+1],
                    [prev_index+2, prev_index+3, curr_index+3],
                    [prev_index+2, curr_index+3, curr_index+2],
                    [prev_index+3, prev_index, curr_index],
                    [prev_index+3, curr_index, curr_index+3]
                ])
        # last face to close the object
        prev_index = i * 4
        curr_index = 0
        # Create faces between current and previous cross sections
        faces.extend([
            [prev_index, prev_index+1, curr_index+1],
            [prev_index, curr_index+1, curr_index],
            [prev_index+1, prev_index+2, curr_index+2],
            [prev_index+1, curr_index+2, curr_index+1],
            [prev_index+2, prev_index+3, curr_index+3],
            [prev_index+2, curr_index+3, curr_index+2],
            [prev_index+3, prev_index, curr_index],
            [prev_index+3, curr_index, curr_index+3]
        ])
        self.vertices = np.array(vertices).reshape(-1, 3)
        self.faces = np.array(faces)

    def create_model(self):
        if self.path_points is None:
            self._generate_path_points()
        if self.faces is None:
            self.extrude_path()
        if self.faces is None or self.vertices is None:
            # this line is mainly here to let the linter chill...
            raise ValueError("Something went wrong while creating faces and vertices!")
        self.model = mesh.Mesh(np.zeros(self.faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(self.faces):
            for j in range(3):
                self.model.vectors[i][j] = self.vertices[face[j], :]
        # if not self.model.is_closed(exact=True):
        #     raise ValueError("Model not closed, aborting!")
        
    def save_stl(self, filename):
        if self.model is None:
            self.create_model()
        if not isinstance(self.model, mesh.Mesh):
            # this line is mainly here to let the linter chill...
            raise ValueError("Something went wrong while creating the Model object!")
        self.model.update_normals()
        self.model.save(filename, mode=Mode.AUTOMATIC, update_normals=False)

    def display_model(self, block=True):
        # Create a new plot
        if self.model is None:
            self.create_model()
        if not isinstance(self.model, mesh.Mesh):
            # this line is mainly here to let the linter chill...
            raise ValueError("Something went wrong while creating the Model object!")

        figure = pyplot.figure()
        axes = figure.add_subplot(projection='3d')

        # Load the STL files and add the vectors to the plot
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.model.vectors))

        # Auto scale to the mesh size
        scale = self.model.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        pyplot.show(block=block)


if __name__ == "__main__":

    width = 5  # Width of the rectangular cross-section
    height = 5  # Height of the rectangular cross-section

    from contours import Circle
    circle = Circle([110,110], 100)
    stl_maker = Stl_maker(circle, width=width, height=height)
    stl_maker.create_model()
    stl_maker.display_model()

    # def custom_contour(s):
    #     return np.array([np.sin(s*2*np.pi), np.cos(s*2*np.pi), np.sin(s*4*np.pi)]) * 100

    # stl_maker = Stl_maker(custom_contour, width=width, height=height, n_segments=101)
    # stl_maker.create_model()
    # stl_maker.display_model()
    # stl_maker.save_stl("3dstl_test.stl")

