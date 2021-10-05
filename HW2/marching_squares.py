import taichi as ti
import numpy as np


@ti.data_oriented
class MarchingSquares():
    def __init__(self, isovalue, nelx, nely, gui_x, gui_y):
        self.nelx = nelx # resolution X
        self.nely = nely # resolution Y
        self.size_x = int(gui_x / nelx) # compute size X
        self.size_y = int(gui_y / nely) # compute size Y

        self.isovalue=isovalue

        self.vertices = ti.Vector.field(2, ti.f32, (self.nelx+1, self.nely+1))
        self.values = ti.field(ti.f32, (self.nelx + 1, self.nely + 1))
        self.cases = ti.field(ti.i32, (self.nelx, self.nely))
        self.edges = ti.Vector.field(2, ti.f32, (self.nelx * self.nely, 4))

    def update(self, stars):
        self.clear_data()
        self.compute_SDF(stars)
        self.compute_cases()
        self.compute_edges()

    def clear_data(self):
        self.values = ti.field(ti.f32, (self.nelx + 1, self.nely + 1))
        self.cases = ti.field(ti.i32, (self.nelx, self.nely))
        self.edges = ti.Vector.field(2, ti.f32, (self.nelx * self.nely, 4))

    @ti.kernel
    def initialize(self):
        for i,j in self.cases:
            scale = ti.Vector([self.nelx+1,self.nely+1])
            self.vertices[i, j] = ti.Vector([i, j]) / scale
            self.vertices[i, j+1] = ti.Vector([i, j + self.size_y]) / scale
            self.vertices[i+1, j+1] = ti.Vector([i + self.size_x, j + self.size_y]) / scale
            self.vertices[i+1, j] = ti.Vector([i + self.size_x, j]) / scale

    @ti.kernel
    def compute_SDF(self, stars: ti.template()):
        for i,j in self.vertices:
            v = self.vertices[i, j]

            for k in range(stars.Number()):
                diff = stars.Pos()[k] - v
                dist = diff.norm(1e-2)
                self.values[i, j] += 1.0/dist

            # sign distance value
            if self.values[i, j] < self.isovalue:
                self.values[i, j] *= -1

    @ti.kernel
    def compute_cases(self):
        for i, j in self.cases:
            if self.values[i, j] < 0.: self.cases[i, j] += 1
            if self.values[i, j + 1] < 0.: self.cases[i, j] += 2
            if self.values[i + 1, j + 1] < 0.: self.cases[i, j] += 4
            if self.values[i + 1, j] < 0.: self.cases[i, j] += 8

    @ti.func
    def linear_interpolation(self, a, b) -> ti.f32:
        return (self.isovalue - a) / (a - b);

    @ti.kernel
    def compute_edges(self):
        for i, j in self.cases:
            v0 = self.vertices[i, j]
            v1 = self.vertices[i, j + 1]
            v2 = self.vertices[i + 1, j + 1]
            v3 = self.vertices[i + 1, j]
            # reference: http://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/
            if (self.cases[i, j] == 1) | (self.cases[i, j] == 14):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v0, v1)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v3, v0)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif (self.cases[i, j] == 2) | (self.cases[i, j] == 13):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v3, v0)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v2, v3)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif (self.cases[i, j] == 3) | (self.cases[i, j] == 12):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v0, v1)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v2, v3)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif (self.cases[i, j] == 4) | (self.cases[i, j] == 11):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v1, v2)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v2, v3)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif self.cases[i, j] == 5:
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v0, v1)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v1, v2)
                self.edges[i * self.nely + j, 2] = self.linear_interpolation(v3, v0)
                self.edges[i * self.nely + j, 3] = self.linear_interpolation(v2, v3)
            elif (self.cases[i, j] == 6) | (self.cases[i, j] == 9):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v1, v2)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v3, v0)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif (self.cases[i, j] == 7) | (self.cases[i, j] == 8):
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v0, v1)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v1, v2)
                self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
                self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])
            elif self.cases[i, j] == 10:
                self.edges[i * self.nely + j, 0] = self.linear_interpolation(v3, v0)
                self.edges[i * self.nely + j, 1] = self.linear_interpolation(v0, v1)
                self.edges[i * self.nely + j, 2] = self.linear_interpolation(v1, v2)
                self.edges[i * self.nely + j, 3] = self.linear_interpolation(v2, v3)

    def draw_contours(self, gui, radius, color):
        edges = self.edges.to_numpy()
        for i in range(edges.shape[0]):
            gui.line(edges[i, 0], edges[i, 1], radius=radius, color=color)
            if not(np.isclose(edges[i, 2], np.array([0., 0.])).all()):
                gui.line(edges[i, 2], edges[i, 3], radius=radius, color=color)