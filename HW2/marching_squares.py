import taichi as ti


@ti.data_oriented
class MarchingSquares():
    def __init__(self, isovalue, nelx, nely, gui_x, gui_y):
        self.isovalue = isovalue
        self.nelx = nelx # resolution X
        self.nely = nely # resolution Y
        self.size_x = int(gui_x / nelx) # compute size X
        self.size_y = int(gui_y / nely) # compute size Y
        self.vertices = ti.Vector.field(2, ti.f32, (self.nelx+1, self.nely+1))
        self.values = ti.field(ti.f32, (self.nelx + 1, self.nely + 1))
        self.cases = ti.field(ti.i32, (self.nelx, self.nely))
        self.edges = ti.Vector.field(2, ti.f32, (self.nelx * self.nely, 4))

    def update(self, pos):
        self.clear_data()
        self.compute_SDF(pos)
        self.compute_cases()
        self.compute_edges()

    @ti.kernel
    def clear_data(self):
        for i, j in self.values:
            self.values[i, j] = 0.
        for i, j in self.cases:
            self.cases[i, j] = 0
            self.edges[i * self.nely + j, 0] = ti.Vector([0., 0.])
            self.edges[i * self.nely + j, 1] = ti.Vector([0., 0.])
            self.edges[i * self.nely + j, 2] = ti.Vector([0., 0.])
            self.edges[i * self.nely + j, 3] = ti.Vector([0., 0.])

    @ti.kernel
    def initialize(self):
        for i, j in self.cases:
            scale = ti.Vector([self.nelx-1,self.nely-1])
            self.vertices[i, j] = ti.Vector([i, j]) / scale
            self.vertices[i, j + 1] = ti.Vector([i, j + self.size_y]) / scale
            self.vertices[i + 1, j+1] = ti.Vector([i + self.size_x, j + self.size_y]) / scale
            self.vertices[i + 1, j] = ti.Vector([i + self.size_x, j]) / scale

    @ti.kernel
    def compute_SDF(self, star: ti.template()):
        for i, j in self.vertices:
            v = self.vertices[i, j]

            diff = star - v
            dist = diff.norm(1e-2)
            self.values[i, j] = dist

            # sign distance value
            if self.values[i, j] < self.isovalue:
                self.values[i, j] = -self.values[i, j]

    @ti.kernel
    def compute_cases(self):
        for i, j in self.cases:
            flag = 0
            if self.values[i, j] < 0.: flag += 1
            if self.values[i, j + 1] < 0.: flag += 2
            if self.values[i + 1, j + 1] < 0.: flag += 4
            if self.values[i + 1, j] < 0.: flag += 8
            self.cases[i, j] = flag

    @ti.func
    def linear_interpolation(self, a, b):
        return (self.isovalue - a) / (a - b);

    @ti.kernel
    def compute_edges(self):
        for i, j in self.cases:
            v0 = self.vertices[i, j]
            v1 = self.vertices[i, j + 1]
            v2 = self.vertices[i + 1, j + 1]
            v3 = self.vertices[i + 1, j]

            vl0 = self.values[i, j]
            vl1 = self.values[i, j + 1]
            vl2 = self.values[i + 1, j + 1]
            vl3 = self.values[i + 1, j]

            ud_v0 = v0 + (v0 - v1) * self.linear_interpolation(vl0, vl1)
            ud_v1 = v1 + (v1 - v2) * self.linear_interpolation(vl1, vl2)
            ud_v2 = v2 + (v2 - v3) * self.linear_interpolation(vl2, vl3)
            ud_v3 = v3 + (v3 - v0) * self.linear_interpolation(vl3, vl0)

            # reference: http://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/
            if (self.cases[i, j] == 1) | (self.cases[i, j] == 14):
                self.edges[i * self.nely + j, 0] = ud_v0
                self.edges[i * self.nely + j, 1] = ud_v3
            elif (self.cases[i, j] == 2) | (self.cases[i, j] == 13):
                self.edges[i * self.nely + j, 0] = ud_v3
                self.edges[i * self.nely + j, 1] = ud_v2
            elif (self.cases[i, j] == 3) | (self.cases[i, j] == 12):
                self.edges[i * self.nely + j, 0] = ud_v0
                self.edges[i * self.nely + j, 1] = ud_v2
            elif (self.cases[i, j] == 4) | (self.cases[i, j] == 11):
                self.edges[i * self.nely + j, 0] = ud_v1
                self.edges[i * self.nely + j, 1] = ud_v2
            elif self.cases[i, j] == 5:
                self.edges[i * self.nely + j, 0] = ud_v0
                self.edges[i * self.nely + j, 1] = ud_v1
                self.edges[i * self.nely + j, 2] = ud_v3
                self.edges[i * self.nely + j, 3] = ud_v2
            elif (self.cases[i, j] == 6) | (self.cases[i, j] == 9):
                self.edges[i * self.nely + j, 0] = ud_v1
                self.edges[i * self.nely + j, 1] = ud_v3
            elif (self.cases[i, j] == 7) | (self.cases[i, j] == 8):
                self.edges[i * self.nely + j, 0] = ud_v0
                self.edges[i * self.nely + j, 1] = ud_v1
            elif self.cases[i, j] == 10:
                self.edges[i * self.nely + j, 0] = ud_v3
                self.edges[i * self.nely + j, 1] = ud_v0
                self.edges[i * self.nely + j, 2] = ud_v1
                self.edges[i * self.nely + j, 3] = ud_v2

    def draw_contours(self, gui, radius, color):
        edges = self.edges.to_numpy()
        cases = self.cases.to_numpy()
        for i in range(cases.shape[0]):
            for j in range(cases.shape[1]):
                gui.line(edges[i * self.nely + j, 0], edges[i * self.nely + j, 1], radius=radius, color=color)
                if (cases[i, j] == 5) | (cases[i, j] == 10):
                    gui.line(edges[i * self.nely + j, 2], edges[i * self.nely + j, 3], radius=radius, color=color)

    def draw_vertices(self, gui, radius, color):
        vertices = self.vertices.to_numpy()
        for i in range(vertices.shape[0]):
            for j in range(vertices.shape[1]):
                gui.circle(vertices[i, j], radius=radius, color=color)