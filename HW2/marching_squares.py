import taichi as ti


@ti.data_oriented
class MarchingSquares():
    def __init__(self, isovalue, nelx, nely, gui_x, gui_y):
        self.isovalue = isovalue
        self.nelx = nelx  # resolution X
        self.nely = nely  # resolution Y
        self.size_x = int(gui_x / nelx)  # compute size X
        self.size_y = int(gui_y / nely)  # compute size Y

        self.vertices = ti.Matrix.field(4, 2, ti.f32, (self.nelx, self.nely))
        self.values = ti.Vector.field(4, ti.f32, (self.nelx, self.nely))
        self.cases = ti.field(ti.i32, (self.nelx, self.nely))
        self.edges = ti.Matrix.field(4, 2, ti.f32, (self.nelx, self.nely))

    def update(self, pos):
        self.clear_data()
        self.compute_SDF(pos)
        self.compute_cases()
        self.compute_edges()

    @ti.kernel
    def clear_data(self):
        for i, j in self.values:
            self.values[i, j] = ti.Vector([0., 0., 0., 0.])
            self.cases[i, j] = 0
            self.edges[i, j] = ti.Matrix([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])


    @ti.kernel
    def initialize(self):
        for i, j in self.vertices:
            scale = ti.Vector([self.nelx, self.nely])
            a = ti.Vector([i, j]) / scale
            b = ti.Vector([i, j + self.size_y]) / scale
            c = ti.Vector([i + self.size_x, j + self.size_y]) / scale
            d = ti.Vector([i + self.size_x, j]) / scale
            self.vertices[i, j] = ti.Matrix([[a[0], a[1]], [b[0], b[1]], [c[0], c[1]], [d[0], d[1]]])

    @ti.kernel
    def compute_SDF(self, star: ti.template()):
        for i, j in self.vertices:
            for k in ti.static(range(4)):
                diff = star - ti.Vector([self.vertices[i, j][k, 0], self.vertices[i, j][k, 1]])
                dist = diff.norm(1e-2)
                self.values[i, j][k] = dist

                # sign distance value
                if self.values[i, j][k] < self.isovalue:
                    self.values[i, j][k] *= -1

    @ti.kernel
    def compute_cases(self):
        for i, j in self.cases:
            flag = 0
            if self.values[i, j][0] < 0.: flag += 1
            if self.values[i, j][1] < 0.: flag += 2
            if self.values[i, j][2] < 0.: flag += 4
            if self.values[i, j][3] < 0.: flag += 8
            self.cases[i, j] = flag

    @ti.func
    def linear_interpolation(self, a, b):
        return (self.isovalue - a) / (a - b);

    @ti.kernel
    def compute_edges(self):
        for i, j in self.cases:
            v0 = ti.Vector([self.vertices[i, j][0, 0], self.vertices[i, j][0, 1]])
            v1 = ti.Vector([self.vertices[i, j][1, 0], self.vertices[i, j][1, 1]])
            v2 = ti.Vector([self.vertices[i, j][2, 0], self.vertices[i, j][2, 1]])
            v3 = ti.Vector([self.vertices[i, j][3, 0], self.vertices[i, j][3, 1]])

            vl0 = self.values[i, j][0]
            vl1 = self.values[i, j][1]
            vl2 = self.values[i, j][2]
            vl3 = self.values[i, j][3]

            ud_v0 = v0 + (v0 - v1) * self.linear_interpolation(vl0, vl1)
            ud_v1 = v1 + (v1 - v2) * self.linear_interpolation(vl1, vl2)
            ud_v2 = v2 + (v2 - v3) * self.linear_interpolation(vl2, vl3)
            ud_v3 = v3 + (v3 - v0) * self.linear_interpolation(vl3, vl0)

            # reference: http://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/
            if (self.cases[i, j] == 1) | (self.cases[i, j] == 14):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v3[0], ud_v3[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 2) | (self.cases[i, j] == 13):
                self.edges[i, j] = ti.Matrix([[ud_v3[0], ud_v3[1]], [ud_v2[0], ud_v2[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 3) | (self.cases[i, j] == 12):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v2[0], ud_v2[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 4) | (self.cases[i, j] == 11):
                self.edges[i, j] = ti.Matrix([[ud_v1[0], ud_v1[1]], [ud_v2[0], ud_v2[1]], [0., 0.], [0., 0.]])
            elif self.cases[i, j] == 5:
                self.edges[i, j] = ti.Matrix(
                    [[ud_v0[0], ud_v0[1]], [ud_v1[0], ud_v1[1]], [ud_v3[0], ud_v3[1]], [ud_v2[0], ud_v2[1]]])
            elif (self.cases[i, j] == 6) | (self.cases[i, j] == 9):
                self.edges[i, j] = ti.Matrix([[ud_v1[0], ud_v1[1]], [ud_v3[0], ud_v3[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 7) | (self.cases[i, j] == 8):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v1[0], ud_v1[1]], [0., 0.], [0., 0.]])
            elif self.cases[i, j] == 10:
                self.edges[i, j] = ti.Matrix(
                    [[ud_v3[0], ud_v3[1]], [ud_v0[0], ud_v0[1]], [ud_v1[0], ud_v1[1]], [ud_v2[0], ud_v2[1]]])

    def draw_contours(self, gui, radius, color):
        edges = self.edges.to_numpy()
        cases = self.cases.to_numpy()
        values = self.values.to_numpy()
        for i in range(cases.shape[0]):
            for j in range(cases.shape[1]):
                gui.line([edges[i, j][0, 0], edges[i, j][0, 1]], [edges[i, j][1, 0], edges[i, j][1, 1]], radius=radius, color=color)
                if (cases[i, j] == 5) | (cases[i, j] == 10):
                    gui.line([edges[i, j][2, 0], edges[i, j][2, 1]], [edges[i, j][3, 0], edges[i, j][3, 1]], radius=radius, color=color)

    def draw_vertices(self, gui, radius, color):
        vertices = self.vertices.to_numpy()
        values = self.values.to_numpy()
        for i in range(vertices.shape[0]):
            for j in range(vertices.shape[1]):
                gui.circles(vertices[i, j], radius=radius, color=color)
                for k in range(4):
                    gui.text(f'({values[i, j][k]:.3})', [vertices[i, j][k, 0], vertices[i, j][k, 1]])