import taichi as ti


@ti.data_oriented
class MarchingSquares():
    def __init__(self, isovalue, nelx, nely):
        self.isovalue = isovalue
        self.nelx = nelx  # resolution X
        self.nely = nely  # resolution Y
        self.size_x = 1.0  # compute size X
        self.size_y = 1.0  # compute size Y

        self.vertices = ti.Vector.field(2, ti.f32, (self.nelx + 1, self.nely + 1))
        self.values = ti.field(ti.f32, (self.nelx + 1, self.nely + 1))
        self.cases = ti.field(ti.i32, (self.nelx, self.nely))
        self.edges = ti.Matrix.field(4, 2, ti.f32, (self.nelx, self.nely))

    def update(self, pos, isovalue):
        if isovalue <= 0.01:
            isovalue = 0.01
        self.isovalue = isovalue
        self.clear_data()
        self.compute_SDF(pos)
        self.compute_cases(isovalue)
        self.compute_edges(isovalue)

    @ti.kernel
    def clear_data(self):
        for i, j in self.values:
            self.values[i, j] = 0.
            self.cases[i, j] = 0
            self.edges[i, j] = ti.Matrix([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])

    @ti.kernel
    def initialize(self):
        for i, j in self.vertices:
            scale = ti.Vector([self.nelx, self.nely])
            self.vertices[i, j] = ti.Vector([i, j]) / scale

    @ti.kernel
    def compute_SDF(self, star: ti.template()):
        for i, j in self.vertices:
            diff = star - self.vertices[i, j]
            dist = diff.norm_sqr()
            self.values[i, j] = dist


    @ti.kernel
    def compute_cases(self, isovalue:ti.f32):
        for i, j in self.cases:
            flag = 0
            if self.values[i, j] < isovalue: flag += 1
            if self.values[i, j + 1] < isovalue: flag += 2
            if self.values[i + 1, j + 1] < isovalue: flag += 4
            if self.values[i + 1, j] < isovalue: flag += 8
            self.cases[i, j] = flag

    @staticmethod
    @ti.func
    def linear_interpolation(a, b, isovalue):
        return (isovalue - a) / (a - b);

    @ti.kernel
    def compute_edges(self, isovalue: ti.f32):
        for i, j in self.cases:
            v0 = self.vertices[i, j]
            v1 = self.vertices[i, j + 1]
            v2 = self.vertices[i + 1, j + 1]
            v3 = self.vertices[i + 1, j]

            vl0 = self.values[i, j]
            vl1 = self.values[i, j + 1]
            vl2 = self.values[i + 1, j + 1]
            vl3 = self.values[i + 1, j]

            ud_v0 = v0 + (v0 - v1) * self.linear_interpolation(vl0, vl1, isovalue)
            ud_v1 = v1 + (v1 - v2) * self.linear_interpolation(vl1, vl2, isovalue)
            ud_v2 = v2 + (v2 - v3) * self.linear_interpolation(vl2, vl3, isovalue)
            ud_v3 = v3 + (v3 - v0) * self.linear_interpolation(vl3, vl0, isovalue)

            # reference: http://jamie-wong.com/2014/08/19/metaballs-and-marching-squares/
            if (self.cases[i, j] == 1) | (self.cases[i, j] == 14):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v3[0], ud_v3[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 2) | (self.cases[i, j] == 13):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v1[0], ud_v1[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 3) | (self.cases[i, j] == 12):
                self.edges[i, j] = ti.Matrix([[ud_v3[0], ud_v3[1]], [ud_v1[0], ud_v1[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 4) | (self.cases[i, j] == 11):
                self.edges[i, j] = ti.Matrix([[ud_v1[0], ud_v1[1]], [ud_v2[0], ud_v2[1]], [0., 0.], [0., 0.]])
            elif self.cases[i, j] == 5:
                self.edges[i, j] = ti.Matrix(
                    [[ud_v0[0], ud_v0[1]], [ud_v1[0], ud_v1[1]], [ud_v2[0], ud_v2[1]], [ud_v3[0], ud_v3[1]]])
            elif (self.cases[i, j] == 6) | (self.cases[i, j] == 9):
                self.edges[i, j] = ti.Matrix([[ud_v0[0], ud_v0[1]], [ud_v2[0], ud_v2[1]], [0., 0.], [0., 0.]])
            elif (self.cases[i, j] == 7) | (self.cases[i, j] == 8):
                self.edges[i, j] = ti.Matrix([[ud_v2[0], ud_v2[1]], [ud_v3[0], ud_v3[1]], [0., 0.], [0., 0.]])
            elif self.cases[i, j] == 10:
                self.edges[i, j] = ti.Matrix(
                    [[ud_v0[0], ud_v0[1]], [ud_v3[0], ud_v3[1]], [ud_v1[0], ud_v1[1]], [ud_v2[0], ud_v2[1]]])

    def draw_contours(self, gui, radius, color):
        edges = self.edges.to_numpy()
        cases = self.cases.to_numpy()
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
                gui.circle(vertices[i, j], radius=radius, color=color)
                gui.text(f'({values[i, j]:.3})', vertices[i, j])
                #     # gui.text(f'({vertices[i, j][k, 0]:.3}, {vertices[i, j][k, 1]:.3})', [vertices[i, j][k, 0], vertices[i, j][k, 1]])
