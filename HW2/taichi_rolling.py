import taichi as ti

ti.init(arch=ti.cpu)

n = 512
x = ti.field(dtype=ti.i32) # 像素
res = n + n // 4 + n // 16 + n // 64 # 最小解析度
img = ti.field(dtype=ti.f32, shape=(res, res))

block1 = ti.root.pointer(ti.ij, n // 64) # 第一个层级，8*8的block1，每个block包含64*64的小数据
block2 = block1.pointer(ti.ij, 4) # 每个Block1分成4*4的小block2
block3 = block2.pointer(ti.ij, 4) # 每个小格子再分成4*4的小block3
block3.dense(ti.ij, 4).place(x) # 每个block3 含有4个pixel


@ti.kernel
def activate(t: ti.f32):
    for i, j in ti.ndrange(n, n):
        p = ti.Vector([i, j]) / n
        p = ti.Matrix.rotation2d(ti.sin(t)) @ (p - 0.5) + 0.5

        if ti.taichi_logo(p) == 0:
            x[i, j] = 1


@ti.func
def scatter(i):
    return i + i // 4 + i // 16 + i // 64+2


@ti.kernel
def paint():
    for i, j in ti.ndrange(n, n):
        t = x[i, j]
        # Use ti.rescale_index(descendant_snode/field, ancestor_snode, index) to compute the ancestor index given a descendant index.
        block1_index = ti.rescale_index(x, block1, [i, j])
        block2_index = ti.rescale_index(x, block2, [i, j])
        block3_index = ti.rescale_index(x, block3, [i, j])
        # Use ti.is_active(snode, [i, j, ...]) to query if snode[i, j, ...] is active or not.
        t += ti.is_active(block1, block1_index)
        t += ti.is_active(block2, block2_index)
        t += ti.is_active(block3, block3_index)
        img[scatter(i), scatter(j)] = 1 - t / 4


img.fill(0.05)

gui = ti.GUI('Sparse Grids', (res, res))
for i in range(100000):
    block1.deactivate_all()
    activate(i * 0.05) # 旋转
    paint()
    gui.set_image(img)
    gui.show()