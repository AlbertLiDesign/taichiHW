from marching_squares import *
from celestial_objects import *

nelx = 10
nely = 10
gui_x = 800
gui_y = 800

gui = ti.GUI("MS-galaxy", res=(gui_x, gui_y))

x, y = 0.5, 0.5
delta = 0.01
radius = 8
isovalue = 0.2

if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    ms = MarchingSquares(0.2, nelx, nely)
    ms.initialize()

    while gui.running:
        while gui.get_event(ti.GUI.PRESS, ti.GUI.MOTION):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == ti.GUI.RMB:
                x, y = gui.event.pos
            elif gui.event.key == ti.GUI.WHEEL:
                x, y = gui.event.pos
                dt = gui.event.delta
                # delta is 2-dim vector (dx, dy)
                # dt[0] denotes the horizontal direction, and dt[1] denotes the vertical direction
                if dt[1] > 0:
                    radius += 10
                elif dt[1] < 0:
                    radius = max(8, radius - 10)

        if gui.is_pressed(ti.GUI.LEFT, 'a'):
            x -= delta
        if gui.is_pressed(ti.GUI.RIGHT, 'd'):
            x += delta
        if gui.is_pressed(ti.GUI.UP, 'w'):
            y += delta
        if gui.is_pressed(ti.GUI.DOWN, 's'):
            y -= delta
        if gui.is_pressed(ti.GUI.DOWN, 'q'):
            isovalue += delta * 2
        if gui.is_pressed(ti.GUI.DOWN, 'e'):
            isovalue -= delta * 2
        if gui.is_pressed(ti.GUI.LMB):
            x, y = gui.get_cursor_pos()

        ms.draw_vertices(gui, radius=3, color=0xffffff)
        ms.update(ti.Vector([x, y]), isovalue)
        gui.circle((x, y), radius=10, color=0xffd500)
        ms.draw_contours(gui, radius=2, color=0xffd500)
        # gui.text(f'({x:.3}, {y:.3})', (x, y)) # display [x, y]
        gui.show()