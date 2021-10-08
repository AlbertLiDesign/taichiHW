from marching_squares import *
from celestial_objects import *

nelx = 80
nely = 80
gui_x = 800
gui_y = 800

x, y = 0.5, 0.5
delta = 0.01
radius = 8
isovalue = 0.05
h = 5e-5  # time-step size
i = 0

# control
paused = False


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    gui = ti.GUI("MS-galaxy", res=(gui_x, gui_y))
    video_manager = ti.VideoManager(output_dir='./galaxy_img', framerate=24, automatic_build=False)

    ms = MarchingSquares(isovalue, nelx, nely)
    ms.initialize()

    # initialize two stars
    stars = Star(N=3, mass=1000)
    stars.initialize(0.5, 0.5, 0.2, 10)
    planets = Planet(N=1000, mass=1)
    planets.initialize(0.5, 0.5, 0.4, 10)

    while gui.running:
        while gui.get_event(ti.GUI.PRESS, ti.GUI.MOTION):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == ti.GUI.RMB:
                x, y = gui.event.pos
            elif gui.event.key == ti.GUI.WHEEL:
                x, y = gui.event.pos
                dt = gui.event.delta
                if dt[1] > 0:
                    radius += 10
                elif dt[1] < 0:
                    radius = max(8, radius - 10)
            elif gui.event.key == ti.GUI.SPACE:
                paused = not paused
                print("paused =", paused)
            elif gui.event.key == 'r':
                stars.initialize(0.5, 0.5, 0.2, 10)
                planets.initialize(0.5, 0.5, 0.4, 10)
                i = 0
            elif gui.event.key == 'i':
                export_images = not export_images

        if not paused:
            stars.computeForce()
            planets.computeForce(stars)
            for celestial_obj in (stars, planets):
                celestial_obj.update(h)
            i += 1

        if gui.is_pressed(ti.GUI.DOWN, 'q'):
            isovalue += delta
        if gui.is_pressed(ti.GUI.DOWN, 'e'):
            isovalue -= delta

        stars.display(gui, radius=10,color=0xffd500)
        planets.display(gui)
        ms.update(stars.Pos(), isovalue)
        ms.draw_contours(gui, radius=2, color=0xffd500)
        video_manager.write_frame(gui.get_image())
        gui.show()
    video_manager.make_video(gif=True)
    gui.close()