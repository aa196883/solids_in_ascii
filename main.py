import numpy as np
import curses
import time
from solides import solids
from draw import draw_line_wu, draw_line, draw_line_w_slope, draw_faces

def rotate(vertices, angle_x, angle_y, angle_z):
    # Matrices de rotation
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    # Application des rotations
    rotated = vertices @ rx @ ry @ rz
    return rotated

def project(vertices, width, height, fov, viewer_distance):
    projected = []
    aspect_ratio = 0.5  # Prend en compte le ratio d'un caractère ASCII sur l'écran
    for vertex in vertices:
        x, y, z = vertex
        factor = fov / (viewer_distance + z)
        x_proj = x * factor + width / 2
        y_proj = y * factor * aspect_ratio + height / 2
        projected.append([x_proj, y_proj, z])
    return np.array(projected)

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)  # Rendre les appels non bloquants
    width = curses.COLS
    height = curses.LINES
    angle_x = angle_y = angle_z = 0

    # Sélection du solide
    solid_name = 'dodecahedron'
    solid = solids.get(solid_name)
    vertices = solid['vertices']
    faces = solid['faces']

    view_vector = np.array([0, 0, 1])  # La caméra regarde dans la direction +z

    while True:
        stdscr.clear()
        # Gestion de la fermeture avec la touche 'q'
        key = stdscr.getch()
        if key == ord('q'):
            break

        rotated_vertices = rotate(vertices, angle_x, angle_y, angle_z)
        projected_vertices = project(rotated_vertices, width, height, fov=100, viewer_distance=4)

        draw_faces(faces, view_vector, projected_vertices, stdscr)

        stdscr.refresh()
        time.sleep(0.015)
        angle_x += 0.05
        angle_y += 0.025
        angle_z += 0.0125

curses.wrapper(main)
