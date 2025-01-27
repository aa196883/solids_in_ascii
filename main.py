import numpy as np
import curses
import time
from solides import solids
from draw import draw_line_wu, draw_line, draw_line_w_slope, draw_faces, toggle_frame_storage, get_stored_frame_chars, copy_frame_storage, reset_frame_storage, draw_point, debug_print
import math
import random
from matching import generate_intermediate_frames, compute_matching

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

def initialize_rotation():
    rotation_axes = random.sample(['x', 'y', 'z'], random.randint(1, 3))  # Choisir 1 à 3 axes
    rotation_speeds = {
        axis: random.uniform(0.005, 0.03) * random.choice([-1, 1])  # Choix aléatoire du signe
        for axis in ['x', 'y', 'z']
    }
    # Mettre à 0 les vitesses pour les axes non actifs
    for axis in ['x', 'y', 'z']:
        if axis not in rotation_axes:
            rotation_speeds[axis] = 0
    return rotation_speeds

def main(stdscr):
    # Configuration de curses
    curses.curs_set(0)
    stdscr.nodelay(1)
    width = curses.COLS
    height = curses.LINES
    angle_x = angle_y = angle_z = 0

    # Prétraitement pour normaliser les sommets
    factor = 2.0
    for solid_name, solid_data in solids.items():
        vertices = solid_data['vertices']
        
        # Normaliser chaque sommet pour que sa distance à l'origine soit égale à 1
        normalized_vertices = []
        for vertex in vertices:
            distance_to_origin = np.linalg.norm(vertex)  # Calculer la norme (distance à l'origine)
            if distance_to_origin > 0:
                normalized_vertex = tuple(coord / distance_to_origin * factor for coord in vertex)  # Normaliser
                normalized_vertices.append(normalized_vertex)
            else:
                # Si un sommet est exactement à l'origine (très rare), laisser tel quel
                normalized_vertices.append(vertex)

        # Mettre à jour le dictionnaire
        solid_data['vertices'] = normalized_vertices

    # Liste des solides disponibles
    solid_names = list(solids.keys())
    solid_name = random.choice(solid_names)
    solid = solids[solid_name]
    vertices = solid['vertices']
    faces = solid['faces']

    view_vector = np.array([0, 0, 1])
    transition_in_progress = False
    intermediate_frames = []
    frame_index = 0
    last_switch_time = time.time()

    # Initialiser les rotations aléatoires
    rotation_speeds = initialize_rotation()
    angle_x = angle_y = angle_z = 0

    while True:
        stdscr.clear()
        key = stdscr.getch()
        if key == ord('q'):
            break

        current_time = time.time()
        
        if transition_in_progress:
            # Transition entre solides
            if frame_index < len(intermediate_frames):
                # Afficher l'état intermédiaire
                for char, (x, y) in intermediate_frames[frame_index]:
                    draw_point(stdscr, y, x, char)
                frame_index += 1
            else:
                # Fin de la transition
                transition_in_progress = False
                frame_index = 0
                last_switch_time = current_time

                # Nouvelle rotation aléatoire après la transition
                rotation_speeds = initialize_rotation()
        else:
            # Phase de rendu du solide actuel
            rotated_vertices = rotate(vertices, angle_x, angle_y, angle_z)
            projected_vertices = project(rotated_vertices, width, height, fov=100, viewer_distance=4)
            
            # Changer de solide toutes les 10 secondes
            if current_time - last_switch_time >= 5:
                # Sauvegarder la position des points actuels
                toggle_frame_storage()
                draw_faces(faces, view_vector, projected_vertices, stdscr)
                current_frame = copy_frame_storage()
                reset_frame_storage()

            if current_time - last_switch_time >= 5:
                # Passer au solide suivant
                new_solid_name = random.choice([name for name in solid_names if name != solid_name])
                solid_name = new_solid_name
                solid = solids[solid_name]
                vertices = solid['vertices']
                faces = solid['faces']

                # Sauvegarder les points du nouveau solide sans tracer à l'écran
                rotated_vertices = rotate(vertices, angle_x, angle_y, angle_z)
                projected_vertices = project(rotated_vertices, width, height, fov=100, viewer_distance=4)
                draw_faces(faces, view_vector, projected_vertices, stdscr, draw=False)
                next_frame = copy_frame_storage()
                reset_frame_storage()

                # Calculer les correspondances et les frames intermédiaires
                matchings, to_remove, to_add = compute_matching(current_frame, next_frame)
                intermediate_frames = generate_intermediate_frames(matchings, 60)

                # Activer la transition
                transition_in_progress = True

                toggle_frame_storage()
            else:
                draw_faces(faces, view_vector, projected_vertices, stdscr)

            # Mettre à jour les angles en fonction des vitesses de rotation
            angle_x += rotation_speeds['x']
            angle_y += rotation_speeds['y']
            angle_z += rotation_speeds['z']

        stdscr.refresh()
        time.sleep(1.0 / 30)

if __name__ == "__main__":
    curses.wrapper(main)
