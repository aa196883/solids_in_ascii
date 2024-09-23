import numpy as np
import curses
import math

def draw_point(stdscr, x, y, char):
    # Afficher le point
    try:
        stdscr.addch(y, x, char)
    except curses.error:
        pass  # Ignorer les erreurs hors écran

def draw_line(stdscr, x0, y0, x1, y1):
    # Convertir les coordonnées en entiers
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    
    # Déterminer la direction du tracé
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # erreur initiale

    x_current, y_current = x0, y0

    while x_current != x1 or y_current != y1:
        draw_point(stdscr, x_current, y_current, '.')

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x_current += sx
        if e2 <= dx:
            err += dx
            y_current += sy
    
    draw_point(stdscr, x0, y0, '*')
    draw_point(stdscr, x1, y1, '*')

def draw_point_w_intensity(stdscr, x, y, intensity):
    try:
        if intensity > 0.75:
            char = 'A'
        elif intensity > 0.5:
            char = 'a'
        elif intensity > 0.25:
            char = '*'
        else:
            char = ' '
        stdscr.addch(y, x, char)
    except curses.error:
        pass

def draw_line_wu(stdscr, x0, y0, x1, y1):
    def fpart(x):
        return x - int(x)

    def rfpart(x):
        return 1 - fpart(x)

    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1

    # Première extrémité
    x_end = round(x0)
    y_end = y0 + gradient * (x_end - x0)
    x_gap = rfpart(x0 + 0.5)
    x_pixel1 = x_end
    y_pixel1 = int(y_end)

    if steep:
        draw_point_w_intensity(stdscr, y_pixel1, x_pixel1, rfpart(y_end) * x_gap)
        draw_point_w_intensity(stdscr, y_pixel1 + 1, x_pixel1, fpart(y_end) * x_gap)
    else:
        draw_point_w_intensity(stdscr, x_pixel1, y_pixel1, rfpart(y_end) * x_gap)
        draw_point_w_intensity(stdscr, x_pixel1, y_pixel1 + 1, fpart(y_end) * x_gap)

    intery = y_end + gradient

    # Deuxième extrémité
    x_end = round(x1)
    y_end = y1 + gradient * (x_end - x1)
    x_gap = fpart(x1 + 0.5)
    x_pixel2 = x_end
    y_pixel2 = int(y_end)

    if steep:
        draw_point_w_intensity(stdscr, y_pixel2, x_pixel2, rfpart(y_end) * x_gap)
        draw_point_w_intensity(stdscr, y_pixel2 + 1, x_pixel2, fpart(y_end) * x_gap)
    else:
        draw_point_w_intensity(stdscr, x_pixel2, y_pixel2, rfpart(y_end) * x_gap)
        draw_point_w_intensity(stdscr, x_pixel2, y_pixel2 + 1, fpart(y_end) * x_gap)

    # Partie centrale
    if steep:
        for x in range(x_pixel1 + 1, x_pixel2):
            y = int(intery)
            draw_point_w_intensity(stdscr, y, x, rfpart(intery))
            draw_point_w_intensity(stdscr, y + 1, x, fpart(intery))
            intery += gradient
    else:
        for x in range(x_pixel1 + 1, x_pixel2):
            y = int(intery)
            draw_point_w_intensity(stdscr, x, y, rfpart(intery))
            draw_point_w_intensity(stdscr, x, y + 1, fpart(intery))
            intery += gradient

def get_char_for_slope(slope):
    if slope == float('inf'):
        return '|'
    angle = math.degrees(math.atan(slope))

    if -15 <= angle <= 15:
        return '-'
    elif 15 < angle <= 75:
        return '/'
    elif 75 < angle <= 105:
        return '|'
    elif 105 < angle <= 165:
        return '\\'
    elif -75 <= angle < -15:
        return '\\'
    elif -105 <= angle < -75:
        return '|'
    elif -165 <= angle < -105:
        return '/'
    elif angle > 165 or angle < -165:
        return '-'
    else:
        return '.'

def draw_line_w_slope(stdscr, x0, y0, x1, y1):
    
    x0, y0 = float(x0), float(y0)
    x1, y1 = float(x1), float(y1)
    
    dx = x1 - x0
    dy = y1 - y0

    # Gérer le cas d'un point unique
    if dx == 0 and dy == 0:
        char = '.'
        draw_point(stdscr, int(round(x0)), int(round(y0)), char)
        return

    # Calculer la pente une fois au début
    if dx != 0:
        slope = dy / dx
    else:
        slope = float('inf')  # Pente infinie pour les lignes verticales

    # Sélectionner le caractère en fonction de la pente
    char = get_char_for_slope(abs(slope))

    # Déterminer si la ligne est plus "raide" (plus verticale qu'horizontale)
    steep = abs(dy) > abs(dx)

    # Échanger x et y si la ligne est "raide"
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    # S'assurer que x0 < x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    # Calculer le gradient
    if dx != 0:
        gradient = dy / dx
    else:
        gradient = 0

    x = x0
    y = y0

    # Tracer la ligne
    while x <= x1:
        if steep:
            draw_point(stdscr, int(round(y)), int(round(x)), char)
        else:
            draw_point(stdscr, int(round(x)), int(round(y)), char)
        x += 1
        y += gradient

def calculate_normal(face_vertices):
    p1, p2, p3 = face_vertices[:3]
    # Convertir les points en vecteurs 3D
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
    v2 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])
    # Produit vectoriel pour obtenir la normale
    normal = np.cross(v1, v2)
    # Normaliser le vecteur
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm
    else:
        normal = np.array([0, 0, 0])
    return normal

def is_face_visible(normal, view_vector):
    dot_product = np.dot(normal, view_vector)
    return dot_product < 0  # La face est visible si le produit scalaire est négatif

def draw_faces(faces, view_vector, projected_vertices, stdscr):
    for face in faces:
        # Sommets de la face (après rotation et projection)
        face_vertices_rotated_projected = [projected_vertices[i] for i in face]
        # Calcul de la normale
        normal = calculate_normal(face_vertices_rotated_projected)
        # Détermination de la visibilité
        if is_face_visible(normal, view_vector):
            # Dessiner les arêtes de la face
            for i in range(len(face)):
                v1 = projected_vertices[face[i]]
                v2 = projected_vertices[face[(i + 1) % len(face)]]
                draw_line_wu(stdscr, v1[0], v1[1], v2[0], v2[1])
            
if __name__ == "__main__":
    for pente in range(-100000, 100000, 1000):
        print('pente=',pente)
        print('symbole=',get_char_for_slope(pente))