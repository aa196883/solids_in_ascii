import numpy as np
import curses
from draw import draw_point
import time

# Sous-fonction pour regrouper les éléments par caractères
def group_by_characters(frame):
    grouped = {}
    for char, coord in frame:
        if char not in grouped:
            grouped[char] = []
        grouped[char].append(coord)
    return grouped

# Sous-fonction pour calculer les correspondances entre deux listes de coordonnées
def match_coordinates(current_coords, next_coords):
    """
    Calcule les correspondances entre deux listes de coordonnées.
    Permet les correspondances multiples après le matching unitaire.

    :param current_coords: Liste des coordonnées actuelles [(x, y), ...].
    :param next_coords: Liste des coordonnées suivantes [(x, y), ...].
    :return:
        - matchings: Liste des correspondances [(i, j)], où i est un indice de current_coords
                     et j un indice de next_coords (inclut 1-à-1, 1-à-N, et N-à-1).
        - unmatched_current: Indices des éléments de current_coords non matchés.
        - unmatched_next: Indices des éléments de next_coords non matchés.
    """
    current_coords = np.array(current_coords)
    next_coords = np.array(next_coords)

    # Calculer les distances entre chaque paire de points
    distance_matrix = np.linalg.norm(current_coords[:, None] - next_coords, axis=2)

    # Phase 1 : Matching unitaire (1-à-1)
    matchings = []
    used_next = np.zeros(len(next_coords), dtype=bool)
    used_current = np.zeros(len(current_coords), dtype=bool)

    for i in range(len(current_coords)):
        # Trouver l'indice du point le plus proche parmi ceux disponibles
        if np.all(used_next):
            break
        distances = distance_matrix[i]
        min_index = np.argmin(distances + used_next * 1e9)  # Éviter les doublons
        if not used_next[min_index]:
            matchings.append((i, min_index))  # Stocker les indices correspondants
            used_next[min_index] = True
            used_current[i] = True

    # Phase 2 : Matching multiple (1-à-N et N-à-1)
    for i in range(len(current_coords)):
        if not used_current[i]:  # Si un élément de current_frame n'a pas été matché
            distances = distance_matrix[i]
            # Trouver tous les points de next_frame les plus proches, non utilisés
            for j in np.argsort(distances):
                # if not used_next[j]:  # Ajouter une correspondance 1-à-N
                matchings.append((i, j))
                used_current[i] = True
                break

    for j in range(len(next_coords)):
        if not used_next[j]:  # Si un élément de next_frame n'a pas été matché
            distances = distance_matrix[:, j]
            # Trouver tous les points de current_frame les plus proches, non utilisés
            for i in np.argsort(distances):
                # if not used_current[i]:  # Ajouter une correspondance N-à-1
                matchings.append((i, j))
                used_next[j] = True
                break

    unmatched_current = ~used_current
    unmatched_next = ~used_next

    return matchings, unmatched_current, unmatched_next

# Fonction principale pour effectuer le matching
def compute_matching(current_frame, next_frame):
    """
    Calcule les correspondances entre deux frames.
    
    :param current_frame: Liste de tuples [(char, (x, y)), ...] pour la frame actuelle.
    :param next_frame: Liste de tuples [(char, (x, y)), ...] pour la frame suivante.
    :return: 
        - all_matchings: Liste des correspondances [(char, (x_start, y_start), (x_end, y_end)), ...].
        - to_remove: Liste des éléments de current_frame non correspondants [(char, (x, y)), ...].
        - to_add: Liste des éléments de next_frame non correspondants [(char, (x, y)), ...].
    """

    # Grouper les éléments par caractères
    current_grouped = group_by_characters(current_frame)
    next_grouped = group_by_characters(next_frame)

    # Initialiser les structures de sortie
    all_matchings = []
    to_remove = []
    to_add = []

    # Effectuer le matching pour chaque caractère
    for char in set(current_grouped.keys()).union(next_grouped.keys()):
        current_coords = current_grouped.get(char, [])
        next_coords = next_grouped.get(char, [])

        if not current_coords and next_coords:
            # Tous les nouveaux points sont à ajouter
            to_add.extend([(char, coord) for coord in next_coords])
        elif not next_coords and current_coords:
            # Tous les anciens points sont à supprimer
            to_remove.extend([(char, coord) for coord in current_coords])
        elif current_coords and next_coords:
            # Calculer le matching
            matchings, unmatched_current, unmatched_next = match_coordinates(current_coords, next_coords)

            # Stocker les correspondances
            all_matchings.extend([
                (char, current_coords[i], next_coords[j]) for i, j in matchings
            ])

            # Points à supprimer (non matchés dans current)
            to_remove.extend([(char, current_coords[i]) for i in np.where(unmatched_current)[0]])

            # Points à ajouter (non matchés dans next)
            to_add.extend([(char, next_coords[j]) for j in np.where(unmatched_next)[0]])
    return all_matchings, to_remove, to_add

def generate_intermediate_frames(matchings, steps):
    """
    Génère les états intermédiaires d'une transition.
    
    :param matchings: Liste de tuples [(char, (x_start, y_start), (x_end, y_end)), ...]
                      contenant les correspondances entre points.
    :param steps: Nombre total d'étapes de la transition.
    :return: Liste de liste, où chaque sous-liste contient les coordonnées des éléments en mouvement.
    """
    frames = [[] for _ in range(steps)]  # Liste contenant les états intermédiaires

    # Calculer la distance maximale parcourue par un élément
    max_distance = 0
    for _, start, end in matchings:
        start = np.array(start)
        end = np.array(end)
        distance = np.linalg.norm(end - start)
        max_distance = max(max_distance, distance)

    # Déterminer la vitesse globale (distance par étape)
    global_speed = max_distance / steps

    for char, start, end in matchings:
        start = np.array(start)
        end = np.array(end)
        
        # Calculer la distance totale parcourue par cet élément
        distance = np.linalg.norm(end - start)

        # Nombre d'étapes nécessaires pour atteindre la destination à la vitesse globale
        if distance == 0:
            num_steps = 0
        else:
            num_steps = int(np.ceil(distance / global_speed))

        # Calculer le vecteur de déplacement
        if num_steps > 0:
            step_vector = (end - start) / num_steps
        else:
            step_vector = np.zeros_like(start)

        # Générer les positions intermédiaires
        for i in range(steps):
            if i < num_steps:
                current_position = start + step_vector * i
                frames[i].append((char, tuple(np.round(current_position).astype(int))))
            else:
                # Rester immobile après avoir atteint la destination
                frames[i].append((char, tuple(np.round(end).astype(int))))

    # Mettre les coordonnées dans le bon sens
    frames = [[(char, (x,y)) for (char, (y,x)) in frame] for frame in frames]

    return frames


def main(stdscr):
    # Exemple d'utilisation
    matchings = [
        ('/', (1, 1), (10, 10)),  # Distance 12.73
        ('\\', (2, 2), (3, 3)),  # Distance 1.41
        ('|', (3, 3), (6, 8))    # Distance 5.83
    ]

    steps = 60

    # intermediate_frames = [
    #     [('/', (135, 52)), ('/', (136, 51)), ('/', (137, 51)), ('/', (138, 50)), ('/', (139, 50)), ('/', (140, 49)), ('/', (141, 49)), ('/', (142, 48)), ('/', (143, 48)), ('/', (144, 47)), ('/', (145, 47)), ('/', (146, 46)), ('/', (147, 46)), ('/', (148, 45)), ('/', (149, 45)), ('/', (150, 44)), ('/', (151, 44)), ('/', (152, 43)), ('/', (153, 43)), ('/', (154, 42)), ('/', (155, 42)), ('/', (156, 41)), ('/', (157, 41)), ('/', (158, 40)), ('/', (113, 30)), ('/', (113, 31)), ('/', (112, 32)), ('/', (112, 33)), ('/', (112, 34)), ('/', (111, 35)), ('/', (111, 36)), ('/', (110, 37)), ('/', (110, 38)), ('/', (110, 39)), ('/', (109, 40)), ('/', (109, 41)), ('/', (109, 42)), ('/', (108, 43)), ('/', (108, 44)), ('/', (107, 45)), ('/', (107, 46)), ('/', (107, 47)), ('/', (123, 14)), ('/', (122, 15)), ('/', (121, 16)), ('/', (121, 17)), ('/', (120, 18)), ('/', (120, 19)), ('/', (119, 20)), ('/', (119, 21)), ('/', (118, 22)), ('/', (117, 23)), ('/', (117, 24)), ('/', (116, 25)), ('/', (116, 26)), ('/', (115, 27)), ('/', (115, 28)), ('/', (114, 29)), ('/', (113, 30)), ('/', (78, 23)), ('/', (79, 22)), ('/', (80, 21)), ('/', (81, 21)), ('/', (82, 20)), ('/', (83, 20)), ('/', (84, 19)), ('/', (85, 19)), ('/', (86, 18)), ('/', (87, 18)), ('/', (88, 17)), ('/', (89, 17)), ('/', (90, 16)), ('/', (91, 16)), ('/', (92, 15)), ('/', (93, 15)), ('/', (94, 14)), ('/', (95, 14)), ('/', (96, 13)), ('/', (97, 13)), ('/', (98, 12)), ('/', (99, 12)), ('/', (113, 30)), ('/', (113, 31)), ('/', (112, 32)), ('/', (112, 33)), ('/', (112, 34)), ('/', (111, 35)), ('/', (111, 36)), ('/', (110, 37)), ('/', (110, 38)), ('/', (110, 39)), ('/', (109, 40)), ('/', (109, 41)), ('/', (109, 42)), ('/', (108, 43)), ('/', (108, 44)), ('/', (107, 45)), ('/', (107, 46)), ('/', (107, 47)), ('/', (76, 36)), ('/', (77, 36)), ('/', (78, 36)), ('/', (79, 35)), ('/', (80, 35)), ('/', (81, 35)), ('/', (123, 14)), ('/', (124, 13)), ('/', (125, 12)), ('/', (126, 11)), ('/', (127, 11)), ('/', (123, 14)), ('/', (124, 13)), ('/', (125, 12)), ('/', (126, 11)), ('/', (127, 11)), ('/', (123, 14)), ('/', (122, 15)), ('/', (121, 16)), ('/', (121, 17)), ('/', (120, 18)), ('/', (120, 19)), ('/', (119, 20)), ('/', (119, 21)), ('/', (118, 22)), ('/', (117, 23)), ('/', (117, 24)), ('/', (116, 25)), ('/', (116, 26)), ('/', (115, 27)), ('/', (115, 28)), ('/', (114, 29)), ('/', (113, 30)), ('/', (135, 52)), ('/', (136, 51)), ('/', (137, 51)), ('/', (138, 50)), ('/', (139, 50)), ('/', (140, 50)), ('/', (141, 49)), ('/', (142, 49)), ('/', (143, 48)), ('/', (144, 48)), ('/', (145, 48)), ('/', (146, 47)), ('/', (147, 47)), ('/', (148, 46)), ('/', (148, 46)), ('/', (149, 46)), ('/', (150, 45)), ('/', (151, 44)), ('/', (152, 44)), ('/', (153, 43)), ('/', (154, 43)), ('/', (155, 42)), ('/', (156, 41)), ('/', (157, 41)), ('/', (135, 52)), ('/', (136, 51)), ('/', (137, 51)), ('/', (138, 50)), ('/', (139, 50)), ('/', (140, 49)), ('/', (141, 49)), ('/', (142, 48)), ('/', (143, 48)), ('/', (144, 47)), ('/', (145, 47)), ('/', (146, 46)), ('/', (147, 46)), ('/', (148, 45)), ('/', (149, 45)), ('/', (150, 44)), ('/', (151, 44)), ('/', (152, 43)), ('/', (153, 43)), ('/', (154, 42)), ('/', (155, 42)), ('/', (156, 41)), ('/', (157, 41)), ('/', (158, 40)), ('/', (76, 36)), ('/', (77, 36)), ('/', (78, 36)), ('/', (79, 35)), ('/', (80, 35)), ('/', (81, 35)), ('-', (150, 26)), ('-', (151, 26)), ('-', (152, 26)), ('-', (153, 26)), ('-', (154, 26)), ('-', (155, 26)), ('-', (156, 26)), ('-', (157, 26)), ('-', (158, 26)), ('-', (159, 26)), ('-', (160, 26)), ('-', (161, 26)), ('-', (106, 48)), ('-', (107, 48)), ('-', (108, 48)), ('-', (109, 48)), ('-', (110, 48)), ('-', (111, 48)), ('-', (112, 49)), ('-', (113, 49)), ('-', (114, 49)), ('-', (115, 49)), ('-', (116, 49)), ('-', (117, 49)), ('-', (118, 49)), ('-', (119, 50)), ('-', (120, 50)), ('-', (121, 50)), ('-', (122, 50)), ('-', (123, 50)), ('-', (124, 50)), ('-', (125, 50)), ('-', (126, 50)), ('-', (127, 51)), ('-', (128, 51)), ('-', (129, 51)), ('-', (130, 51)), ('-', (131, 51)), ('-', (132, 51)), ('-', (133, 51)), ('-', (134, 52)), ('-', (107, 53)), ('-', (108, 53)), ('-', (109, 53)), ('-', (110, 53)), ('-', (111, 53)), ('-', (112, 53)), ('-', (113, 53)), ('-', (114, 52)), ('-', (115, 52)), ('-', (116, 52)), ('-', (117, 52)), ('-', (118, 52)), ('-', (119, 52)), ('-', (120, 52)), ('-', (121, 52)), ('-', (122, 52)), ('-', (123, 52)), ('-', (124, 52)), ('-', (125, 52)), ('-', (126, 52)), ('-', (127, 52)), ('-', (128, 52)), ('-', (129, 52)), ('-', (130, 52)), ('-', (131, 52)), ('-', (132, 52)), ('-', (133, 52)), ('-', (134, 52)), ('-', (113, 30)), ('-', (114, 30)), ('-', (115, 30)), ('-', (116, 30)), ('-', (117, 30)), ('-', (118, 30)), ('-', (119, 30)), ('-', (120, 29)), ('-', (121, 29)), ('-', (122, 29)), ('-', (123, 29)), ('-', (124, 29)), ('-', (125, 29)), ('-', (126, 29)), ('-', (127, 29)), ('-', (128, 29)), ('-', (129, 28)), ('-', (130, 28)), ('-', (131, 28)), ('-', (132, 28)), ('-', (133, 28)), ('-', (134, 28)), ('-', (135, 28)), ('-', (136, 28)), ('-', (137, 28)), ('-', (138, 28)), ('-', (139, 27)), ('-', (140, 27)), ('-', (141, 27)), ('-', (142, 27)), ('-', (143, 27)), ('-', (144, 27)), ('-', (145, 27)), ('\\', (150, 26)), ('\\', (150, 27)), ('\\', (151, 28)), ('\\', (151, 29)), ('\\', (152, 30)), ('\\', (153, 31)), ('\\', (153, 32)), ('\\', (154, 33)), ('\\', (154, 34)), ('\\', (155, 35)), ('\\', (156, 36)), ('\\', (156, 37)), ('\\', (157, 38)), ('\\', (157, 39)), ('\\', (150, 26)), ('\\', (150, 27)), ('\\', (151, 28)), ('\\', (151, 29)), ('\\', (152, 30)), ('\\', (153, 31)), ('\\', (153, 32)), ('\\', (154, 33)), ('\\', (154, 34)), ('\\', (155, 35)), ('\\', (156, 36)), ('\\', (156, 37)), ('\\', (157, 38)), ('\\', (157, 39)), ('\\', (81, 35)), ('\\', (82, 35)), ('\\', (83, 36)), ('\\', (84, 36)), ('\\', (85, 37)), ('\\', (86, 37)), ('\\', (87, 38)), ('\\', (88, 38)), ('\\', (89, 39)), ('\\', (90, 39)), ('\\', (91, 40)), ('\\', (92, 40)), ('\\', (93, 41)), ('\\', (94, 41)), ('\\', (95, 42)), ('\\', (96, 42)), ('\\', (97, 43)), ('\\', (98, 44)), ('\\', (99, 44)), ('\\', (100, 45)), ('\\', (101, 45)), ('\\', (102, 46)), ('\\', (103, 46)), ('\\', (104, 47)), ('\\', (105, 47)), ('\\', (106, 48)), ('\\', (81, 35)), ('\\', (82, 35)), ('\\', (83, 36)), ('\\', (84, 36)), ('\\', (85, 37)), ('\\', (86, 37)), ('\\', (87, 38)), ('\\', (88, 38)), ('\\', (89, 39)), ('\\', (90, 39)), ('\\', (91, 40)), ('\\', (92, 40)), ('\\', (93, 41)), ('\\', (94, 41)), ('\\', (95, 42)), ('\\', (96, 42)), ('\\', (97, 43)), ('\\', (98, 44)), ('\\', (99, 44)), ('\\', (100, 45)), ('\\', (101, 45)), ('\\', (102, 46)), ('\\', (103, 46)), ('\\', (104, 47)), ('\\', (105, 47)), ('\\', (106, 48)), ('\\', (76, 36)), ('\\', (77, 37)), ('\\', (78, 38)), ('\\', (79, 38))]
    # ]
    intermediate_frames = []
    with open("output.txt", "r") as file:
        intermediate_frames = eval(file.read())

    intermediate_frames = [[(char, (x,y)) for (char, (y,x)) in frame] for frame in intermediate_frames]

    curses.curs_set(0)

    for frame in intermediate_frames:
        stdscr.clear()

        for char, (x, y) in frame:
            draw_point(stdscr, y, x, char)

        stdscr.refresh()
        time.sleep(1.0/30)

    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)