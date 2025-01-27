import numpy as np
import math
from itertools import combinations, permutations, product
import time
from collections import Counter

# Définition des sommets du dodécaèdre
phi = (1 + np.sqrt(5)) / 2      # Nombre d'or ≈ 1.61803
inv_phi = 1 / phi               # Inverse du nombre d'or ≈ 0.61803

def calculate_edges(vertices, edge_lengths):
    """
    Génère les arêtes en vérifiant si la distance entre deux sommets
    correspond à l'une des longueurs d'arête spécifiques.
    
    Paramètres:
    - vertices: Liste ou tableau NumPy des coordonnées des sommets.
    - edge_lengths: Liste des longueurs d'arête possibles.
    
    Retourne:
    - edges: Liste des arêtes sous forme de tuples d'indices de sommets (i, j).
    """
    # Tolérance pour la comparaison des distances
    tolerance = 1e-5

    edges = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            distance = np.linalg.norm(vertices[i] - vertices[j])
            for edge_length in edge_lengths:
                if abs(distance - edge_length) <= tolerance:
                    edges.append((i, j))
                    break  # On peut sortir de la boucle des longueurs d'arête
    return edges

def calculate_edge_lengths(vertices, n):
    """
    Calcule les n distances les plus petites (longueurs d'arête) entre le premier sommet
    et les autres sommets.

    Paramètres:
    - vertices: Liste ou tableau NumPy des coordonnées des sommets.
    - n: Nombre de distances les plus petites à retourner.

    Retourne:
    - edge_lengths: Liste des n distances les plus petites.
    """
    if len(vertices) < 2:
        raise ValueError("La liste des sommets doit contenir au moins deux sommets.")
    if n < 1:
        raise ValueError("Le paramètre n doit être au moins égal à 1.")
    if n > len(vertices) - 1:
        raise ValueError("Le paramètre n ne peut pas être supérieur au nombre de sommets moins un.")

    first_vertex = vertices[0]
    distances = []
    for i in range(1, len(vertices)):
        distance = np.linalg.norm(first_vertex - vertices[i])
        distances.append(distance)
    # Supprimer les doublons et trier les distances
    unique_distances = sorted(set(distances))

    # Retourner les n distances les plus petites
    edge_lengths = unique_distances[:n]
    return edge_lengths

def build_adjacency_dict(edges):
    adjacency = dict()
    for edge in edges:
        i, j = edge
        if i not in adjacency.keys():
            adjacency[i] = set()
        adjacency[i].add(j)
        if j not in adjacency.keys():
            adjacency[j] = set()
        adjacency[j].add(i)
    return adjacency

def calculate_faces(adjacency, center, face_sizes, all_vertices):
    faces = set()
    for face_size in face_sizes:
        for vertices in permutations(adjacency.keys(), face_size):
            # Vérifier que tous les sommets sont connectés pour former une face
            is_face = True
            for i in range(len(vertices)):
                if vertices[(i+1)%len(vertices)] not in adjacency[vertices[i]]:
                    is_face = False
                    break
            if is_face:
                # Orienter les faces vers l'exterieur
                oriented_face = orient_face(all_vertices, vertices, center)
                # Normaliser l'ordre des sommets pour éviter les doublons
                min_index = oriented_face.index(min(oriented_face))
                ordered_oriented_face = oriented_face[min_index:] + oriented_face[:min_index]
                faces.add(tuple(ordered_oriented_face))
    return [list(face) for face in faces]

def calculate_normal(face_vertices):
    # Obtenir trois points de la face
    p1, p2, p3 = face_vertices[:3]
    # Vecteurs entre les points
    v1 = p2 - p1
    v2 = p3 - p1
    # Produit vectoriel pour obtenir la normale
    normal = np.cross(v1, v2)
    # Normaliser le vecteur
    normal = normal / np.linalg.norm(normal)
    return normal

def calculate_center(vertices):
    return np.mean(vertices, axis=0)

def orient_face(vertices, face, center):
    # Obtenir les positions des sommets de la face
    face_vertices = [vertices[i] for i in face]
    # Calculer la normale de la face
    normal = calculate_normal(face_vertices)
    # Calculer un vecteur du centre du solide vers un point de la face
    to_face_vector = face_vertices[0] - center
    # Calculer le produit scalaire
    dot_product = np.dot(normal, to_face_vector)
    if dot_product < 0:
        # Si le produit scalaire est négatif, inverser l'ordre des sommets
        face = face[::-1]
    return face

def remove_sublists_with_common_elements(list_of_lists):
    unique_lists = []
    
    for sublist in list_of_lists:
        # Vérifier si la sous-liste partage au moins trois éléments avec une autre sous-liste dans unique_lists de même longueur
        is_duplicate = False
        for unique_sublist in unique_lists:
            if len(sublist) == len(unique_sublist):  # Comparer seulement les sous-listes de même longueur
                common_elements = set(sublist) & set(unique_sublist)
                if len(common_elements) >= 3:
                    is_duplicate = True
                    break

        # Ajouter sublist à unique_lists si elle n'a pas au moins trois éléments communs avec une autre sous-liste de même longueur
        if not is_duplicate:
            unique_lists.append(sublist)
    
    return unique_lists

def is_coplanar(points):
    if len(points) < 4:
        # Moins de 4 points sont toujours coplanaires
        return True
    
    # Prendre les trois premiers points comme référence
    p1, p2, p3 = points[:3]
    
    for p in points[3:]:
        # Créer des vecteurs à partir de p1
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        v3 = np.array(p)  - np.array(p1)
        
        # Calcul du volume du parallélotope (produit mixte)
        volume = np.dot(np.cross(v1, v2), v3)
        
        # Si le volume est non nul, les points ne sont pas coplanaires
        if not np.isclose(volume, 0):
            return False
    
    return True

def find_cycles_from_vertex(adjacency, start_vertex, max_length, face_sizes, vertices):
    """
    Trouve tous les cycles de longueurs spécifiques commençant par le sommet start_vertex.
    
    adjacency : dict, un dictionnaire représentant les connexions entre les sommets
    start_vertex : int, le sommet de départ pour la recherche
    max_length : int, la longueur maximale d'un chemin
    face_sizes : list, liste des tailles de faces recherchées (par exemple, [4, 6])
    """
    def dfs(current_vertex, path, visited):
        if (len(path) - 1) > max_length:
            return
        
        # Si on retourne au sommet de départ avec la bonne longueur
        if (len(path) - 1) in face_sizes and current_vertex == start_vertex:
            path = path[:-1]
            # Orienter les faces vers l'exterieur
            oriented_face = orient_face(vertices, path, center)
            # Normaliser l'ordre des sommets pour éviter les doublons
            min_index = oriented_face.index(min(oriented_face))
            ordered_oriented_face = oriented_face[min_index:] + oriented_face[:min_index]
            cycles.add(tuple(ordered_oriented_face))
            return
        
        # Explorer les sommets adjacents
        for neighbor in adjacency[current_vertex]:
            if neighbor not in visited or (neighbor == start_vertex and len(path) in face_sizes):
                points = [vertices[index] for index in path]
                if is_coplanar(points):
                    dfs(neighbor, path + [neighbor], visited | {neighbor})

    # Ensemble pour stocker les cycles uniques
    cycles = set()

    # Pour normaliser les faces
    center = calculate_center(vertices)

    # Lancer la recherche en profondeur
    dfs(start_vertex, [start_vertex], {start_vertex})

    return cycles

# Exécuter la recherche pour tous les sommets
def find_all_cycles(adjacency, face_sizes, vertices):
    all_cycles = set()
    for vertex in adjacency:
        all_cycles.update(find_cycles_from_vertex(adjacency, vertex, max(face_sizes), face_sizes, vertices))

    all_cycles = [list(cycle) for cycle in all_cycles]

    return all_cycles

def generate_circular_permutations(coordinate_sets):
    circular_permutations = set()

    for coords in coordinate_sets:
        # Générer les permutations circulaires
        for i in range(3):
            permuted_coords = coords[i:] + coords[:i]
            circular_permutations.add(permuted_coords)

    return list(circular_permutations)

def main():
    phi = (1 + math.sqrt(5)) / 2
    phi_squared = phi ** 2

    # coordinate_sets = [
    #     (0, 1/phi, (2 + phi)),
    #     ((2 + phi), 0, 1/phi),
    #     (1/phi, (2 + phi), 0),
    #     (1/phi, phi, 2*phi),
    #     (2*phi, 1/phi, phi),
    #     (phi, 2*phi, 1/phi),
    #     (phi, 2, phi_squared),
    #     (phi_squared, phi, 2),
    #     (2, phi_squared, phi)
    # ]
    coordinate_sets = [
        (1/phi, 1/phi, (3 + phi)),
        (2/phi, phi, (1 + 2*phi)),
        (1/phi, phi_squared, (-1 + 3*phi)),
        ((2*phi - 1),2, (2 + phi)),
        (phi, 3, 2*phi)
    ]

    fact = 3.0
    coordinate_sets = [(val1/fact, val2/fact, val3/fact) for (val1, val2, val3) in coordinate_sets]
    coordinate_sets = generate_circular_permutations(coordinate_sets)

    # Générer toutes les combinaisons de signes pour chaque ensemble de coordonnées
    vertices = set()
    for coords in coordinate_sets:
        for signs in product(*[[-c, c] for c in coords]):  # Appliquer ± à chaque coordonnée
            vertices.add(signs)

    vertices = np.array(list(vertices))
    print(len(vertices))

    ##########################################################################################

    distances = calculate_edge_lengths(vertices, 1)

    edges = calculate_edges(vertices, distances)

    adjacency = build_adjacency_dict(edges)
    face_sizes = [10, 6, 4]
    faces = find_all_cycles(adjacency, face_sizes, vertices)

    vertices = [list(vertex) for vertex in list(vertices)]

    print(f"Nombre de sommets : {len(vertices)}")
    print(f"Nombre d'arrêtes : {len(edges)}")
    print(f"Nombre de faces : {len(faces)}")

    with open('res.txt', 'w') as file:
        file.write(f"'vertices' : {vertices},\n")
        file.write(f"'edges' : {edges},\n")
        file.write(f"'faces' : {faces}")
    
    # Calculer la taille de chaque sous-liste
    sizes = [len(sublist) for sublist in faces]
    # Compter le nombre de sous-listes pour chaque taille
    size_counts = Counter(sizes)
    print(size_counts)

if __name__ == "__main__":
    main()