import numpy as np
from itertools import combinations, permutations


# Définition des sommets du dodécaèdre
phi = (1 + np.sqrt(5)) / 2      # Nombre d'or ≈ 1.61803
inv_phi = 1 / phi               # Inverse du nombre d'or ≈ 0.61803

def calculate_edges(vertices, edge_length):
    # Tolérance pour la comparaison des distances
    tolerance = 1e-5

    # Génération des arêtes
    edges = []
    num_vertices = len(vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            distance = np.linalg.norm(vertices[i] - vertices[j])
            if abs(distance - edge_length) < tolerance:
                edges.append((i, j))

    return edges

def build_adjacency_list(edges):
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

def find_faces(adjacency, face_size):
    faces = set()
    for vertices in permutations(adjacency.keys(), face_size):
        # Vérifier que tous les sommets sont connectés pour former une face
        is_face = True
        for i in range(len(vertices)):
            if vertices[(i+1)%len(vertices)] not in adjacency[vertices[i]]:
                is_face = False
                break
        if is_face:
            # Normaliser l'ordre des sommets pour éviter les doublons
            min_index = vertices.index(min(vertices))
            ordered_vertices = vertices[min_index:] + vertices[:min_index]
            faces.add(tuple(ordered_vertices))
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

def main():
    vertices = np.array([
            # Sommets (0, ±1, ±phi)
            [0, -1, -phi], [0, -1, phi], [0, 1, -phi], [0, 1, phi],
            # Sommets (±1, ±phi, 0)
            [-1, -phi, 0], [-1, phi, 0], [1, -phi, 0], [1, phi, 0],
            # Sommets (±phi, 0, ±1)
            [-phi, 0, -1], [-phi, 0, 1], [phi, 0, -1], [phi, 0, 1]
    ])

    # Calculer le centre du solide
    center = calculate_center(vertices)

    edges = calculate_edges(vertices, edge_length=2)
    adj_list = build_adjacency_list(edges=edges)
    print(edges)
    # print(adj_list)

    faces = find_faces(adj_list, face_size=3)
    
    oritend_faces = []
    for face in faces:
        oritend_faces.append(orient_face(vertices, face, center))
    
    print(oritend_faces)


if __name__ == "__main__":
    main()