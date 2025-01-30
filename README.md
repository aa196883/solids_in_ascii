# Animation ASCII de Solides en Rotation

Ce projet propose une animation ASCII en terminal affichant des **solides réguliers en rotation**, avec une transition fluide entre les changements de solides.

L'affichage utilise plusieurs méthodes de tracé de lignes et une transition procédurale entre les formes.

---

##  **Fonctionnalités**
###  **Affichage et Tracés**
- **Rotation et projection** des solides en 3D.
- **Trois modes de tracé des arêtes**, modifiables dans `draw_faces` de `draw.py` :
  1. **Points (`.`)**
  2. **Anti-aliasing**
  3. **Caractères directionnels**

###  **Transitions entre les solides**
- Les **caractères se déplacent indépendamment** vers les nouvelles positions.

###  **Définition des solides**
- Les coordonnées canoniques des points ont été **trouvées sur Wikipédia**.
- Des **fonctions génériques et  assez optimisées** identifient les faces des solides à partir de leurs sommets.
- Une version précalculée de ces éléments est disponible dans `solids.py`.

---

##  **Structure du projet**
| Fichier | Description |
|---------|------------|
| `main.py` | Orchestration de l'animation : rotation, projection, transitions. |
| `draw.py` | Gestion des tracés des solides. |
| `generate_solids_infos.py` | Génération des coordonnées et des faces des solides. |
| `solides.py` | Contient les données des solides pré-calculées. |
| `matching.py` | Gère les animations de transition entre les solides. |

---

##  **Installation**
Aucune installation spécifique n'est requise, à part Python et les bibliothèques mentionnées dans `requirements.txt`.

###  **Installation des dépendances**
```sh
pip install -r requirements.txt
```
_(Gère automatiquement les différences entre Windows et Linux pour l'affichage en terminal)_

###  **Exécution**
```sh
python main.py
```

---

##  **Modifier le style de tracé**
Dans `draw.py`, la fonction `draw_faces` permet de **changer le mode de tracé** en **commentant/décommentant** des lignes :

```python
# Mode anti-aliasing
# draw_line_wu(stdscr, v1[0], v1[1], v2[0], v2[1], draw)
# Mode caractères directionnels
# draw_line_w_slope(stdscr, v1[0], v1[1], v2[0], v2[1], draw)
# Mode point simple
draw_line(stdscr, v1[0], v1[1], v2[0], v2[1], draw)
```

---

##  **Notes**
- Ce projet est écrit pour fonctionner **à 30 FPS**.
- **L’animation tourne indéfiniment** avec des rotations aléatoires.
- **Appuyez sur `q`** pour quitter proprement.