# À soumettre (mis à jour le 09/10 à 3 h 20)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31** (cassées).

Scores du 08/10 : version 41 (E75) **0,940** = notre meilleur modèle · 40 et 42 → 0,934 · 39 → 0,933 · blend 8 → 0,943 (= la base publique 0,943 seule : un poids unique n'apporte rien).

## Le 09/10 (5 soumissions, à partir de 2 h du matin) : blends sur la nouvelle base publique à 0,950

La base à 0,950 mélange deux modèles publics, avec un poids différent pour chaque anomalie. Les versions 14, 16 et 15
glissent nos modèles dans ce mélange, avec la même logique. Les versions 11 et 9 les ajoutent avec le même poids partout.
Un autre participant a mesuré qu'un poids unique partout fait baisser ce genre de mélange (0,949 → 0,948) : d'où l'essai.

| Notebook | Version | Contenu (base publique 0,950 + nos modèles) | Question |
|---|---|---|---|
| knee-blend | **14** | nos modèles (E75, E56, E64, E60, E81) prennent la moitié de la place du 2e modèle public | dépasse-t-il 0,950 ? |
| knee-blend | **16** | comme le 14, avec 50 % de place en plus pour ces deux modèles | faut-il leur donner plus de poids ? |
| knee-blend | **15** | nos modèles remplacent complètement le 2e modèle public | les nôtres valent-ils mieux que lui ? |
| knee-blend | **11** | mêmes modèles, ajoutés à 15 % partout | témoin : le même poids partout |
| knee-blend | **9** | nos modèles entraînés sur nos seuls labels (E64, E60, E65), à 15 % partout | témoin : quel lot de modèles ? |

Les versions 10, 12 et 13 (même poids partout) restent en réserve.
Les tests de modèles seuls (knee-submit 43, 38, 36, 37, 33) attendent : on ne les soumet que s'ils peuvent changer ce
qu'on met dans les blends.
