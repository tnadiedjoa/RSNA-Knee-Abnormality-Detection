# À soumettre (mis à jour le 08/10 à 22 h 40)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31** (cassées).

Scores du 08/10 : version 41 (E75) **0,940** = notre meilleur modèle · 40 et 42 → 0,934 · 39 → 0,933 · blend 8 en attente.

## Le 09/10 (5 soumissions, à partir de 2 h du matin) : uniquement des blends sur la nouvelle base publique à 0,950

| Notebook | Version | Contenu (base publique 0,950 + nos modèles) | Question |
|---|---|---|---|
| knee-blend | **9** | + nos modèles entraînés sur nos seuls labels (E64, E60, E65) à 15 % | dépasse-t-il 0,950 ? |
| knee-blend | **10** | + nos meilleurs modèles 384 px (E75, E56, E59) à 15 % | lequel des deux lots apporte le plus ? |
| knee-blend | **11** | + un mélange des deux (E75, E56, E64, E60, E81) à 15 % | mieux que 9 et 10 ? |
| knee-blend | **12** | même mélange que le 11, mais à 25 % | faut-il donner plus de poids à nos modèles ? |
| knee-blend | **13** (je confirme ce soir) | même mélange que le 11, à 10 % | ou moins de poids ? |

Les tests de modèles seuls (knee-submit 43, 38, 36, 37, 33) attendent : on ne les soumet que s'ils peuvent changer ce qu'on met dans les blends.
