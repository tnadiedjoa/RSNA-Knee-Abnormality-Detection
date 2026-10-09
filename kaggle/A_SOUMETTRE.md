# À soumettre (mis à jour le 09/10 à 14 h 50)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31**, ni knee-blend **18, 19, 20 ni 21** (cassées).

Scores du 09/10 : knee-blend 9, 11, 14 et 16 → **0,950** (= la base publique seule) · 15 → 0,949. Nos modèles n'ajoutent
rien de visible à cette base. Notre meilleur score au classement passe de 0,944 à 0,950 (328e sur 5 576).

## Le 10/10 (5 soumissions, à partir de 2 h du matin) : nouvelle base publique (heliosli, annoncée à 0,951)

Le notebook public de heliosli mélange le modèle public à 0,949 (70 %) avec son propre modèle ConvNeXt (30 %).
Les versions 23 à 25 y ajoutent le 2e modèle de la base à 0,950 (le « lecteur » 2.5D), avec un poids par anomalie.

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-blend | **23** | heliosli + le lecteur, avec les poids de la base à 0,950 | dépasse-t-il 0,951 ? |
| knee-blend | **24** | pareil, poids du lecteur divisés par deux | moins de lecteur ? |
| knee-blend | **22** | le notebook de heliosli seul | confirme-t-il 0,951 ? |
| knee-blend | **25** | pareil que le 23, poids du lecteur × 1,5 | plus de lecteur ? |
| knee-blend | **17** | la base à 0,950 + le petit frère du modèle à 0,949 (celui à 0,945) | ce petit frère aide-t-il ? |

Les tests de modèles seuls (knee-submit 43, 38, 36, 37, 33) attendent : on ne les soumet que s'ils peuvent changer ce
qu'on met dans les blends.
