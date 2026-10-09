# À soumettre (mis à jour le 09/10 à 18 h 40)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31**, ni knee-blend **18, 19, 20 ni 21** (cassées).

Scores du 09/10 : knee-blend 9, 11, 14 et 16 → **0,950** (= la base publique seule) · 15 → 0,949. Notre meilleur score au
classement passe de 0,944 à 0,950 (328e sur 5 576).

## ⚠️ Changement de liste : ne pas soumettre les versions 22 à 28

heliosli a retiré son modèle à 13 h 50 aujourd'hui. Son jeu de données dit maintenant : « les versions précédentes ont été
retirées et ne sont plus accessibles ». Les versions 22 à 28 utilisent notre copie privée de ce modèle. Le règlement exige
que les modèles externes restent accessibles à tous, et l'auteur ne veut plus qu'on s'en serve : je ne les soumettrais pas,
et on ne pourra de toute façon pas les garder pour la sélection finale. C'est à toi de décider.

## Le 10/10 (5 soumissions, à partir de 2 h du matin) : base publique à 0,950, rien de retiré

La base à 0,950 mélange deux modèles publics, toujours disponibles : le modèle à 0,949 (nartaa) et un « lecteur »
(goodpjw2008), avec un poids fixé à la main pour chaque anomalie. Sur l'arthrose latérale, le modèle à 0,949 est faible
(0,78 sur nos 58 examens, contre 0,86 pour le lecteur), mais le lecteur n'y pèse que 8 %. Un autre participant teste la même
idée sans score publié pour l'instant.

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-blend | **29** | base 0,950, lecteur à 50 % sur l'arthrose latérale (au lieu de 8 %) | le lecteur aide-t-il là ? |
| knee-blend | **30** | comme 29, plus lecteur à 50 % sur la contusion et 40 % sur le LCA | et sur ces deux-là ? |
| knee-blend | **32** | comme 29, plus le 2e modèle public de nartaa (224 px) | ce 2e modèle apporte-t-il ? |
| knee-blend | **31** | comme 29, plus nos modèles | nos modèles aident-ils ici ? |
| knee-blend | **17** | base 0,950 + le 2e modèle de nartaa, sans autre changement | idem, sur la base seule |

Versions 29 à 32 en cours d'envoi et de vérification : je confirme ici quand c'est fait.
