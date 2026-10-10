# À soumettre (mis à jour le 11/10 à 0 h 40)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31**, ni knee-blend **18 à 28** (18–21 cassées ; 22–28 utilisent le modèle de
heliosli, retiré par son auteur le 09/10), ni knee-blend **38** (restée bloquée sur Kaggle, sans fichier de sortie).

Scores du 10/10 : knee-blend 29, 30, 31, 32 et 33 → **0,950** (= la base publique seule). Changer les poids des modèles
publics ou ajouter le 2e modèle de nartaa ne bouge rien ; le « 0,954 » public venait de modèles privés de son auteur.

## Le 11/10 (5 soumissions, à partir de 2 h du matin) : la base à 0,950 + nos modèles « zoomés »

Le seul modèle qui a fait monter le score public (celui de heliosli, +0,002) était du même type que les nôtres mais regardait
des zooms centrés sur le genou. On teste la même idée : nos modèles entraînés sur les 13 cm ou les 10 cm centraux de l'image
au lieu de 15 cm.

| Ordre | Notebook | Version | Contenu | Question |
|---|---|---|---|---|
| 1 | knee-blend | **35** | base 0,950 + nos modèles à 13 cm (E91, E92, E93), mêlés au lecteur | le zoom 13 cm aide-t-il ? |
| 2 | knee-blend | **37** | base 0,950 + nos modèles à 10 cm (E96, E97), mêlés au lecteur | et le zoom 10 cm ? |
| 3 | knee-blend | **40** | base 0,950 + les deux cadrages (5 modèles), mêlés au lecteur | les deux ensemble ? |
| 4 | knee-blend | **36** | base 0,950 à 80 % + nos modèles à 13 cm à 20 % | avec un poids fixe ? |
| 5 | knee-blend | **39** | base 0,950 à 80 % + les deux cadrages à 20 % | idem, les deux ensemble |

Les 5 sont vérifiées sur les 3 examens de test visibles (35–37 le 10/10 au soir, 39 et 40 le 11/10 à 0 h 30).
