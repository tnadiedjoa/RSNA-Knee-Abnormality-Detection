# À soumettre (mis à jour le 09/10 à 16 h 30)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31**, ni knee-blend **18, 19, 20 ni 21** (cassées).

Scores du 09/10 : knee-blend 9, 11, 14 et 16 → **0,950** (= la base publique seule) · 15 → 0,949. Notre meilleur score au
classement passe de 0,944 à 0,950 (328e sur 5 576).

## Le 10/10 (5 soumissions, à partir de 2 h du matin) : base publique de heliosli (annoncée à 0,951)

Le notebook de heliosli mélange le modèle public à 0,949 (70 %) avec son propre modèle ConvNeXt (30 %). Sur les 58 examens
dont on connaît les vrais labels, nos modèles valent mieux que le sien, et les deux ensemble font mieux que chacun seul :
d'où les versions 26 à 28, qui partagent ses 30 % entre son modèle et les nôtres.

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-blend | **26** | modèle à 0,949 (70 %) + modèle de heliosli (15 %) + nos modèles (15 %) | dépasse-t-il 0,951 ? |
| knee-blend | **28** | modèle à 0,949 (70 %) + heliosli, nos modèles et le lecteur public (10 % chacun) | le lecteur en plus aide-t-il ? |
| knee-blend | **23** | heliosli + le lecteur public, sans nos modèles | le lecteur seul fait-il aussi bien ? |
| knee-blend | **27** | comme le 26, en 60 / 20 / 20 | faut-il plus de poids hors du modèle à 0,949 ? |
| knee-blend | **22** | le notebook de heliosli seul | le point de départ (0,951 ?) |

En réserve : 24, 25 (heliosli + lecteur à d'autres poids) et 17. Les tests de modèles seuls (knee-submit) attendent.
