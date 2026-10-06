# À soumettre (mis à jour le 06/10 à 21:50)

Sur Kaggle : Submit → choisir le notebook → **choisir la version dans le menu** (Kaggle propose la dernière par défaut).
Ne jamais soumettre knee-submit **30 ni 31** (cassées).

Aujourd'hui (avant 2h du matin) s'il reste une soumission :

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-submit | **29** | E58 : recette d'E54 (100 % des données) + dosage des labels par maladie | bat-il E54 (0,935) ? |

Demain (5 soumissions), dans cet ordre :

| Notebook | Version | Contenu | Question |
|---|---|---|---|
| knee-submit | **34** | ensemble 100 % : E56 + E59 (meilleure recette 384 px, 6 réseaux) + E60 (ConvNeXt-small 384 px) + E54 | notre meilleur score à nous ? (record 0,936) |
| knee-blend | **7** | public 85 % + l'ensemble de la Version 34 15 % | bat-il 0,944 ? |
| knee-submit | **28** | E56 seul (meilleure recette 384 px, 100 %) | 384 px vs 224 px (E54 0,935) |
| knee-submit | **29** | (si pas soumise aujourd'hui) E58 : dosage des labels par maladie | bat-il E54 (0,935) ? |
| knee-submit | **32** | E61 : poids ×10 pour les 58 études expertes | bat-il E54 (0,935) ? |

Ensuite : knee-submit **33** (E62 : type de séquence IRM indiqué au modèle).

Les knee-submit sont notées en ~1 h (les deux GPU servent maintenant), les knee-blend en ~2 h.
