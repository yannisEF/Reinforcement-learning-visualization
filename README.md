# P_androide
Visualisation du paysage de valeur pour mieux comprendre l’apprentissage par renforcement.

Dernière version de l'outil Vignette disponible dans le dossier TestSAC.

Pour lancer rapidement Vignette (40 min d'éxecution sur nos machines) utiliser la commande suivante :

`python3 Vignette.py --directory Ex_Sauvegarde/Saves --basename save --min_iter 1 --max_iter 2 --eval_maxiter 1 --plot3D True --show3D True`

Pour une exécution encore plus rapide (moins de lignes) :

`python3 Vignette.py --directory Ex_Sauvegarde/Saves --basename save --min_iter 1 --max_iter 2 --eval_maxiter 1 --plot3D True --show3D True --nb_lines 10`

* "--directory", "--min/max_iter" permettent de charger les modèles dans Ex_Sauvegarde/Saves de "save1" à "save2".
* "--eval_maxiter 1" permet une éxecution rapide en évaluant chaque acteur sur seulement 1 épisode.
* "--plot3D True" affiche les plots un par un, cela suspend l'exécution de Vignette d'un fichier à l'autre
* "--show3D True" affiche tous les plots en 3D une fois que tous les fichiers ont été parcouru.
* "--nb_lines" change la taille de la Vignette (60 directions par défaut)

Pour l'instant, seulement quelques options graphiques sont à disposition :
* "--min/max_colormap" change la couleur des Vignettes
* "--x_diff" change l'écart entre les points de chaque ligne (plot3D)
* "--y_diff" change l'écart entre chaque ligne (plot3D)

Le dossier "Old" contient ce qui a été fait l'année dernière.
