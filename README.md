# P_androide
Visualisation du paysage de valeur pour mieux comprendre l’apprentissage par renforcement.

Dernière version de l'outil Vignette disponible dans le dossier TestSAC.

Pour lancer rapidement Vignette (40 min d'éxecution sur nos machines) utiliser la commande suivante :

python3 Vignette.py --directory Ex_Sauvegarde/Saves --basename save --min_iter 1 --max_iter 10 --eval_maxiter 1 --plot3D True --show3D True

Celle-ci applique Vignette centrées en Ex_Sauvegarde/Saves de "save1" à "save10". L'argument "--eval_maxiter 1" permet une éxecution rapide en évaluant chaque acteur sur seulement 1 épisode. "--plot3D True" affiche le plot en 3D de chaque Vignette au fûr et à mesure "--show3D True" affiche tous les plots en 3D une fois que tous les fichiers ont été parcouru.

