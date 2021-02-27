# P_androide/TestSAC
Test des outils en utilisant l'algorithme SAC

<ins>**Code source utilitaire :**</ins>
* slowBar.py contient la class SlowBar permettant d'afficher le temps restant dans une boucle.
* colorTest.py permet de dessiner un gradient de couleur, sauvegardé en color_palette.png. Cette palette de couleur est utilisée par GradientStudy.
* vector_util.py contient des fonctions utilitaires d'opérations sur des couleurs RGB et de tirage de directions.
* *Test_BS3.py est un brouillon*

<ins><h3>**Vignette**</h3></ins>

<ins>**Input de Vignette :**</ins>
* Models doit contenir un fichier .zip d'un modèle entrainé (pour la K-ième sauvegarde : modelNameK.zip)

<ins>**Code source de Vignette :**</ins>
* savedVignette.py contient la class SavedVignette contenant une Vignette et des outils de visualisation
* Vignette.py calcule les Vignettes d'un ensemble de fichiers

<ins>**Output de Vignette :**</ins>
* SavedVignette/ contient la Vignette serialisée
* Vignette_output/ contient les Vignettes en 2D et 3D

<ins><h3>**GradientStudy**</h3></ins>

<ins>**Input de GradientStudy :**</ins>
* Models doit contenir un fichier .zip d'un modèle entrainé (pour la K-ième sauvegarde : modelNameK.zip)

<ins>**Code source de GradientStudy :**</ins>
* GradientStudy.py calcule l'étude de gradient d'un modèle

<ins>**Output de GradientStudy :**</ins>
* Gradient_output/ contient les images de l'étude de gradient
