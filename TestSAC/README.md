# P_androide/TestSAC
Test des outils en utilisant l'algorithme SAC

**Code source utilitaire :**
* slowBar.py contient la class SlowBar permettant d'afficher le temps restant dans une boucle.
* colorTest.py permet de dessiner un gradient de couleur, sauvegardé en color_palette.png. Cette palette de couleur est utilisée par GradientStudy.
* vector_util.py contient des fonctions utilitaires d'opérations sur des couleurs RGB et de tirage de directions.
* trainModel.py allows to train a model and saves its checkpoints in Model

<h3>**Vignette**</h3>

**Input de Vignette :**
* Models doit contenir un fichier .zip d'un modèle entrainé (pour la K-ième sauvegarde : modelNameK.zip)
* See https://drive.google.com/file/d/183-1Im429j5b1UadVArY5YV-shol7lCj/view?usp=drive_web to download a SavedVignette example, trained on Pendulum-v0 (should be put in TestSAC/SavedVignette

**Code source de Vignette :**
* savedVignette.py contient la class SavedVignette contenant une Vignette et des outils de visualisation
* Vignette.py calcule les Vignettes d'un ensemble de fichiers

**Output de Vignette :**
* SavedVignette/ contient la Vignette serialisée
* Vignette_output/ contient les Vignettes en 2D et 3D

<h3>**GradientStudy**</h3>

**Input de GradientStudy :**
* Models doit contenir un fichier .zip d'un modèle entrainé (pour la K-ième sauvegarde : modelNameK.zip)

**Code source de GradientStudy :**
* savedGradient.py contient la class SavedGradient contenant un Gradient et des outils de visualisation
* GradientStudy.py calcule l'étude de gradient d'un modèle

**Output de GradientStudy :**
* Gradient_output/ contient les images de l'étude de gradient
