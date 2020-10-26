# Approche Multiclasse

**Multiclasse :** 
On utilise une approche multiclasse car ici le but n'est pas de séparer un jeu de données en deux classes, mais bien en 10 classes distinctes ({0, 1, ..., 8, 9}).

# Modélisation retenue et paramètres appris

**Modélisation retenue :** La modélisation du réseau de neurones est en 3 layers (ou couches) : un layer d'entrée, un layer caché et un layer de sortie. Pour chaque neurone, nous avons fait le choix d'utiliser une fonction d'activation en sigmoide.

**Paramètres appris :** Le réseau de neurones "apprend" les coefficients des matrices theta associés à chaque layer du réseau.

# Fonction de coût des réseau de neurones

**Réseau de neurones :** On reprend la fonction de coût vue précedemment, mais on ajoute une somme permettant de prendre en compte le coût pour chaque classe ({0, 1, ..., 8, 9}). Cela permet de donner un coût global à tout le réseau de neurones.
