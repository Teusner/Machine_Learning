# Découpe du jeu de donnée

**Beaucoup de données :** On découpe le jeu en 70% pour l'apprentissage et 30% pour l'évaluation.

**Pas beacoup de données :** On scinde le jeu de données en k plis. On reserve le i-ième pli pour l'évaluation et on entraine le modèle avec les autres plis, pou i allant de 1 à k.

**Dans tous les cas :** On n'évalue pas le modèle avec des données qui ont servoes à l'entrainement, sinon on l'impression d'avoir un système trop performant.

# Capacité de prediction d'un modèle

Capacité d'un modèle à donner une sortie cohérente à partir de données qu'il n'a pas vues pendant la phase d'entrainement. 

# Erreur de généralisation

Plus on a de données d'entrainements, meilleure est la généralisation. C'est dû au fait que l'on voit plus de données différentes en entrée, ce qui permet d'avoir un modèle qui a plus d'expérience. L'erreur de généralisation est l'erreur commise sur des nouvelles données jamais vues auparavent.

# Ensemble de validation

Après entrainement, on peut valider le modèle avec des données d'entrainement qui n'ont pas servies à entrainer le modèle. Cela permet de valider le comportement de celui-ci sur des données jamais vues, ce qui permet de vérifier le fait qu'il ne donne pas des sorties erronées.

# Learning curve

La learning curve permet de gérer le ration biai-variance d'un modèle. C'est un indicateur qui permet de déterminer si on est en sous-apprentissage, sur-apprentissage, ou bien trouver un optimum de l'entrainement du modèle. On peut regarder l'erreur en entrainement et en evaluation sur des nouvelles données en fonction du nombre de fois que l'on entraine le modèle. PLe but est d'avoir un réseau de neurones qui ne colle pas trop au jeu de données d'entrainement (sur-apprentissage) mais qui ne généralise pas trop les données du jeu d'entrainement (sous-apprentissage).
