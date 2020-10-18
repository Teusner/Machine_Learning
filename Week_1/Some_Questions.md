# Définitions :

**Approche supervisées :** Approche d'apprentissage à l'aide d'un jeu de données étiquetées. Cela permet de laisser la machine déterminer les caractéristiques de chaque classe lors de la phase d'apprentissage, et ensuite de classer par exemple des nouvelles données.

**Approche non-supervisées :** Approche d'apprentissage dans laquelle la machine doit elle même proposer des classes en fonction des caractéristiques qui se dégagent d'un jeu de données.

**Régression :** Domaine du *Machine Learning* permettant de déterminer la tendance d'un jeu de données, afin de pouvoir estimer une caractéristique d'une nouvelle donnée, en espérant qu'elle suive globalement la même loi que les données d'entrainement.

**Classification :** Domaine du *Machine Learning* permettant de ranger des données dans des classes en fonction de leurs caractéristiques.

# Processus d'apprentissage et de prédiction

**Apprentissage :**
Données d'entraînements -> Modèle statistique -> Analyse d'erreurs

**Prédictions :**
Nouvelles données -> Modèle statistique entraîné -> Prédiction

# Fonctionnement de l'apprentissage

Dans l'approche par apprentissage la machine va analyser toutes les données d'entrée, afin d'entraîner le modèle, c'est à dire de pouvoir soit dégager une tendance dans les données, ou bien classer les données en fonction de leur caractéristiques. Elle stocke ensuite les données apprises sous forme de paramètres / variable numérique qui évolue au cours de l'apprentissage. Le but est de faire converger les valeurs apprises afin d'avoir une régression ou une classification la plus optimale. Pour ce faire, on définie une fonction de coût qui permet de déterminer l'erreur du modèle avec les paramètres actuels.

# Normalisation des descripteurs :

La normalisation des descripteurs permet d'avoir un jeu de données possédant la même dynamique que le jeu de données original, mais permet par exemple de fusionner des données de différentes sources, ne possédant pas spécialement les mêmes caractristiques mathématiques. Il y a aussi d'autres avantages comme par exemple d'avoir une moyenne nulle, mais aussi un écart-type et une variance de 1.