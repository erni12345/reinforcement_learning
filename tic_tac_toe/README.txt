Alpha Toe de Ernesto
Jeu de Morpion contre une Intelligence Artificielle




**********************************************************************************************************

 - Presentation du projet

    Le jeu de Morpion est un jeu simple qui a 5477 differentes possibilites
    Meme si il est imopssible de trouver une strategie qui gagne a chaque fois il est possible d'egaliser ou de gagner a chaque fois.
    Il serait alors interesant de voir si un model de Reinforcement Learning arrive a trouver ces strategies tous seul.

    Alors l'objectif du projet serait de creer une Intelligence Artificielle qui trouve des nouvelles strategies ou bien des nouvelles!


- Description de l’algorithme choisi et fonctions associées

    Tout d'abord l'algorithme choisi est celui de PPO (Proximal Policy Optimization). Celui-ci a pour but de maximiser le "reward" qui est predit a partir
    d'un certain etat et action. Il apprend donc a maximiser la prediction du "reward" et donc a mieu faire la tache.

    Lien pour voir plus sur PPO : https://spinningup.openai.com/en/latest/algorithms/ppo.html

    Le systeme et l'environement est donc simple. L'IA interagit avec l'environement et selon ce qu'elle fait elle recoit un "reward" qui indique si elle
    fait quelle que chose de bien ou pas. Dans se cas le reward est dans [-1;1]. Ensuite elle recois le nouvelle etat de l'environement et tente de nouveau,
    ainsi de suite.

    On a donc un fonction step() qui permet au mouvement de se produire, elle se deroule ainsi:
            1. Verifcation du mouvement
            2. Calcul du "reward" pour le mouvement fait
            3. Faire le mouvement + verification si victoire
            4. Calcul du mouvement de l'adverssaire
            5. Verification de victoire

    Ensuite la fonction reward() regarde l'etat du tableau et determine si il a gagne, perdu ou rien encore:
            si perd, reward = -1
            si gagne, reward = +1
            si rien, reward = 0

        Afin de rendre la tache plus rapide (tentative), j'ai aussi ajouter le fait qu le reward = prob de gagner [0;1] si il a joue le meilleur cout possible


    La fonction qui calcul le meilleur mouvemnt, calcul la probabilite de gagner si on joue dans chaque case vide recursivement.

            On rend la somme des probabilites de victoire des sous couts apres le couts divise par le nombre de couts possible

            De plus, j'ai ajoute de la programation dynamique afin d'eviter de recalculer certain cas a plusieurs reprise qui a rendu la Verification
            +- 100 fois plus rapide




- Quelles ont été les principales difficultés rencontrées ?

    Tout d'abord il fallait que j'apprenne comment tous fonctionne et comment creer mon propre environement. Donc cela a pris beaucoup de temps.
    Mais le plus grand probleme est certainement les maximums locaux. En effet comme l'algo cherche a trouver un maximum il peut facilement rester bloque 
    dans un maximum local et donc plus change de strategie meme si celle-ci n'est pas la bonne. Par example, au debut je metais des "reward" tres grands
    comme 10000 ou bien -10000 car je pensais que cela eviterai completement ces cas. Neanmoins cela a produit des erreurs car l'IA trouve un max local qui
    lui permet d'avoir entre -100 et 0 points qui est beaucoup plus que -10000 donc il considere que c'est la meilleur strategie. 

    De plus une fois apers avoir entraine mon model pendant + de 10h je l'ai sans faire expres effacer ce qui m'a fait perdre beaucoup de temps...

    Finnalement j'ai passe aussi beaucoup de temps a reflechir a comment je pouvais calculer la probabilite de gagner si on joue un mouvement car je n'ai pas
    voulu utiliser des ressources exterieurs.

    


- Quelles notions avez-vous apprises en faisant ce projet ?

    J'ai apris beaucoup sur le Reinforcement Learning, comment sa marche, quand sa sert, quand sa sert pas, comment eviter des bugs, pourquoi ils arrivent
    comment fonctionne les maths derieres (pas encore completement)




- Quelles ressources avez-vous utilisées (liste des sites, manuels… )


    J'ai utilise la documentaion stable_baselines3 ainsi que celle de OpenAi Gym et Pygame. De plus j'ai utilise de nombreux articles et papiers de recherche
    afin de comprendre et appredre.


**********************************************************************************************************



Consignes de lancement
**********************************************************************************************************

- Pour lancer:
    1) lancer module.bat afin d'installer les modules automatiquement
         --> Modules requis : pygame, stablebaselines3, numpy, gym

    2) Lancer main.py afin de tester le Jeu

*Bien verifie que C++ est installer sur ordinateur avec Visual Studio* (En cas d'erreur avec Pytorch)


**********************************************************************************************************



Consignes d'utilisation
**********************************************************************************************************

2 modes --> Normale et IMPOSSIBLE

Normale = Modele entrainer sur environ 3Million de tentative
imopssible = Model qui joue meilleur mouvement a chaque fois

Pour menu utiliser fleches sur clavier et entre
Pour jeu utiliser souri

**********************************************************************************************************
