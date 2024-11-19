# Introdution
Ce projet s’articule autour de l’acquisition et du traitement d’images capturées par l’appareil photo de mon téléphone. La démarche comprend plusieurs étapes cruciales pour l’analyse d’images, notamment l’extraction des canaux RVB et des histogrammes, la transformation en niveaux de gris, la binarisation, ainsi que la détection de contours à l’aide de filtres spatiaux et morphologiques. L’objectif ultime est d’obtenir des mesures précises de chaque élément présent dans l’image, établissant ainsi une approche complète pour l’analyse visuelle et la caractérisation des contenus photographiques.
 
 => Vous trouverez toute la démarche et les résultats dans le rapport (fichier PDF)\n
 
# Plan du rapport : 
 Introduction 3
 1 appareilphotoetscènecapturée 4
 1.1 Appareilphoto:RedmiNote9S . . . . . . . . . . . . . . . . . . . . . . . . . . 4
 1.2 Scènecapturée: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
 2 extractiondescanauxRGBetseshistogrammes 6
 2.1 DifférentscanauxRBG: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
 2.2 HistogrammesdescanauxRGB: . . . . . . . . . . . . . . . . . . . . . . . . . . 7
 2.3 Interprétation: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
 3 Transformationenniveauxdegrisetbinarisation 8
 3.1 Transformationnenniveauxdegris . . . . . . . . . . . . . . . . . . . . . . . . . 8
 3.2 Binarisation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
 4 détectiondecontour 10
 4.1 Filtragespatial . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
 4.2 Filtragemorphologique . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
 4.3 Interprétation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
 5 Étiquetageetcalculdelatailledesobjets 14
 5.1 Déterminationdelarésolutionspatiale . . . . . . . . . . . . . . . . . . . . . . . 14
 5.2 Déterminationdestaillesdesobjets . . . . . . . . . . . . . . . . . . . . . . . . . 15
 6 Conclusion
