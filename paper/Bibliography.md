
Active vision
-------------

-   Karl Friston et al. **Perceptions as hypotheses: Saccades as
    experiments**. In: Frontiers in Psychology 3.MAY (2012), pp. 1--20.
    ISSN : 16641078. DOI : 10.3389/fpsyg.2012.00151 . arXiv: NIHMS150003

-   **Klaus Obermayer** : The successful candidate will explore the
    hypothesis that object-level attentional units are essential
    mid-level factors which guide human eye-movements in visual scene
    analysis. Based on eye-fixation data from visual search tasks she/he
    will first build computational models to emulate the measured
    fixation sequences, to quantify the influence of different low- and
    high-level visual features, and to characterize the influence of
    task-driven changes in object-based attention processes. In a second
    step, plausible models will be integrated as "attentional modules"
    into a computer vision system for visual scene analysis and will be
    evaluated in terms of task success and the number of computations
    involved. Potential achievement of the project is an efficient
    real-time analysis of dynamic visual scenes.

-   **Constantin Rothkopf** at the Technical University Darmstadt. The
    position is initially funded by the German Research Foundation DFG
    for three years. The project \'Active vision: control of
    eye-movements and probabilistic planning\' aims to investigate how
    humans carry out eye-movements across different tasks and how this
    can be understood with models of probabilistic planning. Recent
    publications of the lab on this topic include the following papers:
    -   Hoppe, D., & Rothkopf, C. A. (2016). Learning rational temporal
        eye movement strategies. Proceedings of the National Academy of
        Sciences, 113(29), 8332-8337.
    -   Hoppe, D., & Rothkopf, C. A. (2019). Multi-step planning of eye
        movements in visual search. Scientific reports, 9(1), 144.

Visual search
-------------

- http://www.scholarpedia.org/article/Visual_search

-     Treisman, A., & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology, 12, 97-136. doi:10.1016/0010-0285(80)90005-5.  https://www.sciencedirect.com/science/article/pii/0010028580900055

- https://en.wikipedia.org/wiki/Visual_search



Saliency map models
-------------------

-   Laurent Itti and Christof Koch. **A saliency-based search mechanism
    for overt and covert shifts of visual attention**. In: Vision
    Research 40.10-12 (2000), pp. 1489--1506.

Control models
--------------

-   Najemnik / Where is Waldo? / accuracy model
    -   J Najemnik and Wilson S. Geisler. **Optimal eye movement
        strategies in visual search**. In: Nature reviews. Neuroscience
        434 (2005)
-   Infomax model :
    -   Nicholas J Butko and Javier R Movellan. **Infomax control of eye
        movements**. In: Autonomous Mental Development, IEEE
        Transactions on 2.2 (2010)

Deep / transformer networks {#deep--transformer-networks}
---------------------------

ID deep learning = feed forward alors que la vision est active :

-   vision = multi-steps process
-   approche basée sur le contrôle

**Spatial transformer tutorial** :
<https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html>

par contre c\'est une transfo affine, alors que nous, on pourraitait
faire une transfo log-polaire\...

-   Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
    **Spatial Transformer Networks**. <https://arxiv.org/abs/1506.02025>

-   M. Kümmerer, L. Theis, and M. Bethge **Deep Gaze I: Boosting
    Saliency Prediction with Feature Maps Trained on ImageNet** ICLR
    Workshop, 2015
    -   <https://arxiv.org/pdf/1411.1045.pdf>

Fovea models
------------

-   Philip Kortum and Wilson S. Geisler. **Implementation of a foveated
    image coding system for image bandwidth reduction**. In: SPIE
    Proceedings 2657 (1996)

V1 models
---------

-   Nicholas J. Priebe. **Mechanisms Of Orientation Selectivity In
    Primary Visual Cortex** Annual Review of Vision Science. 2016, 2(1)
    .

-   A. S. Ecker, F. H. Sinz, E. Froudarakis, P. G. Fahey, S. A.
    Cadena, E. Y. Walker, E. Cobos, J. Reimer, et al. **A
    rotation-equivariant convolutional neural network model of primary
    visual cortex** International Conference on Learning Representations
    (ICLR), 2019.
    -   <https://openreview.net/pdf?id=H1fU8iAqKX>

Crowding
--------

**Hypothèse:** un système pré-attentif permet un gain de performance en
classification par rapport au système par défaut et qu\'en cas de
crowding, ce gain est perdu.

-   Ziskind, A.J., Hénaff, O., LeCun, Y., & Pelli, D.G. (2014) **The
    bottleneck in human letter recognition: A computational model**.
    Vision Sciences Society, St. Pete Beach, Florida, May 16-21, 2014,
    56.583. <http://f1000.com/posters/browse/summary/1095738>

-   D. Pelli (2018) **Despite a 100-fold drop in cortical magnification,
    a fixed-size letter is recognized equally well at eccentricities of
    0 to 20 deg. How can this be?.** Journal of Vision 2018;18(10):26.
    <https://jov.arvojournals.org/article.aspx?articleid=2699020&resultClick=1>

-   J. Zhou, N. Benson, J. Winawer, D. Pelli (2018) **Conservation of
    crowding distance in human V4**. Journal of Vision2018;18(10):856.
    <https://jov.arvojournals.org/article.aspx?articleid=2699845&resultClick=1>

Attention network
-----------------


Saliency maps
-------------

Benchmarking
------------

-   M. Kümmerer, T. Wallis, and M. Bethge. **Information-theoretic model
    comparison unifies saliency metrics** Proceedings of the National
    Academy of Science, 112(52), 16054-16059, 2015
    -   <http://www.pnas.org/content/112/52/16054.abstract>

-   **Pysaliency** : M. Kümmerer, T. S. A. Wallis, and M. Bethge
    **Saliency Benchmarking Made Easy: Separating Models, Maps and
    Metrics** The European Conference on Computer Vision (ECCV), 2018.
    -   <https://github.com/matthias-k/pysaliency>
    -   <http://bethgelab.org/media/publications/1704.08615.pdf>

Datasets
--------

-   <https://etra.acm.org/2019/challenge.html>
