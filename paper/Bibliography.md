General
-----------

- Vision UCL talks : [https://www.ucl.ac.uk/ioo/research/research-labs-and-groups/child-vision-lab/event/visionucl-talks]
- Danilo Rezende: [https://www.youtube.com/watch?v=oRJ1hJTRs-0]

Active vision
-------------

-   Karl Friston et al. **Perceptions as hypotheses: Saccades as
    experiments**. In: Frontiers in Psychology 3.MAY (2012), pp. 1--20.
    ISSN : 16641078. DOI : 10.3389/fpsyg.2012.00151 . arXiv: NIHMS150003
    [https://www.frontiersin.org/articles/10.3389/fpsyg.2012.00151/full]

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
    <https://arxiv.org/pdf/1411.1045.pdf>
    
-   A Unified Theory of Early Visual Representations from Retina to Cortex 
    through Anatomically Constrained Deep CNNs <https://openreview.net/forum?id=S1xq3oR5tQ>

Fovea models
------------

-   Philip Kortum and Wilson S. Geisler. **Implementation of a foveated
    image coding system for image bandwidth reduction**. In: SPIE
    Proceedings 2657 (1996)
    
-   En lisant la page 476 de : <https://www.asc.ohio-state.edu/golubitsky.4/reprintweb-0.5/output/papers/6120261.pdf>, 
    il y a des nombres intéressants à utiliser pour notre transformation log-polaire...

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



What/Where 
-----------------

- Denil, M., Bazzani, L., Larochelle, H., & de Freitas, N. (2012). Learning where to attend with deep architectures for image tracking. Neural computation, 24(8), 2151-2184.

-    Benjamin de Haas, Justus-Liebig-Universität Gießen : 'Where' in the ventral stream? Pointers from feature-location tuning and individual salience

- Kruger, N., Janssen, P., Kalkan, S., Lappe, M., Leonardis, A., Piater, J., ... & Wiskott, L. (2012). Deep hierarchies in the primate visual cortex: What can we learn for computer vision?. IEEE transactions on pattern analysis and machine intelligence, 35(8), 1847-1871.




Saliency maps / Attention network
-----------------

LEARNING  WHAT AND WHERE TO ATTEND
Drew Linsley, Dan Shiebler, Sven Eberhardt and Thomas Serre (2019) ICLR

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

-   Explaining the Human Visual Brain Challenge: <http://algonauts.csail.mit.edu/>
-   <https://etra.acm.org/2019/challenge.html>


ICANN reports
--------

Dear Authors,

We have received comments of reviewers and, unfortunately, your article has not been recommended for the acceptance.

Please, find enclosed reviewer’s comments, which we hope will be useful for you to improve your work.

We hope that this rejection will not influence your support of ICANN conferences and we wish you a success in your scientific work.

Thank you for considering ICANN2019 for publication of your article.

Best regards,
Igor Tetko on behalf of ICANN2019 organizers


Reports:

---------------

ARGUMENTATION:

The authors propose a framework to address the problem of visual search via active inference. They evaluate the approach on MNIST. In their experiments, the authors show the advantages of explicitly integrating a foveal mechanism into their system.

Generally, the intro and related work section give a good motivation and overview of other methods. However, a clear separation into an introduction section and related work section would be desirable. Also, the whole field of attentional mechanisms for neural networks is ignored [Denil et al. "Learning where to Attend with Deep Architectures for Image Tracking", Larochelle and Hinton "Learning to combine foveal glimpses with a third-order Boltzmann machine"], even though this field is highly relevant for the proposed approach.

The Principles section gives a good formal problem formulation. The authors should refrain from general and unfounded statements like "[...] even in simple cases such as vision. [...]". The Implementation section gives many details, which should help to reproduce results. However, the system design is arbitrary in many points, such as choosing LeNet for the "What" pathway. This specific neural network design is over 20 years old and the authors do not motivate why it might be beneficial to use this exact design. It is noteworthy that the authors provide their code, unfortunately, the provided link was not reachable at the time of this review.

The experiments section is missing any comparisons to other comparable approaches as baselines. This makes it hard to see how well the proposed approach is actually performing. The experiments are done using digits from MNIST, which is a much simpler case than in the introduction described rededection of a familiar face in a cluttered environment. Of course, the experiments can be seen as a proof of concept, but as stated by the authors, the decoupling into two information streams relies on the naive Bayes assumption. This assumption does not hold true in a more complicated scenario, therefore it is questionable if the approach can be extended to more realistic scenarios.

The approach and motivation of this work are interesting, but considering all points listed previously, I recommend to reject the paper.


---------------

---------------

ARGUMENTATION:

The authors present a foveated vision architecture where one neural network determines where to look in a picture and another one determines what is seen in the picture.

The paper is well written, but it is quite hard to understand what the actual research questions are. The introduction is quite lengthy and does not clearly state in a simple single sentence what the problem is that the authors want to solve, what their hypothesis is, and how they want to address the hypothesis. I am not an expert in biomimetic vision, but I know that there are at least a handful of other approaches that provide a foveal vision approach. The authors should compare their work with at least the 3-4 closest approaches and say what their delta is. Since the authors fail to do this, and since the authors fail to clearly state their research goals and questions, the scientific contribution of this paper is not visible to me.

I do appreciate the computational machinery, i.e., the simulated foveal views, that the authors provide, and I think that the topic is very important. Therefore, I suggest that the authors re-submit the paper to a similar venue with a more extensive evaluation and a better motivation wrt. the state of the art. However, in this form, I do not recommend to accept the paper.


---------------

---------------

ARGUMENTATION:

The paper presents a method for active visual search. The problem is an old one in computer vision and the authors do not seem to be completely aware of all the previous results. The theoretical aspects and discussed in details and some experimental results are proposed. Comparisons with existing methods should be provided.

