Cerebellar Atlases
==================

The functional atlases that came with the SUIT toolbox are now maintained seperately in the `cerebellar atlas repository`_.

.. _a link: https://github.com/DiedrichsenLab/cerebellar_atlases/

You can download the repository into a folder of your choice, or use the SUITPy function :func:`suit.fetch_atlas` to download individual atlases on demand. For example, to download the Diedrichsen_2009 probabilistic atlas, you can run:

.. code::

   suit.fetch_atlas('Diedrichsen_2009')

For this command you can specify a atlas directory, specify the specific maps you want, and define the atlas space that you would like to use. For example, to download the symmetric, 32 parcel, Nettekoven_2024 atlas in SUIT space, you can run:

.. code::

   suit.fetch_atlas('Nettekoven_2024', maps='NettekovenSym32',space='SUIT')

Template spaces
---------------

We are providing the atlas and data maps in three template spaces. All three templates are provided in the `tpl-` directory in a cerebellar-only version.

* MNI152NLin6AsymC: The non-linear asymmetric MNI template used for example in FSL (short MNI)
* MNI152NLin2009cSymC: The 2000c symmetric MNI template (short MNISym)
* SUIT: The original cerebellar-only template (Diedrichsen, 2005)

For every template space, we provide the following files:

* .._T1w: T1-weighted template image
* .._desc-pcereb.nii: Probabilistic mask
* .._desc-cereb_mask.nii: hard mask
* .._xfm.nii: Transform files between different atlas spaces
* ..label-GMc_probseg.nii: Gray matter probability
* ..label-WMc_probseg.nii: White matter probability

Atlases
-------
For every maps, we provide some the following files:

* ..._space-MNI.nii: volume file aligned to MNI152NLin6AsymC
* ..._space-MNISym.nii: volume file aligned to MNI152NLin2009cSymC
* ..._space-SUIT.nii: volume file aligned to SUIT space
* ...tsv: Color and label lookup table for parcellation
* ...lut: Color and label lookup table for FSLeyes
* ....gii: Data projected to surface-based representation of the cerebellum (Diedrichsen & Zotow, 2015).

The atlases are organized by the first author / year of the main paper

The maps can also be viewed online using our [cerebellar atlas viewer](https://www.diedrichsenlab.org/imaging/AtlasViewer).

Diedrichsen_2009: Probabilistic atlas for cerebellar lobules and nuclei
-----------------------------------------------------------------------

.. image:: images/Diedrichsen_2009.png
   :width: 300px
   :align: left

The anatomical definitions are based on the fMRI atlas of an individual cerebellum by Schmahmann et al. (2000). We manually identified the main lobules on MRI scans of 20 healthy young participants (ROIs 1-28). Using a different set of 23 participants, we also identified the deep cerebellar nuclei (ROIs 29-34).

* atl-Anatom:    Number of most probable compartment, Lobules and Nuclei

References and Links:
* Diedrichsen, J., Balsters, J. H., Flavell, J., Cussans, E., & Ramnani, N. (2009). A probabilistic atlas of the human cerebellum. Neuroimage.
* Diedrichsen, J., Maderwald, S., Kuper, M., Thurling, M., Rabe, K., Gizewski, E. R., et al. (2011). Imaging the deep cerebellar nuclei: A probabilistic atlas and normalization procedure. Neuroimage.
* http://www.diedrichsenlab.org/imaging/propatlas.htm


Buckner_2011: Resting state network parcellation
---------------------------------------------------------

.. image:: images/Buckner_2011.png
   :width: 300px
   :align: left


Buckner et al. (2011) presented the first comprehensive functional atlas of the human cerebellum, based on the correlation of each cerebellar voxel and with the 7 or 17 cortical resting state networks, described in Yeo et al. Parcellation is based on the data from 1000 subjects.

* atl-Buckner7:    Assignment of cerebellar voxels to the 7 network parcellation
* atl-Buckner17:    Assignment of cerebellar voxels to the 17 network parcellation

References and Links:
* Buckner, R. L., Krienen, F. M., Castellanos, A., Diaz, J. C. & Yeo, B. T. (2011). The organization of the human cerebellum estimated by intrinsic functional connectivity. J Neurophysiol 106, 2322–2345.


Xue_2021: Individual resting state parcellation
--------------------------------------------------------

.. image:: images/Xue_2021.png
   :width: 300px
   :align: left

Xue et al. (2021) provided two individual parcellations based on resting state data from 31 sessions for each. 10 Cortical networks were estimated using a hierarchical Bayesian model (Kong et al. 2019) and the cerebellum labeled based on the highest correlation with these networks.

* atl-Xue10Sub1:    Individual parcellation for subject 1
* atl-Xue10Sub2:    Individual parcellation for subject 2

References and Links:
* Xue, A., Kong, R., Yang, Q., Eldaief, M. C., Angeli, P. A., Dinicola, L. M., … Yeo, B. T. T. (2021). The detailed organization of the human cerebellum estimated by intrinsic functional connectivity within the individual. https://doi.org/10.1152/jn.00561.2020


Ji_2019: Subcortical resting state parcellation
-----------------------------------------------

.. image:: images/Ji_2019.png
   :width: 300px
   :align: left

Ji et al. (2019) presented a parcellation of subcortical structures based on correlation with 10 cortical networks, based on the HCP resting state data.

* atl-Ji10:    Subcortical resting state parcellation in 10 networks

References and Links:

* Ji, J. L., Spronk, M., Kulkarni, K., Repovš, G., Anticevic, A., & Cole, M. W. (2019). Mapping the human brain's cortical-subcortical functional network organization. Neuroimage, 185, 35-57.


King_2019:Multi-domain task battery (MDTB) parcellation and contrast maps
---------------------------------------------------------------------------------------------

.. image:: images/King_2019.png
   :width: 300px
   :align: left

King et al. (2019) provided an extensive characterization of the functional organization of the cerebellum of 24 healthy, young participants. The contrast are for for 47 task conditions, accounted for the activity caused by left hand, right hand, and eye movements. All contrast maps are relative to the mean activity across all tasks. The parcellation into 10 regions is defined from the task-evoked activity across all tasks.

* atl-MDTB10:    MDTB parcellation into 10 regions
* con-MDTB01LeftHandMovement:    Activity across tasks accounted for by left hand movements
* con-MDTB02RightHandMovement:    Activity across tasks accounted for by right hand movements
* con-MDTB03Saccades:    Activity across tasks accounted for by saccadic eye movements
* con-MDTB04NoGo:    Go-Nogo task with words: No-go
* con-MDTB05Go:    Go-Nogo task with words: go
* con-MDTB06TheoryOfMind:    2 AFC task to indicate if a short story contains true or false belief
* con-MDTB07ActionObservation:    Passive viewing of knots being tied
* con-MDTB08VideoKnots:    Passive viewing of static knots
* con-MDTB09UnpleasantScenes:    IAPS affective pictures: Unpleasant scenes
* con-MDTB10PleasantScenes:    IAPS affective pictures: Pleasant scenes
* con-MDTB11Math:    Simple multiplication equations: Judge true or false
* con-MDTB12DigitJudgement:    Control task for Math: detect 1 within 4 digits
* con-MDTB13ObjectViewing:    Passive viewing of objects or checkerboard patterns
* con-MDTB14SadFaces:    IAPS affective pictures: Sad facial expressions
* con-MDTB15HappyFaces:    IAPS affective pictures: Happy facial expressions
* con-MDTB16IntervalTiming:    Auditory temporal judgement task between short (100ms) and long (175ms)
* con-MDTB17MotorImagery:    Imagine playing a game of tennis
* con-MDTB18FingerSimple:    Series of six simple key presses of same finger
* con-MDTB19FingerSequence:    Bimanual sequence of six key press
* con-MDTB20Verbal2Back-:    Working memory 2-back task with words: no target
* con-MDTB21Verbal2Back+:    Working memory 2-back task with words: target
* con-MDTB22Object2Back-:    Working memory 2-back task with pictures: no target
* con-MDTB23Object2Back+:    Working memory 2-back task with pictures: target
* con-MDTB24SpatialImagery:    Imagine to walk from kitchen to bathroom in your childhood home
* con-MDTB25StroopIncongruent:    Stroop task: Incongruent trials
* con-MDTB26StroopCongruent:    Stroop task: Congruent trials
* con-MDTB27VerbGeneration:    Generate a verb for a displayed noun (dog->bark)
* con-MDTB28WordReading:    Read the displayed noun: control for verb generation
* con-MDTB29VisualSearchSmall:    Find a target ('T') among distractors ('L'): 4 items
* con-MDTB30VisualSearchMedium:    Find a target ('T') among distractors ('L'): 8 items
* con-MDTB31VisualSearchLarge:    Find a target ('T') among distractors ('L'): 12 items
* con-MDTB32Rest:    Passive viewing of fixation cross
* con-MDTB33CPRO:    Concrete Permuted Rules Operations: Apply set of rules to 2 stimuli
* con-MDTB34PredictionTrue:    Predicting the end of a sequentially presented sentence: fulfilled prediction
* con-MDTB35PredictionViolated:    Predicting the end of a sequentially presented sentence: violated prediction
* con-MDTB36PredictionScrambles:    Predicting the end of a sequentially presented sentence: scrambled sentence
* con-MDTB37SpatialMapEasy:    Memorize a spatial map of numbers for subsequent recall: 1 item
* con-MDTB38SpatialMapMedium:    Memorize a spatial map of numbers for subsequent recall: 4 items
* con-MDTB39SpatialMapHard:    Memorize a spatial map of numbers for subsequent recall: 7 items
* con-MDTB40NatureMovie:    Passive viewing of "Planet Earth II: Islands" movie: Animal movements
* con-MDTB41AnimatedMovie:    Passive viewing of "Up" pixar movie: Social interactions
* con-MDTB42LandscapeMovie:    Passive viewing of movie: Landscape scenes
* con-MDTB43MentalRotationEasy:    Mental rotation task between two objects: 0 degrees
* con-MDTB44MentalRotationMedium:    Mental rotation task between two objects: 50 degrees
* con-MDTB45MentalRotationHard:    Mental rotation task between two objects: 150 degrees
* con-MDTB46BiologicalMotion:    Point light walker: Judge whether gait is happy or sad
* con-MDTB47ScrambledMotion:    Point light walker: Judge whether scrambled control stimulus moves fast or slow
* con-MDTB48ResponseAlternativesEasy:    Execute fast keypress to imparative signal: 1 cued position
* con-MDTB49ResponseAlternativesMedium:    Execute fast keypress to imparative signal: 2 cued positions
* con-MDTB50ResponseAlternativesHard:    Execute fast keypress to imparative signal: 4 cued position

References and Links:
* King, M., Hernandez-Castillo, C.R., Poldrack, R.R., Ivry, R., and Diedrichsen, J. (2019). Functional Boundaries in the Human Cerebellum revealed by a Multi-Domain Task Battery. Nat. Neurosci.


Nettekoven_2024: Hierarchical functional cerebellar atlas data fusion
---------------------------------------------------------------------

.. image:: images/Nettekoven_2024.png
   :width: 300px
   :align: left

Functional parcellation into 4 domains, 32 regions or 68 subregions (symmetric or asymmetric). The three levels make up a nested hierarchical organization. An additional version with 128 regions that subdivides the 32 regions spatially into 4 regions (s: superior, i: inferior, t: tertiary, v: vermal) is available. The maps are based on the probabilistic integration of 7 task-based datasets. The color scheme reflects the functional similarity of different regions.

* atl-NettekovenSym32:    Symmetric 32-region parcellation
* atl-NettekovenAsym32:    Asymmetric 32-region parcellation
* atl-NettekovenSym68:    Symmetric 68-region parcellation (functional subregions)
* atl-NettekovenAsym68:    Asymmetric 68-region parcellation (functional subregions)
* atl-NettekovenSym128:    Symmetric 128-region parcellation (spatial subregions)
* atl-NettekovenAsym128:    Asymmetric 128-region parcellation (spatial subregions)

References and Links:

* Nettekoven, C. et al. A hierarchical atlas of the human cerebellum for functional precision mapping. Nature Communications (2024).


