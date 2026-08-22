About RegVelo
-------------

Understanding cellular dynamics and regulatory interactions is crucial for decoding the complex processes that govern cell fate and differentiation. 
Traditional RNA velocity methods capture dynamic cellular transitions by modeling changes in spliced and unspliced mRNA but lack integration with gene regulatory networks (GRNs), omitting critical regulatory mechanisms underlying cellular decisions. 
Conversely, GRN inference techniques map regulatory connections but fail to account for the temporal dynamics of gene expression.

With RegVelo, developed by `Wang et al. (Cell, 2026) <https://www.cell.com/cell/fulltext/S0092-8674(26)00457-5>`_,
the research gap is bridged through combining RNA velocity's temporal insights with a regulatory framework to model transcriptome-wide splicing kinetics informed by GRNs.
This extend current RNA velocity framework to a full mechanism model, allowing more complicated development process to be modeled (see the :ref:`RegVelo model details <model-index>` for a detailed explanation).
Further, by coupling with CellRank `Weiler et al. (Nature Methods, 2024) <https://www.nature.com/articles/s41592-024-02303-9>`_, RegVelo expands its capabilities to include robust perturbation predictions, linking regulatory changes to long-term cell fate decisions. 
CellRank employs velocity-based state transition probabilities to predict terminal cell states, and RegVelo enhances this framework with GRN-informed splicing kinetics, 
enabling precise simulations of transcription factor (TF) knockouts. This synergy allows for the identification of lineage drivers and the prediction of cell fate changes upon genetic perturbations (see :ref:`Perturbation prediction <model-index>` for more details).

