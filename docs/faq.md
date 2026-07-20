# Frequently Asked Questions

#### Q1: My commitment scores look inverted — stem/progenitor cells score low, differentiated cells score high, and in disease samples nearly all cell types are elevated. Is this a RegVelo bug?

**A:** Not necessarily. In most cases this reflects the inferred **velocity field**, not the
commitment score itself. CellRank derives commitment probabilities from the velocity
transition matrix, so a reversed or unreliable velocity field will directly produce
unexpected commitment scores.

Before concluding that RegVelo is at fault, validate whether the velocity field is
reasonable.

#### How do I check whether the velocity field is the problem?

Run one or more simpler velocity models on the same dataset and compare the velocity
streams side by side with RegVelo:

- scVelo (stochastic)
- scVelo (dynamical)
- veloVI

These methods rest on simpler assumptions, which makes the comparison diagnostic:

- **Simpler models also fail to recover the expected trajectory** → the limitation is
  likely the dataset (sparsity, weak kinetics, or the underlying biology), not RegVelo.
- **Simpler models recover a reasonable trajectory but RegVelo doesn't** → the issue
  more likely lies in the GRN prior, model fitting, or RegVelo-specific settings.

```{admonition} Rule of thumb
:class: tip

RegVelo still depends on the RNA velocity signal. If scVelo/veloVI can't recover the
trajectory, RegVelo generally won't either, because the underlying dynamics aren't
identifiable.
```

#### What could cause this?

- The dataset is very sparse, making velocity estimation unreliable.
- Transcriptional dynamics are weak or ambiguous, so RNA velocity can't robustly resolve
  developmental direction.
- The system isn't a simple differentiation process — e.g., cycling populations,
  bidirectional transitions, or strong disease-induced perturbations — which complicates
  terminal-state identification.

In these cases the velocity field, and therefore the commitment score, may not match
prior biological expectations.

#### What should I include when reporting this issue?

- The biological system (in vivo, organoid, cell line, etc.)
- Velocity stream plots from RegVelo, scVelo (stochastic and/or dynamical), and veloVI if
  available
- CellRank terminal-state identification
- UMAP colored by cell type and by commitment score
- Dataset size and sequencing protocol, if relevant

These make it much easier to tell whether the problem originates in velocity inference or
in downstream CellRank analysis.

```{admonition} Bottom line
:class: note

When unsure whether RegVelo suits a dataset, benchmark with scVelo or veloVI first. If
they recover the expected trajectory, RegVelo generally can too (given an appropriate GRN
prior and configuration). If none of them do, the limitation is most likely the dataset
rather than RegVelo.
```
