# RegVelo: gene-regulatory-informed dynamics of single cells

<img src="https://github.com/theislab/regvelo/blob/main/docs/_static/img/overview_fig.png?raw=true" alt="RegVelo" width="600" />

**RegVelo** is an end-to-end framework to infer regulatory cellular dynamics through coupled splicing dynamics. 
See our [RegVelo manuscript](https://www.biorxiv.org/content/10.1101/2024.12.11.627935v1) and [documentation](https://regvelo.readthedocs.io/en/latest/index.html) to learn more. 

Feel free to open an [issue](https://github.com/theislab/regvelo/issues/new) if you encounter a bug, need our help or just want to make a comment/suggestion.

RegVelo's key applications
--------------------------
- Estimate RNA velocity governed by gene regulation.
- Infer latent time to indicate the cellular differentiation process.
- Estimate intrinsic and extrinsic velocity uncertainty [Gayoso et al. (2024)](https://www.nature.com/articles/s41592-023-01994-w).
- Estimate regulon perturbation effects via CellRank framework ([Lange et al. (2022)](https://www.nature.com/articles/s41592-021-01346-6), [Weiler et al. (2024)](https://www.nature.com/articles/s41592-024-02303-9)).


## Getting started

We have [tutorials](https://regvelo.readthedocs.io/en/latest/tutorials/index.html) to help you get started.


## Installation

You need to have Python 3.10 or newer installed on your system.

There are several options to intall regvelo:

1. Install the latest release of `regvelo` from PyPI via

```bash
  pip install regvelo
```

2. Install the latest development version via

```bash
  pip install git+https://github.com/theislab/regvelo.git@main
```

## Citation

If you find RegVelo useful for your research, please consider citing our work as:

```
  @article{wang2026regvelo,
      title={RegVelo: gene-regulatory-informed dynamics of single cells},
      author={Wang, Weixu and Hu, Zhiyuan and Weiler, Philipp and Mayes, Sarah and Lange, Marius and Wang, Jingye and Xue, Zhengyuan and Sauka-Spengler, Tatjana and Theis, Fabian J},
      journal={Cell},
      year={2026},
      publisher={Elsevier},
      doi={10.1016/j.cell.2026.04.022}
    }
```

