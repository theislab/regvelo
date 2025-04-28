import logging
import warnings
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from scvi.data import AnnDataManager
from scvi.data.fields import (LayerField,ObsmField)
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import setup_anndata_dsp

from scipy.spatial.distance import cdist

from ._constants import REGISTRY_KEYS
from ._module import VELOVAE

logger = logging.getLogger(__name__)

## TODO: add preprocessing and network perturbation today
## network perturbation including regulon perturbation, TF-target perturbation, customs network motif perturbation

def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv

## TODO: modified the TrainingPlan and generate the new classes
class ModifiedTrainingPlan(TrainingPlan):
    """
    A TrainingPlan class modified to pass additional attributes to the module during training.

    This class modifies the training_step method to set current_epoch and global_step on the module.

    Attributes:
    module (LightningModule): The module being trained.
    plan_kwargs (dict): Additional keyword arguments for the TrainingPlan.
    """

    def __init__(self, module, **plan_kwargs):
        super().__init__(module, **plan_kwargs)

    def training_step(self, batch, batch_idx):
        # the modification:
        self.module.current_epoch = self.current_epoch
        self.module.global_step = self.global_step

        # as before:
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return scvi_loss.loss

class REGVELOVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """Regulatory Velocity Variational Inference.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~regvelo.REGVELOVI.setup_anndata`.
    W
        Prior gene regulatory graph (torch.Tensor), with row indicating targets, column indicating regulators.
    regulators
        Transcription factor list.
    lam
        Regularization parameter for controling the strengths of adding prior knowledge.
    lam2
        Regularization parameter for controling the strengths of L1 regularization to the Jacobian matrix.
    vector_constraint
        Regularization on velocity.
    bias_constraint
        Regularization on bias term (base transcription rate).
    simple_dynamics
        Use simple version of RegVelo.
    activate
        Activation function used for modeling transcription rate.
    base_alpha
        Adding base transcription rate
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    gamma_init_data
        Initialize gamma using the data-driven technique.
    linear_decoder
        Use a linear decoder from latent space to time.
    soft_constraint
        Use soft constraint mode or hard constraint mode.
    auto_regulation
        Estimate self-regulation links in the GRN.
    **model_kwargs
        Keyword args for :class:`~regvelo.VELOVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        W: torch.Tensor = None,
        regulators: list = None,
        lam: float = 1,
        lam2: float = 0,
        vector_constraint: bool = True,
        bias_constraint: bool = True,
        simple_dynamics: bool = False,
        x0: np.ndarray = None,
        t0: np.ndarray = None,
        activate: Literal["sigmoid", "softplus"] = "softplus",
        base_alpha: bool = True,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        soft_constraint: bool = True,
        auto_regulation: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_latent = n_latent
        
        ## TODO:determine the batch size
        self.batch_size = adata.shape[0]

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()
        
        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        
        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        ## initialization GRN attribute value (corr_m, regulator/target_index)
        #if "regulators" not in adata.uns.keys() or "targets" not in adata.uns.keys():
        #    raise ValueError("adata need to save the names of regulators and targets")
        adata.uns["regulators"] = adata.var.index.values
        adata.uns["targets"] = adata.var.index.values

        regulator_index = [i in adata.uns["regulators"] for i in adata.var.index.values]
        target_index = [i in adata.uns["targets"] for i in adata.var.index.values]
        
        ##
        alpha_1_unconstr = alpha_1_unconstr[target_index]
        alpha_unconstr = alpha_unconstr[target_index]

        if W is None:
            W = torch.zeros([len(adata.var.index.values),len(adata.var.index.values)])
        
        ### adding regulator list
        if regulators is not None:
            regulator_list = [index for index, gene in enumerate(adata.var_names) if gene in regulators]
        else:
            regulator_list = None
            
        if simple_dynamics:
            vector_constraint = False
            bias_constraint = False

        self.module = VELOVAE(
            n_input=self.summary_stats["n_vars"],
            regulator_index = regulator_index,
            target_index = target_index,
            skeleton = W,
            regulator_list = regulator_list,
            activate = activate,
            base_alpha = base_alpha,
            n_hidden = n_hidden,
            n_latent = n_latent,
            n_layers = n_layers,
            lam = lam,
            lam2 = lam2,
            vector_constraint = vector_constraint,
            bias_constraint = bias_constraint,
            dropout_rate = dropout_rate,
            gamma_unconstr_init = gamma_unconstr,
            alpha_unconstr_init = alpha_unconstr,
            alpha_1_unconstr_init = alpha_1_unconstr,
            x0 = x0,
            t0 = t0,
            linear_decoder = linear_decoder,
            soft_constraint = soft_constraint,
            auto_regulation = auto_regulation,
            **model_kwargs,
        )
        self._model_summary_string = (
            "REGVELOVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: Optional[int] = 1500,
        lr: float = 1e-2,
        weight_decay: float = 1e-5,
        eps: float = 1e-16,
        train_size: float = 0.9,
        batch_size: Optional[int] = None,
        validation_size: Optional[float] = None,
        early_stopping: bool = True,
        gradient_clip_val: float = 10,
        plan_kwargs: Optional[dict] = None,
        optimizer: str = "AdamW",
        **trainer_kwargs,
    ):
        """Train the model.
        The code is adapted from veloVI repository (https://github.com/YosefLab/velovi/)

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. Defaults to 1500
        lr
            Learning rate for optimization
        weight_decay
            Weight decay for optimization
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        gradient_clip_val
            Val for gradient clipping
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        self.module.max_epochs = max_epochs
        if batch_size is None:
            batch_size = self.batch_size
        
        user_plan_kwargs = plan_kwargs.copy() if isinstance(plan_kwargs, dict) else {}
        plan_kwargs = {"lr": lr, "weight_decay": weight_decay, "optimizer": optimizer, "eps": eps}
        plan_kwargs.update(user_plan_kwargs)

        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = {"gradient_clip_val": gradient_clip_val}
        trainer_kwargs.update(user_train_kwargs)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = ModifiedTrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        trainer_kwargs["early_stopping_patience"] = 45
        trainer_kwargs["early_stopping_monitor"] = 'elbo_validation'
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **trainer_kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time. 
        The code is adapted from veloVI repository (https://github.com/YosefLab/velovi/)

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                ind_time = generative_outputs["px_rho"] * self.module.t_max

                output = (
                    ind_time
                    # + steady_prob * switch_time
                    # + rep_steady_prob * self.module.t_max
                )

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[self.module.target_index],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.inference_mode()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                beta = inference_outputs["beta"]
                gamma = inference_outputs["gamma"]
                px_rho = generative_outputs["px_rho"]

                ind_t = self.module.t_max * px_rho
                mean_u, mean_s,_ = self.module._get_induction_unspliced_spliced(
                    ind_t
                )
                velo = beta * mean_u - gamma * mean_s
                # expectation
                output = (
                    velo
                )
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        spliced = spliced[:,self.module.target_index]
        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[self.module.target_index],
                index=adata.obs_names[indices],
            )
        else:
            return velos
        
    @torch.inference_mode()
    def rgv_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                px_rho = generative_outputs["px_rho"]

                ind_t = self.module.t_max * px_rho
                fit_u, fit_s, _ = self.module._get_induction_unspliced_spliced(
                    ind_t
                )

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.detach().cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.detach().cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    def compute_shared_time(self, t, perc=None, norm=True):
        nans = np.isnan(np.sum(t, axis=0))
        if np.any(nans):
            t = np.array(t[:, ~nans])
        t -= np.min(t)

        tx_list = np.percentile(t, [15, 25, 50, 75, 85] if perc is None else perc, axis=1)
        tx_max = np.max(tx_list, axis=1)
        tx_max += tx_max == 0
        tx_list /= tx_max[:, None]

        mse = []
        for tx in tx_list:
            tx_ = np.sort(tx)
            linx = np.linspace(0, 1, num=len(tx_))
            mse.append(np.sum((tx_ - linx) ** 2))
        idx_best = np.argsort(mse)[:2]

        t_shared = tx_list[idx_best].sum(0)
        if norm:
            t_shared = (t_shared - t_shared.min()) / (t_shared.max() - t_shared.min())

        return t_shared

    @torch.inference_mode()
    def add_regvelo_outputs_to_adata(
        self,
        n_samples: int = 30,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> AnnData:
        """Returns the processed anndata object containing velocity and latent time layers
        The code is adapted from veloVI repository (https://github.com/YosefLab/velovi/)

        Parameters
        ----------
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        adata = self._validate_anndata(adata)
        latent_time = self.get_latent_time(n_samples=n_samples, batch_size = batch_size)
        velocities = self.get_velocity(n_samples=n_samples, batch_size = batch_size)

        t = latent_time.values.copy()
        scaling = 20 / t.max(0)
        adata_target = adata[:,self.module.target_index].copy()
    
        adata_target.layers["velocity"] = velocities / scaling
        adata_target.layers["latent_time_regvelo"] = latent_time
        
        adata_target.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
        adata_target.var['fit_scaling'] = 1.0
        
        return adata_target

    @torch.inference_mode()
    def get_rates(self):
        gamma, beta, alpha_1= self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha_1": alpha_1.cpu().numpy(),
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: Optional[str] = None,
        unspliced_layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression.

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)   

    def get_directional_uncertainty(
        self,
        adata: Optional[AnnData] = None,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):
        adata = self._validate_anndata(adata)

        logger.info("Sampling from model...")
        velocities_all = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        return df, cosine_sims

    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None
    ) -> Tuple[pd.DataFrame, AnnData]:
        """Compute permutation scores.

        Parameters
        ----------
        labels_key
            Key in adata.obs encoding cell types
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.

        Returns
        -------
        Tuple of DataFrame and AnnData. DataFrame is genes by cell types with score per cell type.
        AnnData is the permutated version of the original AnnData.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        # shuffle spliced then unspliced
        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.X_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.U_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        ms_ = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_ = adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        ms_p = bdata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_p = bdata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        spliced_, unspliced_ = self.get_expression_fit(adata, n_samples=10)
        root_squared_error = np.abs(spliced_ - ms_)
        root_squared_error += np.abs(unspliced_ - mu_)

        spliced_p, unspliced_p = self.get_expression_fit(bdata, n_samples=10)
        root_squared_error_p = np.abs(spliced_p - ms_p)
        root_squared_error_p += np.abs(unspliced_p - mu_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            for g in adata.var_names.tolist():
                x = root_squared_error_p[g][adata.obs[labels_key] == ct]
                y = root_squared_error[g][adata.obs[labels_key] == ct]
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata

    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        # get registry info to later set data back in bdata
        # in a way that doesn't require actual knowledge of location
        unspliced = bdata_manager.get_from_registry(registry_key)
        u_registry = bdata_manager.data_registry[registry_key]
        attr_name = u_registry.attr_name
        attr_key = u_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            unspliced_ct = unspliced[mask].copy()
            unspliced_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=unspliced_ct
            )
            unspliced[mask] = unspliced_ct
        # e.g., if using adata.X
        if attr_key is None:
            setattr(bdata, attr_name, unspliced)
        # e.g., if using a layer
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = unspliced
            setattr(bdata, attr_name, attribute)

        return bdata


def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    logger.info("Computing the uncertainties...")
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    # cells by samples
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims


def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal function for parallelization.

    Parameters
    ----------
    tensor
        Shape of samples by genes for a given cell.
    """
    n_samples = tensor.shape[0]
    # over samples axis
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
    ]
    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )


def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
