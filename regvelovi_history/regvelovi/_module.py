"""Main module."""
from typing import Callable, Iterable, Literal, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl
import torchode as to
from ._constants import REGISTRY_KEYS

torch.backends.cudnn.benchmark = True


class DecoderVELOVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # categorical pi
        # 3 states
        # induction, repression, repression steady state
        self.px_pi_decoder = nn.Linear(n_hidden, 3 * n_output)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
        
        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        # The decoder returns values for the parameters of the ZINB distribution
        rho_first = self.rho_first_decoder(z_in)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp()
                + self.linear_scaling_tau_intercept
            )

        # cells by genes by 3
        pi_first = self.pi_first_decoder(z)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 3))
        )

        return px_pi, px_rho, px_tau


## define a new class velocity encoder
class velocity_encoder(nn.Module):
    """ 
    encode the velocity
    time dependent transcription rate is determined by upstream emulator
    velocity could be build on top of this

    merge velocity encoder and emulator class
    """                 
    def __init__(
        self,
        n_int: int = 5,
        W: torch.Tensor = (torch.FloatTensor(5, 5).uniform_() > 0.5).int(),
        W_int: torch.Tensor = None,
        log_h_int: torch.Tensor = None,
    ):
        device = W.device
        super().__init__()
        self.n_int = n_int
        self.log_h = torch.nn.Parameter(log_h_int.repeat(W.shape[0],1)*W)
        self.log_phi = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)
        self.tau = torch.nn.Parameter(torch.ones(W.shape).to(device)*W*10)
        self.o = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)

        self.mask_m = W

        ## initialize grn
        self.grn = torch.nn.Parameter(W_int*self.mask_m)
        
        ## initilize gamma and beta
        #self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_int))
        #self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_int))
        #self.alpha_unconstr_max = torch.nn.Parameter(torch.randn(n_int))
        #self.alpha_unconstr = torch.nn.Parameter(torch.tensor(alpha_unconstr_init).to(device))
        
        ## initialize logic gate multivariate bernoulli distribution
        ## using hook to mask the gradients
        ### define hook to froze the parameters
    def _set_mask_grad(self):
        self.hooks_grn = []
        self.hooks_log_h = []
        self.hooks_log_phi = []
        self.hooks_tau = []
        self.hooks_o = []
        #mask_m = self.mask_m
        
        def _hook_mask_no_regulator(grad):
            return grad * self.mask_m
        

        w_grn = self.grn.register_hook(_hook_mask_no_regulator)
        self.hooks_grn.append(w_grn)
        
        w_log_h = self.log_h.register_hook(_hook_mask_no_regulator)
        w_log_phi = self.log_phi.register_hook(_hook_mask_no_regulator)
        w_tau = self.tau.register_hook(_hook_mask_no_regulator)
        w_o = self.o.register_hook(_hook_mask_no_regulator)

        self.hooks_log_h.append(w_log_h)
        self.hooks_log_phi.append(w_log_phi)
        self.hooks_tau.append(w_tau)
        self.hooks_o.append(w_o)

    def emulation_all(self,t: torch.Tensor = None):

        emulate_m = []

        h = torch.exp(self.log_h)
        phi = torch.exp(self.log_phi)
        for i in range(t.shape[1]):
            # for each time stamps, predict the emulator predict value
            tt = t[:,i]
            emu = h * torch.exp(-phi*(tt.reshape((len(tt),1))-self.tau)**2) + self.o
            #emu = emu * self.mask_m
            emulate_m.append(emu)

        return torch.stack(emulate_m,2)

    def forward(self,t, y):
        ## split x into unspliced and spliced readout
        ## x is a matrix with (G*2), in which row is a subgraph (batch)
        #print(y)
        #assert torch.jit.isinstance(args, torch.Tensor)
        #assert torch.jit.isinstance(t, torch.Tensor)
        #assert torch.jit.isinstance(y, torch.Tensor)
        
        if len(y.shape) == 1:
            u = y[0]
            s = y[1]
        else:  
            u = y[:,0]
            s = y[:,1]

        ## calculate emulator value
        ## t is a vector per gene (G*1)
        ## extract the corresponding gene
        #u = u[locate]
        #s = s[locate]
        #T = t[locate]

        h = torch.exp(self.log_h)
        phi = torch.exp(self.log_phi)
        #emu = h[locate,:] * torch.exp(-phi[locate,:]*(T.reshape((dim,1))-self.tau[locate,:])**2) + self.o[locate,:]
        emu = h * torch.exp(-phi*(t.reshape((-1,1)) - self.tau)**2) + self.o

        ## Use the Emulator matrix to predict alpha
        #emu = emu * self.grn[locate,:]
        emu = emu * self.grn
        
        alpha_unconstr = emu.sum(dim=1)
        #alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias[locate]
        alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias

        ## Generate kinetic rate
        #beta = torch.clamp(F.softplus(self.beta_mean_unconstr[locate]), 0, 50)
        #gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr[locate]), 0, 50)
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        alpha = torch.clamp(alpha_unconstr,1e-3,) ## set a minimum transcriptional rate to prevent it can't be trained
        alpha = F.softsign(alpha)*torch.clamp(F.softplus(self.alpha_unconstr_max), 0, 50)

        ## Predict velocity
        du = alpha - beta*u
        ds = beta*u - gamma*s
        
        du = du.reshape((-1,1))
        ds = ds.reshape((-1,1))

        v = torch.concatenate([du,ds],axis = 1)

        if len(y.shape) == 1:
            v = v.view(-1)

        return v
    
# VAE model
class VELOVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the veloVI model descibed in :cite:p:`GayosoWeiler2022`

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    alpha_1 represent the maximum transcription rate once could reach in induction stage
    """

    def __init__(
        self,
        n_input: int,
        regulator_index,
        target_index,
        skeleton,
        corr_m,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        log_h_int: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 1/3,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = True,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2
        n_targets = sum(target_index)
        n_regulators = sum(regulator_index)
        self.n_targets = int(n_targets) 
        self.n_regulators = int(n_regulators)
        self.regulator_index = regulator_index
        self.target_index = target_index
        
        # switching time for each target gene
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_targets))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation for each target gene
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_targets))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # splicing for each target gene
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_targets))

        # transcription (bias term for target gene transcription rate function)
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_targets))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )
            
        # TODO: Add `require_grad`
        ## The maximum transcription rate (alpha_1) for each target gene 
        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(torch.ones(n_targets))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_1_unconstr_init)
            )
            self.alpha_1_unconstr.data = self.alpha_1_unconstr.data.float()

        # likelihood dispersion
        # for now, with normal dist, this is just the variance for target genes
        self.scale_unconstr_targets = torch.nn.Parameter(-1 * torch.ones(n_targets*2, 3))
        
        ## TODO: use normal dist to model the emulator preduction
        #self.scale_unconstr_regulators = torch.nn.Parameter(-1 * torch.ones(n_regulators, 1))
        """
        need discussion
        self.scale_unconstr_regulators = torch.nn.Parameter(-1 * torch.ones(n_regulators*2, 3))
        """
        
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_target-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_targets,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
        )
        
        # define velocity encoder, define velocity vector for target genes
        self.v_encoder = velocity_encoder(n_int = n_targets, 
                                          W = skeleton, W_int = corr_m, log_h_int= log_h_int)
        # saved kinetic parameter in velocity encoder module
        self.v_encoder.beta_mean_unconstr = self.beta_mean_unconstr
        self.v_encoder.gamma_mean_unconstr = self.gamma_mean_unconstr
        self.v_encoder.alpha_unconstr_max = self.alpha_1_unconstr
        self.v_encoder.alpha_unconstr_bias = self.alpha_unconstr
        # initilize grn (masked parameters)
        self.v_encoder._set_mask_grad()

    def _get_inference_input(self, tensors):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]
        #regulator_spliced = spliced[:,self.regulator_index]
        #target_spliced = spliced[:,self.target_index,:]
        #target_unspliced = unspliced[:,self.target_index,:]
        
        input_dict = {
            "spliced": spliced,
            "unspliced": unspliced,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha_1 = inference_outputs["alpha_1"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "beta": beta,
            "alpha_1": alpha_1,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, beta, alpha_1 = self._get_rates()

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "gamma": gamma,
            "beta": beta,
            "alpha_1": alpha_1,
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation for each target gene
        gamma = torch.clamp(F.softplus(self.v_encoder.gamma_mean_unconstr), 0, 50)
        # splicing for each target gene
        beta = torch.clamp(F.softplus(self.v_encoder.beta_mean_unconstr), 0, 50)
        # transcription for each target gene (bias term)
        #alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)
        if self.time_dep_transcription_rate:
            ## the maximum transcription rate for each target gene
            alpha_1 = torch.clamp(F.softplus(self.v_encoder.alpha_unconstr_max), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr

        return gamma, beta, alpha_1

    @auto_move_data
    def generative(self, z, gamma, beta, alpha_1, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(decoder_input, latent_dim=latent_dim)
        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr_targets
        scale = F.softplus(scale_unconstr)

        ####################
        #scale_unconstr_regulators = self.scale_unconstr_regulators
        #scale_regulators = F.softplus(scale_unconstr_regulators)
        ####################


        mixture_dist_s, mixture_dist_u, emulation = self.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha_1,
        )

        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "scale": scale,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u": mixture_dist_u,
            "mixture_dist_s": mixture_dist_s,
            "emulation": emulation,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        ### extract spliced, unspliced readout
        regulator_spliced = spliced[:,self.regulator_index]
        target_spliced = spliced[:,self.target_index]
        target_unspliced = unspliced[:,self.target_index]
        
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        #end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        ##################
        # dist_emulate = generative_outputs["dist_emulate"]
        ##################

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(target_spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(target_unspliced)
        
        ############################
        #reconst_loss_regulator = -dist_emulate.log_prob(regulator_spliced)
        #reconst_loss_regulator = reconst_loss_regulator.sum(dim=-1)
        ############################

        reconst_loss_target = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)
        
        ### calculate the reconstruct loss for emulation
        emulation = generative_outputs["emulation"].T
        recon_loss_reg = F.mse_loss(regulator_spliced, emulation, reduction='none').sum(-1)

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss_target + recon_loss_reg + weighted_kl_local)
        # recon_loss_all = local_loss - kl_local
        # add L1 loss to grn
        # L1_loss = torch.abs(self.v_encoder.grn).sum()
        loss = local_loss

        loss_recoder = LossOutput(
            loss=loss, reconstruction_loss=recon_loss_reg, kl_local=torch.tensor(0)
        )

        return loss_recoder

    @auto_move_data
    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha_1,
    ) -> torch.Tensor:
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # predict the abundance in induction phase for target genes
        ind_t = t_s * px_rho
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            ind_t
        )
        
        # get the emulation results of each regulator to the inductive latent time
        emulation = self.v_encoder.emulation_all(ind_t.T)
        emulation = emulation.mean(dim=0)
        
        #######################
        #scale_regulators = scale_regulators.expand(n_cells, self.n_regulators, 1).sqrt()
        #######################

        #if self.time_dep_transcription_rate:
        #    mean_u_ind_steady = (alpha_1 / beta).expand(n_cells, self.n_targets)
        #    mean_s_ind_steady = (alpha_1 / gamma).expand(n_cells, self.n_targets)
        
        ### only have three cell state
        # induction
        # repression
        # repression steady state
        scale_u = scale[: self.n_targets, :].expand(n_cells, self.n_targets, 3).sqrt()

        # calculate the initial state for repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            t_s.reshape(1,len(t_s))
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_targets :, :].expand(n_cells, self.n_targets, 3).sqrt()

        #end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
        #    (s_0 - self.switch_spliced).pow(2)
        #).sum()

        # unspliced
        mean_u = torch.stack(
            (
                mean_u_ind,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            dim=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        scale_s = torch.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            dim=2,
        )
        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        # emulation
        #dist_emulate = Normal(emulation.T,scale_regulators[...,0])
        #############

        return mixture_dist_s, mixture_dist_u, emulation

    def _get_induction_unspliced_spliced(
        self, t, eps=1e-6
    ):
        """
        this function aim to calculate the spliced and unspliced abundance for target genes
        
        alpha_1: the maximum transcription rate during induction phase for each target gene
        beta: the splicing parameter for each target gene
        gamma: the degradation parameter for each target gene
        
        ** the above parameters are saved in v_encoder
        t: target gene specific latent time
        """
        device = self.device
        t = t.T    
        
        if t.shape[1] > 1:
            t_eval, index = torch.sort(t, dim=1)
            index2 = (t_eval[:,:-1] != t_eval[:,1:])
            subtraction_values,_ = torch.where((t_eval[:,1:] - t_eval[:,:-1])>0, (t_eval[:,1:] - t_eval[:,:-1]), torch.inf).min(axis=1)
            subtraction_values[subtraction_values == float("Inf")] = 0
            
            true_tensor = torch.ones((t_eval.shape[0],1), dtype=torch.bool)
            index2 = torch.cat((index2, true_tensor.to(index2.device)),dim=1) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
                
            subtraction_values = subtraction_values[None, :].repeat(index2.shape[1], 1).T
            t_eval[index2 == False] -= subtraction_values[index2 == False]*0.1
            ## extract initial target gene expression value
            #x0 = torch.cat((target_unspliced[:,0].reshape((target_unspliced.shape[0],1)),target_spliced[:,0].reshape((target_spliced.shape[0],1))),dim = 1)
            x0 = torch.zeros((t.shape[0],2)).to(self.device)
            #x0 = x0.double()
            t_eval = torch.cat((torch.zeros((t_eval.shape[0],1)).to(self.device),t_eval),dim=1)
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            
            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            dt0 = torch.full((x0.shape[0],), 1)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval), dt0=dt0)
        else:
            t_eval = t
            t_eval = torch.cat((torch.zeros((t_eval.shape[0],1)).to(self.device),t_eval),dim=1)
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            x0 = torch.zeros((t.shape[0],2)).to(self.device)
            #x0 = x0.double()

            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            dt0 = torch.full((x0.shape[0],), 1)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval), dt0=dt0)

        ## generate predict results
        # the solved results are saved in sol.ys [the number of subsystems, time_stamps, [u,s]]
        pre_u = sol.ys[:,1:,0]
        pre_s = sol.ys[:,1:,1]     
        
        if t.shape[1] > 1:
            unspliced = torch.zeros_like(pre_u)
            spliced = torch.zeros_like(pre_s)   
            for i in range(index.size(0)):
                unspliced[i][index[i]] = pre_u[i]
                spliced[i][index[i]] = pre_s[i]
            unspliced = unspliced.T
            spliced = spliced.T
        else:
            unspliced = pre_u.ravel()
            spliced = pre_s.ravel()
    
        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (
            beta * u_0 / ((gamma - beta) + eps)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings


