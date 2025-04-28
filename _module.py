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
from torch import Tensor
import torch.nn.utils.prune as prune

torch.backends.cudnn.benchmark = True

def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv

class ThresholdPruning(prune.BasePruningMethod):
	PRUNING_TYPE = "unstructured"

	def __init__(self, threshold):
		self.threshold = threshold

	def compute_mask(self, tensor, default_mask):
		return torch.abs(tensor) > self.threshold
    
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
        self.n_output = n_output
        self.linear_decoder = linear_decoder
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm if not linear_decoder else False,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

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
        else:
            px_rho = nn.Sigmoid()(torch.matmul(z_in,torch.ones([1,self.n_output])))
        
        return px_rho

## define a new class velocity encoder
class velocity_encoder(nn.Module):
    """Encode the velocity

    time dependent transcription rate is determined by upstream regulator, velocity could be build on top of this

    Parameters
    ----------
    activate
        activate function used for modeling transcription rate
    bas_alpha
        adding base transcription rate
    n_int
        number of genes
    """                 
    def __init__(
        self,
        activate: str = "softplus",
        base_alpha: bool = True,
        n_int: int = 5,
    ):
        super().__init__()
        self.n_int = n_int
        self.fc1 = nn.Linear(n_int, n_int)
        self.activate = activate
        self.base_alpha = base_alpha
        
    def _set_mask_grad(self):
        self.hooks = []
 
        def _hook_mask_no_regulator(grad):
            return grad * self.mask_m

        w_grn = self.fc1.weight.register_hook(_hook_mask_no_regulator)
        self.hooks.append(w_grn)

    
    ## TODO: regularizing the jacobian
    def GRN_Jacobian(self,s):
        
        if self.activate is not "OR":
            if self.base_alpha is not True:
                grn = self.fc1.weight
                #grn = grn - self.lamb_I
                alpha_unconstr = torch.matmul(s,grn.T)
            else:
                alpha_unconstr = self.fc1(s)

            if self.activate == "softplus":    
                coef = (torch.sigmoid(alpha_unconstr))
            if self.activate == "sigmoid":
                coef = (torch.sigmoid(alpha_unconstr))*(1 - torch.sigmoid(alpha_unconstr))*self.alpha_unconstr_max
        else:
            coef = (1 / (torch.nn.functional.softsign(s) + 1)) * (1 / (1 + torch.abs(s - 0.5))**2) * torch.exp(self.fc1(torch.log(torch.nn.functional.softsign(s - 0.5)+1)))

        if coef.dim() > 1:
            coef = coef.mean(0)
        Jaco_m = torch.matmul(torch.diag(coef), self.fc1.weight)
            
        
        return Jaco_m
    
    def GRN_Jacobian2(self,s):

        if self.base_alpha is not True:
            grn = self.fc1.weight
            alpha_unconstr = torch.matmul(s,grn.T)
        else:
            alpha_unconstr = self.fc1(s)
            
        if self.activate == "softplus":    
            coef = (torch.sigmoid(alpha_unconstr))
        if self.activate == "sigmoid":
            coef = (torch.sigmoid(alpha_unconstr))*(1 - torch.sigmoid(alpha_unconstr))*self.alpha_unconstr_max
        
        # Perform element-wise multiplication
        Jaco = coef.unsqueeze(-1) * self.fc1.weight.unsqueeze(0)

        # Transpose and reshape to get the final 3D tensor with dimensions (m, n, n)
        Jaco = Jaco.reshape(s.shape[0], s.shape[1], s.shape[1])
        
        return Jaco
    

    def transcription_rate(self,s):
        if self.activate is not "OR":
            if self.base_alpha is not True:
                grn = self.fc1.weight
                #grn = grn - self.lamb_I
                alpha_unconstr = torch.matmul(s,grn.T)
            else:
                alpha_unconstr = self.fc1(s)

            if self.activate == "softplus":
                alpha = torch.clamp(F.softplus(alpha_unconstr),0,50)
            elif self.activate == "sigmoid":
                alpha = torch.sigmoid(alpha_unconstr)*self.alpha_unconstr_max
            else:
                raise NotImplementedError
        elif self.activate is "OR":
            alpha = torch.exp(self.fc1(torch.log(torch.nn.functional.softsign(s - 0.5)+1)))
            
        return alpha

    ## TODO: introduce sparsity in the model
    def forward(self,t, u, s):
        ## split x into unspliced and spliced readout
        ## x is a matrix with (G*2), in which row is a subgraph (batch)
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        if self.activate is not "OR":
            if self.base_alpha is not True:
                grn = self.fc1.weight
                alpha_unconstr = torch.matmul(s,grn.T)
            else:
                alpha_unconstr = self.fc1(s)

            if self.activate == "softplus":
                alpha = torch.clamp(F.softplus(alpha_unconstr),0,50)
            elif self.activate == "sigmoid":
                alpha = torch.sigmoid(alpha_unconstr)*self.alpha_unconstr_max
        elif self.activate is "OR":
            alpha = torch.exp(self.fc1(torch.log(torch.nn.functional.softsign(s - 0.5)+1)))
        else:
            raise NotImplementedError

        ## Predict velocity
        du = alpha - beta*u
        ds = beta*u - gamma*s

        return du,ds

class v_encoder_batch(nn.Module):
    """Batching the velocity

    Parameters
    ----------
    num_g
        number of genes
    """
    
    def __init__(
        self,
        num_g: int = 5,
    ):
        super().__init__()
        self.num_g = num_g

    def forward(self,t,y):
        """
        in which x is a reshape matrix: (g*n) * 2
        we first reshape x into two matrix: unspliced (g*n) and spliced (g*n)
        and calculate velocity
        then shape back to the vector: (g*n) * 2
        the batch number in this case is g*n
        """
        u_v = y[:,0]
        s_v = y[:,1]
        u = u_v.reshape(-1, self.num_g)
        s = s_v.reshape(-1, self.num_g)
        du, ds = self.v_encoder_class(t, u, s)
    
        ## reshape du and ds
        du = du.reshape(-1, 1)
        ds = ds.reshape(-1, 1)
    
        v = torch.concatenate([du,ds],axis = 1)
    
        return v

    
# VAE model
class VELOVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the RegVelo model.

    Parameters
    ----------
    n_input
        Number of input genes.
    regulator_index
        list index for all regulators.
    target_index
        list index for all targets.
    skeleton
        prior gene regulatory graph.
    regulator_list
        a integer list represents where is the regulators.
    activate
        Activation function used for modeling transcription rate.
    base_alpha
        Adding base transcription rate.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    lam
        Regularization parameter for controling the strengths of adding prior knowledge.
    lam2
        Regularization parameter for controling the strengths of L1 regularization to the Jacobian matrix.
    vector_constraint
        Regularization on velocity.
    bias_constraint
        Regularization on bias term (base transcription rate).
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
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        regulator_index: list,
        target_index: list,
        skeleton: torch.Tensor,
        regulator_list: list,
        activate: Literal["sigmoid", "softplus"] = "softplus",
        base_alpha: bool = True,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        lam: float = 1,
        lam2: float = 0,
        vector_constraint: bool = True,
        alpha_constraint: float = 0.1,
        bias_constraint: bool = True,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
        t0: Optional[np.ndarray] = None,
        t_max: float = 20,
        linear_decoder: bool = False,
        soft_constraint: bool = True,
        auto_regulation: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.n_input = n_input
        self.t_max = t_max
        self.lamba = lam
        self.lamba2 = lam2
        self.vector_constraint = vector_constraint
        self.alpha_constraint = alpha_constraint
        self.bias_constraint = bias_constraint
        self.soft_constraint = soft_constraint
        

        n_genes = n_input * 2
        n_targets = sum(target_index)
        n_regulators = sum(regulator_index)
        self.n_targets = int(n_targets) 
        self.n_regulators = int(n_regulators)
        self.regulator_index = regulator_index
        self.target_index = target_index
        
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
            #self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_targets))
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
        self.v_encoder = velocity_encoder(n_int = n_targets,activate = activate,base_alpha = base_alpha)
        self.v_encoder.fc1.weight = torch.nn.Parameter(0 * torch.ones(self.v_encoder.fc1.weight.shape))
        # saved kinetic parameter in velocity encoder module
        self.v_encoder.regulator_index = self.regulator_index
        self.v_encoder.beta_mean_unconstr = self.beta_mean_unconstr
        self.v_encoder.gamma_mean_unconstr = self.gamma_mean_unconstr
        self.v_encoder.register_buffer("alpha_unconstr_max", torch.tensor(10.0))
    
        # initilize grn (masked parameters)
        if self.soft_constraint is not True:
            self.v_encoder.register_buffer("mask_m", skeleton)
            self.v_encoder._set_mask_grad()
        else:
            if regulator_list is not None:
                skeleton_ref = torch.zeros(skeleton.shape)
                skeleton_ref[:,regulator_list] = 1
            else:
                skeleton_ref = torch.ones(skeleton.shape)
            if not auto_regulation:
                skeleton_ref[range(skeleton_ref.shape[0]), range(skeleton_ref.shape[1])] = 0
            self.v_encoder.register_buffer("mask_m", skeleton_ref)
            self.v_encoder._set_mask_grad()
            self.v_encoder.register_buffer("mask_m_raw", skeleton)
            
        
        ## define batch velocity vector for numerical solver
        self.v_batch = v_encoder_batch(num_g = n_targets)
        self.v_batch.v_encoder_class = self.v_encoder
        
        ## register variable for torchode
        if x0 is not None:
            self.register_buffer("x0", torch.tensor(x0))
        else:
            self.register_buffer("x0", torch.zeros([n_targets,2]))
            
        ## TODO: follow xiaojie suggestion, update x0 estimate
            
        if t0 is not None:
            self.register_buffer("t0", torch.tensor(t0).reshape(-1,1))
        else:
            self.register_buffer("t0", torch.zeros([n_targets,1]))
            
        self.register_buffer("dt0", torch.ones([1]))
        #self.register_buffer("t0", torch.zeros([1]))
        self.register_buffer("target_m",torch.zeros(self.v_encoder.fc1.weight.data.shape))


    def _get_inference_input(self, tensors):
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]
        
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
            "qzm": qz_m,
            "qzv": qz_v,
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
        alpha_1 = self.alpha_1_unconstr

        return gamma, beta, alpha_1

    @auto_move_data
    def generative(self, z, gamma, beta, alpha_1, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z

        ## decoder directly decode the latent time of each gene
        px_rho = self.decoder(decoder_input, latent_dim=latent_dim)

        scale_unconstr = self.scale_unconstr_targets
        scale = F.softplus(scale_unconstr)

        dist_s, dist_u, index, ind_t = self.get_px(
            px_rho,
            scale,
            gamma,
            beta,
            alpha_1,
        )

        return {
            "px_rho": px_rho,
            "scale": scale,
            "dist_u": dist_u,
            "dist_s": dist_s,
            "t_sort": index,
            "ind_t": ind_t,
        }
    
    def pearson_correlation_loss(self, tensor1, tensor2, eps=1e-6):
        # Calculate means
        mean1 = torch.mean(tensor1, dim=0)
        mean2 = torch.mean(tensor2, dim=0)
        
        # Calculate covariance
        covariance = torch.mean((tensor1 - mean1) * (tensor2 - mean2), dim=0)
        
        # Calculate standard deviations
        std1 = torch.std(tensor1, dim=0,correction = 0)
        std2 = torch.std(tensor2, dim=0,correction = 0)
        
        # Calculate correlation coefficients
        correlation_coefficients = covariance / (std1 * std2 + eps)
        
        # Convert NaNs to 0 (when std1 or std2 are 0)
        correlation_coefficients[torch.isnan(correlation_coefficients)] = 0
        
        # Calculate loss (1 - correlation_coefficient) to minimize correlation
        loss = - correlation_coefficients
        
        return loss

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
        beta = inference_outputs["beta"]

        dist_s = generative_outputs["dist_s"]
        dist_u = generative_outputs["dist_u"]
        t = generative_outputs["ind_t"]
        t_sort = generative_outputs["t_sort"].T

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -dist_s.log_prob(target_spliced)
        reconst_loss_u = -dist_u.log_prob(target_unspliced)
        
        reconst_loss_target = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)
        
        
        alpha = self.v_encoder.transcription_rate(regulator_spliced)
        target_unspliced_sort = torch.gather(target_unspliced,0,t_sort) ## measure the lag target unspliced readout (t+1)
        alpha_sort = torch.gather(alpha,0,t_sort) ## measure the transcription activity (t)
        alpha_loss = self.alpha_constraint * self.pearson_correlation_loss(target_unspliced_sort,alpha_sort).sum() / alpha.shape[1]

        ## add velocity constraint
        ## regularize the inferred velocity has both negative and positive compartments

        if self.vector_constraint:
            du,_ = self.v_encoder(t = 0, u = unspliced, s = spliced)
            alpha = self.v_encoder.transcription_rate(s = spliced)
            du = alpha - beta * unspliced
            velo_loss =  100 * torch.norm(du,dim=1)
        else:
            velo_loss = 0

        ## add graph constraint
        if self.soft_constraint:
            ## Using norm function to perform graph regularization
            mask_m = 1 - self.v_encoder.mask_m_raw
            graph_constr_loss =  self.lamba * torch.norm(self.v_encoder.fc1.weight * mask_m)
        else:
            graph_constr_loss = 0
        
        Jaco = self.v_encoder.GRN_Jacobian(dist_s.mean.mean(0))
        loss = torch.nn.L1Loss(reduction = "sum")
        L1_loss = (self.lamba2)*loss(Jaco,self.target_m)
    
        ## regularize bias need to be negative
        if self.bias_constraint:
            bias_regularize = torch.norm(self.v_encoder.fc1.bias + 10)
        else:
            bias_regularize = 0
        
        # local loss
        kl_local = kl_divergence_z
        weighted_kl_local = (kl_divergence_z) * kl_weight
        local_loss = torch.mean(reconst_loss_target + weighted_kl_local + velo_loss)

        # total loss
        loss = local_loss + alpha_loss + L1_loss + graph_constr_loss + bias_regularize
        
        loss_recoder = LossOutput(
            loss=loss, reconstruction_loss=reconst_loss_target, kl_local=kl_local
        )

        return loss_recoder

    @auto_move_data
    def get_px(
        self,
        px_rho,
        scale,
        gamma,
        beta,
        alpha_1,
    ) -> torch.Tensor:

        # predict the abundance in induction phase for target genes
        ind_t = self.t_max * px_rho
        n_cells = px_rho.shape[0]
        mean_u, mean_s, index = self._get_induction_unspliced_spliced(
            ind_t
        )
        
        ### only consider induction phase
        scale_u = scale[: self.n_targets, 0].expand(n_cells, self.n_targets).sqrt()
        scale_s = scale[self.n_targets :, 0].expand(n_cells, self.n_targets).sqrt()

        dist_u = Normal(mean_u, scale_u)
        dist_s = Normal(mean_s, scale_s)
        
        return dist_s, dist_u, index, ind_t
    
    def root_time(self, t, root=None):
        """TODO."""
        t_root = 0 if root is None else t[root]
        o = (t >= t_root).int()
        t_after = (t - t_root) * o
        t_origin,_ = torch.max(t_after, axis=0)
        t_before = (t + t_origin) * (1 - o)

        t_rooted = t_after + t_before

        return t_rooted

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
        #t = t.T    
        
        if t.shape[0] > 1:
            ## define parameters
            _, index = torch.sort(t, dim=0)
            index = index.T
            dim = t.shape[0] * t.shape[1]
            t0 = self.t0.repeat(t.shape[0],1)
            dt0 = self.dt0.expand(dim)
            x0 = self.x0.repeat(t.shape[0],1)
            
            t_eval = t.reshape(-1,1)
            t_eval = torch.cat((t0,t_eval),dim=1)
            
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            
            term = to.ODETerm(self.v_batch)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval), dt0 = dt0)
        else:
            t_eval = t
            t_eval = torch.cat((self.t0,t_eval),dim=1)
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            #x0 = x0.double()

            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            sol = solver.solve(to.InitialValueProblem(y0=self.x0, t_eval=t_eval), dt0 = self.dt0)

        ## generate predict results
        # the solved results are saved in sol.ys [the number of subsystems, time_stamps, [u,s]]
        pre_u = sol.ys[:,1:,0]
        pre_s = sol.ys[:,1:,1]     
        
        if t.shape[1] > 1:
            unspliced = pre_u.reshape(-1,t.shape[1])
            spliced = pre_s.reshape(-1,t.shape[1])
        else:
            unspliced = pre_u.ravel()
            spliced = pre_s.ravel()
    
        return unspliced, spliced, index

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

    def freeze_mapping(self):
        for param in self.z_encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False