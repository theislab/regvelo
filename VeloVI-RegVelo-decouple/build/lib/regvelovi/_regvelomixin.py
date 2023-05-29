import logging
import warnings
from copy import deepcopy
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

from scvi import REGISTRY_KEYS, settings
from scvi.data import _constants
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.model._utils import parse_use_gpu_arg
from scvi.nn import FCLayers
from scvi.model.base import BaseModelClass

#from ._base_model import BaseModelClass
from scvi.model.base._utils import _initialize_model, _load_saved_files, _validate_var_names

logger = logging.getLogger(__name__)

MIN_VAR_NAME_RATIO = 0.8


class RegVeloMixin:
    """Freeze time(encoder and decoder), RBF emulate function"""
    """Only update GRN, kinetic params and switch time"""

    @classmethod
    def load_pretrain_model(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        use_gpu: Optional[Union[str, int, bool]] = None,
        unfrozen: bool = False,
        freeze_expression: bool = True,
        freeze_time_decoder: bool = True,
        freeze_rbf_params: bool = True,
        freeze_grn: bool = False,
        freeze_kinetic_params: bool = False,
    ):
        """Online update of a reference model with scArches algorithm :cite:p:`Lotfollahi21`.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Either an already instantiated model of the same class, or a path to
            saved outputs for reference model.
        
        """
        _, _, device = parse_use_gpu_arg(use_gpu)

        attr_dict, var_names, load_state_dict = _get_loaded_data(
            reference_model, device=device
        )

    
        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError(
                "It appears you are loading a model from a different class."
            )

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        model.model_stage = "refining"

        model.to_device(device)
        # import parameters
        model.module.load_state_dict(load_state_dict)
        model.module.eval()

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_expression = freeze_expression,
            freeze_time_decoder = freeze_time_decoder,
            freeze_rbf_params = freeze_rbf_params,
            freeze_grn = freeze_grn,
            freeze_kinetic_params = freeze_kinetic_params,
        )
        model.is_trained_ = False

        return model



def _set_params_online_update(
    module,
    unfrozen,
    freeze_expression,
    freeze_time_decoder,
    freeze_rbf_params,
    freeze_grn,
    freeze_kinetic_params,
):
    """Freeze parts of network for scArches."""
    # do nothing if unfrozen
    if unfrozen:
        return

    def requires_grad(key):
        mod_name = key.split(".")[0]
        # linear weights and bias that need grad
        one = two = False
        if freeze_expression:
            one = "z_encoder" in key
        if freeze_time_decoder:
            two = "decoder" in key
        
        if one or two:
            return False
        else:
            return True

    for key, par in module.named_parameters():
        if requires_grad(key):
            par.requires_grad = True
        else:
            par.requires_grad = False
    
    ### tweaking the dynamics parameters
    if freeze_rbf_params:
        module.v_encoder.log_h.requires_grad = False
        module.v_encoder.log_phi.requires_grad = False
        module.v_encoder.tau.requires_grad = False
        module.v_encoder.o.requires_grad = False
        module.switch_time_unconstr.requires_grad = False
    
    if freeze_kinetic_params:
        module.v_encoder.beta_mean_unconstr.requires_grad = False
        module.v_encoder.gamma_mean_unconstr.requires_grad = False
        module.v_encoder.alpha_unconstr_max.requires_grad = False
        module.v_encoder.alpha_unconstr_bias.requires_grad = False
        module.scale_unconstr_targets.requires_grad = False
    
    if freeze_grn:
        module.v_encoder.grn.requires_grad = False
        

def _get_loaded_data(reference_model, device=None):
    if isinstance(reference_model, str):
        attr_dict, var_names, load_state_dict, _ = _load_saved_files(
            reference_model, load_adata=False, map_location=device
        )
    else:
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
        var_names = reference_model.adata.var_names
        load_state_dict = deepcopy(reference_model.module.state_dict())

    return attr_dict, var_names, load_state_dict