import numpy as np
import pandas as pd
import scipy
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
from scipy.stats import ttest_rel

import cellrank as cr
import scanpy as sc
import scvi
from ._model import REGVELOVI
import scvelo as scv


from .tools.set_output import set_output
from .tools._tsi import get_tsi_score


# Packages used for data type validation
from typing import List, Optional, Union, Dict
from anndata import AnnData

class ModelComparison:
    """Compare different types of RegVelo models : cite:p: `Wang2025`.
        
        This class is used to compare different RegVelo models with different optimization mode (soft, hard, soft_regularized) and under different normalization factor lamda2.
        User can evaluate and visulize competence of different types of models based on various side information (Real time, Pseudo Time, Stemness Score, Terminal States Identification, Cross Boundary Correctness) of cell.
        Finally, it will return a barplot with best performed model marked, and its performance will also be highlighted by significance test.
        
        Examples
        ----------
        See notebook.
        
        """
    def __init__(self,
                 terminal_states: List = None,
                 state_transition: Dict = None,
                 n_states: int = None):
        """Initialize parameters in comparision object.
        
        Parameters
        ----------
        terminal_states
            A list records all terminal states among all cell types. 
            This parameter is not necessary if you don't use TSI as side_information. Please make sure they are consistent with information stored in 'side_key' under TSI mode.
        state_transition
            A dict records all possible state transition relationships among all cell types.
            This parameter is not necessary if you don't use CBC as side_information. Please make sure they are consistent with information stored in 'side_key' under CBC mode.
        n_state
            A integer provide the number of cell clusters in total.
            This parameter is not necessary if you don't use TSI as side_information.
            
        Returns
        ----------
        An comparision object. You can deal with more operations as follows.

        """
        self.TERMINAL_STATES = terminal_states
        self.STATE_TRANSITION = state_transition
        self.N_STATES = n_states
        
        self.METHOD = None
        self.MODEL_LIST = None
        
        self.side_info_dict = {
            'Pseudo_Time':'dpt_pseudotime',
            'Stemness_Score': 'ct_score',
            'Real_Time': None,
            'TSI': None,
            'CBC': None 
        }
        self.MODEL_TRAINED = {}
    
    def validate_input(self,
                       adata: AnnData,
                       model_list: List[str] = None,
                       side_information: str = None,
                       lam2: Union[List[float], float] = None,
                       side_key: str = None) -> None:
        
        # 1.Validate adata
        if not isinstance(adata, AnnData):
            raise TypeError(f"Expected AnnData object, got {type(adata).__name__}")
        layers = ['Ms', 'Mu']
        for layer in layers:
            if layer not in adata.layers:
                raise ValueError(f"Missing required layer: {layer}")
        if 'skeleton' not in adata.uns:
            raise ValueError("Missing required 'skeleton' in adata.uns")
        
        if 'TF' not in adata.var:
            raise ValueError("Missing required 'TF' column in adata.var") 
        
        # 2.Validate Model_list
        if model_list is not None:
            valid_models = ['hard', 'soft', 'soft_regularized']
            if not isinstance(model_list, list) or len(model_list) == 0:
                raise ValueError("model_list must be a non-empty list")
            for model in model_list:
                if model not in valid_models:
                    raise ValueError(f"Invalid model: {model}. Valid models are {valid_models}")
                if model == 'soft_regularized' and lam2 is None:
                    raise ValueError(f"Under 'soft_regularized' mode, lam2 must be given")
            if lam2 is not None:
                if not isinstance(lam2, (float, list)):
                    raise TypeError('lam2 must be a float or a list of floats')
                if isinstance(lam2, list):
                    if len(lam2) == 0:
                        raise ValueError('lam2 list can not be empty')
                    for num in lam2:
                        if not isinstance(num, float):
                            raise ValueError('All elements in lam2 list must be float')
                        if not 0.0 < num <= 1.0:
                            raise ValueError('lam2 is expected to be in range of (0,1)')
                    
        # 3.Validate side_information
        if side_information is not None:
            if not isinstance(side_information, str):
                raise TypeError(f"side_information must be a string")
            
            if side_information not in self.side_info_dict.keys():
                raise ValueError(f"Valid side_information are {self.side_info_dict.keys()}")
            
            if side_key is not None:
                if not isinstance(side_key, str):
                    raise TypeError(f"side_key must be a string")
                if side_key not in adata.obs:
                    raise TypeError(f"side_key: {side_key} not found in adata.obs.")
            if side_key is None:
                side_key = self.side_info_dict[side_information]
                if side_key is not None:
                    if side_key not in adata.obs:
                        raise TypeError(f"Default side_key: {side_key} not found in adata.obs, please input it manualy with parameter: side_key")
    
    def min_max_scaling(self,x):
        return (x - np.min(x)) / (np.max(x) - np.min(x)) 
    
    def train(
        self,
        adata: AnnData,
        model_list: List[str],
        lam2: Union[List[float], float] = None,
        n_repeat: int = 1,
        batch_size=None
    ) -> List:
        """Train all the possible models given by users, and stored them in a dictionary, where users can reach them easily and deal with them in batch.If there are already model trained and saved before, they won't be removed.
        
        Parameters
        ----------
        adata
            The annotated data matrix. After input of adata, the object will store it as self variable.
        model_list
            The list of valid model type, including 'Soft', 'Hard', 'Soft_regularized'
        lam2
            Normalization factor used under 'soft_regularized' mode. A float or a list of float number in range of (0,1)
        batch_size
            Training batch size. This enable user to adjust batch size according to data size.
            
        Returns
        ----------
        A dictionary key names, represent to all models trained in this step. 
        
        """
        self.validate_input(adata, model_list = model_list, lam2 = lam2)
        self.ADATA = adata
        
        if not isinstance(n_repeat, int) or n_repeat < 1:
            raise ValueError("n_repeat must be a positive integer")
        
        W = adata.uns["skeleton"].copy()
        W = torch.tensor(np.array(W)).int()
        TF = adata.var_names[adata.var['TF']]
        REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
        
        
        for model in model_list:
            for nrun in range(n_repeat):
                scvi.settings.seed = nrun
                if model == 'soft_regularized':
                    if isinstance(lam2,list):
                        for lambda2 in lam2:
                            vae = REGVELOVI(
                                adata,
                                W=W.T,
                                regulators=TF,
                                lam2 = lambda2
                            )
                            vae.train(batch_size=batch_size)
                            self.MODEL_TRAINED[f"{model}\nlam2:{lambda2}_{nrun}"] = vae
                    else:
                        vae = REGVELOVI(
                            adata,
                            W=W.T,
                            regulators=TF,
                            lam2=lam2
                        )
                        vae.train(batch_size=batch_size)
                        self.MODEL_TRAINED[f"{model}_{nrun}"] = vae
                else:
                    vae = REGVELOVI(
                        adata,
                        W=W.T,
                        regulators=TF,
                        soft_constraint=(model == 'soft')
                    )
                    vae.train(batch_size=batch_size)
                    self.MODEL_TRAINED[f"{model}_{nrun}"] = vae

        return list(self.MODEL_TRAINED.keys())
    
    def evaluate(
        self,
        side_information: str,
        side_key:str = None
        ) -> pd.DataFrame:
        """Evaluate all of trained model under one specific side_information mode, For example, if user know the exact time or stage of cells, user can choose 'Real_Time' as reference; If users has used Pseudotime calculator such as CellRank beforehand, they can also choose 'Pseudo_Time' as reference.
        
        Parameters
        ----------
        side_information
            User can choose perspectives to compare RegVelo models, including 'Real_Time', 'Pseudo_Time', 'Stemness_Score','TSI','CBC'.
        side_key
            Column name of adata.obs which used to store information of selected side_information. For 'Pseudo_Time' and 'Stemness_Score', we provide default side_key, but you can also choose your own side_key as input.
        
        Returns
        ----------
        A dataframe records evaluation performance of all models.
        """
        self.validate_input(self.ADATA, side_information=side_information, side_key=side_key)
        correlations = []
    
        for model, vae in self.MODEL_TRAINED.items():
            set_output(self.ADATA, vae, n_samples = 30, batch_size = self.ADATA.n_obs)
            fit_t_mean = self.ADATA.layers['fit_t'].mean(axis = 1)
            self.ADATA.obs["latent_time"] = self.min_max_scaling(fit_t_mean)
            corr = np.abs(self.calculate(
                self.ADATA, side_information, side_key
            ))
            correlations.append({
                'Model': model[:-2],
                'Corr':corr,
                'Run':model[-1]
            })
        df_name = f"df_{side_information}"
        df = pd.DataFrame(correlations)
        setattr(self, df_name, df)
        return df_name, df
    
    def calculate(
        self,
        adata: AnnData,
        side_information: str,
        side_key: str = None
    ):
        if side_information in ['Pseudo_Time', 'Stemness_Score', 'Real_Time']:
            if side_information in ['Pseudo_Time', 'Stemness_Score'] and side_key is None:
                side_key = self.side_info_dict[side_information]
            return scipy.stats.spearmanr(self.ADATA.obs[side_key].values, self.ADATA.obs['latent_time'])[0]
        elif side_information == 'TSI':
            thresholds = np.linspace(0.1,1,21)[:20]
            vk = cr.kernels.VelocityKernel(self.ADATA)
            vk.compute_transition_matrix()
            ck = cr.kernels.ConnectivityKernel(self.ADATA).compute_transition_matrix()
            kernel = 0.8 * vk + 0.2 * ck
            estimator = cr.estimators.GPCCA(kernel)
            estimator.compute_macrostates(n_states=self.N_STATES, n_cells=30, cluster_key=side_key)
            return np.mean(get_tsi_score(self.ADATA, thresholds, side_key, self.TERMINAL_STATES, estimator))
        elif side_information == 'CBC':
            self.ADATA.obs['CBC_key'] = self.ADATA.obs[side_key].astype(str)
            vk = cr.kernels.VelocityKernel(self.ADATA)
            vk.compute_transition_matrix()
            ck = cr.kernels.ConnectivityKernel(self.ADATA).compute_transition_matrix()
            kernel = 0.8 * vk + 0.2 * ck
            cbc_values = []
            for source, target in self.STATE_TRANSITION:
                cbc = kernel.cbc(source = source, target=target, cluster_key='CBC_key', rep = 'X_pca')
                cbc_values.append(np.mean(cbc))
            return np.mean(cbc_values)
    
    def plot_results(
        self,
        side_information,
        figsize = (6, None),
        palette = 'lightpink'
    ):
        """Visualize comparision result by barplot with scatters. The significant mark will only show with n_repeats more than 3, and p < 0.05.
        
        Paramters
        ----------
        side_information
            Here choose the side_information you wish to visulize, which must be performed in 'evaluation' step in advance.
        figsize
            You can choose the size of figure. Default is (6,None), which means the height of the plot are set to change with the number of models.
        palette
            You can choose the color of barplot.
        
        Returns
        ----------
        A barplot with scatters, represent performance of all models.
        """
        df_name = f"df_{side_information}"
        data = getattr(self, df_name)
        
        model_order = data.groupby('Model')['Corr'].mean().sort_values(ascending=False).index.tolist()
        num_models = len(model_order)
        fig_height = 2 + num_models * 0.5
        figsize = (figsize[0], fig_height)

        sns.set(style='whitegrid', font_scale=1.2)
        fig, ax = plt.subplots(figsize=figsize)
        

        sns.barplot(
            y="Model",
            x="Corr",
            data=data,
            order = model_order,
            width=0.3,
            ci="sd",
            capsize=0.1,
            errwidth=2,
            color=palette,
            ax = ax)
        sns.stripplot(
            y="Model",
            x="Corr",
            data=data,
            order = model_order,
            dodge=True,
            jitter=0.25,
            color="black",
            size = 4,
            alpha=0.8,
            ax = ax
        )
        
        model_means = data.groupby('Model')['Corr'].mean()
        ref_model = model_means.idxmax()
        
        ref_data = data[data["Model"] == ref_model]["Corr"]
        y_ticks = ax.get_yticks()
        model_positions = dict(zip(model_order, y_ticks))
        
        base_line = 1
        for target_model in model_order:
            if target_model == ref_model:
                continue
            target_data = data[data["Model"] == target_model]["Corr"]
            
            if len(target_data) < 3:
                continue
            
            try:
                t_stat, p_value = scipy.stats.ttest_rel(
                    ref_data, 
                    target_data, 
                    alternative="greater"
                )
            except ValueError as e:
                print(f"Significance test: {ref_model} vs {target_model} failed：{str(e)}")
                continue

            if p_value < 0.05:
                base_line += 0.05
                significance = self.get_significance(p_value)
                self._draw_significance_marker(
                    ax=ax,
                    base_line=base_line,
                    start=model_positions[ref_model],
                    end=model_positions[target_model],
                    significance=significance,
                    bracket_height=0.05
                )
        
        ax.set_title(
            f"Prediction based on {side_information}", 
            fontsize=12,
            wrap = True
        )
        ax.set_ylabel('')
        ax.set_xlabel(
            "Prediction Score",
            fontsize=12,
            labelpad=10
        )
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.show()
        
    def get_significance(self, pvalue):
        if pvalue < 0.001:
            return "***"
        elif pvalue < 0.01:
            return "**"
        elif pvalue < 0.05:
            return "*"
        else:
            return "ns"
    
    def _draw_significance_marker(
            self,
            ax,
            base_line,
            start,
            end,
            significance,
            bracket_height=0.05,
            linewidth=1.2,
            text_offset=0.03):
        
        if start > end:
            start, end = end, start
        x_max = ax.get_xlim()[1]
        bracket_level = base_line + bracket_height
        
        ax.plot(
            [bracket_level-0.02, bracket_level, bracket_level, bracket_level-0.02],
            [start, start, end, end],
            color='black',
            lw=linewidth,
            solid_capstyle="butt",
            clip_on=False
        )
        
        ax.text(
            bracket_level + text_offset,
            (start + end)/2,
            significance,
            ha='center',
            va='baseline',
            color='black',
            fontsize=10,
            fontweight='bold',
            rotation = 90
        )     
