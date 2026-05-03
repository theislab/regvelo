from pathlib import Path
from typing import Literal

from anndata import AnnData
import pandas as pd
from scanpy import read

# Remote data URLs
url_adata = "https://drive.google.com/uc?id=1Nzq1F6dGw-nR9lhRLfZdHOG7dcYq7P0i&export=download"
url_grn = "https://drive.google.com/uc?id=1ci_gCwdgGlZ0xSn6gSa_-LlIl9-aDa1c&export=download/"
url_perturb = "https://drive.usercontent.google.com/download?id=1CKADDykadX0CLjYGPgc3YXA-AGRSaA1k&export=download&authuser=1&confirm=t&uuid=37eebdae-5f53-4e53-8d60-a5f52b528720&at=ALBwUgk1uT7pQx7RpXiGyK2yFpBI:1777810924901"

url_adata_murine_processed = "https://drive.usercontent.google.com/download?id=19bNQfW3jMKEEjpjNdUkVd7KDTjJfqxa5&export=download&authuser=1&confirm=t&uuid=4fdf3051-229b-4ce2-b644-cb390424570a&at=APcmpoxgcuZ5r6m6Fb6N_2Og6tEO:1745354679573"
url_adata_murine_normalized = "https://drive.usercontent.google.com/download?id=1xy2FNYi6Y2o_DzXjRmmCtARjoZ97Ro_w&export=download&authuser=1&confirm=t&uuid=12cf5d23-f549-48d9-b7ec-95411a58589f&at=APcmpoyexgouf243lNygF9yRUkmi:1745997349046"
url_adata_murine_velocyto = "https://drive.usercontent.google.com/download?id=18Bhtb7ruoUxpNt8WMYSaJ1RyoiHOCEjd&export=download&authuser=1&confirm=t&uuid=ecc42202-bc82-4ab1-b2c3-bfc31c99f0df&at=APcmpozsh6tBzkv8NSIZW0VipDJa:1745997422108"

url_adata_humanlimb = "https://drive.usercontent.google.com/download?id=17ZaqWx91w--iTtwWMR-SCh6wZO3bzMtM&export=download&authuser=0&confirm=t&uuid=7bdbe7b7-380d-4075-bbe2-1bb65fc0b0a2&at=AN8xHoor6c_hR8_eEd7EENc6dV9M:1750766911809"

url_hindbrain_grn = "https://drive.usercontent.google.com/download?id=11H_7YDhK6EmgZUd6tiyrPBsjl6ASRxH4&export=download&authuser=0&confirm=t&uuid=07c28f5a-bdb5-44ac-93d2-a190e0224e05&at=AN8xHoqX5ROfB0ojYrzisLWDAlnO:1755459638605"
url_adata_hindbrain = "https://drive.usercontent.google.com/download?id=1g0TE-Fx1DxqLmZe5Hp1F6hYWXq4gpa6N&export=download&authuser=1&confirm=t&uuid=1b60b506-43df-4632-a5a6-f4b1f1780a5f&at=AN8xHopDbZ1_ZR20eBBMGi33rse-:1754605312727"
url_adata_hindbrain_processed = "https://drive.usercontent.google.com/download?id=1F2A3kUcHmYJR3v61u6Py3Bmo2Zxl4eUC&export=download&authuser=1&confirm=t&uuid=577ac34b-4fb7-49b9-91e9-9868a9e0f9e4&at=AN8xHorPStlDZePwNziyUuAJmZSQ:1754605238549"

url_adata_schwann = "https://drive.usercontent.google.com/download?id=1ZpG1gz0jjPNuXc_bsz3XInVtHooHU02M&export=download&authuser=1&confirm=t&uuid=eeed9d4b-c938-40c9-af24-b248fd29e161&at=AN8xHoqO3RF2oroUCmn7jFURR16T:1754642525195"




def zebrafish_nc(file_path: str | Path = "data/zebrafish_nc/adata_zebrafish_preprocessed.h5ad") -> AnnData:
    r"""Load zebrafish neural crest (NC) single-cell RNA-seq dataset.

    This dataset contains Smart-seq3 data across seven time points during NC development,
    including four distinct phases:

    1. Specification at the neural plate border (NPB),
    2. Epithelial-to-mesenchymal transition (EMT) from the neural tube,
    3. Peripheral migration, and
    4. Differentiation into distinct cell types.

    Parameters
    ---------
    file_path
        Path to local dataset. Will download from remote URL if not found.

    Returns
    -------
    Annotated data object of zebrafish NC cells.
    """
    adata = read(file_path, backup_url=url_adata, sparse=True, cache=True)
    return adata

def zebrafish_grn(file_path: str | Path = "data/zebrafish_nc/prior_GRN.csv") -> pd.DataFrame:
    r"""Load prior gene regulatory network (GRN) for zebrafish neural crest cells.

    Parameters
    ---------
    file_path
        Path to save the GRN dataset locally and to read it from.

    Returns
    -------
    DataFrame representing the GRN.
    """
    grn = pd.read_csv(url_grn, index_col = 0)
    grn.to_csv(file_path)
    return grn

def zebrafish_perturb(file_path: str | Path = "data/zebrafish_nc/perturbseq_all.h5ad") -> AnnData:                          
      r"""Load zebrafish neural crest Perturb-seq dataset.                      
                                                                                
      This dataset contains single-cell CRISPR screening (Perturb-seq) data from zebrafish neural crest (NC) cells, comprising 12,393 cells and 27,599 genes across 9 pooled samples. Cells were perturbed with sgRNAs targeting 22 transcription factors — including ETS-family members (*fli1a*, *elk3*, *erf*, *etv2*, *ets1*, *elf1*), pigment regulators (*mitfa*, *tfec*, *tfeb*), and additional NC regulators (*nr2f2*, *nr2f5*, *rarga*, *rxraa*, *smarcc1a*, *ebf3a*) — as well as combinatorial double or triple knockouts. Unperturbed control and negative-control cells are also included.
                                                                                
      Cell type annotations span the major stages of NC development:
                                                                                
      - Neural plate border (NPB) progenitors (``NPB_hox``, ``NPB_nohox``),     
      - Migratory NC cells across axial levels (``mNC_arch1``, ``mNC_arch2``, ``mNC_head_mesenchymal``, ``mNC_hox34``, ``mNC_nohox``, ``mNC_trunk``, ``mNC_vagal``),                                                         
      - Differentiating NC cells (``dNC_hox34``, ``dNC_nohox``),                
      - Pigment cell lineages (``Pigment``, ``Pigment_1``, ``Pigment_gch2_high``), and                                                   
      - Mutant populations (``Mutant``, ``Mutant_hox23``).
                                                                                
      The object includes PCA, PHATE, and t-SNE embeddings, as well as imputed latent time and terminal state assignments.                               
                                                                                
      Parameters  
      ----------                                                                
      file_path   
          Path to local dataset. Will download from remote URL if not found.
                                                                                
      Returns
      -------                                                                   
      Annotated data object of zebrafish NC Perturb-seq cells.
      """                                                                       
      adata = read(file_path, backup_url=url_perturb, sparse=True, cache=True)
      return adata

def murine_nc(data_type: Literal["preprocessed", "normalized", "velocyto"] = "preprocessed") -> AnnData:
    r"""Load mouse neural crest single-cell RNA-seq dataset (subset of Qiu et al.).

    The gene regulatory network (GRN) is saved in `adata.uns['skeleton']`, 
    which is learned via pySCENIC.

    Parameters
    ----------
    data_type
        Data version to load. Options are: 
        
        - "preprocessed" (default),
        - "normalized",
        - "velocyto".

    Returns
    -------
    Annotated data object.
    """
    valid_types = ["preprocessed", "normalized", "velocyto"]
    if data_type not in valid_types:
        raise ValueError(f"Invalid data_type: '{data_type}'. Must be one of {valid_types}.")

    file_path = ["data/murine_nc/adata_preprocessed.h5ad","data/murine_nc/adata_gex_normalized.h5ad","data/murine_nc/adata_velocity.h5ad"]

    if data_type == "preprocessed":
        adata = read(file_path[0], backup_url=url_adata_murine_processed, sparse=True, cache=True)
    elif data_type == "normalized":
        adata = read(file_path[1], backup_url=url_adata_murine_normalized, sparse=True, cache=True)
    elif data_type == "velocyto":
        adata = read(file_path[2], backup_url=url_adata_murine_velocyto, sparse=True, cache=True)

    return adata

def human_limb(file_path: str | Path = "data/humanlimb/adata_humanlimb_preprocessed.h5ad") -> AnnData:
    """Load human limb single-cell RNA-seq dataset.

    Parameters
    ----------
    file_path
        Path to local dataset. Will download from remote URL if not found.
        
    Returns
    -------
    Annotated data object of human limb cells.
    """
    adata = read(file_path, backup_url=url_adata_humanlimb, sparse=True, cache=True)
    return adata

def hindbrain(data_type: Literal["original", "preprocessed"] = "preprocessed") -> AnnData:
    """Load human embryonic hindbrain single-cell RNA-seq dataset.

    Parameters
    ----------
    data_type
        Data version to load. Options are: 
        
        - "original",
        - "preprocessed" (default).

    Returns
    -------
    Annotated data object.
    """
    valid_types = ["original", "preprocessed"]
    if data_type not in valid_types:
        raise ValueError(f"Invalid data_type: '{data_type}'. Must be one of {valid_types}.")

    file_path = ["data/hindbrain/adata_hindbrain_original.h5ad", "data/hindbrain/adata_hindbrain_preprocessed.h5ad"]

    if data_type == "original":
        adata = read(file_path[0], backup_url=url_adata_hindbrain, sparse=True, cache=True)
    elif data_type == "preprocessed":
        adata = read(file_path[1], backup_url=url_adata_hindbrain_processed, sparse=True, cache=True)

    return adata

def hindbrain_grn(file_path: str | Path = "data/hindbrain/prior_GRN.csv") -> pd.DataFrame:
    r"""Load prior gene regulatory network (GRN) for human embryonic hindbrain.

    Parameters
    ---------
    file_path
        Path to save the GRN dataset locally and to read it from.

    Returns
    -------
    DataFrame representing the GRN.
    """
    grn = pd.read_csv(url_hindbrain_grn, sep="\t")
    grn.to_csv(file_path)
    return grn

def schwann(file_path: str | Path = "data/schwann/adata_schwann_preprocessed.h5ad") -> AnnData:
    """Load mouse Schwann cell scRNA-seq dataset.

    Parameters
    ----------
    file_path
        Path to local dataset. Will download from remote URL if not found.
    
    Returns
    -------
    Annotated data object of mouse Schwann cells.
    """

    adata = read(file_path, backup_url=url_adata_schwann, sparse=True, cache=True)
    return adata
