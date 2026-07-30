from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, Crippen
import pandas as pd
import numpy as np


def get_lipinski_descriptors(smiles, verbose=False) -> dict:
    """
    Calculate Lipinski's Rule of Five descriptors for a given SMILES string.
    Parameters:
        smiles (str): The SMILES string of the molecule.
        verbose (bool): Whether to print verbose output.

    Returns:
        dict: A dictionary containing the calculated descriptors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        if verbose:
            print(f"Invalid SMILES: {smiles}")
        return None
    return {
        "MW": Descriptors.MolWt(mol),                                   # Molecular Weight
        "LogP": Descriptors.MolLogP(mol),                               # Lipophilicity - the logarithm of the partition coefficient
        "NumHDonors": Descriptors.NumHDonors(mol),                      # Number of hydrogen bond donors
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),                # Number of hydrogen bond acceptors
        "TPSA": Descriptors.TPSA(mol),                                  # Topological Polar Surface Area
        "RotatableBonds": Descriptors.NumRotatableBonds(mol)            # Number of rotatable bonds
    }


def convert_ic50_to_pic50(df: pd.DataFrame, ic50_col: str = 'standard_value', pic50_col: str = 'pIC50') -> pd.DataFrame:
    """
    Convert IC50 values to pIC50 values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    ic50_col (str): The column name for the IC50 values.
    pic50_col (str): The column name for the pIC50 values.

    Returns:
    pd.DataFrame: The DataFrame with the pIC50 column added.

    Constraints: IC50 values must be positive numbers.
    """
    ic50 = pd.to_numeric(df[ic50_col], errors='coerce')
    # Standardise the standard_value such that vals > 100,000,000 are replaced with 100,000,000
    ic50.clip(lower=None, upper=1e8, inplace=True)
    df[pic50_col] = -np.log10(ic50 * 1e-9)
    return df