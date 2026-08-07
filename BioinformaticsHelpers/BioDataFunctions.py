from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem, Crippen, rdFMCS, Geometry
from copy import deepcopy
import pandas as pd
import numpy as np


def get_lipinski_descriptors(smiles: str, verbose: bool = False) -> dict:
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


def draw_ranked_molecules(molecules: pd.DataFrame, sort_by_column: str) -> Draw.MolsToGridImage:
    """
    Draw molecules sorted by a given column.

    Parameters
    ----------
    molecules : pandas.DataFrame
        Molecules (with "ROMol" and "name" columns and a column to sort by.
    sort_by_column : str
        Name of the column used to sort the molecules by.

    Returns
    -------
    Draw.MolsToGridImage
        2D visualization of sorted molecules.
    """

    molecules_sorted = molecules.sort_values([sort_by_column], ascending=False).reset_index()
    return Draw.MolsToGridImage(
        molecules_sorted["ROMol"],
        legends=[
            f"#{index+1} {molecule['name']}, similarity={molecule[sort_by_column]:.2f}"
            for index, molecule in molecules_sorted.iterrows()
        ],
        molsPerRow=3,
        subImgSize=(450, 150),
    )

def get_enrichment_data(molecules: pd.DataFrame, similarity_measure: str, pic50_cutoff: float) -> pd.DataFrame:
    """
    Calculates x and y values for enrichment plot:
        x - % ranked dataset
        y - % true actives identified

    Parameters
    ----------
    molecules : pandas.DataFrame
        Molecules with similarity values to a query molecule.
    similarity_measure : str
        Column name which will be used to sort the DataFrame．
    pic50_cutoff : float
        pIC50 cutoff value used to discriminate active and inactive molecules.

    Returns
    -------
    pandas.DataFrame
        Enrichment data: Percentage of ranked dataset by similarity vs. percentage of identified true actives.
    """

    # Get number of molecules in data set
    molecules_all = len(molecules)

    # Get number of active molecules in data set
    actives_all = sum(molecules["pIC50"] >= pic50_cutoff)

    # Initialize a list that will hold the counter for actives and molecules while iterating through our dataset
    actives_counter_list = []

    # Initialize counter for actives
    actives_counter = 0

    # Note: Data must be ranked for enrichment plots:
    # Sort molecules by selected similarity measure
    molecules.sort_values([similarity_measure], ascending=False, inplace=True)

    # Iterate over the ranked dataset and check each molecule if active (by checking bioactivity)
    for value in molecules["pIC50"]:
        if value >= pic50_cutoff:
            actives_counter += 1
        actives_counter_list.append(actives_counter)

    # Transform number of molecules into % ranked dataset
    molecules_percentage_list = [i / molecules_all for i in range(1, molecules_all + 1)]

    # Transform number of actives into % true actives identified
    actives_percentage_list = [i / actives_all for i in actives_counter_list]

    # Generate DataFrame with x and y values as well as label
    enrichment = pd.DataFrame(
        {
            "% ranked dataset": molecules_percentage_list,
            "% true actives identified": actives_percentage_list,
        }
    )
    return enrichment


def calculate_enrichment_factor_random(ranked_dataset_percentage_cutoff: float) -> float:
    """
    Get the random enrichment factor for a given percentage of the ranked dataset.

    Parameters
    ----------
    ranked_dataset_percentage_cutoff : float or int
        Percentage of ranked dataset to be included in enrichment factor calculation.

    Returns
    -------
    float
        Random enrichment factor.
    """

    enrichment_factor_random = round(float(ranked_dataset_percentage_cutoff), 1)
    return enrichment_factor_random


def calculate_enrichment_factor_optimal(molecules: pd.DataFrame, ranked_dataset_percentage_cutoff: float, pic50_cutoff: float) -> float:
    """
    Get the optimal random enrichment factor for a given percentage of the ranked dataset.

    Parameters
    ----------
    molecules : pandas.DataFrame
        the DataFrame with all the molecules and pIC50.
    ranked_dataset_percentage_cutoff : float or int
        Percentage of ranked dataset to be included in enrichment factor calculation.
    activity_cutoff: float
        pIC50 cutoff value used to discriminate active and inactive molecules

    Returns
    -------
    float
        Optimal enrichment factor.
    """

    ratio = sum(molecules["pIC50"] >= pic50_cutoff) / len(molecules) * 100
    if ranked_dataset_percentage_cutoff <= ratio:
        enrichment_factor_optimal = round(100 / ratio * ranked_dataset_percentage_cutoff, 1)
    else:
        enrichment_factor_optimal = 100.0
    return enrichment_factor_optimal


def tanimoto_distance_matrix(fp_list: list) -> list:
    """Calculate Tanimoto similarity and distance matrix for fingerprint list"""
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

from rdkit.DataStructs import BulkTanimotoSimilarity


def tanimoto_distance_matrix(fp_list):
    """Calculate condensed distance matrix for fingerprint list."""
    n = len(fp_list)
    dissimilarity_matrix = []

    for i in range(1, n):
        # Calculates similarity of fp_list[i] against fp_list[0...i-1]
        similarities = BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1.0 - x for x in similarities])

    return dissimilarity_matrix


# Helper: Draw molecules with highlighted MCS
def highlight_molecule_substructure(molecules: list, mcs: rdFMCS.MCSResult, number: int, label: bool=True, same_orientation: bool=True, **kwargs):
    """Highlight the MCS in our query molecules 
    
    Args:
        molecules (list): List of RDKit molecule objects
        mcs (rdFMCS.MCSResult): MCS result object from rdFMCS
        number (int): Number of molecules to display
        label (bool, optional): Whether to label the molecules with their names. Defaults to True.
        same_orientation (bool, optional): Whether to align the molecules in the same orientation. Defaults to True.
        **kwargs: Additional keyword arguments for Draw.MolsToGridImage
    Returns:
        PIL.Image: Image of the molecules with highlighted MCS
    """
    molecules = deepcopy(molecules)
    # convert MCS to molecule
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    # find the matching atoms in each molecule
    matching = [molecule.GetSubstructMatch(pattern) for molecule in molecules[:number]]

    legends = None
    if label:
        legends = [molname.GetProp("_Name") for molname in molecules]

    # Align by matched substructure so they are depicted in the same orientation
    # Adapted from: https://gist.github.com/greglandrum/82d9a86acb3b00d3bb1df502779a5810
    if same_orientation:
        mol, match = molecules[0], matching[0]
        AllChem.Compute2DCoords(mol)
        coords = [mol.GetConformer().GetAtomPosition(x) for x in match]
        coords2D = [Geometry.Point2D(pt.x, pt.y) for pt in coords]
        for mol, match in zip(molecules[1:number], matching[1:number]):
            if not match:
                continue
            coord_dict = {match[i]: coord for i, coord in enumerate(coords2D)}
            AllChem.Compute2DCoords(mol, coordMap=coord_dict)

    return Draw.MolsToGridImage(
        molecules[:number],
        legends=legends,
        molsPerRow=5,
        highlightAtomLists=matching[:number],
        subImgSize=(200, 200),
        **kwargs,
    )