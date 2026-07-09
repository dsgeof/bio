# BIOINFORMATICS AND DRUG DISCOVERY

## Notebooks ~/notebooks
Exploratory projects and general learning notes are stored in ~/notebooks; specific projects and notes are contained in subdirectories:

- Acetylcholinesterase (AChE) - Neurodegenerative Diseases (e.g. Alzheimer's) 
    /notebooks/target-acetylcholinesterase-alzheimers



A hands-on companion to *Biology for Drug Discovery ML*. By the end you will have:

1. Loaded a **real single-target bioactivity dataset** (BACE-1 inhibitors, 1,513 compounds).
2. Used **RDKit** to parse SMILES, draw molecules, and compute druglikeness (Lipinski).
3. Converted **IC50 ↔ pIC50** by hand so the label stops being a black box.
4. Built a **baseline QSAR model** (Morgan fingerprints → Random Forest).
5. Run the experiment that matters: **random split vs. scaffold split**, and watched the score collapse.

> **The punchline up front:** the *same model* on the *same data* scores **R² ≈ 0.69 on a random split** and **R² ≈ 0.44 on a scaffold split**. The random number is a lie — it's measuring memorization of near-duplicate molecules. The scaffold number is what you'd actually get on new chemistry. Reporting the first one is the single most common way drug-discovery ML quietly fails.

*Runs as-is in Google Colab. ~2 minutes end to end.*