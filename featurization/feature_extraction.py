from typing import Dict, Iterable, Union
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Fragments,
    GraphDescriptors,
    Lipinski,
    MolSurf,
)
from rdkit.Chem.EState import EState
from rdkit.Chem.EState.EState_VSA import (
    EState_VSA1,
    EState_VSA2,
    EState_VSA3,
    EState_VSA4,
    EState_VSA5,
    EState_VSA6,
    EState_VSA7,
    EState_VSA8,
    EState_VSA9,
    EState_VSA10,
    EState_VSA11,
    VSA_EState1,
    VSA_EState2,
    VSA_EState3,
    VSA_EState4,
    VSA_EState5,
    VSA_EState6,
    VSA_EState7,
    VSA_EState8,
    VSA_EState9,
    VSA_EState10,
)

try:
    from rdkit.Chem import rdFingerprintGenerator
    _MORGAN_GENERATOR_AVAILABLE = True
except ImportError:
    _MORGAN_GENERATOR_AVAILABLE = False


ECFP4_RADIUS = 2
ECFP4_N_BITS = 2048

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned_data"
FEATURIZED_DIR = PROJECT_ROOT / "data" / "featurized_data"
# Default when --input is omitted. Other CSVs in cleaned_data are selected with --input.
DEFAULT_CLEANED_NAME = "balanced_pulmonary_adr_dataset.csv"

# Columns written by data_merging/02_build_dataset.py. Kept in the feature CSVs
# when the input has no single 'class' column.
PULMONARY_LABEL_COLS = [
    "Label_ILD",
    "Label_Edema",
    "Label_Pleural",
    "Label_Embolism",
]

OUTPUT_FILENAMES = {
    "rdkit": "rdkit_md_196.csv",
    "ecfp4": "ecfp4_2048.csv",
    "both": "both.csv",
}


def ecfp4_column_names(n_bits: int = ECFP4_N_BITS):
    return [f"ecfp_bit{i}" for i in range(n_bits)]


ECFP4_COLUMNS = ecfp4_column_names()


class molecular_descriptor:
    """Utility class for extracting RDKit molecular descriptors."""

    def __init__(self, smiles: str):
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError("Invalid SMILES string provided.")

    def get_chi(self) -> Dict[str, float]:
        """Chi molecular connectivity indices."""
        return {
            "BalabanJ": GraphDescriptors.BalabanJ(self.mol),
            "BertzCT": GraphDescriptors.BertzCT(self.mol),
            "Chi0": GraphDescriptors.Chi0(self.mol),
            "Chi0n": GraphDescriptors.Chi0n(self.mol),
            "Chi0v": GraphDescriptors.Chi0v(self.mol),
            "Chi1": GraphDescriptors.Chi1(self.mol),
            "Chi1n": GraphDescriptors.Chi1n(self.mol),
            "Chi1v": GraphDescriptors.Chi1v(self.mol),
            "Chi2n": GraphDescriptors.Chi2n(self.mol),
            "Chi2v": GraphDescriptors.Chi2v(self.mol),
            "Chi3n": GraphDescriptors.Chi3n(self.mol),
            "Chi3v": GraphDescriptors.Chi3v(self.mol),
            "Chi4n": GraphDescriptors.Chi4n(self.mol),
            "Chi4v": GraphDescriptors.Chi4v(self.mol),
        }

    def get_estate_vsa(self) -> Dict[str, float]:
        """EState VSA (Van der Waals Surface Area) descriptors."""
        return {
            "EState_VSA1": EState_VSA1(self.mol),
            "EState_VSA2": EState_VSA2(self.mol),
            "EState_VSA3": EState_VSA3(self.mol),
            "EState_VSA4": EState_VSA4(self.mol),
            "EState_VSA5": EState_VSA5(self.mol),
            "EState_VSA6": EState_VSA6(self.mol),
            "EState_VSA7": EState_VSA7(self.mol),
            "EState_VSA8": EState_VSA8(self.mol),
            "EState_VSA9": EState_VSA9(self.mol),
            "EState_VSA10": EState_VSA10(self.mol),
            "EState_VSA11": EState_VSA11(self.mol),
        }

    def get_topological(self) -> Dict[str, float]:
        """Topological descriptors - basic molecular structure properties."""
        return {
            "NumAtoms": float(self.mol.GetNumAtoms()),
            "NumBonds": float(self.mol.GetNumBonds()),
        }

    def get_mol(self) -> Dict[str, float]:
        """General molecular descriptors."""
        return {
            "ExactMolWt": Descriptors.ExactMolWt(self.mol),
            "FractionCSP3": Lipinski.FractionCSP3(self.mol),
            "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(self.mol),
            "Ipc": Descriptors.Ipc(self.mol),
            "Kappa1": GraphDescriptors.Kappa1(self.mol),
            "Kappa2": GraphDescriptors.Kappa2(self.mol),
            "Kappa3": GraphDescriptors.Kappa3(self.mol),
            "LabuteASA": MolSurf.LabuteASA(self.mol),
            "MaxAbsEStateIndex": EState.MaxAbsEStateIndex(self.mol),
            "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(self.mol),
            "MaxEStateIndex": EState.MaxEStateIndex(self.mol),
            "MaxPartialCharge": Descriptors.MaxPartialCharge(self.mol),
            "MinAbsEStateIndex": EState.MinAbsEStateIndex(self.mol),
            "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(self.mol),
            "MinEStateIndex": EState.MinEStateIndex(self.mol),
            "MinPartialCharge": Descriptors.MinPartialCharge(self.mol),
            "MolLogP": Crippen.MolLogP(self.mol),
            "MolMR": Crippen.MolMR(self.mol),
            "MolWt": Descriptors.MolWt(self.mol),
        }

    def get_lipinski(self) -> Dict[str, float]:
        """Lipinski's Rule of Five and related descriptors."""
        return {
            "NHOHCount": Lipinski.NHOHCount(self.mol),
            "NOCount": Lipinski.NOCount(self.mol),
            "NumAliphaticCarbocycles": Lipinski.NumAliphaticCarbocycles(self.mol),
            "NumAliphaticHeterocycles": Lipinski.NumAliphaticHeterocycles(self.mol),
            "NumAliphaticRings": Lipinski.NumAliphaticRings(self.mol),
            "NumAromaticCarbocycles": Lipinski.NumAromaticCarbocycles(self.mol),
            "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(self.mol),
            "NumAromaticRings": Lipinski.NumAromaticRings(self.mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(self.mol),
            "NumHDonors": Lipinski.NumHDonors(self.mol),
            "NumHeteroatoms": Lipinski.NumHeteroatoms(self.mol),
            "NumRadicalElectrons": Descriptors.NumRadicalElectrons(self.mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(self.mol),
            "NumSaturatedCarbocycles": Lipinski.NumSaturatedCarbocycles(self.mol),
            "NumSaturatedHeterocycles": Lipinski.NumSaturatedHeterocycles(self.mol),
            "NumSaturatedRings": Lipinski.NumSaturatedRings(self.mol),
            "NumValenceElectrons": Descriptors.NumValenceElectrons(self.mol),
            "RingCount": Lipinski.RingCount(self.mol),
        }

    def get_peoe_vsa(self) -> Dict[str, float]:
        """PEOE VSA (Partial Equalization of Orbital Electronegativities) descriptors."""
        return {
            "PEOE_VSA1": MolSurf.PEOE_VSA1(self.mol),
            "PEOE_VSA2": MolSurf.PEOE_VSA2(self.mol),
            "PEOE_VSA3": MolSurf.PEOE_VSA3(self.mol),
            "PEOE_VSA4": MolSurf.PEOE_VSA4(self.mol),
            "PEOE_VSA5": MolSurf.PEOE_VSA5(self.mol),
            "PEOE_VSA6": MolSurf.PEOE_VSA6(self.mol),
            "PEOE_VSA7": MolSurf.PEOE_VSA7(self.mol),
            "PEOE_VSA8": MolSurf.PEOE_VSA8(self.mol),
            "PEOE_VSA9": MolSurf.PEOE_VSA9(self.mol),
            "PEOE_VSA10": MolSurf.PEOE_VSA10(self.mol),
            "PEOE_VSA11": MolSurf.PEOE_VSA11(self.mol),
            "PEOE_VSA12": MolSurf.PEOE_VSA12(self.mol),
            "PEOE_VSA13": MolSurf.PEOE_VSA13(self.mol),
            "PEOE_VSA14": MolSurf.PEOE_VSA14(self.mol),
        }

    def get_smr_vsa(self) -> Dict[str, float]:
        """SMR VSA (Molar Refractivity) descriptors."""
        return {
            "SMR_VSA1": MolSurf.SMR_VSA1(self.mol),
            "SMR_VSA2": MolSurf.SMR_VSA2(self.mol),
            "SMR_VSA3": MolSurf.SMR_VSA3(self.mol),
            "SMR_VSA4": MolSurf.SMR_VSA4(self.mol),
            "SMR_VSA5": MolSurf.SMR_VSA5(self.mol),
            "SMR_VSA6": MolSurf.SMR_VSA6(self.mol),
            "SMR_VSA7": MolSurf.SMR_VSA7(self.mol),
            "SMR_VSA8": MolSurf.SMR_VSA8(self.mol),
            "SMR_VSA9": MolSurf.SMR_VSA9(self.mol),
            "SMR_VSA10": MolSurf.SMR_VSA10(self.mol),
        }

    def get_slogp_vsa(self) -> Dict[str, float]:
        """SlogP VSA (LogP) descriptors."""
        return {
            "SlogP_VSA1": MolSurf.SlogP_VSA1(self.mol),
            "SlogP_VSA2": MolSurf.SlogP_VSA2(self.mol),
            "SlogP_VSA3": MolSurf.SlogP_VSA3(self.mol),
            "SlogP_VSA4": MolSurf.SlogP_VSA4(self.mol),
            "SlogP_VSA5": MolSurf.SlogP_VSA5(self.mol),
            "SlogP_VSA6": MolSurf.SlogP_VSA6(self.mol),
            "SlogP_VSA7": MolSurf.SlogP_VSA7(self.mol),
            "SlogP_VSA8": MolSurf.SlogP_VSA8(self.mol),
            "SlogP_VSA9": MolSurf.SlogP_VSA9(self.mol),
            "SlogP_VSA10": MolSurf.SlogP_VSA10(self.mol),
            "SlogP_VSA11": MolSurf.SlogP_VSA11(self.mol),
            "SlogP_VSA12": MolSurf.SlogP_VSA12(self.mol),
        }

    def get_vsa_estate(self) -> Dict[str, float]:
        """VSA EState descriptors."""
        return {
            "TPSA": Descriptors.TPSA(self.mol),
            "VSA_EState1": VSA_EState1(self.mol),
            "VSA_EState2": VSA_EState2(self.mol),
            "VSA_EState3": VSA_EState3(self.mol),
            "VSA_EState4": VSA_EState4(self.mol),
            "VSA_EState5": VSA_EState5(self.mol),
            "VSA_EState6": VSA_EState6(self.mol),
            "VSA_EState7": VSA_EState7(self.mol),
            "VSA_EState8": VSA_EState8(self.mol),
            "VSA_EState9": VSA_EState9(self.mol),
            "VSA_EState10": VSA_EState10(self.mol),
        }

    def get_fragments(self) -> Dict[str, int]:
        """Molecular fragment descriptors."""
        return {
            "fr_Al_COO": Fragments.fr_Al_COO(self.mol),
            "fr_Al_OH": Fragments.fr_Al_OH(self.mol),
            "fr_Al_OH_noTert": Fragments.fr_Al_OH_noTert(self.mol),
            "fr_ArN": Fragments.fr_ArN(self.mol),
            "fr_Ar_COO": Fragments.fr_Ar_COO(self.mol),
            "fr_Ar_N": Fragments.fr_Ar_N(self.mol),
            "fr_Ar_NH": Fragments.fr_Ar_NH(self.mol),
            "fr_Ar_OH": Fragments.fr_Ar_OH(self.mol),
            "fr_COO": Fragments.fr_COO(self.mol),
            "fr_COO2": Fragments.fr_COO2(self.mol),
            "fr_C_O": Fragments.fr_C_O(self.mol),
            "fr_C_O_noCOO": Fragments.fr_C_O_noCOO(self.mol),
            "fr_C_S": Fragments.fr_C_S(self.mol),
            "fr_HOCCN": Fragments.fr_HOCCN(self.mol),
            "fr_Imine": Fragments.fr_Imine(self.mol),
            "fr_NH0": Fragments.fr_NH0(self.mol),
            "fr_NH1": Fragments.fr_NH1(self.mol),
            "fr_NH2": Fragments.fr_NH2(self.mol),
            "fr_N_O": Fragments.fr_N_O(self.mol),
            "fr_Ndealkylation1": Fragments.fr_Ndealkylation1(self.mol),
            "fr_Ndealkylation2": Fragments.fr_Ndealkylation2(self.mol),
            "fr_Nhpyrrole": Fragments.fr_Nhpyrrole(self.mol),
            "fr_SH": Fragments.fr_SH(self.mol),
            "fr_aldehyde": Fragments.fr_aldehyde(self.mol),
            "fr_alkyl_carbamate": Fragments.fr_alkyl_carbamate(self.mol),
            "fr_alkyl_halide": Fragments.fr_alkyl_halide(self.mol),
            "fr_allylic_oxid": Fragments.fr_allylic_oxid(self.mol),
            "fr_amide": Fragments.fr_amide(self.mol),
            "fr_amidine": Fragments.fr_amidine(self.mol),
            "fr_aniline": Fragments.fr_aniline(self.mol),
            "fr_aryl_methyl": Fragments.fr_aryl_methyl(self.mol),
            "fr_azide": Fragments.fr_azide(self.mol),
            "fr_azo": Fragments.fr_azo(self.mol),
            "fr_barbitur": Fragments.fr_barbitur(self.mol),
            "fr_benzene": Fragments.fr_benzene(self.mol),
            "fr_benzodiazepine": Fragments.fr_benzodiazepine(self.mol),
            "fr_bicyclic": Fragments.fr_bicyclic(self.mol),
            "fr_diazo": Fragments.fr_diazo(self.mol),
            "fr_dihydropyridine": Fragments.fr_dihydropyridine(self.mol),
            "fr_epoxide": Fragments.fr_epoxide(self.mol),
            "fr_ester": Fragments.fr_ester(self.mol),
            "fr_ether": Fragments.fr_ether(self.mol),
            "fr_furan": Fragments.fr_furan(self.mol),
            "fr_guanido": Fragments.fr_guanido(self.mol),
            "fr_halogen": Fragments.fr_halogen(self.mol),
            "fr_hdrzine": Fragments.fr_hdrzine(self.mol),
            "fr_hdrzone": Fragments.fr_hdrzone(self.mol),
            "fr_imidazole": Fragments.fr_imidazole(self.mol),
            "fr_imide": Fragments.fr_imide(self.mol),
            "fr_isocyan": Fragments.fr_isocyan(self.mol),
            "fr_isothiocyan": Fragments.fr_isothiocyan(self.mol),
            "fr_ketone": Fragments.fr_ketone(self.mol),
            "fr_ketone_Topliss": Fragments.fr_ketone_Topliss(self.mol),
            "fr_lactam": Fragments.fr_lactam(self.mol),
            "fr_lactone": Fragments.fr_lactone(self.mol),
            "fr_methoxy": Fragments.fr_methoxy(self.mol),
            "fr_morpholine": Fragments.fr_morpholine(self.mol),
            "fr_nitrile": Fragments.fr_nitrile(self.mol),
            "fr_nitro": Fragments.fr_nitro(self.mol),
            "fr_nitro_arom": Fragments.fr_nitro_arom(self.mol),
            "fr_nitro_arom_nonortho": Fragments.fr_nitro_arom_nonortho(self.mol),
            "fr_nitroso": Fragments.fr_nitroso(self.mol),
            "fr_oxazole": Fragments.fr_oxazole(self.mol),
            "fr_oxime": Fragments.fr_oxime(self.mol),
            "fr_para_hydroxylation": Fragments.fr_para_hydroxylation(self.mol),
            "fr_phenol": Fragments.fr_phenol(self.mol),
            "fr_phenol_noOrthoHbond": Fragments.fr_phenol_noOrthoHbond(self.mol),
            "fr_phos_acid": Fragments.fr_phos_acid(self.mol),
            "fr_phos_ester": Fragments.fr_phos_ester(self.mol),
            "fr_piperdine": Fragments.fr_piperdine(self.mol),
            "fr_piperzine": Fragments.fr_piperzine(self.mol),
            "fr_priamide": Fragments.fr_priamide(self.mol),
            "fr_prisulfonamd": Fragments.fr_prisulfonamd(self.mol),
            "fr_pyridine": Fragments.fr_pyridine(self.mol),
            "fr_quatN": Fragments.fr_quatN(self.mol),
            "fr_sulfide": Fragments.fr_sulfide(self.mol),
            "fr_sulfonamd": Fragments.fr_sulfonamd(self.mol),
            "fr_sulfone": Fragments.fr_sulfone(self.mol),
            "fr_term_acetylene": Fragments.fr_term_acetylene(self.mol),
            "fr_tetrazole": Fragments.fr_tetrazole(self.mol),
            "fr_thiazole": Fragments.fr_thiazole(self.mol),
            "fr_thiocyan": Fragments.fr_thiocyan(self.mol),
            "fr_thiophene": Fragments.fr_thiophene(self.mol),
            "fr_unbrch_alkane": Fragments.fr_unbrch_alkane(self.mol),
            "fr_urea": Fragments.fr_urea(self.mol),
        }

    def get_all_descriptors(
        self, as_array: bool = False, decimal: int = 3
    ) -> Union[Dict[str, float], np.ndarray]:
        """Return all 196 RDKit descriptors."""
        all_descriptors = {}
        all_descriptors.update(self.get_topological())
        all_descriptors.update(self.get_chi())
        all_descriptors.update(self.get_estate_vsa())
        all_descriptors.update(self.get_mol())
        all_descriptors.update(self.get_lipinski())
        all_descriptors.update(self.get_peoe_vsa())
        all_descriptors.update(self.get_smr_vsa())
        all_descriptors.update(self.get_slogp_vsa())
        all_descriptors.update(self.get_vsa_estate())
        all_descriptors.update(self.get_fragments())

        if as_array:
            values = [round(all_descriptors[key], decimal) for key in sorted(all_descriptors.keys())]
            return np.array(values, dtype=np.float64)

        return all_descriptors

    def get_ecfp4(self, n_bits: int = ECFP4_N_BITS) -> np.ndarray:
        """ECFP4 (Morgan radius 2) binary fingerprint."""
        if _MORGAN_GENERATOR_AVAILABLE:
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=ECFP4_RADIUS, fpSize=n_bits
            )
            bitvect = morgan_gen.GetFingerprint(self.mol)
        else:
            bitvect = AllChem.GetMorganFingerprintAsBitVect(
                self.mol, ECFP4_RADIUS, nBits=n_bits
            )
        array = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvect, array)
        return array


DESCRIPTOR_NAMES = list(molecular_descriptor("C").get_all_descriptors().keys())


def _nan_descriptors() -> Dict[str, float]:
    return {name: np.nan for name in DESCRIPTOR_NAMES}


def _nan_ecfp4(n_bits: int = ECFP4_N_BITS) -> np.ndarray:
    return np.full((n_bits,), np.nan, dtype=float)


def extract_rdkit_descriptors(smiles: str) -> Dict[str, float]:
    """Extract 196 RDKit descriptors from one SMILES string."""
    try:
        return molecular_descriptor(smiles).get_all_descriptors()
    except (ValueError, TypeError):
        return _nan_descriptors()


def extract_ecfp4(smiles: str, n_bits: int = ECFP4_N_BITS) -> np.ndarray:
    """Extract a 2048-bit ECFP4 fingerprint from one SMILES string."""
    try:
        return molecular_descriptor(smiles).get_ecfp4(n_bits=n_bits)
    except (ValueError, TypeError):
        return _nan_ecfp4(n_bits)


def resolve_cleaned_input(input_path=None):
    """Resolve a CSV under dird/data/cleaned_data.

    Default is ``balanced_pulmonary_adr_dataset.csv``. Pass another name or
    path with ``--input`` when cleaned_data contains more than one file.
    """
    if input_path is None or str(input_path).strip() == "":
        input_path = DEFAULT_CLEANED_NAME

    raw = Path(input_path)
    if raw.is_file():
        return raw
    candidate = CLEANED_DIR / raw.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"No input CSV at {raw} or {candidate}. "
        f"Expected a file inside {CLEANED_DIR}."
    )


def resolve_smiles_column(df, smiles_col=None):
    """Use 'SMILES' or 'smiles' (or an explicit column name)."""
    if smiles_col:
        if smiles_col not in df.columns:
            raise KeyError(
                f"No SMILES column '{smiles_col}'. Columns: {list(df.columns)}"
            )
        return smiles_col
    for name in ("SMILES", "smiles"):
        if name in df.columns:
            return name
    matches = [c for c in df.columns if str(c).lower() == "smiles"]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"No 'SMILES' or 'smiles' column. Columns: {list(df.columns)}"
    )


def resolve_label_columns(df, label_col=None):
    """Return the label column(s) to keep beside SMILES.

    Prefers an explicit ``--label-col``, then a single ``class``/``label``/
    ``target`` column, then the four pulmonary labels from the cleaned dataset.
    """
    if label_col:
        if label_col not in df.columns:
            raise KeyError(
                f"No target column '{label_col}'. Columns: {list(df.columns)}"
            )
        return [label_col]
    for name in ("class", "Class", "CLASS", "label", "Label", "target", "Target"):
        if name in df.columns:
            return [name]
    matches = [c for c in df.columns if str(c).lower() in {"class", "label", "target"}]
    if len(matches) == 1:
        return matches
    present = [c for c in PULMONARY_LABEL_COLS if c in df.columns]
    if present:
        return present
    raise KeyError(
        "No 'class' column and none of "
        f"{PULMONARY_LABEL_COLS}. Columns: {list(df.columns)}"
    )


def _id_columns(df: pd.DataFrame, smiles_col: str, label_cols) -> pd.DataFrame:
    return df[[smiles_col] + list(label_cols)].copy()


def rdkit_descriptor_frame(
    smiles: Iterable[str],
    index=None,
) -> pd.DataFrame:
    rows = [extract_rdkit_descriptors(smi) for smi in smiles]
    return pd.DataFrame(rows, index=index, columns=DESCRIPTOR_NAMES)


def ecfp4_frame(
    smiles: Iterable[str],
    index=None,
    n_bits: int = ECFP4_N_BITS,
) -> pd.DataFrame:
    rows = [extract_ecfp4(smi, n_bits=n_bits) for smi in smiles]
    return pd.DataFrame(rows, index=index, columns=ecfp4_column_names(n_bits))


def extract_features(
    input_path=None,
    output_dir=None,
    smiles_col=None,
    label_col=None,
    features="both",
):
    """Read a cleaned CSV and write one feature table under featurized_data.

    features:
        rdkit  — 196 RDKit descriptors only
        ecfp4  — 2048-bit ECFP4 only
        both   — descriptors and ECFP4 in one combined file

    Input default: dird/data/cleaned_data/balanced_pulmonary_adr_dataset.csv.
    Pass --input with another filename when cleaned_data has more than one CSV.
    Uses the 'SMILES' or 'smiles' column and keeps 'class' or the four
    Label_* columns from data merging.

    Output is written to dird/data/featurized_data (created if missing):
        rdkit  → rdkit_md_196.csv
        ecfp4  → ecfp4_2048.csv
        both   → both.csv

    Output keeps SMILES and labels so it can be passed to
    ml_experiments/01_data_splitting.py.
    """
    input_path = resolve_cleaned_input(input_path)
    output_dir = Path(output_dir) if output_dir is not None else FEATURIZED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    smiles_col = resolve_smiles_column(df, smiles_col)
    label_cols = resolve_label_columns(df, label_col)

    mode = str(features).strip().lower()
    if mode in {"descriptor", "descriptors"}:
        mode = "rdkit"
    if mode == "ecfp":
        mode = "ecfp4"
    if mode in {"combined", "rdkit_ecfp4"}:
        mode = "both"

    ids = _id_columns(df, smiles_col, label_cols).reset_index(drop=True)
    smiles = df[smiles_col].astype(str)

    if mode == "rdkit":
        features_df = rdkit_descriptor_frame(smiles)
    elif mode == "ecfp4":
        features_df = ecfp4_frame(smiles)
    elif mode == "both":
        features_df = pd.concat(
            [rdkit_descriptor_frame(smiles), ecfp4_frame(smiles)],
            axis=1,
        )
    else:
        raise ValueError("features must be 'rdkit', 'ecfp4', or 'both'")

    out = pd.concat([ids, features_df.reset_index(drop=True)], axis=1)

    # A few RDKit charge descriptors can be infinite for unusual molecules.
    # Replace those values deterministically so downstream estimators receive
    # finite numeric input; the column median is computed from this dataset.
    numeric_cols = features_df.columns
    finite_features = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Guard against pathological descriptors (for example Ipc) that are
    # finite but too large for downstream scalers/kernels.
    finite_features = finite_features.mask(finite_features.abs() > 1e6)
    for column in numeric_cols:
        if finite_features[column].isna().any():
            median = finite_features[column].median()
            finite_features[column] = finite_features[column].fillna(
                0.0 if pd.isna(median) else median
            )
    out.loc[:, numeric_cols] = finite_features

    path = output_dir / OUTPUT_FILENAMES[mode]

    out.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract 196 RDKit descriptors and/or 2048-bit ECFP4 fingerprints."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_CLEANED_NAME,
        help=(
            "CSV in dird/data/cleaned_data (any name). "
            f"Default: {DEFAULT_CLEANED_NAME}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(FEATURIZED_DIR),
        help="Output folder (default: dird/data/featurized_data).",
    )
    parser.add_argument(
        "--smiles-col",
        default=None,
        help="SMILES column name. Default: use 'SMILES' or 'smiles'.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Target column. Default: 'class', or all four Label_* columns.",
    )
    parser.add_argument(
        "--features",
        choices=["rdkit", "ecfp4", "both"],
        default="both",
        help="rdkit, ecfp4, or both combined in one file (default: both).",
    )
    args = parser.parse_args()

    extract_features(
        input_path=args.input,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        features=args.features,
    )
