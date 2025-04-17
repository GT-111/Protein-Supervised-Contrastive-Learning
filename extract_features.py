import os
import math
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter

def entropy_approx(aa_freq):
    """Shannon entropy of amino acid distribution"""
    entropy = -sum(p * math.log2(p) for p in aa_freq.values() if p > 0)
    return entropy

def extract_protein_features(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("X", pdb_path)
    except:
        return None

    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    try:
                        seq += seq1(residue.get_resname())
                    except:
                        continue

    if len(seq) < 30:
        return None

    try:
        analysed = ProteinAnalysis(seq)
        aa_percent = analysed.amino_acids_percent
        features = {
            "length": len(seq),
            "molecular_weight": analysed.molecular_weight(),
            "gravy": analysed.gravy(),
            "aromaticity": analysed.aromaticity(),
            "instability_index": analysed.instability_index(),
            "entropy_approx": entropy_approx(aa_percent)
        }
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            features[f"AA_{aa}"] = aa_percent.get(aa, 0.0)
    except:
        return None

    try:
        dssp = DSSP(structure[0], pdb_path)
        sec_struct = [d[2] for d in dssp]
        total = len(sec_struct)
        helix = sum(s in ['H', 'G', 'I'] for s in sec_struct)
        sheet = sum(s in ['E', 'B'] for s in sec_struct)
        coil = sum(s in ['T', 'S', ' '] for s in sec_struct)

        features["helix_frac"] = helix / total if total else 0.0
        features["sheet_frac"] = sheet / total if total else 0.0
        features["coil_frac"] = coil / total if total else 0.0
    except Exception as e:
        features["helix_frac"] = features["sheet_frac"] = features["coil_frac"] = 0.0

    return features

def load_id_class_map(filepath="id_subclass_map.txt"):
    id_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '\t' in line:
                pdb_id, subclass = line.strip().split("\t")
                id_map[pdb_id.upper()] = subclass.lower()
    return id_map

def batch_extract(folder, label, id_class_map):
    data = []
    for file in tqdm(os.listdir(folder), desc=f"Extracting from {folder}"):
        if file.endswith(".pdb"):
            path = os.path.join(folder, file)
            feats = extract_protein_features(path)
            if feats:
                pdb_id = file.replace(".pdb", "").upper()
                feats["label"] = label
                feats["pdb_id"] = pdb_id
                feats["subclass"] = id_class_map.get(pdb_id, "unknown")
                data.append(feats)
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("🔍 Loading subclass mapping from id_subclass_map.txt...")
    id_class_map = load_id_class_map()

    print("🧪 Extracting enzyme features...")
    df_enzyme = batch_extract("data/enzymes", label=1, id_class_map=id_class_map)

    print("🧬 Extracting non-enzyme features...")
    df_non_enzyme = batch_extract("data/non_enzymes", label=0, id_class_map=id_class_map)

    df = pd.concat([df_enzyme, df_non_enzyme], ignore_index=True)
    df.to_csv("protein_features.csv", index=False)
    print(f"✅ Features saved to protein_features.csv, total samples: {len(df)}")
