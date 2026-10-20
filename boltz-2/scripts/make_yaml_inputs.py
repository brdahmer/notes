#!/usr/bin/env python3
"""Generate one Boltz-2 YAML input file per ligand.

Reads a text file with one ligand per line in the form ``name: SMILES``
(the output of the Open Babel / PubChem steps of the protocol) and writes
``<OUTPUT_DIR>/<name>.yaml`` for each ligand. All files share the same protein,
binding-site constraint, optional template and affinity request.

Usage:
    python make_yaml_inputs.py                 # uses the settings below
    python make_yaml_inputs.py my_ligands.txt  # different input file

Edit the SETTINGS block for your own protein before running.
"""

import re
import sys
from pathlib import Path

# ============================== SETTINGS ==============================

INPUT_FILE = "ligands.txt"      # lines like "ligand1: CC(=O)Oc1ccccc1C(=O)O"
OUTPUT_DIR = "yaml_inputs"      # folder for the generated .yaml files

PROTEIN_SEQUENCE = (
    "GSHMHYSYETFLKDSLELVKQVEQICGVPEALVCVMRGGMTLTHFLSLHWDLREVYGINAIS"
    "YDTTHRQNALKIENIPTIKDHLKTILVVDEIVDSGNSLEAVLKVLQDKHPDKKFYSASLFQK"
    "TSAKYKADAFLKDAPEWIDFFWEVDLKNLKSH"
)
PROTEIN_CHAINS = ["A"]          # ["A"] = monomer, ["A", "B"] = homodimer
LIGAND_CHAIN = "B"              # must differ from all protein chain IDs

# Binding-site residues for the pocket constraint (set to [] to disable).
# Boltz numbers residues by their POSITION in PROTEIN_SEQUENCE, starting at 1.
# If the numbers below come from a PDB file, set SEQUENCE_OFFSET so that
# position = PDB number + SEQUENCE_OFFSET. Here the sequence starts with a
# 3-residue tag (GSH) before Met1, so the offset is 3.
# Check this for your protein; use 0 if the numbers are already sequence positions.
POCKET_RESIDUES = [34, 35, 92, 93, 94, 95, 121, 138, 139, 140, 144]
SEQUENCE_OFFSET = 3
POCKET_MAX_DISTANCE = 5.0       # Å, allowed 4–20 (Boltz default 6)
POCKET_FORCE = True             # enforce the pocket with a potential

# Optional protein template (.cif or .pdb). Set to None to disable.
# The path is resolved relative to the folder where you run `boltz predict`,
# not relative to the YAML file.
TEMPLATE_FILE = "./template.cif"
TEMPLATE_CHAIN = "A"            # protein chain in the template file

COMPUTE_AFFINITY = True

# ======================================================================


def safe_name(name: str) -> str:
    """Make a ligand name safe for use as a file name and Boltz record id."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return cleaned or "ligand"


def check_smiles(smiles: str):
    """Return an error message if RDKit is installed and rejects the SMILES."""
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return None  # RDKit not installed: skip the check (Boltz will check)
    RDLogger.DisableLog("rdApp.*")
    return None if Chem.MolFromSmiles(smiles) is not None else "RDKit could not parse it"


def build_yaml(smiles: str) -> str:
    """Return the YAML text for one ligand."""
    chains = ", ".join(PROTEIN_CHAINS)
    lines = [
        "version: 1",
        "",
        "sequences:",
        "  - protein:",
        f"      id: [{chains}]",
        f"      sequence: {PROTEIN_SEQUENCE}",
        "  - ligand:",
        f"      id: {LIGAND_CHAIN}",
        f"      smiles: '{smiles}'",
    ]
    if POCKET_RESIDUES:
        contacts = ", ".join(
            f"[{PROTEIN_CHAINS[0]}, {r + SEQUENCE_OFFSET}]" for r in POCKET_RESIDUES
        )
        lines += [
            "",
            "constraints:",
            "  - pocket:",
            f"      binder: {LIGAND_CHAIN}",
            f"      contacts: [{contacts}]",
            f"      max_distance: {POCKET_MAX_DISTANCE}",
            f"      force: {str(POCKET_FORCE).lower()}",
        ]
    if TEMPLATE_FILE:
        key = "pdb" if TEMPLATE_FILE.lower().endswith(".pdb") else "cif"
        lines += [
            "",
            "templates:",
            f"  - {key}: {TEMPLATE_FILE}",
            f"    chain_id: {PROTEIN_CHAINS[0]}",   # chain in THIS YAML (must be a protein)
            f"    template_id: {TEMPLATE_CHAIN}",   # chain in the TEMPLATE file
        ]
    if COMPUTE_AFFINITY:
        lines += [
            "",
            "properties:",
            "  - affinity:",
            f"      binder: {LIGAND_CHAIN}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    input_file = Path(sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE)
    out_dir = Path(OUTPUT_DIR)

    # Sanity checks on the settings
    if LIGAND_CHAIN in PROTEIN_CHAINS:
        sys.exit("LIGAND_CHAIN must differ from the protein chain IDs.")
    for r in POCKET_RESIDUES:
        pos = r + SEQUENCE_OFFSET
        if not 1 <= pos <= len(PROTEIN_SEQUENCE):
            sys.exit(f"Pocket residue {r} -> position {pos} is outside the sequence "
                     f"(1-{len(PROTEIN_SEQUENCE)}).")
    if not input_file.exists():
        sys.exit(f"Input file not found: {input_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, seen = 0, 0, set()
    for n, line in enumerate(input_file.read_text(encoding="utf-8").splitlines(), 1):
        if ":" not in line:
            continue  # blank or malformed line
        name, rest = line.split(":", 1)
        tokens = rest.split()
        smiles = tokens[0] if tokens else ""   # keeps only the SMILES, drops e.g. "\t*****"
        name = safe_name(name)

        if not smiles:
            print(f"  skipped line {n} ({name}): no SMILES")
            skipped += 1
            continue
        problem = check_smiles(smiles)
        if problem:
            print(f"  skipped line {n} ({name}): invalid SMILES '{smiles}' ({problem})")
            skipped += 1
            continue
        if name in seen:
            print(f"  skipped line {n} ({name}): duplicate name")
            skipped += 1
            continue
        seen.add(name)

        (out_dir / f"{name}.yaml").write_text(build_yaml(smiles), encoding="utf-8")
        written += 1

    print(f"Wrote {written} YAML file(s) to {out_dir}/ ({skipped} line(s) skipped).")


if __name__ == "__main__":
    main()
