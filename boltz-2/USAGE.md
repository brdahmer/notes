# Boltz-2 Protocol: Protein–Ligand Structure and Affinity Prediction

## Introduction

Molecular docking is a computational technique for predicting the binding pose and affinity of a ligand to a protein. It is widely used to study molecular recognition and in drug discovery.

Boltz-2 is a deep-learning model that approaches the same problem differently. Instead of placing a ligand into a fixed receptor structure, it **co-folds** the protein and the ligand, predicting the 3D structure of the whole complex. It also predicts **binding affinity**, so structure prediction and a docking-like screen happen in a single run. Boltz-2 can also be used for protein structure prediction alone, without ligands.

More information:
- [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
- [Official Boltz repository (GitHub)](https://github.com/jwohlwend/boltz)
- [Boltz prediction documentation (input format, flags, outputs)](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)

## Requirements

- Boltz-2 installed (here in a conda environment called `boltz-cuda`, with GPU/CUDA support)
- Open Babel (for converting ligand files to SMILES)
- Sequence(s) of the protein(s) of interest
- The ligands to be tested

## Useful terminal commands

| Command | What it does |
|---|---|
| `cd folder_name` | Enter a folder (change the working directory) |
| `cd ..` | Go up one folder |
| `ls` | List the files and folders in the current directory |
| `mkdir folder_name` | Create a folder |
| `python file_name.py` | Run a Python script |
| `<command> --help` | Show the help/options for a command |

The template input file and helper scripts are in this repository:

| File | Purpose |
|---|---|
| [`templates/template.yaml`](templates/template.yaml) | Annotated example Boltz-2 input file |
| [`scripts/make_yaml_inputs.py`](scripts/make_yaml_inputs.py) | Creates one YAML input per ligand from a `name: SMILES` list |
| [`scripts/collect_affinity.py`](scripts/collect_affinity.py) | Collects affinity and confidence results from all ligands into one CSV |

> **Note:** remember to change the file names and paths in the commands below to match your own files (shown as placeholders such as `./yaml_inputs/`).

## Procedure

All steps below are run in a terminal.

### 1. Activate the environment

```sh
conda deactivate            # deactivate any active conda environment
conda activate boltz-cuda   # activate the environment set up to run Boltz with CUDA
```

### 2. Convert ligands to SMILES

Boltz-2 takes ligands as **SMILES** strings (or as a PDB Chemical Component Dictionary code, see step 3). Ligand files in **.mol2** or **.cml** format (the default format of newer Avogadro versions) can be converted to SMILES with Open Babel. Ligand names need a different tool (section 2.2).

If your goal is only protein structure prediction, skip to [step 3](#3-prepare-the-yaml-input-files).

#### 2.1 From .mol2 files

From inside the folder that contains the .mol2 files (use `cd`), run:

```sh
> ligands.txt
for f in *.mol2; do
    smiles=$(obabel "$f" -osmi -xn 2>/dev/null | awk '{print $1}')
    echo "${f%.mol2}: $smiles" >> ligands.txt
done
```

How it works:
- `> ligands.txt` creates (or empties) the output file, so re-running the loop does not append duplicates.
- `for f in *.mol2; do` loops over every .mol2 file in the current directory.
- `obabel "$f" -osmi -xn` converts the file to SMILES. `-osmi` sets the output format to SMILES, and `-xn` leaves out the molecule title. Without `-xn`, Open Babel writes the title after the SMILES, separated by a tab. Avogadro names molecules `*****`, which is where the asterisks at the end of the line come from.
- `awk '{print $1}'` keeps only the SMILES string.
- `echo "${f%.mol2}: $smiles" >> ligands.txt` appends the file name (without the extension) and its SMILES to the output file.
- `done` ends the loop.

Output file:

```text
ligand1: CC(=O)OC1=CC=CC=C1C(=O)O
ligand2: C1=CC=C(C=C1)O
ligand3: CN1CCCC1C2=CN=CC=C2
```

#### 2.2 From compound names (IUPAC or common names)

Open Babel **cannot** convert names to structures. `obabel -:"..."` expects a SMILES string, so `obabel -:"benzene"` fails with an "invalid character" error. Use a name-to-structure service instead, such as PubChem's REST API (requires internet).

Create a file `names.txt` with one name per line:

```text
benzene
ethanol
glucose
```

Then run:

```sh
> names_smiles.txt
while IFS= read -r name; do
    query=$(printf '%s' "$name" | sed 's/ /%20/g')
    smiles=$(curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${query}/property/SMILES/TXT" | head -n 1)
    echo "$name: $smiles" >> names_smiles.txt
done < names.txt
```

How it works:
- `> names_smiles.txt` creates (or empties) the output file.
- `while IFS= read -r name; do ... done < names.txt` reads `names.txt` line by line. `IFS=` keeps names with spaces intact, and `read -r name` stores each line in the variable `name`.
- `sed 's/ /%20/g'` makes names with spaces safe to put in a URL.
- The `curl` command asks PubChem for the SMILES of that name. `head -n 1` keeps the first match.
- `echo "$name: $smiles" >> names_smiles.txt` appends a `name: SMILES` line to the output file (`>>` appends instead of overwriting).

Output file:

```text
benzene: C1=CC=CC=C1
ethanol: CCO
glucose: C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O
```

> **Note:** check that names match the intended compound. Common names such as "glucose" can map to a specific stereoisomer or anomer. Names PubChem does not recognise return an error message instead of a SMILES string.

#### 2.3 From .cml files

From inside the folder that contains the .cml files, run:

```sh
> cml_smiles.txt
for f in *.cml; do
    smiles=$(obabel "$f" -osmi -xn 2>/dev/null | awk '{print $1}')
    echo "${f%.cml}: $smiles" >> cml_smiles.txt
done
```

Output file:

```text
molecule1: C1=CC=CC=C1
molecule2: CCO
molecule3: C(CO)O
```

Once the SMILES are collected in a text file, move on to preparing the YAML input files.

### 3. Prepare the YAML input files

Each Boltz-2 run reads one `.yaml` input file ([`templates/template.yaml`](templates/template.yaml)) describing the complex: protein chain(s), ligand(s), optional constraints and templates, and which properties to compute. For a ligand screen, create **one YAML file per ligand** (or per protein).

Example (values in `<...>` and the commented lines must be adapted to your system):

```yaml
version: 1  # input format version

sequences:
  - protein:
      id: [A, B]   # chain IDs; two IDs = two identical copies (homodimer). Use A alone for a monomer
      sequence: GSHMHYSYETFLKDSLELVKQVEQICGVPEALVCVMRGGMTLTHFLSLHWDLREVYGINAISYDTTHRQNALKIENIPTIKDHLKTILVVDEIVDSGNSLEAVLKVLQDKHPDKKFYSASLFQKTSAKYKADAFLKDAPEWIDFFWEVDLKNLKSH  # one line, no spaces or line breaks
  - ligand:
      id: C                    # ligand chain ID (must differ from the protein IDs)
      smiles: 'C1=CC=CC=C1'    # SMILES of the ligand (alternatively: ccd: <CCD code>)

constraints:  # OPTIONAL: define the binding site (pocket-guided prediction)
  - pocket:
      binder: C   # chain that should bind in the pocket (here, the ligand)
      contacts: [
        [A, 37], [A, 38],                     # residues binding pyrophosphate (PPi)
        [A, 95], [A, 96], [A, 97], [A, 98],   # residues binding the phosphate group
        [A, 124], [A, 141], [A, 142], [A, 143], [A, 147]  # residues binding the 6-oxopurine base
      ]   # residue indices = positions in the sequence above, starting at 1 (see note below)
      max_distance: 5.0   # max distance (Å) between the binder and each contact residue (allowed 4–20 Å; default 6 Å)
      force: true         # apply a potential to enforce the pocket constraint

templates:  # OPTIONAL: use an experimental structure as a template for protein chains
  - cif: ./template.cif     # path to an mmCIF file (e.g. downloaded from the PDB); `pdb:` files are also accepted
    chain_id: [A, B]        # which chains in THIS YAML should be templated
    template_id: [A, B]     # which chains in the TEMPLATE file to use for them

properties:  # OPTIONAL: what Boltz should compute besides the structure
  - affinity:
      binder: C   # the ligand chain for which binding affinity is predicted
```

Notes on the input file:

- **Protein only.** To predict only the protein structure, remove the `- ligand:` entry, the `constraints` block and the `properties` block.
- **Chain IDs.** For a monomer use `id: A`. For a homodimer use `id: [A, B]`. Give the ligand an ID that is not used by any protein chain, and reuse exactly that ID in `pocket: binder` and `affinity: binder`.
- **The sequence must be a single line.** If it wraps onto a new line, YAML joins the lines with a space and the sequence becomes invalid.
- **Residue numbering in `contacts`.** Boltz numbers residues by their position in the `sequence` string, starting at 1. This is *not* necessarily the residue numbering of a PDB file. In the example, the sequence starts with an expression-tag remnant (`GSH`) before Met1. If the binding-site residues were read from a PDB entry in which Met1 is residue 1, PDB residue *n* is sequence position *n* + 3. The PDB numbers 34, 35, 92–95, 121, 138–140 and 144 therefore become 37, 38, 95–98, 124, 141–143 and 147. These map onto the expected motifs: R-G for PPi binding, D-S-G-N for the phosphate loop, and the base-binding K and D. Always check this mapping for your own protein.
- **Constraints and templates are optional.** They help when ligands are placed outside the active site. A `pocket` constraint on the active-site residues gives a guided, site-directed prediction.
- **Templates apply to protein chains only.** Any mmCIF (PDBx/mmCIF) file from the PDB can be used, and PDB files are accepted too. Ligands cannot be templated. To specify a known ligand exactly, give its PDB Chemical Component Dictionary code with `ccd:` instead of `smiles:`.
- **Properties.** `affinity` is the only property Boltz-2 computes. Affinity is predicted for **one** small-molecule ligand chain per input, and only against protein targets. The ligand can have at most 128 atoms, and the developers advise against ligands much larger than about 56 atoms.

To create many YAML files at once, use [`scripts/make_yaml_inputs.py`](scripts/make_yaml_inputs.py). First edit the `SETTINGS` block at the top of the script: protein sequence, chain IDs, pocket residues and their numbering offset, and the template file. Then run it in the folder that contains `ligands.txt`:

```sh
python make_yaml_inputs.py ligands.txt
```

This creates one YAML file per ligand in `yaml_inputs/`, each with the same protein and settings and that ligand's SMILES. The script:
- keeps only the SMILES from each line, so a trailing `*****` from Avogadro files does no harm;
- skips lines with no SMILES (e.g. a failed name lookup) and duplicate names;
- replaces spaces and special characters in ligand names, which become file names;
- checks each SMILES with RDKit when RDKit is installed (it is part of the Boltz environment).

You can then start the ligand screen.

### 4. Ligand screening

`boltz predict` accepts either a single YAML file or a **directory**. Given a directory, it runs every `.yaml` file inside it, so a whole screen is a single command:

```sh
boltz predict ./yaml_inputs/ \
    --use_msa_server \
    --use_potentials \
    --recycling_steps 25 \
    --diffusion_samples 10 \
    --sampling_steps_affinity 400 \
    --diffusion_samples_affinity 50 \
    --affinity_mw_correction \
    --max_parallel_samples 10 \
    --override \
    --out_dir ./resultados_screen
```

Alternatively, a simple bash loop runs one file at a time. This gives a line of progress output per ligand, but reloads the model for every file:

```sh
for f in ./yaml_inputs/*.yaml; do
    echo "Running $f..."
    boltz predict "$f" \
        --use_msa_server \
        --use_potentials \
        --recycling_steps 25 \
        --diffusion_samples 10 \
        --sampling_steps_affinity 400 \
        --diffusion_samples_affinity 50 \
        --affinity_mw_correction \
        --max_parallel_samples 10 \
        --override \
        --out_dir ./resultados_screen
done
```

> **Note:** in bash, the backslash `\` must be the **last character** on the line. A comment or a trailing space after it breaks the command. That is why the flags are explained in the table below rather than inline.

| Flag | Meaning | Default |
|---|---|---|
| `--use_msa_server` | Builds the multiple sequence alignment (MSA) automatically with the ColabFold MMseqs2 server. The protein sequence is sent to an external server. | off |
| `--use_potentials` | Applies inference-time physical potentials (e.g. against steric clashes and incorrect stereochemistry) to make the poses more physically plausible. It does not add a binding-energy term. | off |
| `--recycling_steps` | Number of times the model refines its internal representation before generating structures. AlphaFold3 uses 10; higher values cost more time. | 3 |
| `--diffusion_samples` | Number of structural models generated per input | 1 |
| `--sampling_steps_affinity` | Number of sampling steps used for the affinity prediction | 200 |
| `--diffusion_samples_affinity` | Number of diffusion samples used to estimate the affinity | 5 |
| `--affinity_mw_correction` | Adds a molecular-weight correction to the affinity prediction | off |
| `--max_parallel_samples` | Maximum number of samples (models) computed **simultaneously on the GPU**. Lower it if you run out of GPU memory. It does not control how many input files run in parallel. | 5 |
| `--override` | Ignores cached preprocessing and existing predictions in the output folder, and reruns from scratch | off |
| `--out_dir` | Output directory | (input name) |
| `--method` | *Optional.* Conditions the prediction on a structure-determination method, e.g. `x-ray diffraction`, `solution nmr` or `md` (structures resembling molecular-dynamics ensembles). It does **not** run a molecular dynamics simulation. | none |

Many flags are optional or can take other values. See [Boltz-2 prediction docs](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) for the full list.

Boltz writes its results to `<out_dir>/boltz_results_<input name>/`. With a folder as input, this is one `boltz_results_yaml_inputs/` folder containing every ligand. With the loop, it is one `boltz_results_<ligand>/` folder per ligand. Either layout works with the collection script in step 5.

### 5. Analysing the results

Inside each `boltz_results_*` folder, `predictions/` contains one subfolder per input file, with three important kinds of files:

```text
out_dir/
└── boltz_results_[input_name]/
    ├── predictions/
    │   └── [ligand_name]/
    │       ├── [ligand_name]_model_0.cif              # predicted structure(s)
    │       ├── confidence_[ligand_name]_model_0.json  # confidence scores (one per model)
    │       └── affinity_[ligand_name].json            # affinity prediction (one per input)
    └── processed/                                     # preprocessed inputs used by the model
```

#### 5.1 Structure files (.cif)

Open these in PyMOL (or a similar viewer) to inspect the 3D pose of the ligand in the protein. There is one file per model (`--diffusion_samples`), ranked by confidence score, so `model_0` is the highest ranked.

The same files can be used to analyse 2D protein–ligand interactions with tools such as PoseView or PLIP.

#### 5.2 Confidence files (confidence_*.json)

These contain the overall confidence metrics: `confidence_score`, `ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, `complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`, `chains_ptm` and `pair_chains_iptm`. There is one file per model. They describe how reliable the predicted structure is.

- `confidence_score`, `ptm` and the `plddt` scores (and their interface and per-chain versions) range from 0 to 1. Higher values mean higher confidence. `confidence_score` = 0.8 × `complex_plddt` + 0.2 × `iptm` (or `ptm` for single chains), and it is the score used to rank the models.
- `pde` scores are in ångströms. Lower values mean higher confidence.
- For protein–ligand complexes, `ligand_iptm` is the most direct measure of confidence in the protein–ligand interface.

#### 5.3 Affinity files (affinity_*.json)

These contain the affinity predictions: `affinity_pred_value` and `affinity_probability_binary`, each also given separately as `...1` and `...2`. There is one affinity file per input. Values 1 and 2 come from two differently trained models, and the unnumbered values are the ensemble prediction: the mean of models 1 and 2. If the run used `--affinity_mw_correction`, the molecular-weight correction is applied **only** to the ensemble `affinity_pred_value`, not to `affinity_pred_value1` or `affinity_pred_value2`. Use the ensemble values for ranking. The two quantities are trained on largely different datasets with different supervision, and are meant for different purposes:

- **`affinity_probability_binary`** distinguishes binders from decoys, for example in a **hit-discovery** stage.
  - It ranges from 0 to 1 and is the predicted probability that the ligand is a binder. A threshold of 0.5 is a reasonable starting point for ranking candidates. This is a rule of thumb, not an official cut-off.
- **`affinity_pred_value`** measures the specific affinity of different binders and how it changes with small modifications of the molecule. It is meant for **hit-to-lead and lead-optimization** stages, and only for comparing molecules that are already believed to be active, not inactives.
  - It is reported as **log10(IC50)**, with IC50 in μM. Lower values mean stronger predicted binding.
  - To recover IC50 in μM, compute 10^x, where x is `affinity_pred_value` (ensemble or individual model).
  - To express it as pIC50, use pIC50 = 6 − x. An approximate binding free energy in kcal/mol is −1.364 × (6 − x).

| IC50 | IC50 (μM) | `affinity_pred_value` | Interpretation |
|---|---|---|---|
| 10⁻⁹ M | 0.001 | −3 | strong binder |
| 10⁻⁶ M | 1 | 0 | moderate binder |
| 10⁻⁴ M | 100 | 2 | weak binder / decoy |

To collect the results of all ligands into one CSV file, use [`scripts/collect_affinity.py`](scripts/collect_affinity.py):

```sh
python collect_affinity.py resultados_screen            # the folder used as --out_dir
python collect_affinity.py resultados_screen --decimal-comma   # for spreadsheets that use decimal commas
```

The output `affinity_results.csv` has one row per ligand, sorted by `affinity_probability_binary` (highest first), with these columns:
- ensemble and per-model `affinity_probability_binary` and `affinity_pred_value`;
- IC50 (μM), pIC50 and approximate ΔG (kcal/mol), derived from the ensemble `affinity_pred_value`;
- `confidence_score`, `ligand_iptm`, `iptm` and `complex_plddt` of the top-ranked model.

> **Note:** without an argument, the script looks for a single folder whose name contains `resultados` or `results`, and stops if it finds more than one.
>
> **Note:** by default the CSV uses decimal points and comma separators. If your spreadsheet locale uses decimal commas (e.g. Portuguese in Google Sheets), either use `--decimal-comma` or change the spreadsheet locale to English before importing.
