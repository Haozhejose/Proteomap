# Multi-Objective Molecule Generation for LNP with Protein Corona Features

This project is based on the paper "Multi-Objective Molecule Generation using Interpretable Substructures" (Wengong Jin et al., ICML 2020), modified for LNP (Lipid Nanoparticles) molecule generation with reinforcement learning.

## Key Modifications
- Added target efficiency prediction.
- Integrated protein corona features using Mordred descriptors and similarity matching.
- Fine-tuned on LNP dataset (3204 molecules) starting from ChemBL pre-trained model.

## Environment Setup
1. Install Miniconda or Anaconda if not already installed.
2. Create the conda environment:
   ```bash
   conda env create -f env.yml
   ```
3. Activate the environment:
   ```bash
   conda activate proteomap
   ```

## Files
- `train.py`: RL fine-tuning script.
- `generate.py`: Generate molecules from proteomaps using trained model.
- `fuseprop/`: Core model components (GNN, dataset, etc.).
- `scripts/`: Evaluation and utility scripts.
- `data/`: Proteomap and dataset files.
- `ckpt/`: Model checkpoints.
- `env.yml`: Conda environment configuration.
- `model_description.md`: Detailed model description for publication.

## Usage
1. Fine-tune: `python train.py --init_model ckpt/chembl-h400beta0.3/model.20 --save_dir ckpt/lnp_new/ --epoch 40 --proteomap data/lnp/lnp_converted.txt --prop target_efficiency`
2. Generate: `python generate.py --model ckpt/lnp_new/model_best.pt --num_decode 100`
   (If model_best.pt contains proteomaps, --proteomap can be omitted)

## Controlling Molecule Generation Quantity
- `--num_decode`: Number of decoding attempts per proteomap (default 100). Total molecules = number of proteomaps * num_decode.
- `--batch_size`: Batch size for generation (default 20). Affects processing speed but not total count.
- Example: If proteomap file has 10 molecules and num_decode=50, generates ~500 molecules.

Best model is saved as `model_best.pt` based on target efficiency.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.