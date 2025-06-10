#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=todiit00@hs-esslingen.de
#SBATCH --job-name=dvae_train
#SBATCH --output=slurm_%j.out

# CUDA-Toolchain laden
module load devel/cuda/12.4

# Wechsle ins temporäre Verzeichnis
cd $TMPDIR

# Kopiere Projekt + Daten aus dem HOME-Verzeichnis
rsync -avz $HOME/Point-BERT-Khan/ ./Point-BERT-Khan/
cd Point-BERT-Khan

# Erstelle virtuelle Umgebung
python3 -m venv pointbert-env
source pointbert-env/bin/activate

# Installiere Abhängigkeiten
pip install --upgrade pip
pip install -r requirements.txt

# Starte dVAE Training
bash scripts/train.sh 0 \
    --config cfgs/ShapeNet55_models/dvae.yaml \
    --exp_name dvae_training

# Optional: Kopiere Ergebnisse zurück ins HOME-Verzeichnis
mkdir -p $HOME/Point-BERT-Khan/results
rsync -avz checkpoints/ $HOME/Point-BERT-Khan/results/checkpoints_$SLURM_JOB_ID/
