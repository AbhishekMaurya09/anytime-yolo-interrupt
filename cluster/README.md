# Cluster training (TrinityX / SLURM)

Everything lives in `~/anytime_yolo_cs25m201/` on the cluster:

```
env/                         Python 3.11 + PyTorch 2.6 (cu124), separate from the account's own conda
anytime_yolo_interrupt/      project code (copied from the laptop)
data/bdd100k.zip             full BDD100K images + det_v2 labels (unzipped to data/bdd100k_raw/)
data/bdd70k/                 train / calibration / test split built by scripts/build_bdd_subset.py
data/old200, data/bdd1500    earlier subsets: excluded from bdd70k and kept as unseen test sets
runs/anytime_interrupt/      checkpoints and logs (train6_bdd70k)
logs/                        SLURM job output
```

Submit / resume (same command both times):

```bash
cd ~/anytime_yolo_cs25m201 && sbatch cluster/train_bdd70k.sbatch
```

Check: `squeue -u $USER`, `tail -f logs/anytime70k_<jobid>.out`. Cancel: `scancel <jobid>`.

To remove everything afterwards: `rm -rf ~/anytime_yolo_cs25m201`.
