# Data

The raw NASA C-MAPSS files are **not** committed (see `.gitignore`).

1. Download the *Turbofan Engine Degradation Simulation Data Set* from the
   [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
2. Unzip the `train_FD00x.txt`, `test_FD00x.txt` and `RUL_FD00x.txt` files into `data/CMAPSS/`.
3. Train on it: `enginewatch train --data-dir data/CMAPSS --subset FD001`

No download? Use `--synthetic` on the `train` and `simulate` commands.
