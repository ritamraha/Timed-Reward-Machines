# Experimental evaluations of Reinforcement Learning algorithms for Timed Reward Machines

## Install dependencies

- Required Python packages are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

## TRM logs and plot generation

The TensorBoard logs for the TRM experiments used in the paper are included in the `Logs/` folder.

To extract and generate the SVG plots from those logs, run the provided extraction script.
For example (from the repository root):
```bash
python tb_extract.py -b Logs/disc_vs_cont_Taxi
```
will generate the plots for *performance difference for various timed interpretations* (Figure 7).