# Experimental evaluations of Reinforcement Learning algorithms for Timed Reward Machines

## Install dependencies

Good practice is to create a virtual environment first:
```bash
python -m venv trm_env
source trm_env/bin/activate 
```

- Required Python packages are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```



## Command-line arguments (main.py)

The script main.py accepts the following parser arguments:

| Flag(s) | Name | Type / choices | Default | Description |
|---------|------|----------------|---------|-------------|
| -e, --env-type | env_type | taxi, frozen_lake | taxi | Environment to run |
| -t, --trm-name | trm_name | string (path) | env/Taxi/disc_vs_cont.txt | Path to TRM file to load |
| -c, --add-ci | add_ci | 0 or 1 | 1 | Whether to add Counterfactual Imagining (CI) |
| -m, --mode | mode | digital, real | digital | digital or real-time setting for TRM interpretation |
| -d, --discretization-param | discretization_param | float | 1.0 | TRM discretization (use 0 for untimed RM) |
| -tt, --total-timesteps | total_timesteps | int | 300000 | Total timesteps per run |
| -n, --total-runs | total_runs | int | 1 | Number of runs (different seeds) |


Example:
```bash
python main.py -e taxi -t env/Taxi/disc_vs_cont.txt -c 1 -m digital -d 1.0 -tt 300000 -n 1
```

## TRM logs and plot generation
### Quick Visualization
For quick visualization of the training process, you can use TensorBoard.

To view the TensorBoard graphs for runs written to default saved location (q_learning_logs), run:
```bash
tensorboard --logdir q_learning_logs
```

### Generating Graphs from the Article
The TensorBoard logs for the TRM experiments *used in the article* are included in the `Logs/` folder.

To extract and generate the SVG plots from those logs, run the provided extraction script.
For example (from the repository root):
```bash
python tb_extract.py -b Logs/disc_vs_cont_Taxi
```
will generate the plots for *performance difference for various timed interpretations* (Figure 7).