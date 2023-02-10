# BALanCe
This is all the source codes for ICLR2023 paper

BALanCe: Deep Bayesian Active Learning via Equivalence Class Annealing

## How to reproduce the results
Install pytorch and other libs

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

### AL trials on MNIST, EMNIST-Balanced, EMNIST-ByMerge, EMNIST-ByClass and Fashion-MNIST
go to the task folder, e.g., 
```
cd BALanCe/
```

go to data folder, e.g., 
```
cd data/MNIST/
```

download the data, e.g., 
https://github.com/pytorch/examples/tree/master/mnist

split the data
```
python make_lt.py
```

go to task main folder, run the commands, e.g.,
```
python main_eced.py
```

All the paramter are defined inside python file. The default ones are used in the paper.

### Fig. 1 and Fig. 2

go to folder Fig1-2/
```
cd Fig1-2
```

split the data as described by paper
```
python make_pt.py
```

train the BNN and draw 200 MC dropouts
```
python epsilon_net.py
```

run farthest-first traversal and draw the figs
```
python teo-heatmap.py
```

### Plot figures
The scripts for generate the learning curve is in Plot/ folder. You can specify log files and the legend.
