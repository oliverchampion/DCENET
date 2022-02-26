# DCE-NET
Framework for estimating DCE-MRI physiological parameters. 

## Getting Started:
To create the environment **dce** in anaconda, the following command can be used:
```bash
conda env create -f environment.yml
```

## Simulations:
Executing the framework on simulations can be done using **main.py**.
```
usage: main.py [--nn] [--layers] [--lr] [--batch_size] [--attention]
               [--bidirectional] [--supervised] [--results]

optional arguments:
  --nn            neural network to use - linear / lstm / gru
  --layers        number and size of layers - linear:   neurons_layer_1 neurons_layer_2 ...
                                            - lstm/gru: hidden_dimension stacked_layers
  --lr            learning rate - float
  --batch_size    batch size - int
  --attention     option to include attention layers for lstm/gru
  --bidirectional option to include bidirectionality for lstm/gru
  --supervised    option to train on ground truth parameters
  --results       option to perform evaluation on network using different SNR (must be trained first)
```
More hyperparameters are stored in **hyperparameters.py**.

### Showing Results
Results of the trained frameworks can be obtained by executing **results.py**.
