# DCE-NET
Framework for estimating DCE-MRI physiological parameters. 

Based on the master thesis: [DCE-MRI parameter estimation using a physics-based deep learning approach](https://scripties.uba.uva.nl/download?fid=682699)

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

## Patient Data
To construct a dataset, **create_dataset.py** is used that stores DCE-MRI signals with a dimension of (Slices, Width, Height, Length).

Training on the dataset is done with **sim_patients.py**.

```
usage: sim_patients.py [--nn] [--layers] [--lr] [--batch_size] 
                       [--optim] [--dual_path] [--bidirectional] 
                       [--pretrained] [--cpu]

optional arguments:
  --nn            neural network to use - linear / lstm / gru / convlin / unet / convgru
  --layers        number and size of layers - linear:   neuron_layer_1 neuron_layer_2 ...
                                            - lstm/gru: hidden_dimension stacked_layers
                                            - convlin:  input_channels_1 input_channels_2 ...
                                            - unet:     first_input_channel depth
                                            - convgru:  hidden_dimension_1 hidden_dimension_2 ...
  --lr            learning rate - float
  --batch_size    batch size - int
  --optim         optimizer - adam / sgd / adagrad
  --dual_path     option to create a convolution linear architecture with a dual pathway
  --bidirectional option to include bidirectionality for lstm/gru/ConvGRU
  --pretrained    option to train further on a pretrained model
  --cpu           option to switch to cpu
```

### Showing Results
#### Similarity
Executing **similarity.py** calculates the RMSE, nRMSE and SSIM compared to the input signals. The results are stored in _patient_results.csv_

The same arguments are used as in _sim_patients.py_ to specify on which network the similarity is calculated.

#### Parameter maps
Executing **visuals.py** shows the parameter maps created from different specified frameworks.

#### Bland-Altman Plots
Executing **bland_altman.py** creates bland-altman plots from the specified frameworks and the non-linear least squares method.

The same arguments are used as in _sim_patients.py_ to specify from which network the bland_altman plots are created. 
