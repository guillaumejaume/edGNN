# edGNN: A simple and powerful GNN for labeled graphs

The ability of a graph neural network (GNN) to leverage both the graph topology and graph labels is fundamental to build discriminative node and graph embeddings. Building on previous work, we theoretically show that our model for directed labeled graphs, edGNN, is as powerful as the Weisfeiler-Lehman algorithm for graph isomorphism. Our experiments support our theoretical findings, confirming that graph neural networks can be effectively used for inference problems on directed graphs with both node and edge labels.

## Installation

Create a `virtualenv`:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the package as editable and any dependencies:

```sh
pip3 install -e .
```

## Train a model

a. Preprocess:

```sh
preprocess <dataset_family> --dataset <datase_name> --out_folder <path_to_preprocessed_data>
```

Currently, the 2 dataset families are `dortmund` (includes `PTC_FR`, `PTC_FM` and `PTC_MM`, `PTC_MR` and `MUTAG`) for graph classification and `dglrgcn` (includes `mutag` and `aifb`) for node classification. 

b. Run a model:

```sh
run_model --dataset <dataset_name> --config_fpath <path_to_configuration_files> --data_path <path_to_preprocessed_data>
```

Configuration files for our model `edGNN` and `R-GCN` can be found in `core/models/config_files/`
For other options to the `run_model` app, refer to the implementation.

## Examples

### a. Graph classification of the PTC_FR dataset:

(i) with edGNN:

```sh
cd bin && mkdir ../preprocessed_graphs && cd ../preprocessed_graphs && mkdir ptc_fr && cd ../bin/
preprocess dortmund --dataset ptc_fr --out_folder ../preprocessed_graphs/ptc_fr/
run_model --dataset ptc_fr  --config_fpath ../core/models/config_files/config_edGNN_graph_class.json  --data_path ../preprocessed_graphs/ptc_fr/ --n-epochs 40
```

Default config are the same as reported in the paper. `PTC_FR` accuracy around `88%`.  

(ii) with R-GCN:

```sh
run_model --dataset ptc_fr  --config_fpath ../core/models/config_files/config_RGCN_graph_class.json  --data_path ../preprocessed_graphs/ptc_fr/ --n-epochs 40
```

Default config are the same as reported in the paper. `PTC_FR` accuracy around `86%`.  

### b. Node classification of the AIFB dataset:

```sh
cd bin && mkdir ../preprocessed_graphs && cd ../preprocessed_graphs && mkdir aifb && cd ../bin/
preprocess dglrgcn --dataset aifb --out_folder ../preprocessed_graphs/aifb --reverse_edges True
run_model --dataset aifb  --config_fpath ../core/models/config_files/config_edGNN_node_class.json  --data_path ../preprocessed_graphs/aifb/ --n-epochs 400 --weight-decay 0 --lr 0.005
```

Default config are the same as reported in the paper. `AIFB` accuracy around `91%`.  


Note that if you are running the code with a GPU, you should add the argument `--gpu 0` to `run_model`.
