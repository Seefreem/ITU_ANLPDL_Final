# ITU_ANLPDL_Final  

This repository contains the course project for **Advanced Natural Language Processing and Deep Learning (Autumn 2024)** at **IT University of Copenhagen**.  

## Getting Started  

To reproduce the results of the **GMM models**, run the following command:  

```sh
python main.py --dataset coqa --model google/gemma-2-2b-it --judge_model google/gemma-2-9b-it --output_dir ./cache --data_split train
```

This command will:  
1. Download the target models and dataset.  
2. Generate answers for the dataset's questions.  
3. Load the judge model to label responses as *known* or *unknown*.  
4. Extract last-token representations and train the **GMM models**.  

At this stage, only training is performed, and test data is not yet available. To generate ground truth labels for the **test set**, rerun the command with `--data_split test`:  

```sh
python main.py --dataset coqa --model google/gemma-2-2b-it --judge_model google/gemma-2-9b-it --output_dir ./cache --data_split test
```

### Baseline Evaluation  

To compute baseline scores, run:  

```sh
python baseline.py --dataset coqa --model google/gemma-2-2b-it
```

### Probing  

For **MLP probing**, use:  

```sh
python mlp_probe.py --dataset coqa --model google/gemma-2-2b-it
```

For **linear probing**, refer to the `linear_probing.ipynb` notebook.  

