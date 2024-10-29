## TACO
This is the source code for paper ''[A Topology-aware Graph Coarsening Framework for Continual Graph Learning](https://arxiv.org/abs/2401.03077)'' to appear in The Thirty-eighth Annual Conference on Neural Information Processing Systems (Neurips 2024). 

Xiaoxue Han, Zhuo Feng, [Yue Ning](https://yue-ning.github.io/)

## Prerequisites
The code has been successfully tested in the following environment. (For older versions, you may need to modify the code)
- Python 3.8.13
- PyTorch 1.12.1+cu11.6
- dgl 0.9.1
- pygsp 0.5.1
- sklearn 1.1.2


## Getting Started

### Download raw data
- Kindle dataset: https://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt
- DBLP dataset: https://originalstatic.aminer.cn/misc/dblp.v13.7z
- ACM dataset: https://lfs.aminer.org/lab-datasets/citation/acm.v9.zip

Place the download files to `raw-data` folder. The folder structure is as follows:
```sh
- TACO-code
	- raw-data
		- data file
		- ...
	- src
  - ...
```

### Preprocess data
To preprocess the data for each dataset, please run the following commands in order under the `[dataset-name]-data-preprocessing` folder:


```python
python 0-process_dataset.py (only for ACM and DBLP datasets)
```

```python
python 1-build_graph.py
```
```python
python 2-get_largest_connected_component.py
```
```python
python 3-generate_random_masks.py
```

### Training and testing
Please run following commands for training and testing under the `src` folder. We take the dataset `kindle` with GCN as backbone GNN model as the example.


**Evaluate the TACO model**
```python
python -W ignore train.py --dataset kindle --method DYGRA --gnn GCN --reduction_rate 0.5 --buffer_size 200
```

## Cite
Please cite our paper if you find this code useful for your research:


**BibTeX**

```
@misc{han2024topologyawaregraphcoarseningframework,
      title={A Topology-aware Graph Coarsening Framework for Continual Graph Learning}, 
      author={Xiaoxue Han and Zhuo Feng and Yue Ning},
      year={2024},
      eprint={2401.03077},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.03077}, 
}
```
