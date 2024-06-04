# CAReg

The official code of the paper "Few-Shot Anomaly Detection via Category-Agnostic Registration Learning".

## Preparation

### Environment

This code is built on top of [RegAD](https://github.com/MediaBrain-SJTU/RegAD). Based on that project, run `pip install -r requirements.txt` to install a few more packages. Then, you are ready to go.

### Data

Please follow [RegAD](https://github.com/MediaBrain-SJTU/RegAD) (named `Files Preparation`) to set up the datasets.

## Training

```bash
python train.py --obj class_name --shot shot_number --data_path_train train_data_path --data_path_test test_data_path
```
* Replace the `class_name` with the real object class name, e.g.,  `zipper`. 
* Replace the `shot_number` with the shot number, e.g.,  `8`. 
* Replace the `train_data_path` with the training data path, e.g.,  `Dataset/MPDD`. 
* Replace the `class_name` with the testing data path, e.g.,  `Dataset/MVTec`. 

For example,

```bash
python train.py --obj 'zipper' --shot 8 --data_path_train 'Dataset/MPDD' --data_path_test 'Dataset/MVTec'
```