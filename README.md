# FeatNet

Source code and data for the paper *Multi-Granularity based Feature Interaction Pruning Algorithm for CTR  Prediction* which is accepted by CCIR 2022.

## Requirements

```
Python3
Pytorch >= 1.7
tensorboard
tfrecord
```

## Run

After downloading the project, please change the *absPath* in file *"Config/FeatureConfig.json"* to your local address. Other options in FeatureConfig.json:

```python
usage: 	[absPath] 			# Absolute path of the project.
		[epoch]				# The maxium epoches of training.
		[savedModelPath]	# Path to save the trained models.
    	[datasetPath]		# Path to load dataset.
        [modelName]			# The model to train.
```

After setup all options, run the project by:

`````
python main_0.py
`````
