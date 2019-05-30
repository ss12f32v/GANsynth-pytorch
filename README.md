# GANsynth-pytorch
PyTorch  implementation of GANsynth  


## Note
This repo only support the best setting in the GANsynth paper which is a simple version comparing to orginal Tensorflow version by magenta team.
So if you want to test other frequency setting, you may need to modify the code.



## Prepare Data
Use Make Training Data notebook to generate HDF5 file for training. 

## Train a new model
You have to make a directory to save model checkpint and output spectrum first.
```sh
    python3 train.py
```

## Inference 
Use Inference notebook to load the model and generate audio.

## Example Generated Audio
 https://drive.google.com/open?id=1tNnOtcqCpgTTXGmkHJBA4K6MalBjdsPC

## Reference 
- https://github.com/shanexn/pytorch-pggan
- https://arxiv.org/abs/1902.08710