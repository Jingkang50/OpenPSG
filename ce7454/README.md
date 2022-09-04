# CE7454 Assignment 1: PSG Classification

## Get Started

We specially build the tiny codebase here (in this directory) to help our students to quickly get started.

First off, let's clone or fork the codebase and enter in the `ce7454` directory. Don't forget to star the repo if you find the assignment is interesting and instructive.
Then we download the data [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpU1E9PvC1RNrrhrubGs8gMBGy5ayyfPo6I8HcA5BU7g2Q?e=cJjmgy) and unzip the data at the correct place. Eventually, your `ce7454` folder should looks like this:
```
ce7454
├── checkpoints
├── data
│   ├── coco
│   │   ├── panoptic_train2017
│   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   └── val2017
│   └── psg
│       ├── psg_cls_basic.json
│       └── psg_val_advanced.json
├── results
├── dataset.py
├── evaluator.py
├── ...
```


We provide 4500 training data, 500 validation data, and 500 test data.
Notice that there might not be exactly 4500 training images (so are val/test images) as some images are annotated twice, and we consider one annotation as one sample.

Then, we need to setup the environment. We use `conda` to manage our dependencies. The code is heavily dependent on PyTorch.

```bash
conda install python=3.7 pytorch=1.7.0 torchvision=0.8.0 torchaudio==0.7.0 cudatoolkit=10.1
pip install tqdm
```

Finally, make sure your working directory is `ce7454`, and let's train the model!
```bash
python main.py
```

You can explore the project by reading from the `main.py` and dive in. Good Luck!!
