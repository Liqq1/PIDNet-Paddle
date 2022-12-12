
# PIDNet-Paddle
The Paddle Implementation of PIDNet

### Train
Put the [PIDNet_M_ImageNet.pdparams](https://drive.google.com/file/d/1WsklgSkTzAoqa1w4e7P0FtOluEFsvYch/view?usp=share_link) under pretrained_models/imagenet/

For example, train the PIDNet-M on Cityscapes with batch size of 3 on 4 GPUs:
<pre><code> python tools/train_paddle.py --cfg configs/cityscapes/pidnet_medium_cityscapes.yaml GPUS (0,1,2,3) TRAIN.BATCH_SIZE_PER_GPU 3</code></pre>
