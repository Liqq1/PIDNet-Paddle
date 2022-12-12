
# PIDNet-Paddle
The Paddle Implementation of PIDNet

### Train

For example, train the PIDNet-M on Cityscapes with batch size of 3 on 4 GPUs:
<pre><code> python tools/train_paddle.py --cfg configs/cityscapes/pidnet_medium_cityscapes.yaml GPUS (0,1,2,3) TRAIN.BATCH_SIZE_PER_GPU 3</code></pre>
