# RML-CNN

Code covering some of the experiments in the following paper:

[Cevikalp, H., Benligiray, B., Gerek, O. N.. (2019). Semi-Supervised Robust Deep Neural Networks for Multi-Label Image Classification. In Pattern Recognition.](https://www.sciencedirect.com/science/article/abs/pii/S0031320319304649)

Use [this](https://github.com/bbenligiray/nus_wide_formatter_SRN) to create a `nus_wide.h5` and [this](https://github.com/bbenligiray/ms_coco_formatter_SRN) to create a `ms_coco.h5` file. Download `resnet101_weights_tf.h5` from [here](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294). Put these in `/rml-cnn`, then run `run.sh` to run the experiments.

Alternatively, you can use the loss functions in `ml_loss` with any model/dataset you like, see `main.py` for reference. Note that your final layer's activation should be `None` for all loss functions (including softmax).
