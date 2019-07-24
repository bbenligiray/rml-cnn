# RML-CNN

(Some of the) code used in the following paper:

[Cevikalp, H., Benligiray, B., Gerek, O. N., & Saribas, H. (2019). Semi-Supervised Robust Deep Neural Networks for Multi-Label Classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 9-17).](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Deep%20Vision%20Workshop/Cevikalp_Semi-Supervised_Robust_Deep_Neural_Networks_for_Multi-Label_Classification_CVPRW_2019_paper.pdf)

Use [this](https://github.com/bbenligiray/nus_wide_formatter_SRN) to create a `nus_wide.h5` and [this](https://github.com/bbenligiray/ms_coco_formatter_SRN) to create a `ms_coco.h5` file. Download `resnet101_weights_tf.h5` from [here](https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294). Put these in `/rml-cnn`, then run `run.sh` to run the experiments.

Alternatively, you can use the loss functions in `ml_loss` with any model/dataset you like, see `main.py` for reference. Note that your final layer's activation should be `None` for all loss functions (including softmax).
