Implementation of the Deep Adversarial Metric Learning (DAML) algorithm in Chainer v4.2.0 (https://docs.chainer.org/en/v4.2.0/)

Please use the citation provided below if it is useful to your research:

Yueqi Duan, Wenzhao Zheng, Xudong Lin, Jiwen Lu, and Jie Zhou, Deep Adversarial Metric Learning, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 2780-2789.

-- Dependencies:
```bash
pip install cupy==1.0.2 chainer==2.0.2 fuel==0.2.0 tqdm
```

-- Dataset:
Stanford Cars Dataset (Cars196)
Download: https://ai.stanford.edu/~jkrause/cars/car_dataset.html or lib/datasets/cars196_downloader.py
Convert: lib/datasets/cars196_convert.py to .h5py file; put it in lib/datasets/data/cars196/

-- Usage:
```bash
python main_triplet.py
```

-- Baseline Code Reference:
deep\_metric\_learning (https://github.com/ronekko/deep_metric_learning) by [ronekko](https://github.com/ronekko)


