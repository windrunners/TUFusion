# TUFusion
TUFusion is an image fusion algorithm, more precisely, a versatile or universal image fusion algorithm that can be applied across multiple domains. Currently, the algorithm has demonstrated excellent performance in areas such as infrared-visible, multi-exposure, and multi-modal medical image fusion. Unlike other image fusion algorithms, TUFusion requires only one set of weights and models to handle multiple tasks effectively.


# Update
[2023/7] "test_image.py" now supports input of images with any size.  
[2023/6] We've updated some of the files.  
[2022/12] The initial code has been uploaded.


# Citation
```
@article{zhao2023tufusion,
    title={TUFusion: A Transformer-based Universal Fusion Algorithm for Multimodal Images},
    author={Zhao, Yangyang and Zheng, Qingchun and Zhu, Peihao and Zhang, Xu and Ma, Wenpeng},
    journal={IEEE Transactions on Circuits and Systems for Video Technology},
    year={2023},
    publisher={IEEE}
}
```

# data set
## data set for training
Download MSCOCO as a file named "MSCOCO 2014" and place the file in the main folder, then you can train.

## data set for test
Put your own test data into the corresponding file.


# train
Run the "train_tufusion.py" file.

# test
Run the "test_image.py" file.

# Download the best models for our study
(1) Create a new folder named "models" under the main file;
(2) You can download the trained model in this article from the following link:
the url: https://pan.baidu.com/s/1Te-IC0jwQsiSOqpPH8O1_A 
password: rmsj
