# DeepBCTPred

### Framework of bladder cancer tissue prediction
![DeepBCTPred_page-0001](https://github.com/user-attachments/assets/deffa28b-5dfc-47d1-87e4-ae15dc101efd)



### Data availability
All training and independent datasets are given in [Dataset](Dataset) folder and [Bladder Cancer](https://drive.google.com/drive/folders/1Tm3ItdAmjxwEAZNo-CIrujjmXdqGsG4S?usp=sharing)

### Environments
OS: Pop!_OS 22.04 LTS

Python version: Python 3.9.19


Used libraries: 
```
numpy==1.26.4
pandas==2.2.1
pytorch==2.4.1
xgboost==2.0.3
pickle5==0.0.11
scikit-learn==1.2.2
matplotlib==3.8.2
timm==1.0.11
torchvision==0.19.1
pillow==10.4.0
huggingface-hub==0.24.6
torcheval==0.0.7
opencv-python==4.10.0.84
scikit-image==0.24.0
```

### Reproduce results
1.  Reproducable codes are given. Training, validation and testing scripts are also provided in [Training](Training), [Validation](Validation) and [Testing](Testing) folders respectively. See the examples with 'sample_1.png', 'sample_2.png', 'sample_3.png' and 'sample_4.png'.
2.  Additional files that were not given due to the size, are provided in [Google drive](https://drive.google.com/drive/folders/12rSChKSW_HkcQr_-KCXtE42zXRMRMaz0?usp=sharing)

### Prediction
Firsly, you need to add the image file in [Prediction](Prediction) folder. Then, run the [predict.py](predict.py) file.

### Heatmap
Firsly, you need to add the image file in [Heatmap](Heatmap) folder. Then, run the [main.py](main.py) file.

### Reproduce previous paper metrics
In [prev_paper](prev_paper), scripts are provided for reproducing the results of the previous papers.
