
# vehicle-cut-in-detection
Intel Vehicle Cut In Detecteion Project

# [ Vehicle Cut In Detection Using Monocular Camera ]
Purpose is to detect the cut in of vehicles and give warning to the driver  

 

  
# Introduction
>Bounding box and Depth are extracted from image data to predict the distance.    

    
  
# Model Process
> First Vehicle Deetection is Done and then Depth is estimated finally we then calculate the distance between two cars.
> Detr is used to do the vehicle detection and then GLPN model is used to perfrom the monocular depth estimation
  
# Dataset
**KITTI DATASET HAS BEEN USED**  

------------
- **Train data (# number of Data: 21,616)**  

| Model | MAE | RMSE | Accuracy |
| ------------- | ------------- | ------------- | ------------- | 
| `LSTM` | 0.6988 | 1.4736 | 0.9746 |  
  
- **Test data (# number of Data: 2,703)**  

| Model | MAE | RMSE | Accuracy | Pre-trained | scaler file |

| `LSTM` | 1.1658 | 2.1420 | 0.9526 | [LSTM_16]| [scaler]|  

------------

   
1) Make Our Datasets   

```
Splitting of the datasets has been done with 3 csv files of train , test and valid , are Available in  the Datasets Directory
```
  
# Training 
If you want reproduce the training part , the training folder has the LSTMtraining notebook , which will generate the ODD_variable16.pth model file.(Already the file has been provided in the model folder)

# Testing
LSTM_RUNNING is given inside the run folder which can be used to directly run the project without training again


# References
[DETR](https://github.com/facebookresearch/detr)   
[GLP-depth](https://github.com/vinvino02/GLPDepth)   

 
e3c0939 (pushing)

=======
# vehicle-cut-in-detection
Intel Vehicle Cut In Detecteion Project
>>>>>>> origin/main
>>>>>>> 64189c1 (new change)
