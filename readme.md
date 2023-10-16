## How To Run:

Just run the script <br>
```
python3 pollution_script.py
```
in your terminal.<br>

# Beijing Pollution Forecast

## How To Run
**Method 1:** run pollution_script.py in youre preferred IDE <br>
**Method 2:** run *python3 pollution_script.py* in your terminal <br>

## Main Method
I used Long Short-Term Memory(LSTM) to forecast pollution levels of Beijing.

### What is LSTM?
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. The central role of an LSTM model is held by a memory cell known as a ‘cell state’ that maintains its state over time. The cell state is the horizontal line that runs through the top of the below diagram. It can be visualized as a conveyor belt through which information just flows, unchanged. Information can be added to or removed from the cell state in LSTM and is regulated by gates. These gates optionally let the information flow in and out of the cell. It contains a pointwise multiplication operation and a sigmoid neural net layer that assist the mechanism.

### Other methods employed
Data cleaning, Feature Engineering, Data Scaling, Train-test split, Neural Network Building, analyzing loss(the penalty for a bad prediction), testing results using root mean squared error

## Process 

### Goal 
My goal with this project is to create a model that can predict pollution levels given data about the atmosphere.

### A description of the dataset (https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate)
This dataset contains data on pollution levels and different metrics surrounding weather such as temperature, dew point, pressure, wind speed and direction, and precipitation. I found this data on Kaggle while looking at time series data.

### Preprocessing
The dataset did not require a lot of preprocessing. I only had to encode one label: the wind direction. All of the other features of the data were quantitative metrics, so the only preprocessing that was necessary was to scale the data.

## Results

### After Training
<img src=https://github.com/NicoCeresa/PersonalProjects/blob/main/PollutionForecast/Output%20Images/train.png width='300' height='200'/>

### After Validating
<img src=https://github.com/NicoCeresa/PersonalProjects/blob/main/PollutionForecast/Output%20Images/val.png width='300' height='200'/>

### After Testing
<img src=https://github.com/NicoCeresa/PersonalProjects/blob/main/PollutionForecast/Output%20Images/test.png width='300' height='200'/>

## Conclusions

### Summary Statistics
For the summary statistic, I chose to evaluate my model using the Root Mean Squared Error. I chose to use it as my accuracy metric as RMSE gives a relatively high weight to large errors. This means the RMSE is most useful when large errors are particularly undesirable. On the training data, I had an RMSE of 28.9. On the validation data, an RMSE of 26.64. On the testing data, an RMSE of 27.3.

### A contextualization of the results
I think that considering how non-complex my model is, the results were great. Sure, if I was to go deeper into this and really make a super complex model, I could achieve much better results with lower RMSEs, but that is not necessary for what I am predicting. Predicting pollution levels, though requires general accuracy, does not require extreme precision as the difference between 110 and 120 is not very noticeable. So, having RMSEs around 27 is not horrible.
