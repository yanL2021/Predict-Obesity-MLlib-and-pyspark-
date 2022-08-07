# Applied machine learning models with MLlib from pyspark to predict obesity based on people's eating habits

**Purpose**: Using MLlib from pyspark to Fit Machine Learning Models, finding the relationship between obesity and people's eating habits & their physical condition.

**Dataset**: Estimation of obesity levels based on eating habits and physical condition Data Set include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level based on BMI which is calculated by height and weight), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.</b>

The predictors in this dataset are: Frequent consumption of high caloric food (FAVC), Frequency of consumption of vegetables (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Consumption of water daily (CH20), Consumption of alcohol (CALC), Calories consumption monitoring (SCC), Physical activity frequency (FAF), Time using technology devices (TUE), Transportation used (MTRANS), Gender, Age, Height and Weight.

**Data cleaning and modification**: No missing data were detected in the dataset, however, To have a binary response in the latter modeling, we classified Obesity Type I, Obesity Type II and Obesity Type III as obesityYes (obsYes=1), the rest levels of NObesity as obesityYes (obsYes=0).

**Supervised Learning Idea and Data Split**: Supervised learning means that a variable or variables in the data set represent an output or response variable Generally speaking, supervised Learning try to relate predictors to a response variable through a model, including making inference on the model parameters, predicting a value or classifying an observation. The process is applying supervised learning algorithms to take a known set of input data (the learning set) and known responses to the data (the output), and forms models to generate reasonable predictions for the response to the new input data.

To identify the most ideal model for prediction, we split our data into a training and test set. The process of tuning the hyperparameter(s) and estimating the model parameters are only done iteratively in the training data. The test set is used to produce unbiased estimate of the performance for the final model chosen. The testing data can not be untouched or unseen during the training process. We must split our data into a training and test set before the model fitting. Otherwise it may result in overfitting since we have already used the test set to build the final model.

**Models**: We fit the data set with three different classes of models: Logistic model, Classification tree and Random Forest model. Here we are going to brief discuss the general idea of those models and how they work.

Logistic model: Logistic regression models are used mostly as a tool for data analysis and inference, where the main goal is to understand the role of the predictors in explaining the outcome. Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, homoscedasticity, and measurement level. Our data meets all the assumptions for logistic regression. First, the response is binary. Second, the observations are different patients which are independent of each other. Third, there is no multicollinearity among the predictors as we shown in the latter correlation matrix.

Classification Tree model: The basic idea of tree models is to split up predictor space into regions. Each region represents different predictions. Classification tree is to classify or predict group memberships. For a given region, usually use most prevalent class as prediction. One main advantage of trees is that they can be displayed graphically, and are easily interpreted even by a non-expert - this is especially true for small trees. The reason is that trees are very easy to explain to people since they more closely mirror human decision-making. Also, trees can easily handle categorical predictors without the need to create dummy variables. Since our response is binary, we applied a Classification Tree to fit our data.

Random Forest model: Random Forest is based on the bagging algorithm and uses Ensemble Learning technique. Random forests provide an improvement over bagging by decorrelating the trees. It forces each split to consider only a subset of the predictors. As in bagging, we build a number of decision trees on bootstrapped training samples. When building these decision trees, for each split in a tree, a random subset of predictors is chosen as split candidates from the full set of p predictors. Generally, Random Forest will provide a better prediction performance than Classification Tree. However, a disadvantage of random forest is that the resulting model is often difficult or impossible to interpret, as we are averaging many trees rather than looking at a single tree.

**Modules**:

1. `pandas`
2. `pyspark`
3. `matplotlib.pyplot`
4. `pyspark.sql`
5. `os`
6. `sys`
7. `pyspark.ml`
