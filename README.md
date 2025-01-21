![image](https://github.com/user-attachments/assets/54fc5df1-90e4-4b30-9dbb-e2cebd27fa30)


### Comparing Regression and Deep Learning Models on Texas Housing Price Predictor 

 

 

**Introduction**  

This project is a housing price predictor that is able to predict house prices in Texas, specifically the cities of Pflugerville, Del Valle, Austin, driftwood, Manor, Dripping Springs, Manchaca and West Lake Hills. 

 As a former resident who grew up and lived in Texas, I have realized that the Texas housing market is a complex and dynamic, with prices influenced by various factors. Accurately predicting housing prices is crucial for buyers, sellers and investors. However, some   existing models often rely on simplistic approaches, neglecting the intricate relationship between variables.  

The problem is an important one because reliable housing predictors for buyers and sellers enable them reduce the risk of financial loss, accurate prediction help maintain market stability by preventing drastic price fluctuations. Also, predicting housing prices help identify areas with affordable housing options, promoting accessibility for low-income families. 

The inputs to my machine learning algorithm are:  

"zipcode", "latitude", "longitude", "propertyTaxRate", "garageSpaces", "parkingSpaces", "yearBuilt","price","numPriceChanges", "latest_salemonth", "latest_saleyear", "numOfAccessibilityFeatures", "numOfAppliances", "numOfParkingFeatures", "numOfPatioAndPorchFeatures", "numOfSecurityFeatures", "numOfWaterfrontFeatures", "numOfWindowFeatures", "numOfCommunityFeatures", "lotSizeSqFt", "livingAreaSqFt", "numOfBathrooms", "numOfBedrooms", "numOfStories" 

 

I then used Linear, Gradient Boosting, Polynomial Regression Models, Random Forest and Deep learning model to predict the housing prices. From there I compare the prediction of each model  using R2 score , MAE  and RMSE  to see which model performed the best at predicting the housing price.  

 

**Related Work**

I found three additional work and two studies which explored the concept of predicting house prices using regression models vs deep learning models. (Wadkins ) explored the use of Natural Language processing, Regularization Model, K-Nearest Neighbour Model and Support Vector Regression Model which I found to be unique and creative but I noticed the dataset expert used a Recurrent Neural Network (used for NLP and speech) but did not use any Convolutional Neural Network (used to make predictions from texts, numbers and audio) or Artificial Neural Network which would be an ideal application for predicting the house prices in Texas. 

(Al-Qawasmi 60) approach was to do a comparative study of Machine Learning Models- Random Forest, CNN, CNN and Random Forest Combination, ANN and RNN to predict housing prices. He compared the efficacy of the five distinct algorithms and the results showed the measure of their R2 scores.  I found this approach interesting as it showed that the CNN and Random Forest combination had the highest R2 score.  

(Eze et al. 500) states that regression-based methods are imperfect because they suffer from issues such as multicollinearity and heteroscedasticity. Recent years have witnessed the use of machine learning methods but the results are mixed. The paper introduced the application of a new approach using deep learning models to real estate property price prediction. The paper uses a deep learning approach for modeling to improve the accuracy of real estate property price prediction with data representing sales transactions in a large metropolitan area. Three deep learning models, Long Short-Term (LSTM), Gated Recurrent Unit (GRU) and Transformer, were created and compared with other machine learning and traditional models. The results obtained for the data set with all features clearly showed that the Random Forest and Transformer models outperformed the other models. LSTM and GRU models produced the worst results, suggesting that they are perhaps not suitable to predict the real estate price. 

 

**Dataset and Features**

My dataset was sourced from Kaggle.com and consists of 15,171 records of houses in Texas that are situated in the cities listed above. This dataset was first cleaned by removing features of the house that were not directly related to predicting the price of the house. The features that were removed are:  

"homeImage", "numOfPrimarySchools", "numOfElementarySchools", "numOfMiddleSchools",  "numOfHighSchools", "avgSchoolDistance", "avgSchoolRating", "avgSchoolSize", "MedianStudentsPerTeacher", "description", "latestPriceSource", "latest_saledate", "zpid", "numOfPhotos", "streetAddress" 

With fewer, more relevant columns, the computer can now make predictions or decisions more effectively, leading to better and faster results in machine learning.  

After dropping the columns, the dataset is then checked for missing values and duplicate values, which would be dropped if there were any in this dataset but there was none. A total of 239 outliers in the price columns were dropped using the z-scored based method because the housing prices were either extremely high or extremely low.  

Before the last step of preparing the data for machine learning, I had to perform feature scaling on the specific columns below:  

{“city”, ‘hasAssociation',‘hasCooling''hasGarage’,‘hasHeating',‘hasSpa','hasView', 

'homeType'} 

Feature scaling involves changing letter and words such as “Yes” or “No” into numbers. By encoding the Boolean features as 1 or 0, the computer can now interpret and use these features directly in calculations 

Finally, Normalization is the last step I performed before machine learning.  Normalization is a specific form of feature scaling that transforms the range of features to a standard scale. It enhances the model's performance and improves the accuracy. It aids algorithms that rely on distance metrics, such as k-nearest neighbors or support vector machines, by preventing features with larger scales from dominating the learning process.  The normalization process I used is Standard Scaling because it works well when the outliers are removed from the dataset.  

 

 

**Methods**

The learning algorithms used are Linear Regression, Polynomial Regression, Gradient Boosting Regression, Random Forest and Feedforward Deep Learning.  

Linear regression models assume a linear relationship between a dependent variable (y) and one or more independent variables (x). It aims to find the best-fitting line through the data points to predict y based on x. Its Mathematical notation includes: y = mx + b  

y: Dependent variable (predicted value) 

x: Independent variable (predictor) 

m: Slope (how much y changes for a unit change in x) 

b: Intercept (value of y when x is 0) 

 

Gradient boosting is an ensemble learning technique that combines multiple weak models (typically decision trees) to create a strong predictive model. It works by iteratively adding models that focus on the errors made by previous models. It does not include a single mathematical equation, but involves concepts like: Loss Function (Measures the error between predicted and actual values), Gradient Descent (Optimizes the model parameters to minimize the loss function), Weak Learners (Simple models (e.g., decision trees) that are combined) and Ensemble (The final model is a combination of these weak learners) 

 

Polynomial regression models the relationship between a dependent variable and independent variables using polynomial functions. This allows for capturing non-linear relationships between variables. Its mathematical notation include:  

y = b₀ + b₁x + b₂x² + ... + bₙxⁿ  

y: Dependent variable 

x: Independent variable 

b₀, b₁, b₂, ..., bₙ: Coefficients 

 

Random Forest Regression is a versatile ensemble learning method that combines multiple decision trees to make accurate predictions. It's particularly effective for handling complex datasets and reducing overfitting. The mathematical notation for random forest:  

ŷ = (1/N) * Σ(yᵢ) 
 

where: 

ŷ: Predicted value for a given input 

N: Number of data points in the leaf node 

yᵢ: Actual value of the i-th data point in the leaf node 

 

 

To sum it all up for the regression models-linear and polynomial regressions make direct predictions, while gradient boosting and random forest combines predictions from multiple models. 

Lastly, I also used Feedforward Neural Network deep learning model (also known as a fully connected neural network). Feedforward neural networks are the fundamental building blocks of deep learning. Their ability to learn complex patterns and hierarchical representations is the reason I used this deep learning model to predict the Texas house prices.  This model works by first receiving the input data, secondly, processing the input data through multiple layers, each layer learning more complex features and then lastly produces the prediction.  The mathematical notation includes:  

z^(l) = w^(l)^T * a^(l-1) + b^(l) 
a^(l) = f(z^(l)) 

Where: 

l: Layer index 

w^(l): Weight matrix for layer l 

b^(l): Bias vector for layer l 

a^(l-1): Activation output from the previous layer 

 

In order for me to test and compare all of the models amongst each other, I used an R2 score, MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error) in order to come to my conclusions.   

 

 

**Experiments/Results/Discussion**

Based on the R2, MAE and RMSE scores calculated as shown on the graphs below. The graph shows that Random Forest Regression proved to be the best at predicting with Gradient Boosting being the second best. The graph also shows the R2 scores for Linear and Polynomial Regression have a value below 0. This means that these models are performing worse than simply predicting the average value of the target variable, essentially meaning they are not adding any useful information to the prediction.  

Also, I did an evaluation of the RMSE and MAE error valuations for the regression models. Random Forest Regression Model proved to have the least error as can be seen on the graph below. 

 

 

The Deep Learning model learning rate that was chosen is 0.000001. It was chosen by manually changing the rate and watching the models learning rate improved the smaller it got. The graph below shows that the model even though not learning accurately. Its validation loss improved when I increased the epochs to 500. Indicating the deep learning model was having a hard time learning the complexity of the data. 

 

To prevent overfitting, I used an early stopping technique that monitors the validation loss and it stops the training when the validation loss stops improving for a certain number of epochs. 

Finally, I tested the model with a real-life example of a Texas Home located in the city of Pflugerville with data supplied from Zillow.com. The graph below shows that Random Forest and Gradient Boosting predicted the closet to the actual house price. Linear Regression and the Feedforward Deep Learning model perform the worst. 

 

 

**Conclusion**

In concluding, this report has analyzed and test performance of Linear, Polynomial, Gradient Boosting and Feedforward Deep Learning models on predicting the housing prices in Texas. My findings indicate that Random Forest Boosting is the best of the other models when it comes to predicting house prices when working with a medium to small dataset.  To maximize the accuracy of the predictions, I would train the deep learning model, especially the Artificial Neural Network, on a large dataset to see if it would perform better.  While this study has provided valuable insights, further research and large dataset is needed to explore better models that could possibly yield more accurate results. 

 

 

**Team Members**  

Burnie Murray
Derrick Kessie

 

**References**

E. Eze, S. Sujith, M. S. Sharif and W. Elmedany, "A Comparative Study For Predicting House Price Based on Machine Learning," 2023 4th International Conference on Data Analytics for Business and Industry (ICDABI), Bahrain, 2023, pp. 75-81, doi: 10.1109/ICDABI60145.2023.10629399. 
keywords: {Radio frequency;Industries;Analytical models;Linear regression;Linearity;Predictive models;Robustness;Time-series;machine learning;k-nearest neighbours (k-NN); linear regression;decision tree;random forest}, 

 

J. Al-Qawasmi, “Machine Learning Applications in Real Estate: Critical Review of Recent Development”, vol. 647 IFIP. Springer International Publishing, 2022. 

Wadkins, Jen. " Austin Housing - EDA, NLP, Models, Visualizations " Kaggle, 08 August 2021, https://www.kaggle.com/code/threnjen/austin-housing-eda-nlp-models-visualizations/comments 

 
