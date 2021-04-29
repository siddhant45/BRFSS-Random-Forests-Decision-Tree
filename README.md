# BRFSS-Random-Forests-Decision-Tree
In this notebook, we will implement a random forest in Python. With machine learning in Python, it's very easy to build a complex model without having any idea how it works. Therefore, we'll start with a single decision tree and a simple problem, and then work our way to a random forest and a real-world problem.

Once we understand how a single decision tree thinks, we can transfer this knowledge to an entire forest of trees. The problem we’ll solve is a binary classification task with the goal of predicting an individual’s health. The features are socioeconomic and lifestyle characteristics of individuals and the label is 0 for poor health and 1 for good health. This dataset was collected by the Centers for Disease Control and Prevention

https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76

Data Description 
The data has been divided into  groups : 
Variables/Columns to use:
State: _STATE 
Number of Days Physical Health Not Good: PHYSHLTH 
Number of Days Mental Health Not Good: MENTHLTH 
Could Not See Doctor Because of Cost: MEDCOST 
Ever Told Blood Pressure High: BPHIGH4 
Ever Told Had Asthma: ASTHMA3
Education Level: EDUCA 
Income Level: INCOME2 
Exercise in Past 30 Days: EXERANY2
Type of Physical Activity: EXRACT11 
Reference - BRFSS Handbook: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf 

Case scenario 1: We will try to figure out which state or states in the US have the lowest healthcare coverage or relatively lower healthcare coverage penetration respectively. The United States has structural issues in the healthcare delivery system. The availability and affordability of healthcare related services are not the same across regions or states. There are still pockets in the US where citizens are not getting the benefit of any government or private healthcare programs. Our analysis will try to find out such regions or states from the BRFSS survey data. It is important to identify such regions to design and put in place, a better healthcare delivery system.

Case Scenario 2: Our second question will delve to drill down more on the information we already gathered from our first question analysis. Here we will try to understand, for the lowest healthcare covered state (read Texas),which socio-economic class is most adversely impacted. For example we might want to know the income group and race of the people who cannot even afford to visit a doctor even if they need medical care or attention.  In a large state (like Texas), not all socio-economic class will be impacted at same degree due to lower healthcare coverage among participants. There are group of people who will be impacted more adversely. Do they belong to any specific socio-economic stratum.

Case Scenario 3: We will try to find an answer to: Does having a health plan, motivates people to go for a routine checkup within working age group (18-64yrs). We are specifically interested in the population who also reports not to see Dr. because of cost.  Routine or wellness checkup is an important tool to get insight on overall health status and work towards attaining a better state of health over a period of time. Most of the health plans,government or private, come with a free annual routine checkup. Thus there is an incentive for the population having a health plan to go for a routine checkup. It is specifically important for individuals who could not have gone to a doctor due to deductible cost. A routine checkup can help them to get a quick health snapshot and to participate in preventive care.

The training set should be used to build our machine learning models. For the training set we provide the outcome( also known as target or class label) for each data point. 
The test set should be used to see how well our model performs on unseen data. For the test set we do not provide the outcome(target variable) for each data point. For each data point in the test set use the model you trained to predict price range.


Approach & Methodology

The risk factors that impact heart disease to build binary classifiers using Random Forests and Decision Tree Networks. The steps used to accomplish this are as follows:
Dataset and Feature Selection 
Data Preprocessing 
Create 50–50 and 60–40 Datasets 
Model Building

About Decision Tree:

A decision tree is a supervised machine learning algorithm that can be used for both classification and regression problems. A decision tree is simply a series of sequential decisions made to reach a specific result. Here’s an illustration of a decision tree in action.


Let’s understand how this tree works.

First, it checks if the customer has a good credit history. Based on that, it classifies the customer into two groups, i.e., customers with good credit history and customers with bad credit history. Then, it checks the income of the customer and again classifies him/her into two groups. Finally, it checks the loan amount requested by the customer. Based on the outcomes from checking these three features, the decision tree decides if the customer’s loan should be approved or not.

The features/attributes and conditions can change based on the data and complexity of the problem but the overall idea remains the same. So, a decision tree makes a series of decisions based on a set of features/attributes present in the data, which in this case were credit history, income, and loan amount.

Now, we might be wondering - Why did the decision tree check the credit score first and not the income?
This is known as feature importance and the sequence of attributes to be checked is decided on the basis of criteria like Gini Impurity Index or Information Gain.

Gini Impurity: The Gini Impurity of a node is the probability that a randomly chosen sample in a node would be incorrectly labelled if it was labelled by the distribution of samples in the node.

Overfitting occurs when we have a very flexible model (the model has a high capacity) which essentially memorizes the training data by fitting it closely. The problem is that the model learns not only the actual relationships in the training data, but also any noise that is present. A flexible model is said to have high variance because the learned parameters (such as the structure of the decision tree) will vary considerably with the training data. 
On the other hand, an inflexible model is said to have high bias because it makes assumptions about the training data (it’s biased towards pre-conceived ideas of the data.) For example, a linear classifier makes the assumption that the data is linear and does not have the flexibility to fit non-linear relationships. An inflexible model may not have the capacity to fit even the training data and in both cases — high variance and high bias — the model is not able to generalize well to new data. 
The balance between creating a model that is so flexible it memorizes the training data versus an inflexible model that can’t learn the training data is known as the bias-variance trade-off and is a foundational concept in machine learning. 
The reason the decision tree is prone to overfitting when we don’t limit the maximum depth is because it has unlimited flexibility, meaning that it can keep growing until it has exactly one leaf node for every single observation, perfectly classifying all of them.As an alternative to limiting the depth of the tree, which reduces variance (good) and increases bias (bad), we can combine many decision trees into a single ensemble model known as the random forest.

Random Forest: 
The random forest is a model made up of many decision trees. 
Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random: 
Random sampling of training data points when building trees 
Random subsets of features considered when splitting nodes
The decision tree algorithm is quite easy to understand and interpret. But often, a single tree is not sufficient for producing effective results. This is where the Random Forest algorithm comes into the picture. 
Random Forest is a tree-based machine learning algorithm that leverages the power of multiple decision trees for making decisions. As the name suggests, it is a “forest” of trees!
But why do we call it a “random” forest? That’s because it is a forest of randomly created decision trees. Each node in the decision tree works on a random subset of features to calculate the output. The random forest then combines the output of individual decision trees to generate the final output.
In simple words:- The Random Forest Algorithm combines the output of multiple (randomly created) Decision Trees to generate the final output.

This process of combining the output of multiple individual models (also known as weak learners) is called Ensemble Learning.
Random sampling of training observations 
When training, each tree in a random forest learns from a random sample of the data points. The samples are drawn with replacement, known as bootstrapping, which means that some samples will be used multiple times in a single tree. The idea is that by training each tree on different samples, although each tree might have high variance with respect to a particular set of the training data, overall, the entire forest will have lower variance but not at the cost of increasing the bias. 
At test time, predictions are made by averaging the predictions of each decision tree. This procedure of training each individual learner on different bootstrapped subsets of the data and then averaging the predictions is known as bagging, short for bootstrap aggregating. 
Random Subsets of features for splitting nodes
The other main concept in the random forest is that only a subset of all the features are considered for splitting each node in each decision tree. Generally this is set to sqrt(n_features) for classification meaning that if there are 16 features, at each node in each tree, only 4 random features will be considered for splitting the node. (The random forest can also be trained considering all the features at every node as is common in regression. These options can be controlled in the Scikit-Learn Random Forest implementation).
The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by averaging the predictions of each individual tree.

Dataset and Feature selection
Let's Start with Importing Packages & Libraries in Python (Jupyter)

Data Cleaning 
We'll read the data in and do a little cleaning.

df = pd.read_csv('../input/SID/behavioral-risk-factor-surveillance-system/2015.csv').sample(10000, random_state = 50)
df.head()


Label Distribution

The label imbalanced means that accuracy is not the best metric.  
We won't do any data exploration in this notebook, but in general, exploring the data is a best practice. This can help you for feature engineering (which we also won't do here) or by identifying and correcting anomalies / mistakes in the data.





Below, we drop a number of columns that we should not use for modeling (they are different versions of the labels).

Split Data into Training and Testing Set

To assess our predictions, we'll need to use a training and a testing set. The model learns from the training data and then makes predictions on the testing data. Since we have the correct answers for the testing data, we can tell how well the model is able to generalize to new data. It's important to only use the testing set once, because this is meant to be an estimate of how well the model will perform on new data.

We'll save 30% of the examples for testing.


Imputation of Missing values 

We'll fill in the missing values with the mean of the column. It's important to note that we fill in missing values in the test set with the mean of columns in the training data. This is necessary because if we get new data, we'll have to use the training data to fill in any missing values.



Decision Tree on Real Data
First, we'll train the decision tree on the data. Let's leave the depth unlimited and see if we get overfitting!

#Assess Decision Tree Performance
Given the number of nodes in our decision tree and the maximum depth, we expect it has overfit to the training data. This means it will do much better on the training data than on the testing data.



Our model does outperform a baseline guess, but we can see it has severely overfit to the training data, achieving perfect ROC AUC.

Evaluate the Decision Tree 
We'll write a short function that calculates a number of metrics for the baseline (guessing the most common label in the training data), the testing predictions, and the training predictions. The function also plots the ROC curve where a better model is to the left and towards the top.

Now, construct ROC curve and confusion matrix.



There we can see the problem with a single decision tree where the maximum depth is not limited: severe overfitting to the training data.

Another method to inspect the performance of a classification model is by making a confusion matrix.

Confusion matrix


This shows the classifications predicted by the model on the test data along with the real labels. We can see that our model has many false negatives (predicted good health but actually poor health) and false positives (predicted poor health but actually good health).
Feature Importances
Finally, we can take a look at the features considered most important by the Decision Tree. The values are computed by summing the reduction in Gini Impurity over all of the nodes of the tree in which the feature is used.

Visualize Full Tree

As before, we can look at the decision tree on the data. This time, we have to limit the maximum depth otherwise the tree will be too large and cannot be converted and displayed as an image.


We can see that our model is small deep and has some nodes. To reduce the variance of our model, we could limit the maximum depth or the number of leaf nodes. Another method to reduce the variance is to use more trees, each one trained on a random sampling of the observations. This is where the random forest comes into play.

Random forest on real data
Now we can move on to a more powerful model, the random forest. This takes the idea of a single decision tree, and creates an ensemble model out of hundreds or thousands of trees to reduce the variance. Each tree is trained on a random set of the observations, and for each split of a node, only a subset of the features are used for making a split. When making predictions, the random forest averages the predictions for each of the individual decision trees for each data point in order to arrive at a final classification.

Creating and training a random forest is extremely easy in Scikit-Learn.


We can see how many nodes there are for each tree on average and the maximum depth of each tree. There were 100 trees in the forest.

We see that each decision tree in the forest has many nodes and is extremely deep. However, even though each individual decision tree may overfit to a particular subset of the training data, the idea is that the overall random forest should have a reduced variance.

Random Forest Results



The model still achieves perfect measures on the training data, but this time, the testing scores are much better. If we compare the ROC AUC, we see that the random forest does significantly better than a single decision tree.


Compared to the single decision tree, the model has fewer false postives although more false negatives. Overall, the random forest does significantly better than a single decision tree. This is what we expected!

This model does pretty well! Compared to the single decision tree, the random forest is much better able to generalize to new data.

Conclusions
In this python notebook, we built and used a random forest machine learning model in Python. Rather than just writing the code and not understanding the model, we formed an understanding of the random forest by inspecting an individual decision tree and discussing its limitations. We visualized the decision tree to see how it makes decisions and also saw how one decision tree overfit the training data. To overcome the limitations of a single decision tree, we can train hundreds or thousands of them in a single ensemble model. This model, known as a random forest, trains each tree on a different set of the training observations, and makes splits at each node based on a subset of the features leading to a model with reduced variance and better generalization performance on the testing set.
A few key concepts to take away are
Individual decision tree: intuitive model that makes decisions based on a flowchart of questions asked about feature values. Has high variance indicated by overfitting to the training data.
Gini Impurity: Measure that the decision tree tries to minimize when splitting each node. Represents the probability that a randomly selected sample from a node will be incorreclty classified according to the distribution of samples in the node.
Bootstrapping: sampling random sets of observations with replacement. Method used by the random forest for training each decision tree.
Random subsets of features: selecting a random set of the features when considering how to split each node in a decision tree.
Random Forest: ensemble model made of hundreds or thousands of decision trees using bootstrapping, random subsets of features, and average voting to make predictions.
Bias-variance tradeoff: the fundamental issue in machine learning that describes the tradeoff between a model with high complexity that learns the training data very well at the cost of not being able to generalize to the testing data (high variance), and a simple model (high bias) that cannot even learn the training data. A random forest reduces the variance of a single decision tree while also accurately learning the training data leading to better predictions on the testing data.
Hopefully this python notebook has given you not only the code required to use a random forest, but also the background necessary to understand how the model is making decisions. Machine learning is a powerful tool and it's important to not only know how to use the tool, but also to understand how it works!
Code Notebook link for reference - https://www.kaggle.com/siddhant45/random-forests-decision-tree-by-sid
