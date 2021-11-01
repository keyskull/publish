---
tags: [Business Intelligence, BI, Data Mining, SQL Server, algorithm]
name: "Description of Data Mining and Six of the algorithms available in SQL Server"
---

# Description of Data Mining and Six of the algorithms available in SQL Server

Date: 9 March 2021

## Introduction 
This paper will discuss about the definition of Data Mining and the six algorithms available in SQL Server. With my understanding, Data Mining is a way to recognize an order to extract the specific data from enormous data mart. In SQL Server, it is supporting six Data mining algorithms to help us explore the data usefully, which has Microsoft Decision Trees, Microsoft Linear Regression, Microsoft Naïve Bayes, Microsoft Clustering, Microsoft Association Rules, and Microsoft Sequence Clustering, etc.

## Data mining
Data mining enables us to put computing power to work, combing through  mounds of data to find meaningful bits of information. Data mining  takes this number-crunching responsibility out of our hands. We do,  however, need to follow a number of steps to prepare the data and the  algorithm for the mining process. We also need to evaluate the result  to find the gold among the rock produced. (Larson, 2008) With my  understanding, Larson's book that data mining is a multi-step process  which has 5 following steps to deployment the mining program. 

> The steps for data mining are:
1. Problem Definition
2. Data Preparation
3. Training
4. Validation
5. Deployment

Processing these following steps for Data Mining cloud clarify all the processes run correctly.

While we doing the Data Mining process of training, we use selected algorithm to training the data mining model. In SQL Server we have serval data mining algorithms provided.

## Microsoft Decision Tree 
The Microsoft Decision Trees algorithm is one of the easiest algorithms to understand because it creates a tree structure during its training process. (You probably already guessed that from the name.) The tree structure is then used to provide predictions and analysis. (Larson, 2008)

## Microsoft Linear Regression 
 The Microsoft Linear Regression algorithm is a specialized implementation of the Microsoft Decision Trees algorithm. As the name suggests, this algorithm is used to model a linear relationship between two numeric variables. Using that linear relationship, if we know the value of one variable, called the independent variable, we can predict the value of the other variable, called the dependent variable. (Larson, 2008)

## Microsoft Naïve Baye 
Donald Farmer, Principal Program Manager for Microsoft Data Mining, claims because there is a Naïve Bayes algorithm, there must be a "deeply cynical" Bayes algorithm out there somewhere in data mining land. I guess it is needed to bring balance to data mining's version of the "force." We will try not to be too naïve as we explore the benefits and shortcomings of this algorithm. (Larson, 2008)

## Microsoft Clusterin
The Microsoft Clustering algorithm builds clusters of entities as it processes the training data set. This is shown in Figure 13-14. Once the clusters are created, the algorithm analyzes the makeup of each cluster. It looks at the values of each attribute for the entities in the cluster. (Larson, 2008)

## Microsoft Association Rule 
As the name suggests, the Microsoft Association Rules algorithm is used for association. Therefore, to use this algorithm, we must have entities that are grouped into sets within our data. Refer to the section "Association" if you need more information. (Larson, 2008)

## Microsoft Sequence Clustering
The Microsoft Sequence Clustering algorithm was developed by Microsoft Research. As the name implies, the Microsoft Sequence Clustering algorithm is primarily used for sequence analysis, but it has other uses as well. (Larson, 2008)

Data Mining algorithms are having different mainly use and supplely use with the purpose. Here we use a chart to show Data Mining algorithm's feature below:

**Figure 1**

|                               | Classification | Regression | Segmentation | Association | Sequence Analysis |
| ----------------------------- | -------------- | ---------- | ------------ | ----------- | ----------------- |
| Microsoft Decision Trees      | P              | S          |              | S           |                   |
| Microsoft Linear Regression   |                | P          |              |             |                   |
| Microsoft Naïve Bayes         | P              |            |              |             |                   |
| Microsoft Clustering          | S              | S          | P            |             |                   |
| Microsoft Association Rules   |                |            |              | P           |                   |
| Microsoft Sequence Clustering | S              | S          | P            |             | P                  |

	P= Primary, S= Secondary

## Conclusion 
The concept of Data Mining steps is similar to scientific method process, both can use those steps to develop rigorous experimental process. About the Data Mining in SQL Server, Microsoft provided useful tools cloud help us more operational to various data from different data source.

## REFERENCES
Larson, B. (2016, November 4). *Delivering Business Intelligence with Microsoft SQL Server 2016, Fourth Edition, 4th Edition*. *\[\[VitalSource Bookshelf version\]\].* Retrieved from vbk://9781259641497
Larson, B. (2008). *Delivering Business Intelligence with Microsoft SQL Server 2008* (2nd ed.). McGraw-Hill Education.
