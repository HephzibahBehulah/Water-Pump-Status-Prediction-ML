# Pump it Up : Data-Mining the Water Table
#### Link to the Databricks notebook: Click [here](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/167428040012665/848460877589804/8971546509206599/latest.html) <br>

The goal is to predict the operating condition of a waterpoint for each record in the dataset.You are provided the following set of information about the waterpoints

* amount_tsh - Total static head (amount water available to waterpoint) <br>
* date_recorded - The date the row was entered<br>
* funder - Who funded the well<br>
* gps_height - Altitude of the well<br>
* installer - Organization that installed the well<br>
* longitude - GPS coordinate<br>
* latitude - GPS coordinate<br>
* wpt_name - Name of the waterpoint if there is one<br>
* num_private -<br>
* basin - Geographic water basin<br>
* subvillage - Geographic location<br>
* region - Geographic location<br>
* region_code - Geographic location (coded)<br>
* district_code - Geographic location (coded)<br>
* lga - Geographic location<br>
* ward - Geographic location<br>
* population - Population around the well<br>
* public_meeting - True/False<br>
* recorded_by - Group entering this row of data<br>
* scheme_management - Who operates the waterpoint<br>
* scheme_name - Who operates the waterpoint<br>
* permit - If the waterpoint is permitted<br>
* construction_year - Year the waterpoint was constructed<br>
* extraction_type - The kind of extraction the waterpoint uses<br>
* extraction_type_group - The kind of extraction the waterpoint uses<br>
* extraction_type_class - The kind of extraction the waterpoint uses<br>
* management - How the waterpoint is managed<br>
* management_group - How the waterpoint is managed<br>
* payment - What the water costs<br>
* payment_type - What the water costs<br>
* water_quality - The quality of the water<br>
* quality_group - The quality of the water<br>
* quantity - The quantity of water<br>
* quantity_group - The quantity of water<br>
* source - The source of the water<br>
* source_type - The source of the water<br>
* source_class - The source of the water<br>
* waterpoint_type - The kind of waterpoint<br>
* waterpoint_type_group - The kind of waterpoint<br>

## The labels in this dataset

![](http://drivendata.materials.s3.amazonaws.com/pumps/labeldistribution.png)

Distribution of Labels<br>
The labels in this dataset are simple. There are three possible values:<br>

* functional - the waterpoint is operational and there are no repairs needed<br>
* functional needs repair - the waterpoint is operational, but needs repairs<br>
* non functional - the waterpoint is not operational<br>

![](https://github.com/chanddu/Pump-it-Up-Data-Mining-the-Water-Table/blob/master/Workflow.png)


## Coding Language/Environment used:
We have use the Databricks platform to achieve our task. The programming language used is scala and we used Spark's MLlib library to build the model. We ran it on a Community Optimized Spark 2.1 cluster with 6GB of memory.

Initially, we merged the Training set values with the corresponding Training set labels into a single CSV file with the help of ID column. Having the label in the same place as the features makes it easy to build the model. Then, we mapped the class labels, non-functional, functional, functional needs repair to 0, 1, 2 respectively. 

We used 80% of the original data as the training data to build our model. We evaluated the model on rest of the 20% data. We used the Mean Square error metric to do that task. As this is a multi-label classification problem, we experimented with several classification techniques and found Logistic regression to be appropriate. We used the MLlib's Logistic regression, to build our model and we obtained labels for the Test set values CSV file. We used assembler, normalizer and pipeline concepts to build the model.

## Results
We have split the data into 80-20 and calculated the Mean Squared Error with the help of MLlib's RegressionMetrics and found the value to be 0.257, which is decent. <br>
We have classified our test data. A screenshot of a few results is provided below.

![](https://github.com/chanddu/Pump-it-Up-Data-Mining-the-Water-Table/blob/master/Picture1.png)
