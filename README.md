# A-B-testing-using-Python

Analyze A/B Test Results

This project will assure you have mastered the subjects covered in the statistics lessons. We have organized the current notebook into the following sections:

    Introduction
    Part I - Probability
    Part II - A/B Test
    Part III - Regression
    Final Check
    Submission

Specific programming tasks are marked with a ToDo tag.

Introduction

A/B tests are very commonly performed by data analysts and data scientists. For this project, you will be working to understand the results of an A/B test run by an e-commerce website. Your goal is to work through this notebook to help the company understand if they should:

    Implement the new webpage,
    Keep the old webpage, or
    Perhaps run the experiment longer to make their decision.

Each ToDo task below has an associated quiz present in the classroom. Though the classroom quizzes are not necessary to complete the project, they help ensure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the rubric specification.

    Tip: Though it's not a mandate, students can attempt the classroom quizzes to ensure statistical numeric values are calculated correctly in many cases.

Part I - Probability

To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

ToDo 1.1

Now, read in the ab_data.csv data. Store it in df. Below is the description of the data, there are a total of 5 columns:
Data columns 	Purpose 	Valid values
user_id 	Unique ID 	Int64 values
timestamp 	Time stamp when the user visited the webpage 	-
group 	In the current A/B experiment, the users are categorized into two broad groups.
The control group users are expected to be served with old_page; and treatment group users are matched with the new_page.
However, some inaccurate rows are present in the initial data, such as a control group user is matched with a new_page. 	['control', 'treatment']
landing_page 	It denotes whether the user visited the old or new webpage. 	['old_page', 'new_page']
converted 	It denotes whether the user decided to pay for the company's product. Here, 1 means yes, the user bought the product. 	[0, 1]

</center> Use your dataframe to answer the questions in Quiz 1 of the classroom.

    Tip: Please save your work regularly.

a. Read in the dataset from the ab_data.csv file and take a look at the top few rows here:

df = pd.read_csv('ab_data.csv')
df.head()

	user_id 	timestamp 	group 	landing_page 	converted
0 	851104 	2017-01-21 22:11:48.556739 	control 	old_page 	0
1 	804228 	2017-01-12 08:01:45.159739 	control 	old_page 	0
2 	661590 	2017-01-11 16:55:06.154213 	treatment 	new_page 	0
3 	853541 	2017-01-08 18:28:03.143765 	treatment 	new_page 	0
4 	864975 	2017-01-21 01:52:26.210827 	control 	old_page 	1

b. Use the cell below to find the number of rows in the dataset.

df.shape[0]

294478

c. The number of unique users in the dataset.

df.nunique()

user_id         290584
timestamp       294478
group                2
landing_page         2
converted            2
dtype: int64

d. The proportion of users converted.

df['converted'].mean()

0.11965919355605512

e. The number of times when the "group" is treatment but "landing_page" is not a new_page.

df.query('group == "treatment" and landing_page != "new_page"').shape[0]

1965

f. Do any of the rows have missing values?

df.isnull().sum()

user_id         0
timestamp       0
group           0
landing_page    0
converted       0
dtype: int64

ToDo 1.2

In a particular row, the group and landing_page columns should have either of the following acceptable values:
user_id 	timestamp 	group 	landing_page 	converted
XXXX 	XXXX 	control 	old_page 	X
XXXX 	XXXX 	treatment 	new_page 	X

It means, the control group users should match with old_page; and treatment group users should matched with the new_page.

However, for the rows where treatment does not match with new_page or control does not match with old_page, we cannot be sure if such rows truly received the new or old wepage.

Use Quiz 2 in the classroom to figure out how should we handle the rows where the group and landing_page columns don't match?

a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.

# Remove the inaccurate rows, and store the result in a new dataframe df2
df2 = df.drop(df[(df['group'] == "treatment") & (df['landing_page'] != "new_page")].index)
df2.drop(df[(df['group'] == "control") & (df['landing_page'] != "old_page")].index, inplace = True)

df2

	user_id 	timestamp 	group 	landing_page 	converted
0 	851104 	2017-01-21 22:11:48.556739 	control 	old_page 	0
1 	804228 	2017-01-12 08:01:45.159739 	control 	old_page 	0
2 	661590 	2017-01-11 16:55:06.154213 	treatment 	new_page 	0
3 	853541 	2017-01-08 18:28:03.143765 	treatment 	new_page 	0
4 	864975 	2017-01-21 01:52:26.210827 	control 	old_page 	1
5 	936923 	2017-01-10 15:20:49.083499 	control 	old_page 	0
6 	679687 	2017-01-19 03:26:46.940749 	treatment 	new_page 	1
7 	719014 	2017-01-17 01:48:29.539573 	control 	old_page 	0
8 	817355 	2017-01-04 17:58:08.979471 	treatment 	new_page 	1
9 	839785 	2017-01-15 18:11:06.610965 	treatment 	new_page 	1
10 	929503 	2017-01-18 05:37:11.527370 	treatment 	new_page 	0
11 	834487 	2017-01-21 22:37:47.774891 	treatment 	new_page 	0
12 	803683 	2017-01-09 06:05:16.222706 	treatment 	new_page 	0
13 	944475 	2017-01-22 01:31:09.573836 	treatment 	new_page 	0
14 	718956 	2017-01-22 11:45:11.327945 	treatment 	new_page 	0
15 	644214 	2017-01-22 02:05:21.719434 	control 	old_page 	1
16 	847721 	2017-01-17 14:01:00.090575 	control 	old_page 	0
17 	888545 	2017-01-08 06:37:26.332945 	treatment 	new_page 	1
18 	650559 	2017-01-24 11:55:51.084801 	control 	old_page 	0
19 	935734 	2017-01-17 20:33:37.428378 	control 	old_page 	0
20 	740805 	2017-01-12 18:59:45.453277 	treatment 	new_page 	0
21 	759875 	2017-01-09 16:11:58.806110 	treatment 	new_page 	0
23 	793849 	2017-01-23 22:36:10.742811 	treatment 	new_page 	0
24 	905617 	2017-01-20 14:12:19.345499 	treatment 	new_page 	0
25 	746742 	2017-01-23 11:38:29.592148 	control 	old_page 	0
26 	892356 	2017-01-05 09:35:14.904865 	treatment 	new_page 	1
27 	773302 	2017-01-12 08:29:49.810594 	treatment 	new_page 	0
28 	913579 	2017-01-24 09:11:39.164256 	control 	old_page 	1
29 	736159 	2017-01-06 01:50:21.318242 	treatment 	new_page 	0
30 	690284 	2017-01-13 17:22:57.182769 	control 	old_page 	0
... 	... 	... 	... 	... 	...
294448 	776137 	2017-01-12 05:53:12.386730 	treatment 	new_page 	0
294449 	883344 	2017-01-22 23:15:58.645325 	treatment 	new_page 	0
294450 	825594 	2017-01-06 12:37:08.897784 	treatment 	new_page 	0
294451 	875688 	2017-01-14 07:19:49.042869 	control 	old_page 	0
294452 	927527 	2017-01-12 10:52:11.084740 	control 	old_page 	0
294453 	789177 	2017-01-17 18:17:56.215378 	control 	old_page 	0
294454 	937338 	2017-01-19 03:23:22.236666 	treatment 	new_page 	0
294455 	733101 	2017-01-23 12:52:58.711914 	treatment 	new_page 	0
294456 	679096 	2017-01-02 16:43:49.237940 	treatment 	new_page 	0
294457 	691699 	2017-01-09 23:42:35.963486 	treatment 	new_page 	0
294458 	807595 	2017-01-22 10:43:09.285426 	treatment 	new_page 	0
294459 	924816 	2017-01-20 10:59:03.481635 	control 	old_page 	0
294460 	846225 	2017-01-16 15:24:46.705903 	treatment 	new_page 	0
294461 	740310 	2017-01-10 17:22:19.762612 	control 	old_page 	0
294462 	677163 	2017-01-03 19:41:51.902148 	treatment 	new_page 	0
294463 	832080 	2017-01-19 13:18:27.352570 	control 	old_page 	0
294464 	834362 	2017-01-17 01:51:56.106436 	control 	old_page 	0
294465 	925675 	2017-01-07 20:38:26.346410 	treatment 	new_page 	0
294466 	923948 	2017-01-09 16:33:41.104573 	control 	old_page 	0
294467 	857744 	2017-01-05 08:00:56.024226 	control 	old_page 	0
294468 	643562 	2017-01-02 19:20:05.460595 	treatment 	new_page 	0
294469 	755438 	2017-01-18 17:35:06.149568 	control 	old_page 	0
294470 	908354 	2017-01-11 02:42:21.195145 	control 	old_page 	0
294471 	718310 	2017-01-21 22:44:20.378320 	control 	old_page 	0
294472 	822004 	2017-01-04 03:36:46.071379 	treatment 	new_page 	0
294473 	751197 	2017-01-03 22:28:38.630509 	control 	old_page 	0
294474 	945152 	2017-01-12 00:51:57.078372 	control 	old_page 	0
294475 	734608 	2017-01-22 11:45:03.439544 	control 	old_page 	0
294476 	697314 	2017-01-15 01:20:28.957438 	control 	old_page 	0
294477 	715931 	2017-01-16 12:40:24.467417 	treatment 	new_page 	0

290585 rows × 5 columns

# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]

0

ToDo 1.3

Use df2 and the cells below to answer questions for Quiz 3 in the classroom.

a. How many unique user_ids are in df2?

df2.nunique()

user_id         290584
timestamp       290585
group                2
landing_page         2
converted            2
dtype: int64

b. There is one user_id repeated in df2. What is it?

df2['dupli'] = df2.duplicated(subset = 'user_id', keep = 'first')
df2[df2['dupli'] == True]['user_id']

2893    773192
Name: user_id, dtype: int64

c. Display the rows for the duplicate user_id?

df2['dupli'] = df2.duplicated(subset = 'user_id', keep = 'first')
df2[df2['dupli'] == True]

	user_id 	timestamp 	group 	landing_page 	converted 	dupli
2893 	773192 	2017-01-14 02:55:59.590927 	treatment 	new_page 	0 	True

d. Remove one of the rows with a duplicate user_id, from the df2 dataframe.

# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2.drop(df2[(df2['user_id'] == 773192) & (df2['dupli'] == True)].index, inplace = True)
# Check again if the row with a duplicate user_id is deleted or not
df2.count()

user_id         290584
timestamp       290584
group           290584
landing_page    290584
converted       290584
dupli           290584
dtype: int64

ToDo 1.4

Use df2 in the cells below to answer the quiz questions related to Quiz 4 in the classroom.

a. What is the probability of an individual converting regardless of the page they receive?

    Tip: The probability you'll compute represents the overall "converted" success rate in the population and you may call it ppopulation

    .

df2['converted'].mean()

0.11959708724499628

b. Given that an individual was in the control group, what is the probability they converted?

c = df2.query("group == 'control'")["converted"].mean()
c

0.1203863045004612

c. Given that an individual was in the treatment group, what is the probability they converted?

y = df2.query("group == 'treatment'")["converted"].mean()
y

0.11880806551510564

    Tip: The probabilities you've computed in the points (b). and (c). above can also be treated as conversion rate. Calculate the actual difference (obs_diff) between the conversion rates for the two groups. You will need that later.

# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
obs_diff = y - c 
obs_diff

-0.0015782389853555567

d. What is the probability that an individual received the new page?

df2.query("landing_page == 'new_page'").shape[0]/df2['landing_page'].count()

0.50006194422266881

e. Consider your results from parts (a) through (d) above, and explain below whether the new treatment group users lead to more conversions.

    No, The old page makes more convertion

Part II - A/B Test

Since a timestamp is associated with each event, you could run a hypothesis test continuously as long as you observe the events.

However, then the hard questions would be:

    Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?
    How long do you run to render a decision that neither page is better than another?

These questions are the difficult parts associated with A/B tests in general.
ToDo 2.1

For now, consider you need to make the decision just based on all the data provided.

    Recall that you just calculated that the "converted" probability (or rate) for the old page is slightly higher than that of the new page (ToDo 1.4.c).

If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should be your null and alternative hypotheses (H0
and H1

)?

You can state your hypothesis in terms of words or in terms of pold
and pnew

, which are the "converted" probability (or rate) for the old and new pages respectively.

    H0 : P_new <= P_old H1 : P_new > P_old

ToDo 2.2 - Null Hypothesis H0
Testing

Under the null hypothesis H0
, assume that pnew and pold are equal. Furthermore, assume that pnew and pold

both are equal to the converted success rate in the df2 data regardless of the page. So, our assumption is:

pnew
= pold = ppopulation

In this section, you will:

    Simulate (bootstrap) sample data set for both groups, and compute the "converted" probability p

    for those samples.

    Use a sample size for each group equal to the ones in the df2 data.

    Compute the difference in the "converted" probability for the two samples above.

    Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate.

Use the cells below to provide the necessary parts of this simulation. You can use Quiz 5 in the classroom to make sure you are on the right track.

a. What is the conversion rate for pnew

under the null hypothesis?

p_new = df2["converted"].mean()
p_new

0.11959708724499628

b. What is the conversion rate for pold

under the null hypothesis?

p_old = p_new
p_old

0.11959708724499628

c. What is nnew

, the number of individuals in the treatment group?

Hint: The treatment group users are shown the new page.

n_new = len(df2.query("group == 'treatment'"))
n_new

145310

d. What is nold

, the number of individuals in the control group?

n_old = len(df2.query("group == 'control'"))
n_old

145274

e. Simulate Sample for the treatment Group
Simulate nnew
transactions with a conversion rate of pnew under the null hypothesis.

Hint: Use numpy.random.choice() method to randomly generate nnew number of values.
Store these nnew

1's and 0's in the new_page_converted numpy array.

# Simulate a Sample for the treatment Group
n_new_converted = np.random.choice(df2.query("group == 'treatment'")['converted'], 145311)
p_new = n_new_converted.mean()
p_new

0.11834616787442107

f. Simulate Sample for the control Group
Simulate nold
transactions with a conversion rate of pold under the null hypothesis.
Store these nold

1's and 0's in the old_page_converted numpy array.

# Simulate a Sample for the control Group
n_old_converted = np.random.choice(df2.query("group == 'control'")['converted'], 145274)
p_old = n_old_converted.mean()
p_old

0.1215289728375346

g. Find the difference in the "converted" probability (p′new
- p′old)

for your simulated samples from the parts (e) and (f) above.

p_new - p_old

-0.0031828049631135308

h. Sampling distribution
Re-create new_page_converted and old_page_converted and find the (p′new
- p′old)

value 10,000 times using the same simulation process you used in parts (a) through (g) above.


Store all (p′new
- p′old)

values in a NumPy array called p_diffs.

# Sampling distribution 
p_diffs = []
for x in range(10000):
    n_old_converted = np.random.choice(n_old, 145274)
    p_old = n_old_converted.mean()
    n_new_converted = np.random.choice(n_new, 145311)
    p_new = n_new_converted.mean()
    p_diffs.append(p_new - p_old)

i. Histogram
Plot a histogram of the p_diffs. Does this plot look like what you expected? Use the matching problem in the classroom to assure you fully understand what was computed here.

Also, use plt.axvline() method to mark the actual difference observed in the df2 data (recall obs_diff), in the chart.

    Tip: Display title, x-label, and y-label in the chart.

plt.hist(p_diffs)
plt.axvline(obs_diff, c = 'red')

<matplotlib.lines.Line2D at 0x7fe730bc6860>

j. What proportion of the p_diffs are greater than the actual difference observed in the df2 data?

(p_diffs > obs_diff).mean()

0.54359999999999997

k. Please explain in words what you have just computed in part j above.

    What is this value called in scientific studies?
    What does this value signify in terms of whether or not there is a difference between the new and old pages? Hint: Compare the value above with the "Type I error rate (0.05)".

>**P_value/ This value indicates that there is no difference between old page and the new one**

l. Using Built-in Methods for Hypothesis Testing
We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance.

Fill in the statements below to calculate the:

    convert_old: number of conversions with the old_page
    convert_new: number of conversions with the new_page
    n_old: number of individuals who were shown the old_page
    n_new: number of individuals who were shown the new_page

import statsmodels.api as sm

# number of conversions with the old_page
convert_old = df2.query(" landing_page == 'old_page' and converted == 1").shape[0]

# number of conversions with the new_page
convert_new = df2.query(" landing_page == 'new_page' and converted == 1").shape[0]

# number of individuals who were shown the old_page
n_old = df2[df2['group'] == 'control'].shape[0]

# number of individuals who received new_page
n_new = df2[df2['group'] == 'treatment'].shape[0]

m. Now use sm.stats.proportions_ztest() to compute your test statistic and p-value. Here is a helpful link on using the built in.

The syntax is:

proportions_ztest(count_array, nobs_array, alternative='larger')

where,

    count_array = represents the number of "converted" for each group
    nobs_array = represents the total number of observations (rows) in each group
    alternative = choose one of the values from [‘two-sided’, ‘smaller’, ‘larger’] depending upon two-tailed, left-tailed, or right-tailed respectively.

        Hint:
        It's a two-tailed if you defined H1

as (pnew=pold).
It's a left-tailed if you defined H1 as (pnew<pold).
It's a right-tailed if you defined H1 as (pnew>pold)

        .

The built-in function above will return the z_score, p_value.
About the two-sample z-test

Recall that you have plotted a distribution p_diffs representing the difference in the "converted" probability (p′new−p′old)

for your two simulated samples 10,000 times.

Another way for comparing the mean of two independent and normal distribution is a two-sample z-test. You can perform the Z-test to calculate the Z_score, as shown in the equation below:
Zscore=(p′new−p′old)−(pnew−pold)σ2newnnew+σ2oldnold−−−−−−−−√

where,

    p′

is the "converted" success rate in the sample
pnew
and pold
are the "converted" success rate for the two groups in the population.
σnew
and σnew
are the standard deviation for the two groups in the population.
nnew
and nold

    represent the size of the two groups or samples (it's same in our case)

    Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error.

Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values:

    Zscore

Zα
or Z0.05, also known as critical value at 95% confidence interval. Z0.05 is 1.645 for one-tailed tests, and 1.960 for two-tailed test. You can determine the Zα

    from the z-table manually.

Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the null based on the comparison between Zscore
and Zα

.

    Hint:
    For a right-tailed test, reject null if Zscore

> Zα.
For a left-tailed test, reject null if Zscore < Zα

    .

In other words, we determine whether or not the Zscore
lies in the "rejection region" in the distribution. A "rejection region" is an interval where the null hypothesis is rejected iff the Zscore

lies in that region.

Reference:

    Example 9.1.2 on this page/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org

    Tip: You don't have to dive deeper into z-test for this exercise. Try having an overview of what does z-score signify in general.

import statsmodels.api as sm
# ToDo: Complete the sm.stats.proportions_ztest() method arguments
z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old],alternative='larger')
print(z_score, p_value)

-1.31092419842 0.905058312759

n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages? Do they agree with the findings in parts j. and k.?

    Tip: Notice whether the p-value is similar to the one computed earlier. Accordingly, can you reject/fail to reject the null hypothesis? It is important to correctly interpret the test statistic and p-value.

    it means that the conversion rate for the old page is the same for the new one / fail to reject.

Part III - A regression approach
ToDo 3.1

In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.

a. Since each row in the df2 data is either a conversion or no conversion, what type of regression should you be performing in this case?
>**Logistic Regression**

b. The goal is to use statsmodels library to fit the regression model you specified in part a. above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the df2 dataframe:

    intercept - It should be 1 in the entire column.
    ab_page - It's a dummy variable column, having a value 1 when an individual receives the treatment, otherwise 0.

df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2.head()

	user_id 	timestamp 	group 	landing_page 	converted 	dupli 	intercept 	ab_page
0 	851104 	2017-01-21 22:11:48.556739 	control 	old_page 	0 	False 	1 	0
1 	804228 	2017-01-12 08:01:45.159739 	control 	old_page 	0 	False 	1 	0
2 	661590 	2017-01-11 16:55:06.154213 	treatment 	new_page 	0 	False 	1 	1
3 	853541 	2017-01-08 18:28:03.143765 	treatment 	new_page 	0 	False 	1 	1
4 	864975 	2017-01-21 01:52:26.210827 	control 	old_page 	1 	False 	1 	0

c. Use statsmodels to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts.

sample = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = sample.fit()

Optimization terminated successfully.
         Current function value: 0.366118
         Iterations 6

d. Provide the summary of your model below, and use it as necessary to answer the following questions.

results.summary2()

Model: 	Logit 	No. Iterations: 	6.0000
Dependent Variable: 	converted 	Pseudo R-squared: 	0.000
Date: 	2022-03-23 16:59 	AIC: 	212780.3502
No. Observations: 	290584 	BIC: 	212801.5095
Df Model: 	1 	Log-Likelihood: 	-1.0639e+05
Df Residuals: 	290582 	LL-Null: 	-1.0639e+05
Converged: 	1.0000 	Scale: 	1.0000
	Coef. 	Std.Err. 	z 	P>|z| 	[0.025 	0.975]
intercept 	-1.9888 	0.0081 	-246.6690 	0.0000 	-2.0046 	-1.9730
ab_page 	-0.0150 	0.0114 	-1.3109 	0.1899 	-0.0374 	0.0074

e. What is the p-value associated with ab_page? Why does it differ from the value you found in Part II?

Hints:

    What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in Part II?
    You may comment on if these hypothesis (Part II vs. Part III) are one-sided or two-sided.
    You may also compare the current p-value with the Type I error rate (0.05).

    ** p_value = 0.1899--- In part two H0 : P_new = P_old----- In part three H0 : P_new > P_old----- this is a two-sided t-test compared to a one-sided t-test in part II.

f. Now, you are considering other things that might influence whether or not an individual converts. Discuss why it is a good idea to consider other factors to add into your regression model. Are there any disadvantages to adding additional terms into your regression model?

    It is very important to include more factors to reduce the error-------one of the problems that could arise by considering other additional factors may be multicollinearity

g. Adding countries
Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in.

    You will need to read in the countries.csv dataset and merge together your df2 datasets on the appropriate rows. You call the resulting dataframe df_merged. Here are the docs for joining tables.

    Does it appear that country had an impact on conversion? To answer this question, consider the three unique values, ['UK', 'US', 'CA'], in the country column. Create dummy variables for these country columns.

        Hint: Use pandas.get_dummies() to create dummy variables. You will utilize two columns for the three dummy variables.

    Provide the statistical output as well as a written response to answer this question.

# Read the countries.csv
df3 = pd.read_csv('countries.csv')
df3

	user_id 	country
0 	834778 	UK
1 	928468 	US
2 	822059 	UK
3 	711597 	UK
4 	710616 	UK
5 	909908 	UK
6 	811617 	US
7 	938122 	US
8 	887018 	US
9 	820683 	US
10 	697357 	US
11 	748296 	US
12 	666132 	UK
13 	668810 	UK
14 	940939 	US
15 	646414 	US
16 	907385 	US
17 	698200 	US
18 	738692 	US
19 	724651 	US
20 	662682 	US
21 	639818 	US
22 	920941 	US
23 	804632 	US
24 	684798 	UK
25 	766270 	UK
26 	857817 	UK
27 	750698 	UK
28 	721445 	US
29 	744732 	UK
... 	... 	...
290554 	834931 	UK
290555 	667920 	US
290556 	869193 	US
290557 	737522 	US
290558 	937048 	UK
290559 	689274 	UK
290560 	916713 	US
290561 	894324 	US
290562 	816587 	US
290563 	905950 	US
290564 	865612 	US
290565 	766165 	US
290566 	664716 	US
290567 	893381 	UK
290568 	746186 	UK
290569 	815837 	US
290570 	646239 	US
290571 	703088 	US
290572 	758018 	UK
290573 	663071 	UK
290574 	635122 	US
290575 	757673 	UK
290576 	870839 	US
290577 	659679 	US
290578 	674173 	US
290579 	653118 	US
290580 	878226 	UK
290581 	799368 	UK
290582 	655535 	CA
290583 	934996 	UK

290584 rows × 2 columns

# Join with the df2 dataframe
df4 = df2.join(df3.set_index('user_id'), on='user_id')
df4

	user_id 	timestamp 	group 	landing_page 	converted 	dupli 	intercept 	ab_page 	country
0 	851104 	2017-01-21 22:11:48.556739 	control 	old_page 	0 	False 	1 	0 	US
1 	804228 	2017-01-12 08:01:45.159739 	control 	old_page 	0 	False 	1 	0 	US
2 	661590 	2017-01-11 16:55:06.154213 	treatment 	new_page 	0 	False 	1 	1 	US
3 	853541 	2017-01-08 18:28:03.143765 	treatment 	new_page 	0 	False 	1 	1 	US
4 	864975 	2017-01-21 01:52:26.210827 	control 	old_page 	1 	False 	1 	0 	US
5 	936923 	2017-01-10 15:20:49.083499 	control 	old_page 	0 	False 	1 	0 	US
6 	679687 	2017-01-19 03:26:46.940749 	treatment 	new_page 	1 	False 	1 	1 	CA
7 	719014 	2017-01-17 01:48:29.539573 	control 	old_page 	0 	False 	1 	0 	US
8 	817355 	2017-01-04 17:58:08.979471 	treatment 	new_page 	1 	False 	1 	1 	UK
9 	839785 	2017-01-15 18:11:06.610965 	treatment 	new_page 	1 	False 	1 	1 	CA
10 	929503 	2017-01-18 05:37:11.527370 	treatment 	new_page 	0 	False 	1 	1 	UK
11 	834487 	2017-01-21 22:37:47.774891 	treatment 	new_page 	0 	False 	1 	1 	US
12 	803683 	2017-01-09 06:05:16.222706 	treatment 	new_page 	0 	False 	1 	1 	US
13 	944475 	2017-01-22 01:31:09.573836 	treatment 	new_page 	0 	False 	1 	1 	US
14 	718956 	2017-01-22 11:45:11.327945 	treatment 	new_page 	0 	False 	1 	1 	US
15 	644214 	2017-01-22 02:05:21.719434 	control 	old_page 	1 	False 	1 	0 	US
16 	847721 	2017-01-17 14:01:00.090575 	control 	old_page 	0 	False 	1 	0 	US
17 	888545 	2017-01-08 06:37:26.332945 	treatment 	new_page 	1 	False 	1 	1 	US
18 	650559 	2017-01-24 11:55:51.084801 	control 	old_page 	0 	False 	1 	0 	CA
19 	935734 	2017-01-17 20:33:37.428378 	control 	old_page 	0 	False 	1 	0 	US
20 	740805 	2017-01-12 18:59:45.453277 	treatment 	new_page 	0 	False 	1 	1 	US
21 	759875 	2017-01-09 16:11:58.806110 	treatment 	new_page 	0 	False 	1 	1 	UK
23 	793849 	2017-01-23 22:36:10.742811 	treatment 	new_page 	0 	False 	1 	1 	US
24 	905617 	2017-01-20 14:12:19.345499 	treatment 	new_page 	0 	False 	1 	1 	UK
25 	746742 	2017-01-23 11:38:29.592148 	control 	old_page 	0 	False 	1 	0 	US
26 	892356 	2017-01-05 09:35:14.904865 	treatment 	new_page 	1 	False 	1 	1 	UK
27 	773302 	2017-01-12 08:29:49.810594 	treatment 	new_page 	0 	False 	1 	1 	US
28 	913579 	2017-01-24 09:11:39.164256 	control 	old_page 	1 	False 	1 	0 	US
29 	736159 	2017-01-06 01:50:21.318242 	treatment 	new_page 	0 	False 	1 	1 	US
30 	690284 	2017-01-13 17:22:57.182769 	control 	old_page 	0 	False 	1 	0 	US
... 	... 	... 	... 	... 	... 	... 	... 	... 	...
294448 	776137 	2017-01-12 05:53:12.386730 	treatment 	new_page 	0 	False 	1 	1 	US
294449 	883344 	2017-01-22 23:15:58.645325 	treatment 	new_page 	0 	False 	1 	1 	CA
294450 	825594 	2017-01-06 12:37:08.897784 	treatment 	new_page 	0 	False 	1 	1 	UK
294451 	875688 	2017-01-14 07:19:49.042869 	control 	old_page 	0 	False 	1 	0 	US
294452 	927527 	2017-01-12 10:52:11.084740 	control 	old_page 	0 	False 	1 	0 	US
294453 	789177 	2017-01-17 18:17:56.215378 	control 	old_page 	0 	False 	1 	0 	US
294454 	937338 	2017-01-19 03:23:22.236666 	treatment 	new_page 	0 	False 	1 	1 	UK
294455 	733101 	2017-01-23 12:52:58.711914 	treatment 	new_page 	0 	False 	1 	1 	US
294456 	679096 	2017-01-02 16:43:49.237940 	treatment 	new_page 	0 	False 	1 	1 	US
294457 	691699 	2017-01-09 23:42:35.963486 	treatment 	new_page 	0 	False 	1 	1 	US
294458 	807595 	2017-01-22 10:43:09.285426 	treatment 	new_page 	0 	False 	1 	1 	US
294459 	924816 	2017-01-20 10:59:03.481635 	control 	old_page 	0 	False 	1 	0 	US
294460 	846225 	2017-01-16 15:24:46.705903 	treatment 	new_page 	0 	False 	1 	1 	US
294461 	740310 	2017-01-10 17:22:19.762612 	control 	old_page 	0 	False 	1 	0 	US
294462 	677163 	2017-01-03 19:41:51.902148 	treatment 	new_page 	0 	False 	1 	1 	US
294463 	832080 	2017-01-19 13:18:27.352570 	control 	old_page 	0 	False 	1 	0 	US
294464 	834362 	2017-01-17 01:51:56.106436 	control 	old_page 	0 	False 	1 	0 	US
294465 	925675 	2017-01-07 20:38:26.346410 	treatment 	new_page 	0 	False 	1 	1 	US
294466 	923948 	2017-01-09 16:33:41.104573 	control 	old_page 	0 	False 	1 	0 	US
294467 	857744 	2017-01-05 08:00:56.024226 	control 	old_page 	0 	False 	1 	0 	US
294468 	643562 	2017-01-02 19:20:05.460595 	treatment 	new_page 	0 	False 	1 	1 	CA
294469 	755438 	2017-01-18 17:35:06.149568 	control 	old_page 	0 	False 	1 	0 	US
294470 	908354 	2017-01-11 02:42:21.195145 	control 	old_page 	0 	False 	1 	0 	US
294471 	718310 	2017-01-21 22:44:20.378320 	control 	old_page 	0 	False 	1 	0 	US
294472 	822004 	2017-01-04 03:36:46.071379 	treatment 	new_page 	0 	False 	1 	1 	CA
294473 	751197 	2017-01-03 22:28:38.630509 	control 	old_page 	0 	False 	1 	0 	US
294474 	945152 	2017-01-12 00:51:57.078372 	control 	old_page 	0 	False 	1 	0 	US
294475 	734608 	2017-01-22 11:45:03.439544 	control 	old_page 	0 	False 	1 	0 	US
294476 	697314 	2017-01-15 01:20:28.957438 	control 	old_page 	0 	False 	1 	0 	US
294477 	715931 	2017-01-16 12:40:24.467417 	treatment 	new_page 	0 	False 	1 	1 	UK

290584 rows × 9 columns

# Create the necessary dummy variables
df4[['UK','US','CA']] = pd.get_dummies(df4['country'])
df4[['UK']] = pd.get_dummies(df4['country'])['UK']
df4[['US']] = pd.get_dummies(df4['country'])['US']
df4[['CA']] = pd.get_dummies(df4['country'])['CA']
df4

	user_id 	timestamp 	group 	landing_page 	converted 	dupli 	intercept 	ab_page 	country 	UK 	US 	CA
0 	851104 	2017-01-21 22:11:48.556739 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
1 	804228 	2017-01-12 08:01:45.159739 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
2 	661590 	2017-01-11 16:55:06.154213 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
3 	853541 	2017-01-08 18:28:03.143765 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
4 	864975 	2017-01-21 01:52:26.210827 	control 	old_page 	1 	False 	1 	0 	US 	0 	1 	0
5 	936923 	2017-01-10 15:20:49.083499 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
6 	679687 	2017-01-19 03:26:46.940749 	treatment 	new_page 	1 	False 	1 	1 	CA 	0 	0 	1
7 	719014 	2017-01-17 01:48:29.539573 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
8 	817355 	2017-01-04 17:58:08.979471 	treatment 	new_page 	1 	False 	1 	1 	UK 	1 	0 	0
9 	839785 	2017-01-15 18:11:06.610965 	treatment 	new_page 	1 	False 	1 	1 	CA 	0 	0 	1
10 	929503 	2017-01-18 05:37:11.527370 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0
11 	834487 	2017-01-21 22:37:47.774891 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
12 	803683 	2017-01-09 06:05:16.222706 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
13 	944475 	2017-01-22 01:31:09.573836 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
14 	718956 	2017-01-22 11:45:11.327945 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
15 	644214 	2017-01-22 02:05:21.719434 	control 	old_page 	1 	False 	1 	0 	US 	0 	1 	0
16 	847721 	2017-01-17 14:01:00.090575 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
17 	888545 	2017-01-08 06:37:26.332945 	treatment 	new_page 	1 	False 	1 	1 	US 	0 	1 	0
18 	650559 	2017-01-24 11:55:51.084801 	control 	old_page 	0 	False 	1 	0 	CA 	0 	0 	1
19 	935734 	2017-01-17 20:33:37.428378 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
20 	740805 	2017-01-12 18:59:45.453277 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
21 	759875 	2017-01-09 16:11:58.806110 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0
23 	793849 	2017-01-23 22:36:10.742811 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
24 	905617 	2017-01-20 14:12:19.345499 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0
25 	746742 	2017-01-23 11:38:29.592148 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
26 	892356 	2017-01-05 09:35:14.904865 	treatment 	new_page 	1 	False 	1 	1 	UK 	1 	0 	0
27 	773302 	2017-01-12 08:29:49.810594 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
28 	913579 	2017-01-24 09:11:39.164256 	control 	old_page 	1 	False 	1 	0 	US 	0 	1 	0
29 	736159 	2017-01-06 01:50:21.318242 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
30 	690284 	2017-01-13 17:22:57.182769 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
294448 	776137 	2017-01-12 05:53:12.386730 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294449 	883344 	2017-01-22 23:15:58.645325 	treatment 	new_page 	0 	False 	1 	1 	CA 	0 	0 	1
294450 	825594 	2017-01-06 12:37:08.897784 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0
294451 	875688 	2017-01-14 07:19:49.042869 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294452 	927527 	2017-01-12 10:52:11.084740 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294453 	789177 	2017-01-17 18:17:56.215378 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294454 	937338 	2017-01-19 03:23:22.236666 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0
294455 	733101 	2017-01-23 12:52:58.711914 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294456 	679096 	2017-01-02 16:43:49.237940 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294457 	691699 	2017-01-09 23:42:35.963486 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294458 	807595 	2017-01-22 10:43:09.285426 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294459 	924816 	2017-01-20 10:59:03.481635 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294460 	846225 	2017-01-16 15:24:46.705903 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294461 	740310 	2017-01-10 17:22:19.762612 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294462 	677163 	2017-01-03 19:41:51.902148 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294463 	832080 	2017-01-19 13:18:27.352570 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294464 	834362 	2017-01-17 01:51:56.106436 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294465 	925675 	2017-01-07 20:38:26.346410 	treatment 	new_page 	0 	False 	1 	1 	US 	0 	1 	0
294466 	923948 	2017-01-09 16:33:41.104573 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294467 	857744 	2017-01-05 08:00:56.024226 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294468 	643562 	2017-01-02 19:20:05.460595 	treatment 	new_page 	0 	False 	1 	1 	CA 	0 	0 	1
294469 	755438 	2017-01-18 17:35:06.149568 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294470 	908354 	2017-01-11 02:42:21.195145 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294471 	718310 	2017-01-21 22:44:20.378320 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294472 	822004 	2017-01-04 03:36:46.071379 	treatment 	new_page 	0 	False 	1 	1 	CA 	0 	0 	1
294473 	751197 	2017-01-03 22:28:38.630509 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294474 	945152 	2017-01-12 00:51:57.078372 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294475 	734608 	2017-01-22 11:45:03.439544 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294476 	697314 	2017-01-15 01:20:28.957438 	control 	old_page 	0 	False 	1 	0 	US 	0 	1 	0
294477 	715931 	2017-01-16 12:40:24.467417 	treatment 	new_page 	0 	False 	1 	1 	UK 	1 	0 	0

290584 rows × 12 columns

df4['intercept'] = 1
sample = sm.Logit(df4['converted'],df4[['intercept','UK','US']])
results = sample.fit()
results.summary2()

Optimization terminated successfully.
         Current function value: 0.366116
         Iterations 6

Model: 	Logit 	No. Iterations: 	6.0000
Dependent Variable: 	converted 	Pseudo R-squared: 	0.000
Date: 	2022-03-23 17:03 	AIC: 	212780.8333
No. Observations: 	290584 	BIC: 	212812.5723
Df Model: 	2 	Log-Likelihood: 	-1.0639e+05
Df Residuals: 	290581 	LL-Null: 	-1.0639e+05
Converged: 	1.0000 	Scale: 	1.0000
	Coef. 	Std.Err. 	z 	P>|z| 	[0.025 	0.975]
intercept 	-2.0375 	0.0260 	-78.3639 	0.0000 	-2.0885 	-1.9866
UK 	0.0507 	0.0284 	1.7863 	0.0740 	-0.0049 	0.1064
US 	0.0408 	0.0269 	1.5178 	0.1291 	-0.0119 	0.0935

    It seems that the country have an impact on conversion ----- US have higher conversion rate than UK & CA

h. Fit your model and obtain the results

Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion. Create the necessary additional columns, and fit the new model.

Provide the summary results (statistical output), and your conclusions (written response) based on the results.

    Tip: Conclusions should include both statistical reasoning, and practical reasoning for the situation.

    Hints:

        Look at all of p-values in the summary, and compare against the Type I error rate (0.05).
        Can you reject/fail to reject the null hypotheses (regression model)?
        Comment on the effect of page and country to predict the conversion.

# Fit your model, and summarize the results
df4['ab_us'] = df4['ab_page'] * df4['US']
df4['ab_ca'] = df4['ab_page'] * df4['CA']
df4['ab_uk'] = df4['ab_page'] * df4['UK']
sample_2 = sm.Logit(df4['converted'],df4[['intercept','ab_page','US','UK','ab_us','ab_uk']])
results = sample_2.fit()
results.summary2()

Optimization terminated successfully.
         Current function value: 0.366109
         Iterations 6

Model: 	Logit 	No. Iterations: 	6.0000
Dependent Variable: 	converted 	Pseudo R-squared: 	0.000
Date: 	2022-03-23 17:06 	AIC: 	212782.6602
No. Observations: 	290584 	BIC: 	212846.1381
Df Model: 	5 	Log-Likelihood: 	-1.0639e+05
Df Residuals: 	290578 	LL-Null: 	-1.0639e+05
Converged: 	1.0000 	Scale: 	1.0000
	Coef. 	Std.Err. 	z 	P>|z| 	[0.025 	0.975]
intercept 	-2.0040 	0.0364 	-55.0077 	0.0000 	-2.0754 	-1.9326
ab_page 	-0.0674 	0.0520 	-1.2967 	0.1947 	-0.1694 	0.0345
US 	0.0175 	0.0377 	0.4652 	0.6418 	-0.0563 	0.0914
UK 	0.0118 	0.0398 	0.2957 	0.7674 	-0.0663 	0.0899
ab_us 	0.0469 	0.0538 	0.8718 	0.3833 	-0.0585 	0.1523
ab_uk 	0.0783 	0.0568 	1.3783 	0.1681 	-0.0330 	0.1896

 

    According to The P_values ,we fail to reject the null hypotheses....The effect of the new_page and the US country on conversion is higher than the conversion of CA & UK country.. It is recommended not to launch the new page

Final Check!

Congratulations! You have reached the end of the A/B Test Results project! You should be very proud of all you have accomplished!

    Tip: Once you are satisfied with your work here, check over your notebook to make sure that it satisfies all the specifications mentioned in the rubric. You should also probably remove all of the "Hints" and "Tips" like this one so that the presentation is as polished as possible.

Submission

You may either submit your notebook through the "SUBMIT PROJECT" button at the bottom of this workspace, or you may work from your local machine and submit on the last page of this project lesson.

    Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).

    Alternatively, you can download this report as .html via the File > Download as submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.

    Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

0

 

