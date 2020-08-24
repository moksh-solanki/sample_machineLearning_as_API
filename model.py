# python library to generate random numbers 
from random import randint 
from joblib import dump
# the limit within which random numbers are generated 
TRAIN_SET_LIMIT = 1000

# to create exactly 50 data items 
TRAIN_SET_COUNT = 50

# list that contains input and corresponding output 
TRAIN_INPUT = list() 
TRAIN_OUTPUT = list() 

# loop to create 100 data items with three columns each 
for i in range(TRAIN_SET_COUNT): 
	a = randint(0, TRAIN_SET_LIMIT) 
	b = randint(0, TRAIN_SET_LIMIT) 
	c = randint(0, TRAIN_SET_LIMIT) 

# creating the output for each data item 
	op = a + (2 * b) + (3 * c) 
	TRAIN_INPUT.append([a, b, c]) 

# adding each output to output list 
	TRAIN_OUTPUT.append(op) 



################################# train model #################################

# Sk-Learn contains the linear regression model 
from sklearn.linear_model import LinearRegression 

# Initialize the linear regression model 
predictor = LinearRegression(n_jobs =-1) 

# Fill the Model with the Data 
predictor.fit(X = TRAIN_INPUT, y = TRAIN_OUTPUT) 

################################# save model #################################
# Save your model
dump(predictor, 'model.pkl')
print("Model dumped!")