import csv
from math import log
from collections import defaultdict, Counter
from posixpath import split
from re import A, X
import this
from datetime import datetime
import statistics
import string
import pycountry_convert as pc
import random
import pickle


"""
* ECON1660
* PS4: Trees
*
* Fill in the functions that are labaled "TODO".  Once you
* have done so, uncomment (and adjust as needed) the main
* function and the call to main to print out the tree and
* the classification accuracy.
"""


"""
* TODO: Create features to be used in your regression tree.
"""

"""************************************************************************
* function: partition_loss(subsets)
* arguments:
* 		-subsets:  a list of lists of labeled data (representing groups
				   of observations formed by a split)
* return value:  loss value of a partition into the given subsets
*
* TODO: Write a function that computes the loss of a partition for
*       given subsets
************************************************************************"""
def partition_loss(subsets):
	#TODO
	mse = 0
	num_obs = sum(len(subset) for subset in subsets)
	# print("num_obs_total: " + str(num_obs))
	for subset in subsets:
		# print("num_obs_subset: " + str(len(subset)))
		subset_days = [loan[1] for loan in subset]
		mean_val = statistics.mean(subset_days)
		subset_sum = 0
		for loan in subset:
			subset_sum += (loan[1] - mean_val)**2
		mse += subset_sum / num_obs

	return mse


"""************************************************************************
* function: partition_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  a list of lists, where each list represents a subset of
*				 the inputs that share a common value of the given 
*				 attribute
************************************************************************"""
def partition_by(inputs, attribute):
	groups = defaultdict(list)
	for input in inputs:
		key = input[0][attribute]	#gets the value of the specified attribute
		groups[key].append(input)	#add the input to the appropriate group
	return groups


"""************************************************************************
* function: partition_loss_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  the loss value of splitting the inputs based on the
*				 given attribute
************************************************************************"""
def partition_loss_by(inputs, attribute):
	partitions = partition_by(inputs, attribute)
	return partition_loss(partitions.values())


"""************************************************************************
* function:  build_tree(inputs, num_levels, split_candidates = None)
*
* arguments:
* 		-inputs:  labeled data used to construct the tree; should be in the
*				  form of a list of tuples (a, b) where 'a' is a dictionary
*				  of features and 'b' is a label
*		-num_levels:  the goal number of levels for our output tree
*		-split_candidates:  variables that we could possibly split on.  For
*							our first level, all variables are candidates
*							(see first two lines in the function).
*			
* return value:  a tree in the form of a tuple (a, b) where 'a' is the
*				 variable to split on and 'b' is a dictionary representing
*				 the outcome class/outcome for each value of 'a'.
* 
* TODO:  Write a recursive function that builds a tree of the specified
*        number of levels based on labeled data "inputs"
************************************************************************"""
def build_tree(inputs, num_levels, split_candidates = None):
	#TODO

	#if first pass, all keys are split candidates
	if split_candidates == None:
		split_candidates = inputs[0][0].keys()
	
	input_days = [label[1] for label in inputs]
	
	# If the data all have the same number of days, then create a leaf node that predicts that number of days and then stop.
	if all(input_day == input_days[0] for input_day in input_days):
		# print(int(input_days[0]))
		return int(input_days[0])
	# If the list of attributes is empty OR reached max num_levels, then create a leaf node that predicts the mean number of days and then stop
	elif num_levels == 0 or len(split_candidates) == 0:
		# return mode(labels)
		# print(int(statistics.mean(input_days)))
		return int(statistics.mean(input_days))
	# Partition the data by each of the attributes and choose the partition with the lowest loss
	else:
		chosen_candidate = split_candidates[0]
		chosen_candidate_loss = partition_loss_by(inputs, chosen_candidate)
		candidates = split_candidates[1:]

		for next_candidate in candidates:
			next_candidate_loss = partition_loss_by(inputs, next_candidate)
			if next_candidate_loss < chosen_candidate_loss:
				chosen_candidate = next_candidate
				chosen_candidate_loss = next_candidate_loss

		updated_candidates = [x for x in split_candidates if x != chosen_candidate]
		partitions = partition_by(inputs, chosen_candidate)
		candidate_dictionary = {
			1:build_tree(partitions[1], (num_levels-1), updated_candidates),
			0:build_tree(partitions[0], (num_levels-1), updated_candidates)
		}

		return (chosen_candidate, candidate_dictionary)


"""************************************************************************
* function:  classify(tree, to_classify)
*
* arguments:
* 		-tree:  a tree built with the build_tree function
*		-to_classify:  a dictionary of features
*
* return value:  a value indicating a prediction of days_until_funded

* TODO:  Write a recursive function that uses "tree" and the values in the
*		 dictionary "to_classify" to output a predicted value.
************************************************************************"""
def classify(tree, to_classify):
	#TODO
	# base case where we hit a leaf
	if type(tree) == int:
		return tree
	attribute = tree[0]
	input_attribute = to_classify[attribute]

	branches = tree[1]
	sub_tree = branches[input_attribute]

	return classify(sub_tree, to_classify)

def find_accuracy(data, tree):
	# outputs the accuracy of our tree
	correct_predictions = 0
	sqr_err = 0
	for i in range(len(data)):
		classification = classify(tree, data[i][0])
		actual = data[i][1]
		if classification == data[i][1]:
			correct_predictions += 1
		sqr_err += (actual-classification)**2
	# print("total correct: " + str(correct_predictions))
	mse = sqr_err/len(data)
	accuracy = correct_predictions / float(len(data))
	return accuracy, mse

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    #country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    #return country_continent_name
    return country_continent_code

"""************************************************************************
* function:  load_data()
* arguments:  N/A
* return value:  a list of tuples representing the loans data
* 
* TODO:  Read in the loans data from the provided csv file.  Store the
* 		 observations as a list of tuples (a, b), where 'a' is a dictionary
*		 of features and 'b' is the value of the days_until_funded variable
************************************************************************"""
def load_data(file, train = True):
	#TODO
	fileReader = open(file, "rt", encoding="utf8")
	csvReader  = csv.reader(fileReader)

	data = list()
	nObservations = 0
	cHeader = next(csvReader)
	cHeader = [x for x in cHeader if x != 'days_until_funded']
	numAttributes = len(cHeader)
	numerical = ['id','pictured','loan_amount','repayment_term']
	exclude = set(string.punctuation)
	
	for row in csvReader:
		if nObservations % 30000 == True:
			print("loaded " + str(nObservations) + " loans")
		dictionary = {}
		for attribute in range(numAttributes):

			if cHeader[attribute] in numerical:
				dictionary.update({cHeader[attribute]:int(row[attribute])})
			elif cHeader[attribute] == 'posted_date':
				date_obj = datetime.strptime(row[attribute], '%Y-%m-%dT%H:%M:%SZ')
				dictionary.update({cHeader[attribute]:date_obj})
			elif cHeader[attribute] == 'description':
				# Parse out punctuation & make all lowercase
				st = ''.join(ch for ch in row[attribute] if ch not in exclude).lower()
				dictionary.update({cHeader[attribute]:set(st.split())})
			elif cHeader[attribute] == 'languages':
				st = row[attribute].split('|')
				dictionary.update({cHeader[attribute]:set(filter(None, st))})
			else:
				dictionary.update({cHeader[attribute]:row[attribute]})

			nObservations += 1

		if train == True:
			data.append((dictionary, int(row[numAttributes])))
		else:
			data.append((dictionary, 0))

	fileReader.close()

	return data, cHeader

def add_features(loans):
	# add binary divided features to the dataset for prediction
	# Use if-statements for binary divide in features
	for loan in loans:
		loan_dict = loan[0]

		## gender
		if(loan_dict['gender'] == 'M'):
			loan_dict.update({'gender_male':1})
		else:
			loan_dict.update({'gender_male':0})
		
		if(loan_dict['gender'] == 'F'):
			loan_dict.update({'gender_female':1})
		else:
			loan_dict.update({'gender_female':0})
		

		## description
		positive_words = set(['good','support', 'expertise', 'happy','responsible', 'trustful', 'honor', 'polite', 'leader', 'president', 'represent', 'nice',
								'grow', 'improve', 'improving', 'great', 'help', 'motivate', 'first', 'grow', 'enjoy', 'photo', 'skill', 'dream', 'new', 'bless' ])
		negative_words = set(['quit','unreliable', 'four','five', 'six','seven','eight', 'hard', 'without', 'maybe', 'unstable', 'party', 
								'already', 'previous', 'supplement', 'another', 'additional', 'enough', 'increase', 'increasing', 'neither', 'nor',
								'little', 'continue', 'far', 'but', 'struggle', 'risk', 'lack'])
		supportive_personal_background = set(['married', 'husband', 'wife', 'children', 'help', 'house', 'town', 'job', 'income'])
		unsupportive_personal_background = set(['children', 'single mother', 'single', 'divorce', 'widow', 'rural', 'village', 'volatile', 'challenge', 'unemploy', 'young'])
		
		

		positive_word_presence = {i for i in loan_dict['description'] if any(j in i for j in positive_words)}
		positive_word_count = len(positive_word_presence)
		if positive_word_count > 6:
			loan_dict.update({'positive_description':1})
		else:
			loan_dict.update({'positive_description':0})	
		
		negative_word_presence = {i for i in loan_dict['description'] if any(j in i for j in negative_words)}
		negative_word_count = len(negative_word_presence)
		if negative_word_count > 5:
			loan_dict.update({'negative_description':1})
		else:
			loan_dict.update({'negative_description':0})
		
		good_background_presence = {i for i in loan_dict['description'] if any(j in i for j in supportive_personal_background)}
		good_background_word_count = len(good_background_presence)
		if good_background_word_count > 2:
			loan_dict.update({'good_background':1})
		else:
			loan_dict.update({'good_background':0})
		
		bad_background_presence = {i for i in loan_dict['description'] if any(j in i for j in unsupportive_personal_background)}
		bad_background_word_count = len(bad_background_presence)
		if bad_background_word_count > 2:
			loan_dict.update({'bad_background':1})
		else:
			loan_dict.update({'bad_background':0})
		
		## sector
		if(loan_dict['sector'] == 'Agriculture'):
			loan_dict.update({'sector_ag':1})
		else:
			loan_dict.update({'sector_ag':0})

		if(loan_dict['sector'] == 'Food'):
			loan_dict.update({'sector_food':1})
		else:
			loan_dict.update({'sector_food':0})
		
		if(loan_dict['sector'] == 'Retail'):
			loan_dict.update({'sector_retail':1})
		else:
			loan_dict.update({'sector_retail':0})
		
		if(loan_dict['sector'] == 'Services'):
			loan_dict.update({'sector_service':1})
		else:
			loan_dict.update({'sector_service':0})
		
		if(loan_dict['sector'] == 'Clothing'):
			loan_dict.update({'sector_clothing':1})
		else:
			loan_dict.update({'sector_clothing':0})
		
		if(loan_dict['sector'] == 'Housing'):
			loan_dict.update({'sector_housing':1})
		else:
			loan_dict.update({'sector_housing':0})
		
		other_sector_checker = set(['Agriculture', 'Food', 'Retail', 'Services', 'Clothing', 'Housing'])

		if(set([loan_dict['sector']]).issubset(other_sector_checker)) == False:
			loan_dict.update({'sector_others':1})
		else:
			loan_dict.update({'sector_others':0})
		

		## divide country into continent groups
		if(loan_dict['country'] == 'Congo (Dem. Rep.)' or loan_dict['country'] == 'Congo (Rep.)') or loan_dict['country'] == 'The Democratic Republic of the Congo':
			loan_dict.update({'africa_requests':1})
			loan_dict.update({'america_requests':0})
			loan_dict.update({'asia_requests':0})
			loan_dict.update({'aus_europe_requests':0})
		elif(loan_dict['country'] == 'Timor-Leste'):
			loan_dict.update({'africa_requests':0})
			loan_dict.update({'america_requests':0})
			loan_dict.update({'asia_requests':1})
			loan_dict.update({'aus_europe_requests':0})
		elif(loan_dict['country'] == 'Myanmar (Burma)') or (loan_dict['country'] == 'Cote D\'Ivoire') or (loan_dict['country'] == 'Lao PDR'):
			loan_dict.update({'africa_requests':0})
			loan_dict.update({'america_requests':0})
			loan_dict.update({'asia_requests':1})
			loan_dict.update({'aus_europe_requests':0})
		else:
			if(country_to_continent(loan_dict['country']) == "AF"):
				loan_dict.update({'africa_requests':1})
			else:
				loan_dict.update({'africa_requests':0})
			
			if(country_to_continent(loan_dict['country']) == "NA") or (country_to_continent(loan_dict['country']) == "SA"):
				loan_dict.update({'america_requests':1})
			else:
				loan_dict.update({'america_requests':0})
			
			if(country_to_continent(loan_dict['country']) == "AS"):
				loan_dict.update({'asia_requests':1})
			else:
				loan_dict.update({'asia_requests':0})
			
			if(country_to_continent(loan_dict['country']) == "OC") or (country_to_continent(loan_dict['country']) == "EU"):
				loan_dict.update({'aus_europe_requests':1})
			else:
				loan_dict.update({'aus_europe_requests':0})

		## languages
		if(loan_dict['languages'] == 'en'):
			loan_dict.update({'english_only_description':1})
		else:
			loan_dict.update({'english_only_description':0})

		if(len(loan_dict['languages']) > 1):
			loan_dict.update({'bilingual_description':1})
		else:
			loan_dict.update({'bilingual_description':0})
		
		if(set(['es']).issubset(loan_dict['languages'])):
			loan_dict.update({'has_spanish_description':1})
		else:
			loan_dict.update({'has_spanish_description':0})
		

	return loans

def add_features_loan_amt(loans, interval1, interval2, interval3):
	print("Adding Features: Loan Amounts")
	new_features = []
	for loan in loans:
		loan_dict = loan[0]
		for amt in range (0, 1000, interval1):
			label = 'loan_amt_' + str(amt) + '-' + str(amt+interval1)
			if(loan_dict['loan_amount'] >= amt) and (loan_dict['loan_amount'] < amt + interval1):
				loan_dict.update({label:1})
			else:
				loan_dict.update({label:0})
		for amt in range (1000, 1800, interval2):
			label = 'loan_amt_' + str(amt) + '-' + str(amt+interval2)
			if(loan_dict['loan_amount'] >= amt) and (loan_dict['loan_amount'] < amt + interval2):
				loan_dict.update({label:1})
			else:
				loan_dict.update({label:0})
		for amt in range (1800, 4800, interval3):
			label = 'loan_amt_' + str(amt) + '-' + str(amt+interval3)
			if(loan_dict['loan_amount'] >= amt) and (loan_dict['loan_amount'] < amt + interval3):
				loan_dict.update({label:1})
			else:
				loan_dict.update({label:0})
		
		if(loan_dict['loan_amount'] >= 4800):
			loan_dict.update({'loan_amt_4800+':1})
		else:
			loan_dict.update({'loan_amt_4800+':0})

	for amt in range (0, 1000, interval1):
		label = 'loan_amt_' + str(amt) + '-' + str(amt+interval1)
		new_features.append(label)
	for amt in range (1000, 1800, interval2):
		label = 'loan_amt_' + str(amt) + '-' + str(amt+interval2)
		new_features.append(label)
	for amt in range (1800, 4800, interval3):
		label = 'loan_amt_' + str(amt) + '-' + str(amt+interval3)
		new_features.append(label)
	new_features.append('loan_amt_4800+')
		
	return loans, new_features

def add_features_repayment_term(loans, interval):
	print("Adding Features: Repayment Term")
	new_features = []
	for loan in loans:
		loan_dict = loan[0]
		for amt in range (4, 28, interval):
			label = 'repayment_term_' + str(amt) + '-' + str(amt+interval)
			if(loan_dict['repayment_term'] >= amt) and (loan_dict['repayment_term'] < amt + interval):
				loan_dict.update({label:1})
			else:
				loan_dict.update({label:0})
		if(loan_dict['repayment_term'] >= 28):
			loan_dict.update({'repayment_term_28+':1})
		else:
			loan_dict.update({'repayment_term_28+':0})

	for amt in range (4, 28, interval):
		label = 'repayment_term_' + str(amt) + '-' + str(amt+interval)
		new_features.append(label)
	new_features.append('repayment_term_28+')
		
	return loans, new_features

def add_features_loan_year(loans, yr1, yr2):
	print("Adding Features: Loan Year")
	new_features = ['request_year<' + str(yr1), 'request_year' + str(yr1) + "-" + str(yr2), 'request_year>' + str(yr2)]
	for loan in loans:
		loan_dict = loan[0]
		loan_year =loan_dict['posted_date'].year
		if loan_year < yr1:
			loan_dict.update({'request_year<' + str(yr1):1})
			loan_dict.update({'request_year' + str(yr1) + "-" + str(yr2):0})
			loan_dict.update({'request_year>' + str(yr2):0})
		elif loan_year > yr2:
			loan_dict.update({'request_year<' + str(yr1):0})
			loan_dict.update({'request_year' + str(yr1) + "-" + str(yr2):0})
			loan_dict.update({'request_year>' + str(yr2):1})
		else:
			loan_dict.update({'request_year<' + str(yr1):0})
			loan_dict.update({'request_year' + str(yr1) + "-" + str(yr2):1})
			loan_dict.update({'request_year>' + str(yr2):0})
	return loans, new_features


def split_data(loans):
	# divider = random.randint(1, len(loans)-2)
	divider = int(len(loans)/2)
	random.shuffle(loans)
	return loans[:divider], loans[divider:]

def bootstrap_sample(inputs, length):
	return random.choices(inputs, k=length)

def build_forest_tree(inputs, num_levels, num_split_candidates, split_candidates = None):
	#if first pass, all keys are split candidates
	if split_candidates == None:
		split_candidates = inputs[0][0].keys()
	
	if len(split_candidates) <= num_split_candidates:
		sampled_split_candidates = split_candidates
	else:
		sampled_split_candidates = random.sample(split_candidates, num_split_candidates)
	
	input_days = [label[1] for label in inputs]
	
	# If the data all have the same number of days, then create a leaf node that predicts that number of days and then stop.
	if all(input_day == input_days[0] for input_day in input_days):
		# print(int(input_days[0]))
		return int(input_days[0])
	# If the list of attributes is empty OR reached max num_levels, then create a leaf node that predicts the mean number of days and then stop
	elif num_levels == 0 or len(sampled_split_candidates) == 0:
		# return mode(labels)
		# print(int(statistics.mean(input_days)))
		return int(statistics.mean(input_days))
	# Partition the data by each of the attributes and choose the partition with the lowest loss
	else:
		chosen_candidate = sampled_split_candidates[0]
		chosen_candidate_loss = partition_loss_by(inputs, chosen_candidate)
		candidates = sampled_split_candidates[1:]

		for next_candidate in candidates:
			next_candidate_loss = partition_loss_by(inputs, next_candidate)
			if next_candidate_loss < chosen_candidate_loss:
				chosen_candidate = next_candidate
				chosen_candidate_loss = next_candidate_loss

		updated_candidates = [x for x in sampled_split_candidates if x != chosen_candidate]
		partitions = partition_by(inputs, chosen_candidate)
		# Make a leaf if partitions become too small to prevent overfitting
		if len(partitions[0]) < 100 or len(partitions[1]) < 100:
			return int(statistics.mean(input_days))
		candidate_dictionary = {
			1:build_forest_tree(partitions[1], (num_levels-1), num_split_candidates, updated_candidates),
			0:build_forest_tree(partitions[0], (num_levels-1), num_split_candidates, updated_candidates)
		}

		return (chosen_candidate, candidate_dictionary)

def build_forest(inputs, num_trees, split_candidates):
	forest = []
	for i in range(num_trees):
		print("☘ Planted " + str(i+1) + " trees ☘")
		new_tree = build_forest_tree(bootstrap_sample(inputs, 20000), num_levels=6, num_split_candidates=25, split_candidates = split_candidates)
		forest.append(new_tree)
	return forest

# Takes in a loan tuple
def forest_predict(forest, loan):
	prediction = 0
	predictions = []
	for tree in forest:
		prediction += classify(tree, loan[0])
		predictions.append(classify(tree, loan[0]))
	return prediction / len(forest)

def forest_accuracy(forest, loans):
	correct_predictions = 0
	sqr_err = 0
	for i in range(len(loans)):
		classification = forest_predict(forest, loans[i])
		actual = loans[i][1]
		# print("classification: " + str(classification) + " actual: " + str(actual))
		if classification == actual:
			correct_predictions += 1
		sqr_err += (actual-classification)**2
	# print("total correct: " + str(correct_predictions))
	mse = sqr_err/len(loans)
	accuracy = correct_predictions / float(len(loans))
	return accuracy, mse

def main():
	# loans, attributes = load_data("../loans_AB_labeled.csv")
	# # loans[:100000]
	## ---- processing loans from dataset ab
	# loans = add_features(loans)
	# loans, new_amt_features = add_features_loan_amt(loans, 200, 400, 800)
	# loans, new_repay_features = add_features_repayment_term(loans, 4)
	# loans, new_year_features = add_features_loan_year(loans, 2009, 2012)
	# candidates = [  'gender_male', 'gender_female', 'pictured', 
	# 				'positive_description', 'negative_description', 'good_background', 'bad_background',
	# 				'sector_ag','sector_food', 'sector_clothing', 'sector_housing', 'sector_retail','sector_service', 'sector_others', 
	# 				'africa_requests', 'america_requests', 'asia_requests', 'aus_europe_requests',
	# 				'english_only_description', 'bilingual_description', 'has_spanish_description' ]
	# candidates = candidates + new_amt_features + new_repay_features + new_year_features

	## Save processed loans into file
	# with open('loans_A_processed', 'wb') as file:
	# 	pickle.dump(loans, file)
	# with open('feature_list', 'wb') as file:
	# 	pickle.dump(candidates, file)

	# Load previously processed loans from file
	with open('loans_AB_processed', 'rb') as file:
		loans = pickle.load(file)
	with open('feature_list', 'rb') as file:
		candidates = pickle.load(file)
	
	
	print(len(candidates))
	# a_1, a_2 = split_data(loans)

	forest = build_forest(loans, 150, candidates)
	print("𓋼𓋼𓋼 ~ Forest ~ 𓋼𓋼𓋼")
	# for i in range(len(forest)):
	# 	print("𓋼- Tree " + str(i+1) + " -𓋼")
	# 	print(forest[i])
	f_accuracy, f_mse = forest_accuracy(forest, loans)
	print("𓋼 Forest MSE: " + str(f_mse))

	#-----------
	# tree = build_tree(a_1, 6, candidates)
	# print("tree")
	# print(tree)
	# accuracy, mse = find_accuracy(a_1, tree)
	# print("accuracy A1: " + str(accuracy) + " mse: " + str(mse))
	# accuracy, mse = find_accuracy(a_2, tree)
	# print("accuracy A2: " + str(accuracy) + " mse: " + str(mse))
	
	## ---- processing loans from dataset c
	# predict_loans, attributes = load_data("../loans_C_unlabeled.csv", train=False)
	# predict_loans = add_features(predict_loans)
	# predict_loans, new_amt_features = add_features_loan_amt(predict_loans, 200, 400, 800)
	# predict_loans, new_repay_features = add_features_repayment_term(predict_loans, 4)
	# predict_loans, new_year_features = add_features_loan_year(predict_loans, 2009, 2012)

	# with open('loans_C_processed', 'wb') as file:
	# 	pickle.dump(predict_loans, file)
	with open('loans_C_processed', 'rb') as file:
		predict_loans = pickle.load(file)

	ids = [loan[0]["id"] for loan in predict_loans]
	predictions = []

	# print(predict_loans[32834][0])
	
	for loan in predict_loans:
		prediction = forest_predict(forest, loan)
		predictions.append(prediction)
	
	num_prediction_loans = len(ids)
	# print(num_prediction_loans)

	with open("loans_C_predicted_cx_kw.csv", 'w') as file:
		writer = csv.writer(file)
		writer.writerow(["ID", "days_until_funded_CX_KW"])
		for i in range(num_prediction_loans):
			writer.writerow([str(ids[i]) , str(predictions[i])])

main()
