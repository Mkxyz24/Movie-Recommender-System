from scipy import sparse		#import sparse module from SciPy package
import numpy as np    		#import NumPy
import csv
from sys import maxsize
import time

INT_MIN = -maxsize -1 				



def sim(mat_coo, i, j):
	"""
	input: sparse matrix, item i, item j
	returns similarity of item i and item j
	"""
	r = mat_coo
	i_vector = r.getrow(i)
	j_vector = r.getrow(j)
	if(i_vector.getnnz() == 0 or j_vector.getnnz() == 0):
		return -1
	i_mean = i_vector.sum()/i_vector.getnnz()
	j_mean = j_vector.sum()/j_vector.getnnz()
	i_vector.data -= i_mean
	j_vector.data -= j_mean
	numerator = (i_vector.dot(j_vector.transpose())).data
	if len(numerator) == 0:
		return 0	
	denominator = ((i_vector.dot(i_vector.transpose())).data)*((j_vector.dot(j_vector.transpose())).data)
	similarity = numerator/np.sqrt(denominator)
	similarity = round(float(similarity),2) 
	return similarity


def findrating(mat_coo, user, item, baseline):
	"""
	input: sparse matrix, user id, item id,baseline = true or false
	output : estimated rating of item  by user

	"""
	r = mat_coo.copy()
	m = 0
	bx = 0
	bi = 0
	user_vector = r.getcol(user)
	item_vector = r.getrow(item)
	if baseline == True:
		m = r.sum()/r.getnnz()
		bx = (user_vector.sum()/user_vector.getnnz()) -m
		bi = (item_vector.sum()/item_vector.getnnz()) -m
	bxi = m + bx + bi
	S = np.zeros(mat_coo.shape[0])
	for i in range(mat_coo.shape[0]):
		S[i] = sim(r, item, i)
	S[item] = INT_MIN
	r_mat = r.toarray()
	numerator = 0
	denominator = 0
	for i in range(2):
		max_sim = np.argmax(S)
		temp_vector = r.getrow(max_sim)
		bxj = 0
		bj = 0
		if baseline == True:
			bj = (temp_vector.sum()/temp_vector.getnnz()) -m
		bxj = m + bx + bj
		numerator += (r_mat[max_sim, user] - bxj)*S[max_sim]
		denominator += S[max_sim]
		S[max_sim] = INT_MIN
	if denominator == 0:
		rating = bxi
	else:	
		rating = round(bxi + (numerator/denominator), 2)
	if rating < 0:
		rating = 0
	if rating > 5:
		rating = 5	
	return rating



def complete(mat_coo, util_mat, baseline):
	"""
	input sparse matrix and array
	output: matrix with respective estimated ratings of ratings in given matrix 
	"""
	filled = np.zeros(mat_coo.shape)
	row = mat_coo.row
	column = mat_coo.col
	for i in range(len(row)):
		filled[row[i],column[i]] = findrating(mat_coo,column[i],row[i],baseline)
	
	
	return filled
def rmse_spearman_correlation(true_coo, predicted_array):
	"""
	function to find rmse and spearman rank correlation 
	input: original sparse matrix, predicted matrix

	"""
	true_array = true_coo.toarray()
	row = true_coo.row
	column = true_coo.col
	N = len(row)
	error = 0
	for i in range(N):
		error += (true_array[row[i]][column[i]] - predicted_array[row[i]][column[i]])**2
	rmse = (error/N) ** 0.5
	spearman_correlation = 1 - 6*(error/(N*((N**2) - 1)))
	return round(rmse,5), round(spearman_correlation,5)

def precision_top_k(k, true_urm,predicted_array):
	x = predicted_array.shape[0]
	y = predicted_array.shape[1]
	true_urm = true_urm[:x,:y]
	true_array = true_urm.toarray()
	top_predicted = dict()
	top_user = dict()

	for i in range(y):
		user_row = predicted_array[:,i]
		for ind in range(k):
			temp = []
			index = np.argmax(user_row)
			temp.append(index)
			user_row[index] = INT_MIN
		top_predicted[i] = temp
	for i in range(y):
		user_row =  true_array[:,i]
		for ind in range(k):
			temp = []
			index = np.argmax(user_row)
			user_row[index] = INT_MIN
			temp.append(index)
		top_user[i] = temp
	precision = 0
	for i in range(y):
		lst1 = top_user[i]
		lst2 = top_predicted[i]
		temp = set(lst2) 
		lst3 = [value for value in lst1 if value in temp]
		precision += len(lst3)/k
	return precision/y



if __name__ == "__main__":
	row=[]
	col=[]
	data=[]
	rank=[]
	with open('testfinal.data','rt') as tsv:
		reader = csv.reader(tsv,delimiter='\t',quoting=csv.QUOTE_NONNUMERIC)
		for item in reader:
			row.append(int(item[1]-1))
			col.append(int(item[0]-1))
			data.append(float(item[2]))

	mat_coo = sparse.coo_matrix((data,(row,col)))
	mat_csr = mat_coo.tocsr()
	mat_csc = mat_coo.tocsc()
	util_mat = mat_coo.toarray()

	
	test_coo = mat_csr[:40,:]
	test_coo = test_coo.tocsc()
	test_coo = test_coo[:,:40]
	test_coo = test_coo.tocoo()
	test_util_mat = test_coo.toarray()
	
	t0 = time.time()
	predicted_matrix_f = complete(test_coo, test_util_mat,False)
	t1 = time.time()
	time_f = t1-t0
	print(f"time taken for prediction without baseline:  {time_f}")
	t0 = time.time()
	predicted_matrix_t = complete(test_coo, test_util_mat,True)
	t1 = time.time()
	time_t = t1-t0
	print(f"time taken for prediction with baseline:  {time_t}")

	rmse_f,spc_f = rmse_spearman_correlation(test_coo,predicted_matrix_f)
	rmse_t,spc_t = rmse_spearman_correlation(test_coo,predicted_matrix_t)
	top_10_f = precision_top_k(10,mat_csc,predicted_matrix_f)
	top_10_t = precision_top_k(10,mat_csc,predicted_matrix_t)
	#print(test_util_mat)
	#print(predicted_matrix_f)

	print(f"rmse for collaborative filtering without baseline:  {rmse_f}")
	print(f"rmse for collaborative filtering with baseline:  {rmse_t}")
	print(f"spearman rank correlation for collaborative filtering without baseline:  {spc_f}")
	print(f"spearman rank correlation for collaborative filtering with baseline:  {spc_t}")
	print(f"precision at top 10 without baseline: {top_10_f}")
	print(f"precision at top 10 with baseline: {top_10_t}")


