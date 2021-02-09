import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import math as mt
import csv
import time
from sparsesvd import sparsesvd
from scipy.sparse.linalg import * #used for matrix multiplication
from numpy import linalg as LA
from sys import maxsize


INT_MIN = -maxsize - 1

def readUrm():
	"""
	function to read input file and return sparse matrix
	"""
	row=[]
	col=[]
	data=[]
	with open('/Users/mohith/Desktop/ir ass/finmax.data', 'rt') as trainFile:
		urmReader = csv.reader(trainFile, delimiter='\t')
		for item in urmReader:
			row.append(int(item[0]))
			col.append(int(item[1]))
			data.append(float(item[2]))
	mat_coo = sparse.coo_matrix((data,(row,col)))
	mat_csc = mat_coo.tocsc()

	return mat_csc


def readUsersTest():
	"""
	function to store user id's of test file in a dictionary 

	output: dictionary which stores user id's
	"""

	user_rec_dic = dict()
	with open("/Users/mohith/Desktop/ir ass/testfinal.data", 'rt') as testFile:
		testReader = csv.reader(testFile, delimiter='\t')
		for row in testReader:
			user_rec_dic[int(row[0])] = list()

	return user_rec_dic


def getMoviesSeen():
	"""
	function to store all movies already rated by users in test file

	output: dictionary of movies already rated by users
	"""

	moviesSeen = dict()
	with open("/Users/mohith/Desktop/ir ass/testfinal.data", 'rt') as trainFile:
		urmReader = csv.reader(trainFile, delimiter='\t')
		for row in urmReader:
			try:
				moviesSeen[int(row[0])].append(int(row[1]))
			except:
				moviesSeen[int(row[0])] = list()
				moviesSeen[int(row[0])].append(int(row[1]))

	return moviesSeen


def e_decompose(urm):
	"""
	function to carry the eigen value decomposition of the given sparse matrix
	output :  1)list of sorted eigen values
			  2)matrix corresponding eigen vectors as the columns
	"""

	arr = urm.toarray()
	e_values,e_vectors = LA.eig(arr)
	e_values = e_values.real
	e_vectors = e_vectors.real
	for i in range(len(list(e_values))):
		e_values[i] = round(e_values[i],2)
	for i in range(e_vectors.shape[0]):
		for j in range(e_vectors.shape[1]):
			e_vectors[i][j] = round(e_vectors[i][j],2)
	eigen = dict()
	for i in range(len(e_values)):
		if e_values[i] !=0:
			eigen[e_values[i]] = e_vectors[:,i]
	sorted_eval = sorted(list(eigen.keys()), reverse=True)
	sorted_evec = np.zeros_like(e_vectors)
	for i in range(len(sorted_eval)):
		sorted_evec[:, i] = eigen[sorted_eval[i]]
	sorted_evec = sorted_evec[:,:len(sorted_eval)]

	return sorted_eval,sorted_evec

def computeSVD(urm, dimreduc):
	"""
	function to compute the singular value decomposition of given matrix
	output: U,S,Vt matrices from singular value decompostion
	"""

	arr = urm.toarray()
	AAT = urm.dot(urm.transpose())
	ATA = (urm.transpose()).dot(urm)
	e_val_U, U = e_decompose(AAT)
	e_val_V, V = e_decompose(ATA)

	Vt = V.transpose()
	#print(e_val_U)
	#print(e_val_V)
	S = np.diag([np.sqrt(s) for s in e_val_U])
	if dimreduc != 1:
		total_sigma = np.sum(S ** 2)
		for i in range(S.shape[0]):
			sigma_sum = np.sum(S[:i+1, :i+1] ** 2)
			if sigma_sum > dimreduc * total_sigma:
				S = S[:i, :i]
				U = U[:, :i]
				Vt = Vt[:i, :]
				return U, S, Vt
		"""
		s_len = S.shape[0]
		#print(s_len)
		reduced_len = round(dimreduc * s_len)
		#print(reduced_len)
		S = S[:reduced_len,:reduced_len]
		U = U[:,:reduced_len]
		Vt = Vt[:reduced_len,:]
		"""
	return U,S,Vt



def computeEstimatedRatings(urm, U, S, Vt, user_rec_dic, moviesSeen, test):
	"""
	function to estimate the ratings of user and reccomend the top 5 movies which are not rated by user
	output: dictionary with keys as user id and values as recommended movies
	"""
	rightTerm = np.dot(S,Vt) 
	MAX_UID = np.shape(urm)[0]
	MAX_PID = np.shape(urm)[1]
	estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
	for userTest in user_rec_dic:
		prod = np.dot(U[userTest, :],rightTerm)
		#we convert the vector to dense format in order to get the indices of the movies with the best estimated ratings 
		estimatedRatings[userTest, :] = prod
		recom = (-estimatedRatings[userTest, :]).argsort()[:250]
		for r in recom:
			if r not in moviesSeen[userTest]:
				user_rec_dic[userTest].append(r)

				if len(user_rec_dic[userTest]) == 5:
					break
			
	#print(estimatedRatings)
	return user_rec_dic


def rmse_spearman_correlation(true_urm, predicted_array):
	"""
	function to calculate rmse and spearman rank correlation of the original matrix and estimated matrix
	output: rmse value and spearman rank correlation value
	"""
	x = predicted_array.shape[0]
	y = predicted_array.shape[1]
	true_urm = true_urm[:x,:y]
	true_coo = true_urm.tocoo()
	true_array = true_coo.toarray()
	row = true_coo.row
	column = true_coo.col
	N = len(row)
	error = 0
	for i in range(N):
		error += (true_array[row[i]][column[i]] - predicted_array[row[i]][column[i]])**2
	rmse = (error/N) ** 0.5
	error = 6*error
	error = error/N
	error = error/((N**2)-1)
	spearman_correlation = 1 - error
	return round(rmse,5), round(spearman_correlation,5) 


def precision_top_k(k, true_urm,predicted_array):
	"""
	function to calculate the precision at top k of the given original and estimated matrix
	finds precision as ratio of # of relevant recommended and #relevant
	output: precision at top k 
	"""

	x = predicted_array.shape[0]
	y = predicted_array.shape[1]
	true_urm = true_urm[:x,:y]
	true_array = true_urm.toarray()
	top_predicted = dict()
	top_user = dict()

	for i in range(x):
		user_row = predicted_array[i,:]
		for ind in range(k):
			temp = []
			index = np.argmax(user_row)
			temp.append(index)
			user_row[index] = INT_MIN
		top_predicted[i] = temp
	for i in range(x):
		user_row =  true_array[i,:]
		for ind in range(k):
			temp = []
			index = np.argmax(user_row)
			user_row[index] = INT_MIN
			temp.append(index)
		top_user[i] = temp
	precision = 0
	for i in range(x):
		lst1 = top_user[i]
		lst2 = top_predicted[i]
		temp = set(lst2) 
		lst3 = [value for value in lst1 if value in temp]
		precision += len(lst3)/k
	return precision/x
		





if __name__ == "__main__":
	
	urm = readUrm()												## <------original
	urm_coo = urm.tocoo()
	test_error_urm = urm[:300,:500]								## <------test matrix
	test_error_urm_coo = test_error_urm.tocoo()
	U, S, Vt = computeSVD(urm, 1)
	
	
	user_rec_dic = readUsersTest()
	moviesSeen = getMoviesSeen()
	
	
	t0 = time.time()
	Utest,Stest,Vttest = computeSVD(test_error_urm,1)
	predict_mat = np.dot(np.dot(Utest,Stest),Vttest)
	print(f"time taken for SVD prediction:  {time.time()-t0}")

	t0 = time.time()
	Utest90,Stest90,Vttest90 = computeSVD(test_error_urm,0.8)
	predic_mat_90 = np.dot(np.dot(Utest90,Stest90),Vttest90)
	print(f"time taken for SVD prediction with 90% retained energy:  {time.time()-t0}")
	
	rmse_val,spc = rmse_spearman_correlation(urm, predict_mat)
	rmse_val_90,spc_90 = rmse_spearman_correlation(urm, predic_mat_90)
	top_10 = precision_top_k(10, urm, predict_mat)
	top_10_90 = precision_top_k(10, urm, predic_mat_90)
	user_rec_dic = computeEstimatedRatings(urm, U, S, Vt, user_rec_dic, moviesSeen, True)
	
	print(f"rmse value for SVD is: {rmse_val}")
	print(f"rmse value for SVD with 90% retained energy : {rmse_val_90}")
	print(f"spearman rank correlation for SVD: {spc}")
	print(f"spearman rank correlation for SVD with 90% retained energy: {spc_90}")
	print(f"precision at top 10 for SVD is: {top_10}")
	print(f"precision at top 10 for SVD with 90% retained energy: {top_10_90}")
	for u in user_rec_dic:
		if u<10:
			print(f"movie recommendations for user {u}: {user_rec_dic[u]} \n")


