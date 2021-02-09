import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import math as mt
import csv
from sparsesvd import sparsesvd
from scipy.sparse.linalg import * #used for matrix multiplication
from numpy import linalg as LA
import random
import time


def readUrm():
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

def e_decompose(arr):
	
	e_values,e_vectors = LA.eig(arr)
	e_values = e_values.real
	e_vectors = e_vectors.real
	columns = list()
	#print(len(list(e_values)))
	#print(e_values)

	if 0 not in columns:
		columns.append(0)
	for i in range(len(list(e_values))):
		e_values[i] = round(e_values[i],2)
	for i in range(e_vectors.shape[0]):
		for j in range(e_vectors.shape[1]):
			e_vectors[i][j] = round(e_vectors[i][j],2)
	eigen = dict()
	for i in range(len(e_values)):
		if(e_values[i] != 0):
			eigen[e_values[i]] = e_vectors[:,i]
			columns.append(i)
	print(len(columns))
	#print(len(eigen))
	sorted_eval = sorted(list(eigen.keys()), reverse=True)
	sorted_evec = np.zeros_like(e_vectors)
	for i in range(len(sorted_eval)):
		sorted_evec[:, i] = eigen[sorted_eval[i]]
	sorted_evec = sorted_evec[:,:len(sorted_eval)]
	#print(len(sorted_eval))
	return sorted_eval,sorted_evec,columns

def computeSVD(urm_arr, retain = 1):
	
	AAT = np.matmul(urm_arr,urm_arr.T)
	ATA = np.matmul(urm_arr.T,urm_arr)
	e_val_U, U, columns_U = e_decompose(AAT)
	e_val_V, V, columns_V = e_decompose(ATA)
	Vt = V.transpose()
	#print(e_val_U)
	#print(e_val_V)
	S = list()
	for s in e_val_U:
		S.append(np.sqrt(s))
	if retain != 1:
		s_len = len(S)
		#print(s_len)
		reduced_len = round(retain * s_len)
		#print(reduced_len)
		S = S[:reduced_len]
		U = U[:,:reduced_len]
		Vt = Vt[:reduced_len,:]
	return U,S,Vt,columns_U

def cur(M, c, r, repeat=None, retain = 1):
	"""
	CUR function returns C,U,R

	Input:
	@M: input numpy array
	@c: Number of column selections
	@r: Number of row selections
	@repeat: Repetition allowed
	"""
	m_square_sum = np.sum(M ** 2)
	#print("m_square_sum " + str(m_square_sum))
	M_col, cols_sel = column_selection(M, m_square_sum, c, repeat=repeat)
	M_row, rows_sel = row_selection(M, m_square_sum, r, repeat=repeat)
	#print(rows_sel)
	#print(cols_sel)
	W = M[rows_sel, :]
	W = W[:, cols_sel]
	#print(W)
	_, W, _, columns= computeSVD(W,retain=retain)
	M_col = M_col[:, columns]
	M_row = M_row[columns, :]
	x = M_col.shape[1]
	y = M_col.shape[1]
	M_col = M_col[1:,:]
	M_row = M_row[1:,:]
	print(M_col.shape)
	print(M_row.shape)
	for i in range(len(W)):
		if W[i] != 0:
			W[i] = 1 / W[i]
	if retain != 1.0:
		
		s_len = len(W)
		#print(s_len)
		reduced_len = round(retain * s_len)
		#print(reduced_len)
		W = W[:reduced_len]
		M_col = M_col[:, :reduced_len]
		M_row = M_row[:reduced_len, :]
	W = np.diag(W)
	W = np.matmul(W,W)
	M_p = np.matmul(np.matmul(M_col,  W), M_row)
	#print("M_p")
	#print(M_p)
	return M_p


def column_selection(M, m_square_sum, c, repeat=None):
	"""
	Column selection algorithm

	Input:
	@M: Input numpy matrix M
	@m_square_sum: Sum of squares of elements of M
	@c: number of columns to select
	@repeat: Repetition allowed
	"""
	m_prob = np.zeros_like(M, dtype=np.float32)
	col_value = 0
	for i in range(M.shape[1]):
		col_value =np.sum(M.T[i] ** 2)
		col_value = float(col_value) / float(m_square_sum)
		m_prob.T[i] = col_value
	if repeat:
		column_selections = set()
		while len(column_selections) < c:
			column_selections.add(random.randint(0, M.shape[1]-1))
		column_selections = list(column_selections)
	else:
		column_selections = list()
		for i in range(c):
			column_selections.append(random.randint(0, M.shape[1]-1))
	m_prob = m_prob[:, column_selections]
	M = M[:, column_selections]
	for i in range(M.shape[1]):
		M.T[i] = M.T[i] / ((c*m_prob[0][i])**0.5)
	return [M , column_selections]


def row_selection(M, m_square_sum, r, repeat=None):
	"""
	Row selection algorithm

	Input:
	@M: Input numpy matrix M
	@m_square_sum: Sum of squares of elements of M
	@r: number of rows to select
	@repeat: Repetition allowed
	"""
	m_prob = np.zeros_like(M, dtype=np.float32)
	row_value = 0
	for i in range(M.shape[0]):
		row_value =np.sum(M[i] ** 2)
		row_value = float(row_value) / float(m_square_sum)
		m_prob[i] = row_value
	if repeat:
		row_selections = set()
		while len(row_selections) < r:
			row_selections.add(random.randint(0, M.shape[0]-1))
		row_selections = list(row_selections)
	else:
		row_selections = list()
		for i in range(r):
			row_selections.append(random.randint(0, M.shape[0]-1))
	m_prob = m_prob[row_selections, :]
	M = M[row_selections, :]
	for i in range(M.shape[0]):
		M[i] = M[i] / ((r*m_prob[i][0])**0.5)
	return M , row_selections

def rmse_spearman_correlation(true_urm, predicted_array):
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
	urm_csc = readUrm()
	urm_coo = urm_csc.tocoo()
	urm_arr = urm_coo.toarray()
	repeat = None
	retain = 0.8
	t0 = time.time()
	predict_mat = cur(urm_arr,200,200,repeat=repeat,retain=1)
	print(f"time taken for prediction CUR: {time.time() - t0}")
	t0 = time.time()
	predict_mat_90 = cur(urm_arr,200,200,repeat=repeat,retain=retain)
	print(f"time taken for prediction CUR with 90% retained energy: {time.time() - t0}")

	rmse_val,spc = rmse_spearman_correlation(urm_csc,predict_mat)
	rmse_val_90,spc_90 = rmse_spearman_correlation(urm_csc,predict_mat_90)
	top_10 = precision_top_k(10,urm_csc,predict_mat)
	top_10_90 = precision_top_k(10,urm_csc,predict_mat_90)

	print(f"rmse value for CUR is: {rmse_val}")
	print(f"rmse value for CUR with 90% retained energy : {rmse_val_90}")
	print(f"spearman rank correlation for CUR: {spc}")
	print(f"spearman rank correlation for CUR with 90% retained energy: {spc_90}")
	print(f"precision at top 10 for CUR is: {top_10}")
	print(f"precision at top 10 for cur with 90% retained energy: {top_10_90}")



	