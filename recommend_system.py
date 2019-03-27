import numpy as np

def jac(ss1, ss2):
	return len(set(ss1) & set(ss2)) / len(set(ss1) | set(ss2))

def sim(mat):
	n_mat = len(mat)
	data_matrix = np.zeros((n_mat, n_mat))
	for i in range(n_mat - 1):
		for j in range(i + 1, n_mat):
			data_matrix[i][j] = jac(mat[i], mat[j])
	return data_matrix

def rec(mat2):
	mat3 = sim(mat2)
	mat3 = mat3 + mat3.T
	arr_total = []
	for i in range(len(mat3)):
		mat_temp = mat3[i]
		sort = np.sort(mat_temp)
		sort_index = np.argsort(mat_temp)
		arr_temp = [[sort_index[::-1][j], sort[::-1][j]] for j in range(len(sort))]
		arr_total.append(arr_temp)

	return arr_total

# %%
if __name__ == '__main__':
	result = rec(np.array([[1,2,3,4],[5,6,7,8],[1,2,5,6],[1,3,2,4,8,6]]))
	print(result)
