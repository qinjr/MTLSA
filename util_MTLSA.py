import numpy as np
import scipy.io
import random
from sklearn.metrics import *
import h5py


def sample_data(sample_rate, input_file, output_file):
	fin = open(input_file, 'r')
	lines = fin.readlines()
	data_amt = len(lines)

	sample_num = data_amt * sample_rate

	s = set()
	while len(s) < sample_num:
		candidate = random.randint(0, data_amt - 1)
		s.add(candidate)

	out = []#lines[0:int(len(lines)*sample_rate)]
	for i in s:
		out.append(lines[i])

	fout = open(output_file, 'w')
	fout.writelines(out)

	fin.close()
	fout.close()

def get_zplus1(csv_file_z, csv_file_zp1):
	fin = open(csv_file_z, 'r')
	lines = fin.readlines()
	newlines = []
	for line in lines:
		items = line.split(',')
		items[-2] = str(int(float(items[-2])) + 1)
		newline = ','.join(items)
		newlines.append(newline)
	fout = open(csv_file_zp1, 'w')
	fout.writelines(newlines)
	fout.close()
	fin.close()

def getANLP(campaign, csv_file, MTLSA_res_file_z, MTLSA_res_file_z1, best_auc_index):
	fin = open(csv_file, 'r')
	lines = fin.readlines()
	price = []
	label = []
	for line in lines:
		items = line.split(',')
		label.append(1 - int(float(items[-1][:-1])))
		price.append(int(float(items[-2])) - 1)
	fin.close()
	label = np.array(label)
	price = np.array(price)[label == 1]
	price_1 = price + 1


	mat_z = h5py.File(MTLSA_res_file_z, 'r')['win_rate'][()]
	mat_z1 = h5py.File(MTLSA_res_file_z1, 'r')['win_rate'][()]

	s_z_prices = mat_z.reshape(mat_z.shape[2], mat_z.shape[1], mat_z.shape[0])[best_auc_index][label == 1]
	s_z1_prices = mat_z1.reshape(mat_z1.shape[2], mat_z1.shape[1], mat_z1.shape[0])[best_auc_index][label == 1]

	s_z = np.zeros([s_z_prices.shape[0]])
	s_z1 = np.zeros([s_z1_prices.shape[0]])
	for i in range(s_z_prices.shape[0]):
		s_z[i] = s_z_prices[i, price[i]]
		s_z1[i] = s_z1_prices[i, price_1[i]]

	p = s_z - s_z1
	p[p <= 0] = 1e-8
	anlp = np.mean(-np.log(p))

	print(campaign, anlp)
	return anlp


def getAUC(campaign, csv_file, MTLSA_res_file):
	fin = open(csv_file, 'r')
	lines = fin.readlines()
	price = []
	label = []
	for line in lines:
		items = line.split(',')
		label.append(1 - int(float(items[-1][:-1])))
		price.append(int(float(items[-2])) - 1)

	fin.close()

	price = np.array(price)
	mat = h5py.File(MTLSA_res_file, 'r')['win_rate'][()]#scipy.io.loadmat(MTLSA_res_file)
	mat = mat.reshape(mat.shape[2], mat.shape[1], mat.shape[0])
	best_auc = 0
	best_auc_index = 0
	cross_entropy = 0
	for index in range(mat.shape[0]):
		preds_prices = mat[index]
		preds = np.zeros([preds_prices.shape[0],])
		for i in range(price.shape[0]):
			preds[i] = preds_prices[i, price[i]]

		#preds = 1 - preds
		label = 1 - np.array(label).reshape(-1,)
		auc = roc_auc_score(label, preds)
		if auc > best_auc:
			best_auc_index = index
			best_auc = auc
			cross_entropy = log_loss(label, preds)
	print(campaign, best_auc)
	print(campaign, cross_entropy)
	return best_auc, best_auc_index


if __name__ == '__main__':
	campaign_list = ['2261']#2261
	for campaign in campaign_list:
		print(campaign)
		#sample_data(0.1, 'data/deep-bid-lands-data/' + campaign + '/train.emb.csv', 'data/deep-bid-lands-data/' + campaign + '/train.emb.mini.csv')
		#sample_data(0.1, 'data/deep-bid-lands-data/' + campaign + '/test.emb.csv', 'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z.csv')
		#get_zplus1('data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z.csv', 'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z1.csv')
		auc, best_auc_index = getAUC(campaign, 'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z.csv', 'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z0.01_z_result.mat')
		anlp = getANLP(campaign, 'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z.csv', \
								'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z0.01_z_result.mat', \
								'data/deep-bid-lands-data/' + campaign + '/test.emb.mini.z10.01_z1_result.mat', best_auc_index)
		fout = open('MTLSA_res.txt', 'a')
		fout.write(campaign + '\t' + str(auc) + '\t' + str(anlp) + '\n')
