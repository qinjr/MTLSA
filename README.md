
# BASELINE4DESA_MTLSA
This repo is forked from [MTLSA](https://github.com/MLSurvival/MTLSA), I modified some code in the repo to make it a baseline model for [DESA](https://github.com/qinjr/deep-bid-lands/tree/master/published_code), which is the implementation of the model proposed in a KDD'18 submitted paper, "Deep Survival Analysis for Fine-grained Bid Landscape Forecasting in Real-time Bidding Advertising".
Many thanks to the authors of `MTLSA`.

### Data Preparation
We have upload a tiny data sample for training and evaluation.
The full dataset for this project can be download from this [link](http://apex.sjtu.edu.cn/datasets/13).
After download please replace the sample data in `data/deep-bid-lands-data/` folder with the full data files.

### Installation and Running
Please install `h5py` first.
The code contains two parts, so you need to run it by following the steps here:
* step1:
	If you just want to run the demo, you should execute the run.m in MATLAB, and that is enough 		for step1. If you want to run it with full volume data, you should follow the instructions in previous section and modifiy the run.m script to use different campaign's data. It is very simple.
* step2:
	As step1, if you just want to run the demo, just execute:
	```
	python3 util_MTLSA.py
	```
  and it will give the AUC, log-loss and ANLP value trained on the sample data.
  If you want to get full volume data result, do step1 and modify `util_MTLSA.py` in `campaign_list` variable.

Be patient, to run the code with demo data we need about 30 minutes. Much slower when running it with full volume dataset.
