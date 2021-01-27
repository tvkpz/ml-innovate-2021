import argparse
import bz2
import json
import os
import pickle
import random
import tempfile
import urllib.request
import pandas as pd
import glob
import pickle as pkl
import numpy as np
import boto3
import logging
from botocore.exceptions import ClientError

import xgboost
from sklearn import metrics
#from smdebug import SaveConfig
#from smdebug.xgboost import Hook
from sklearn.model_selection import StratifiedKFold
from collections import namedtuple
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
from fairlearn.metrics import selection_rate

metric = 'f1'

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--eta", type=float, default=1)  # 0.2
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_round", type=int, default=30)
    parser.add_argument("--metric", type=str, default='f1')
    parser.add_argument("--protected", type=str, default='Gender')
    parser.add_argument("--thresh", type=float, default=0.5)
    

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--s3_bucket', type=str, default=None)

    args = parser.parse_args()

    return args

def custom_asymmetric_objective(y_pred, dtrain):
    #y_pred[y_pred < -1] = -1 + 1e-6
    y_true = dtrain.get_label()
    #print(min(y_pred), max(y_pred))
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred = [1.0 if y > 0.5 else 0.0 for y in y_pred]
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual<0, -2*10.0*residual, -2*residual)
    hess = np.where(residual<0, 2*10.0, 2.0)
    return grad, hess

def huber_approx_obj(preds, dtrain):
    #print('obj:', min(preds), max(preds))
    preds = 1.0 / (1.0 + np.exp(-preds))
    preds = [1.0 if y > 0.5 else 0.0 for y in preds]
    d = dtrain.get_label()-preds #remove .get_labels() for sklearn
    #print('obj-d:', min(d), max(d))
    h = 1  #h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_1 = 1 + (d / (4+h)) ** 2
    scale_sqrt = np.sqrt(scale)
    scale_sqrt_1 = np.sqrt(scale_1)
    grad = np.where(d<0, -d / scale_sqrt_1, -d / scale_sqrt)
    hess = np.where(d<0, 1 / scale_1 / scale_sqrt_1, 1 / scale / scale_sqrt)
    return grad, hess

#difference in statistical parity
def fair_metrics(bst,data,column, thresh):
    tr = list(data.get_label())
    best_iteration = bst.best_ntree_limit
    pred=bst.predict(data, ntree_limit=best_iteration)
    pred = [1 if p > thresh else 0 for p in pred]
    na0=0
    na1=0
    nd0=0
    nd1=0
    for p,c in zip(pred,column):
        if (p==1 and c==0):
            nd1 += 1
        if (p==1 and c==1):
            na1 += 1
        if (p==0 and c==0):
            nd0 += 1
        if (p==0 and c==1):
            na0 += 1
    Pa1, Pd1, Pa0, Pd0 = na1/(na1+na0), nd1/(nd1+nd0), na0/(na1+na0), nd0/(nd1+nd0)
    dsp_metric = np.abs(Pd1-Pa1)
    #dsp_metric = np.abs((first-second)/(first+second))
    sr_metric = selection_rate(tr, pred, pos_label=1)
    dpd_metric = demographic_parity_difference(tr, pred, sensitive_features=column)
    dpr_metric = demographic_parity_ratio(tr, pred, sensitive_features=column)
    eod_metric = equalized_odds_difference(tr, pred, sensitive_features=column)
    
    return dsp_metric, sr_metric, dpd_metric, dpr_metric, eod_metric
    
def eval_fun(y_pred, dtrain):
    #y_pred[y_pred < -1] = -1 + 1e-6
    y_true = list(dtrain.get_label())
    #print('eval:', min(y_pred), max(y_pred))
    #y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred = [1.0 if y > 0.5 else 0.0 for y in y_pred]
    y_pred = list(y_pred)
    d = y_pred - dtrain.get_label()
    #print('---', Counter(d))
    #print(y_true)
    #print(y_pred)
    tp = sum([1 if (t==1 and p==1) else 0 for t,p in zip(y_true,y_pred)])
    fn = sum([1 if (t==1 and p==0) else 0 for t,p in zip(y_true,y_pred)])
    fp = sum([1 if (t==0 and p==1) else 0 for t,p in zip(y_true,y_pred)])
    tn = sum([1 if (t==0 and p==0) else 0 for t,p in zip(y_true,y_pred)])
    #print('eval-d:', tp/sum(y_true), fn/sum(y_true))
    precision = 0 if (tp+fp)==0 else tp / (tp + fp) # positive predictive value
    recall = 0 if (tp+fn)==0 else tp / (tp + fn) # true_positive rate
    false_negative_rate = 0 if (fn+tp)==0 else fn / (fn+tp)
    false_positive_rate = 0 if (fp+tn)==0 else fp / (fp+tn)
    f1 = 0 if (precision+recall)==0 else 2 * precision * recall / (precision + recall)
    
    if metric == 'recall':
        return "recall", recall
    if metric == 'f1':
        return "f1", f1

def eval_f1(y_pred, dtrain):
    #y_pred[y_pred < -1] = -1 + 1e-6
    y_true = list(dtrain.get_label())
    y_pred = [1.0 if y > 0.5 else 0.0 for y in y_pred]
    y_pred = list(y_pred)
    tp = sum([1 if (t==1 and p==1) else 0 for t,p in zip(y_true,y_pred)])
    fn = sum([1 if (t==1 and p==0) else 0 for t,p in zip(y_true,y_pred)])
    fp = sum([1 if (t==0 and p==1) else 0 for t,p in zip(y_true,y_pred)])
    tn = sum([1 if (t==0 and p==0) else 0 for t,p in zip(y_true,y_pred)])
    #print('eval-d:', tp/sum(y_true), fn/sum(y_true))
    precision = 0 if (tp+fp)==0 else tp / (tp + fp) # positive predictive value
    recall = 0 if (tp+fn)==0 else tp / (tp + fn) # true_positive rate
    f1 = 0 if (precision+recall)==0 else 2 * precision * recall / (precision + recall)

    return "f1", f1

def eval_auc(y_pred, dtrain):
    y_true = list(dtrain.get_label())
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return "auc", auc
    
def write_metadata(path, train_recall, val_recall):
    metrics = {
        'metrics': [{
            'name': 'train-recall',  # The name of the metric. Visualized as the column name in the runs table.
            'numberValue': train_recall,  # The value of the metric. Must be a numeric value.
            'format': "PERCENTAGE",  # The optional format of the metric.
        },
        {
            'name': 'val-recall',
            'numberValue': val_recall,
            'format': "PERCENTAGE"
        }]
    }
    
    logging.info("Succeed in Writing Training Metrics")
    with open('/opt/ml/output/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    divmod_output = namedtuple('evalMetrics', ['train_recall', 'val_recall', 'mlpipeline_metrics'])
    return divmod_output(train_recall, val_recall, json.dumps(metrics))


def xgb_evaluate(bst, data):
    y = list(data.get_label())
    best_iteration = bst.best_ntree_limit
    #print('best_iteration:',best_iteration)
    pred=bst.predict(data, ntree_limit=best_iteration)
    #pred = 1.0 / (1.0 + np.exp(-pred))
    pred = [1 if p > 0.5 else 0 for p in pred]
    tp = sum([1 if (t==1 and p==1) else 0 for t,p in zip(y,pred)])
    fn = sum([1 if (t==1 and p==0) else 0 for t,p in zip(y,pred)])
    fp = sum([1 if (t==0 and p==1) else 0 for t,p in zip(y,pred)])
    tn = sum([1 if (t==0 and p==0) else 0 for t,p in zip(y,pred)])
    recall = 0 if (tp+fn)==0 else tp / (tp + fn)
    precision = 0 if (tp+fp)==0 else tp / (tp + fp)
    f1 = 0 if (precision+recall)==0 else 2 * precision * recall / (precision + recall)
    
    if metric == 'recall':
        return recall
    if metric == 'f1':
        return f1
    
def main():

    args = parse_args()
    train_files_path = args.train

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        #"silent": args.silent,
        "tree_method": 'hist',
        #"disable_default_eval_metric": 1,
        "objective": args.objective,
        #"num_class": args.num_class
    }
    
    job_name = json.loads(os.environ['SM_TRAINING_ENV'])['job_name']
    
    train_files_list = glob.glob(train_files_path + '/*.*')
    print(train_files_list)
    
    print('Loading training data...')
    df_train = pd.concat(map(pd.read_csv, train_files_list))
    print('Data loading completed.')
    
    y = df_train.Target.values
    pcol = df_train[args.protected].values
    X =  df_train.drop(['Target'], axis=1).values

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    split_count = 1
    best_model = []
    if metric == 'auc':
        auc = 0
    if metric == 'f1':
        f1 = 0
    if metric == 'recall':
        recall = 0

    best_train_metric = []
    best_val_metric = []
    best_dsp_train = []
    best_dsp_val = []
    best_sr_train = []
    best_sr_val = []
    best_dpd_train = []
    best_dpd_val = []
    best_dpr_train = []
    best_dpr_val = []
    best_eod_train = []
    best_eod_val = []
    best_iteration = 1
    
    path = os.path.join(args.s3_bucket, job_name, "hpo-debug", str(split_count))

    for train_index, test_index in skf.split(X, y):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pcol_train, pcol_test = pcol[train_index], pcol[test_index]
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test, label=y_test)

        '''
        hook = Hook(
            out_dir=path,  
            include_collections=['feature_importance', 'full_shap', 'average_shap', 'labels', 'predictions'],
            train_data=dtrain,
            validation_data=dtest,
            hyperparameters=params,
            save_config=SaveConfig(save_interval=10)
        )
        '''      
        watchlist = [(dtrain, "train"), (dtest, "validation")]
        evals_result=dict()
        
        bst = xgboost.train(
            params=params,
            dtrain=dtrain,
            #obj=custom_asymmetric_objective,
            feval=eval_f1,
            maximize=True,
            evals=watchlist,
            early_stopping_rounds=10,
            evals_result=evals_result,
            num_boost_round=args.num_round)
            #callbacks=[hook])
        
        print('evals_result: ', evals_result)
        print('best iteration: ',bst.best_ntree_limit)
        tr_result = xgb_evaluate(bst,dtrain)
        te_result = xgb_evaluate(bst,dtest)
        tr_dsp, tr_sr, tr_dpd, tr_dpr, tr_eod = fair_metrics(bst, dtrain, list(pcol_train), args.thresh)
        te_dsp, te_sr, te_dpd, te_dpr, te_eod = fair_metrics(bst, dtest, list(pcol_test), args.thresh)
        print('Best Stratified Model: train_f1= {}, val_f1= {}'.format(tr_result,te_result))
        print('Fairness Metrics (train): dsp={}, ,sr={}, dpd={}, dpr={}, eod={}'.format(tr_dsp,tr_sr,tr_dpd,tr_dpr,te_eod))
        print('Fairness Metrics (val): dsp={}, sr={}, dpd={}, dpr={}, eod={}'.format(te_dsp,te_sr,te_dpd,te_dpr,te_eod))
        if te_result > f1:
            best_model = bst
            best_train_metric = tr_result
            best_val_metric = te_result
            best_dsp_train = tr_dsp
            best_dsp_val = te_dsp
            best_sr_train = tr_sr
            best_sr_val = te_sr
            best_dpd_train = tr_dpd
            best_dpd_val = te_dpd
            best_dpr_train = tr_dpr
            best_dpr_val = te_dpr
            best_eod_train = tr_eod
            best_eod_val = te_eod
            results=evals_result
            best_iteration = bst.best_ntree_limit
            f1 = best_val_metric
            
        #if evals_result['validation']['recall'][args.num_round-1] > recall:
        #    best_model = bst
        #    best_train_metric = evals_result['train']['recall'][args.num_round-1]
        #    best_val_metric = evals_result['validation']['recall'][args.num_round-1]
        #    recall = evals_result['validation']['recall'][args.num_round-1]
 
    file_path = os.path.join(args.s3_bucket, job_name, "metrics")
    #train_metrics = write_metadata(file_path, best_train_metric, best_val_metric)
    #print("train-dsp:{},validation-dsp:{}".format(best_dsp_train, best_dsp_val))
    #print("train-sr:{},validation-sr:{}".format(best_sr_train, best_sr_val))
    #print("train-dpd:{},validation-dpd:{}".format(best_dpd_train, best_dpd_val))
    #print("train-dpr:{},validation-dpr:{}".format(best_dpr_train, best_dpr_val))
    #print("train-eod:{},validation-eod:{}".format(best_eod_train, best_eod_val))
    print("train-f1: {},validation-f1: {}".format(best_train_metric, best_val_metric))
    print("best iteration: ", best_iteration)
    model_dir = os.environ.get('SM_MODEL_DIR')
    with open(model_dir + '/model.bin', 'wb') as f:
        pkl.dump(best_iteration, f)
        pkl.dump(best_model, f)
    with open(model_dir + '/best_metrics.pkl', 'wb') as f:
        pkl.dump(best_train_metric, f)
        pkl.dump(best_val_metric, f)
        pkl.dump(best_dsp_train, f)
        pkl.dump(best_dsp_val, f)
        pkl.dump(best_sr_train, f)
        pkl.dump(best_sr_val, f)
        pkl.dump(best_dpd_train, f)
        pkl.dump(best_dpd_val, f)
        pkl.dump(best_dpr_train, f)
        pkl.dump(best_dpr_val, f)
        pkl.dump(best_eod_train, f)
        pkl.dump(best_eod_val, f)
    
    #return train_metrics

if __name__ == "__main__":
    main()