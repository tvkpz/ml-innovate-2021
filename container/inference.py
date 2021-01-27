from io import BytesIO
import numpy as np
import xgboost as xgb
import pickle as pkl
import os

def input_fn(request_body, request_content_type):
    """An input_fn that loads a numpy array"""
    print('content_type', request_content_type)
    print('actual_content_type', type(request_body))
    print(request_body)
    if request_content_type == "text/csv":
        data = []
        examples = request_body.split('\n')
        for e in examples:
            arr = e.split(',')
            print(arr)
            array = [int(x) for x in arr]
            print(array)
            data.append(array)
            #array = array[np.newaxis, :]
        data = np.array(data)
        print(data.shape)
        return xgb.DMatrix(data)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass

def model_fn(model_dir):
    print('model_dir', model_dir)
    print('dir_elements', os.listdir(model_dir))
    print('xgb_version', xgb.__version__)
    with open(os.path.join(model_dir, "model.bin"), "rb") as f:
        iteration = pkl.load(f)
        booster = pkl.load(f)
    print('iteration=',iteration)
    print('booster=',booster)
    return (iteration, booster)

def predict_fn(input_data, bst):
    iteration = bst[0]
    model = bst[1]
    print('input_data: ',input_data)
    print('iteration_received=',iteration)
    print('model=',model)
    pred = model.predict(input_data, ntree_limit=iteration)
    print('pred: ', pred)
    res = np.array([1 if p > 0.5 else 0 for p in pred])
    feature_contribs = model.predict(input_data, ntree_limit=iteration, pred_contribs=True)
    print('f_contribs: ', feature_contribs)
    output = np.hstack((pred[:, np.newaxis], feature_contribs))
    print('output: ', output)
    return output