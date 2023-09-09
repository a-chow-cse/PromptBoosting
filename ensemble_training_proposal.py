import numpy as np 
import argparse
import torch
import csv
from libsvm.svmutil import *

import tqdm
import time
import os

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from src.multicls_trainer import PromptBoostingMLTrainer
from src.ptuning import BaseModel, RoBERTaVTuningClassification, OPTVTuningClassification
from src.saver import PredictionSaver, TestPredictionSaver
from src.template import SentenceTemplate, TemplateManager
from src.utils import ROOT_DIR, BATCH_SIZE, create_logger, MODEL_CACHE_DIR
from src.data_util import get_class_num, get_template_list_with_filter, load_dataset, get_task_type, get_template_list

import wandb
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import xgboost as xgb


import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Classifier', add_help=False)
    parser.add_argument("--adaboost_lr", type = float, default = 1.0)
    parser.add_argument("--adaboost_weak_cls", type = int, default = 200)
    parser.add_argument("--dataset", type = str, default = 'sst')
    parser.add_argument("--model", type = str, default = 'roberta')
    parser.add_argument("--template_name", type = str, default = 't5_template3')
    parser.add_argument("--stop_criterior", type = str, default = 'best')
    parser.add_argument("--label_set_size", type = int, default = 5)
    parser.add_argument("--max_template_num", type = int, default = 0)

    parser.add_argument("--pred_cache_dir", type = str, default = '')
    parser.add_argument("--use_logits", action = 'store_true')
    parser.add_argument("--use_wandb", action = 'store_true')
    parser.add_argument("--change_template", action = 'store_true')
    parser.add_argument("--manual", action = 'store_true')

    parser.add_argument("--use_part_templates", action = 'store_true')
    parser.add_argument("--start_idx", type = int, default = 0)
    parser.add_argument("--end_idx", type = int, default = 10)

    parser.add_argument("--second_best", action = 'store_true')
    parser.add_argument("--sort_dataset", action = 'store_true')

    parser.add_argument("--fewshot", action = 'store_true')
    parser.add_argument("--low", action = 'store_true')
    parser.add_argument("--fewshot_k", type = int, default = 0)
    parser.add_argument("--fewshot_seed", type = int, default = 100, choices = [100, 13, 21, 42, 87])
    parser.add_argument("--algorithm", type = str, default = "None", choices = ["rf", "lasso", "svm", "xgboost","tsne"])

    parser.add_argument("--filter_templates", action = 'store_true')

    return parser


def get_promptBoostingFeatures(args):
    device = torch.device('cuda')
    adaboost_lr = args.adaboost_lr
    adaboost_weak_cls = args.adaboost_weak_cls
    template_name = args.template_name
    dataset = args.dataset
    sentence_pair = get_task_type(dataset)
    num_classes = get_class_num(dataset)
    model = args.model

    pred_cache_dir = args.pred_cache_dir
    sort_dataset = args.sort_dataset
    stop_criterior = args.stop_criterior
    use_logits = args.use_logits
    use_wandb = args.use_wandb
    label_set_size = args.label_set_size
    max_template_num = args.max_template_num

    adaboost_maximum_epoch = 20000

    fewshot = args.fewshot
    low = args.low
    fewshot_k = args.fewshot_k
    fewshot_seed = args.fewshot_seed

    assert not (fewshot and low), "fewshot and low resource can not be true!"
    filter_templates = args.filter_templates

    suffix = ""
    if args.use_part_templates:
        suffix = f"({args.start_idx}-{args.end_idx})"
    if filter_templates:
        suffix += f"filtered"

    wandb_name = f"{model}-{dataset}-{suffix}"

    if use_wandb:
        if fewshot:
            wandb_name += f"-{fewshot_k}shot-seed{fewshot_seed}"
        elif low:
            wandb_name += f"-low{fewshot_k}-seed{fewshot_seed}"
        wandb.init(project = f'vtuning-{dataset}', name = f'{wandb_name}')


    train_dataset, valid_dataset, test_dataset = load_dataset(dataset_name = dataset, sort_dataset = sort_dataset, fewshot = fewshot, k = fewshot_k, rand_seed = fewshot_seed,
                                                            low_resource = low)

    num_training = len(train_dataset[0])
    num_valid = len(valid_dataset[0])
    num_test = len(test_dataset[0])
    train_labels = torch.LongTensor(train_dataset[1]).to(device)
    valid_labels = torch.LongTensor(valid_dataset[1]).to(device)
    test_labels = torch.LongTensor(test_dataset[1]).to(device)

    weight_tensor = torch.ones(num_training, dtype = torch.float32).to(device) / num_training

    if model == 'roberta':
        vtuning_model = RoBERTaVTuningClassification(model_type = 'roberta-large', cache_dir = os.path.join(MODEL_CACHE_DIR, 'roberta_model/roberta-large/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    elif model == 'opt-6.7b':
        vtuning_model = OPTVTuningClassification(model_type = 'facebook/opt-6.7b', cache_dir = os.path.join(MODEL_CACHE_DIR, 'opt_model/opt-6.7b/'),
                                                device = device, verbalizer_dict = None, sentence_pair = sentence_pair)
    else:
        raise NotImplementedError

    if filter_templates:
        template_dir_list = get_template_list_with_filter(dataset, fewshot = fewshot, low = low,  fewshot_seed = fewshot_seed, 
                                                          fewshot_k = fewshot_k,  topk = 10, return_source_dir = False)
    else:
        template_dir_list = get_template_list(dataset, model = args.model)
    template_manager = TemplateManager(template_dir_list = template_dir_list, output_token = vtuning_model.tokenizer.mask_token, max_template_num = max_template_num,
                                        use_part_templates = args.use_part_templates, start_idx = args.start_idx, end_idx = args.end_idx)

    dir_list = "\n\t".join(template_manager.template_dir_list)
    print(f"using templates from: {dir_list}",)

    trainer = PromptBoostingMLTrainer(adaboost_lr = adaboost_lr, num_classes = num_classes, adaboost_maximum_epoch = adaboost_maximum_epoch, use_logits=True)

    if pred_cache_dir != '':
        prediction_saver = PredictionSaver(save_dir = os.path.join(ROOT_DIR, pred_cache_dir), model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,
                                            low = low,
                                            )
    else:
        prediction_saver = PredictionSaver(model_name = model,
                                            fewshot = fewshot, fewshot_k = fewshot_k, fewshot_seed = fewshot_seed,        
                                            )
    test_pred_saver = TestPredictionSaver(save_dir = os.path.join(ROOT_DIR, f'cached_test_preds/{dataset}/'), model_name = model)
    train_probs, valid_probs = [],[]

    word2idx = vtuning_model.tokenizer.get_vocab()

    # Obtain features/probs
    all_train_probs = None
    all_valid_probs = None
    all_test_probs = None
    for template in template_manager.template_list:
        template = template_manager.change_template()
        template.visualize()
        cached_preds, flag = prediction_saver.load_preds(template)
        if not flag:
            train_probs = trainer.pre_compute_logits(vtuning_model, template, train_dataset,)
            valid_probs = trainer.pre_compute_logits(vtuning_model, template, valid_dataset,)
            prediction_saver.save_preds(template, train_probs, valid_probs)
        else:
            train_probs, valid_probs = cached_preds
        
        test_probs = trainer.pre_compute_logits(vtuning_model, template, test_dataset)
        #print("Train Probes Shape: ",train_probs.shape)

        if all_train_probs is None:
            all_train_probs = train_probs
            all_valid_probs = valid_probs
            all_test_probs = test_probs
        else:
            all_train_probs = torch.cat((all_train_probs, train_probs), dim=1)
            all_valid_probs = torch.cat((all_valid_probs, valid_probs), dim=1)
            all_test_probs = torch.cat((all_test_probs, test_probs), dim=1)


    # Data
    train_features = all_train_probs.detach().cpu().numpy()
    train_labels = train_labels.detach().cpu().numpy()
    valid_features = all_valid_probs.detach().cpu().numpy()
    valid_labels = valid_labels.detach().cpu().numpy()
    test_features = all_test_probs.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    print("Train Features Shape: ",train_features.shape)
    print("Train Label Shape: ",train_labels.shape)
    print("Test Features Shape : ",test_features.shape)
    print("Test Label Shape :",test_labels.shape)

    return train_features,train_labels, valid_features, valid_labels,test_features,test_labels
"""
class HellingerDistanceCriterion(Criterion):
    def init(self):
        super().init()
        self.hellinger_norm = None

    def init_distributions(self):
        super().init_distributions()
        self.hellinger_norm = np.sqrt(2.0)

    def __call__(self, criterion_input):
        super().__call__(criterion_input)
        n_node_samples = criterion_input.n_node_samples
        n_classes = criterion_input.n_classes
        proba = criterion_input.proba
        value = 0.0

        for i in range(n_classes):
            sqrt_proba = np.sqrt(proba[:, i])
            value += np.square(sqrt_proba.sum())

        value /= (n_node_samples * self.hellinger_norm)

        return -value
        """

def randomForestCheck(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    best_valid_accuracy = 0.0
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': list(range(100, 301, 10)),  # Generate values from 100 to 300 with a step of 10
        'max_depth': list(range(10, 100, 10)),
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"]
    }
    for criter in ["gini", "entropy", "log_loss"]:
        for n_e in range(100,301,10):
            for depth in range(10,101,10):
                for m_f in [ 'sqrt', 'log2']:
                    rf_classifier = RandomForestClassifier(criterion=criter,n_estimators=n_e,max_depth=depth,max_features=m_f)
                    rf_classifier.fit(train_features, train_labels)

                    valid_acc= accuracy_score(valid_labels, rf_classifier.predict(valid_features))
                    
                    if valid_acc >best_valid_accuracy:
                        best_rf_classifier=rf_classifier
                        best_valid_accuracy=valid_acc
                        
                        train_acc= accuracy_score(train_labels, rf_classifier.predict(train_features))
                        test_acc= accuracy_score(test_labels, rf_classifier.predict(test_features))
                        with open("rf_classifier_results.txt", 'a') as file:
                            file.write(f'Criterion : {criter} , n_estimator : {n_e} , max_depth : {depth} , max_features : {m_f} valid_acc : {valid_acc}, train_acc : {train_acc}, test_acc : {test_acc} \n')

def lassoClassifier(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    # 1. Split the data into training, validation, and test sets
    X_train, y_train = train_features, train_labels
    X_valid, y_valid = valid_features, valid_labels
    X_test, y_test = test_features, test_labels

    # 2. Define a grid of hyperparameters to search over
    param_grid = {
        'penalty': ['l1'],
        'C': [ 1, 10, 100,1000],  # Regularization strength (inverse of alpha)
        'tol': [1e-4, 1e-5],  # Tolerance for convergence
        'max_iter': [100, 1000, 10000,100000]  # Maximum number of iterations
    }

    max_valid_acc=0.7604166666666666

    for c in param_grid['C']:
        for t in param_grid['tol']:
            for m_i in param_grid['max_iter']:
                lasso_classifier = LogisticRegression(solver='liblinear',penalty='l1',C=c,tol=t,max_iter=m_i,verbose=2)
                lasso_classifier.fit(X_train, y_train)
                valid_acc= accuracy_score(y_valid, lasso_classifier.predict(X_valid))
                
                if valid_acc >max_valid_acc:
                    best_lasso_classifier=lasso_classifier
                    max_valid_acc=valid_acc
                    valid_acc= accuracy_score(y_valid, lasso_classifier.predict(X_valid))
                    train_acc= accuracy_score(y_train, lasso_classifier.predict(X_train))
                    test_acc= accuracy_score(y_test, lasso_classifier.predict(X_test))
                    with open("Lasso_classifier_results.txt", 'a') as file:
                        file.write(f'C : {c} , tol : {t} , max_iter : {m_i} , valid_acc : {valid_acc}, train_acc : {train_acc}, test_acc : {test_acc} \n')

def lib_svm(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    
    X_train_libsvm = libsvm.svm_sparse_to_libsvm(train_features)
    X_test_libsvm = libsvm.svm_sparse_to_libsvm(test_features)
    X_valid_libsvm = libsvm.svm_sparse_to_libsvm(valid_features)

    C = 1.0  # You can adjust the regularization parameter C as needed
    kernel = 'rbf'  # Use the RBF kernel suitable for many sparse data

    
    model = libsvm.svm_train(X_train_libsvm, train_labels, kernel=kernel, C=C)

    y_pred, _, _ = libsvm.svm_predict(test_labels, X_test_libsvm, model)
    test_accuracy = accuracy_score(test_labels, y_pred)

    y_pred, _, _ = libsvm.svm_predict(valid_labels, X_valid_libsvm, model)
    val_accuracy = accuracy_score(valid_labels, y_pred)

    print("Test Accuracy:", test_accuracy)

def tsne(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    print(train_features[0,:].reshape(1, -1).transpose(1,0).shape)
    tsne = TSNE(n_components=2,perplexity=10)
    embedded_data = tsne.fit_transform(train_features.transpose(1,0))
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('t-SNE Visualization (2D)')
    plt.savefig("tSne.png")

def xgboost(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtest = xgb.DMatrix(test_features, label=test_labels)
    dvalid = xgb.DMatrix(valid_features, label=valid_labels)

    param_grid = {
        'booster': ["gbtree","gblinear","dart"],  # Generate values from 100 to 300 with a step of 10
        'eta': [0.1, 0.2, 0.3,0.4,0.5],
        'gamma': list(range(0, 10, 1)),
        'max_depth': list(range(2, 20, 2)),
        'subsample':[0.5,1],
        'sampling_method': ["uniform", "subsample", "gradient_based"],
        'tree_method': ["auto", "hist", "gpu_hist"],
        'lambda':[0.01, 0.1, 1.0, 10.0] 
    }
    num_round = [10, 50, 100, 200] 

    evallist = [(dtrain, 'train'), (dvalid, 'eval')]

    best_valid_acc=0

    for booster in param_grid['booster']:
        for eta in param_grid['eta']:
            for gamma in param_grid['gamma']:
                for max_depth in param_grid['max_depth']:
                    for subsample in param_grid['subsample']:
                        for sampling_method in param_grid['sampling_method']:
                            for tree_method in param_grid['tree_method']:
                                for lamb_da in param_grid['lambda']:
                                    param = {
                                        'objective': 'multi:softmax',
                                        'num_class':len(np.unique(train_labels)),
                                        'booster': booster,  # Generate values from 100 to 300 with a step of 10
                                        'eta': eta,
                                        'gamma': gamma,
                                        'max_depth': max_depth,
                                        'subsample':subsample,
                                        'sampling_method': sampling_method,
                                        'tree_method': tree_method,
                                        'lambda':lamb_da
                                    }
                                    for n_r in num_round:
                                        bst = xgb.train(param, dtrain, n_r, evallist,early_stopping_rounds=5)
                                        
                                        #print("Pred: ",bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1)))
                                        #print(bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1)).shape)

                                        train_acc=accuracy_score(train_labels, bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1)))
                                        test_acc=accuracy_score(test_labels, bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1)))
                                        valid_acc=accuracy_score(valid_labels, bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1)))

                                        if valid_acc>best_valid_acc:
                                            best_valid_acc=valid_acc
                                            bst.save_model('best_xg.model')

                                            with open("xgboost_classifier_results.txt", 'a') as file:
                                                file.write(f'{param}')
                                                file.write(f'valid_acc : {valid_acc}, train_acc : {train_acc}, test_acc : {test_acc} \n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('PromptBoost training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    train_features,train_labels, valid_features, valid_labels,test_features,test_labels= get_promptBoostingFeatures(args)

    if args.algorithm=="xgboost":
        xgboost(train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
    elif args.algorithm=="rf":
        randomForestCheck(train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
    elif args.algorithm=="lasso":
        lassoClassifier(train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
    #lib_svm(train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
    
    
    exit()

    # SVM
    svc = SVC(gamma='auto')
    svc.fit(train_features, train_labels)

    train_acc = svc.score(train_features, train_labels)
    valid_acc = svc.score(valid_features, valid_labels)
    test_acc = svc.score(test_features, test_labels)

    print(f"SVM | Train: {round(train_acc, 4)*100}% - Val: {round(valid_acc, 4)*100}% - Test: {round(test_acc, 4)*100}%")
        
    # RBF
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(train_features, train_labels)
    
    train_acc = gpc.score(train_features, train_labels)
    valid_acc = gpc.score(valid_features, valid_labels)
    test_acc = gpc.score(test_features, test_labels)
    
    print(f"RBF | Train: {round(train_acc, 4)*100}% - Val: {round(valid_acc, 4)*100}% - Test: {round(test_acc, 4)*100}%")
    
    # XGBoost
    xg = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(train_features, train_labels)
    
    train_acc = xg.score(train_features, train_labels)
    valid_acc = xg.score(valid_features, valid_labels)
    test_acc = xg.score(test_features, test_labels)
    
    print(f"XG | Train: {round(train_acc, 4)*100}% - Val: {round(valid_acc, 4)*100}% - Test: {round(test_acc, 4)*100}%")

    
     