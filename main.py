import pickle
from utils import *
from model import *
from data_loader import *

def main(type_impute, data_name, miss_rate, non_missing_columns, pca_pjt):
    # Load data and introduce missingness
    ori_data_train, ori_data_test, miss_data_x, data_m = data_loader(data_name, miss_rate)
    if type_impute in ['pcaSoft', 'softImputer', 'pcaMice', 'miceImputer', 'pcaGain', 'gainImputer', 'pca_missf', 'missfImputer', 'pca_knn', 'knnImputer']:
        if "pca" in type_impute:
            imputed_data_final, pca_model, time_error = globals()[type_impute](ori_data_train[0], non_missing_columns, miss_rate, pca_pjt)
            print(imputed_data_final)
            x_test, y_test = ori_data_test
            x_train, y_train = ori_data_train
            if x_test is not None:
                x_test = pca_model.transform(x_test)
            x_train = pca_model.transform(x_train)
            ori_data_test = (x_test, y_test)
            ori_data_train = (x_train, y_train)
        else:
            imputed_data_final, time_error = globals()[type_impute](ori_data_train[0], non_missing_columns, miss_rate)
    else:
        imputed_data_final = None
    data = (imputed_data_final, ori_data_train, ori_data_test)
    return data, time_error

# Inputs for the main function
if __name__=='__main__':
    type_impute = ['pcaGain']
    miss_rates = [0.2]
    dataset_name = ["fashion_mnist"]
    results = []

    for impute_method in type_impute:
        for dataset in dataset_name:
            for missing_rate in miss_rates:
                pca_pjt = "pca2"
                if dataset == "parkinson":
                    non_missing_columns = 700
                if dataset == "gene":
                    non_missing_columns = 20000
                else:
                    non_missing_columns = 700
                args = {
                    "type_impute": impute_method,
                    "miss_rate": missing_rate,
                    "data_name": dataset,
                    "pca_pjt": pca_pjt, 
                    "non_missing_columns": non_missing_columns
                }
                data, time_error = main(**args)
                time_take = time_error[1]
                imputed_data, train_data, test_data = data
                x_train, y_train = train_data
                x_test, y_test = test_data
                imputed_model = LogisticRegression(random_state=None).fit(imputed_data, y_train)
                if x_test is None:
                    imputed_acc = np.mean(cross_val_score(imputed_model, x_train, y_train, cv=5))
                else:
                    imputed_acc = imputed_model.score(x_test, y_test)
                coef_impute = imputed_model.coef_
                normal_model = LogisticRegression(random_state=None).fit(x_train, y_train)
                coef_normal = normal_model.coef_ 
                record = calculate_report(coef_normal, coef_impute)
                save_dict = {"methods": args['type_impute'], "Imputed_accuracy": 100*imputed_acc, "records": record,\
                "data": args['data_name'], "missing_rate": args['miss_rate'], "time": time_take, "coefficients": coef_impute}
                results.append(save_dict)
                print(f"Dataset: {dataset} || impute method: {impute_method} || missing rate {missing_rate} || Accuracy: {imputed_acc:.3f}")
                pickle.dump(results, open("results.pkl", "wb"))