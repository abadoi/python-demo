import subprocess
import locale
import os
from os import listdir
from os.path import isfile, join
import json
import pprint
import pandas as pd
import numpy as np
import utils as ut 
from utils import CWE, NumpyEncoder
from enum import Enum
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import joblib
from pipelinehelper import PipelineHelper
import seaborn as sns


from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics, tree, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, accuracy_score, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB 
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression


CLANG = os.environ["CC"]
PATTERN_MATCHING_TOOL = os.environ["PATTERN_MATCHING_TOOL"]
DATA_DIR = os.environ["DATA_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]


def get_labeled_data(parent, filters, binary):
    batch_data = []
    ft_analyzer_file = ut.find_file(parent, "ft-analyzers.json")
    binary_file = ut.find_file(parent, "_llvm.ll")
    binary_retdec_file = ut.find_file(parent, "_retdec.ll")

    if ft_analyzer_file and binary_file and binary_retdec_file:
        instructions = ut.find_file(parent, 'instructions.json')
        basic_blocks = ut.find_file(parent, 'basic_blocks.json')
        if not instructions or not basic_blocks:
            # call patern matching script
            result = ut.pattern_matching(PATTERN_MATCHING_TOOL, ft_analyzer_file, binary_file, binary_retdec_file, parent)
            instructions = ut.find_file(parent, 'instructions.json')
            basic_blocks = ut.find_file(parent, 'basic_blocks.json')
        # load data
        if instructions and basic_blocks:
            labeled_data = ut.filters(ut.load_data(instructions, basic_blocks, unlabeled=False), filters[0], filters[1], filters[2])
            batch_data = ut.create_df_rows(labeled_data, binary = binary)
            
    return batch_data

def get_unlabeled_data(parent, binary = False):
    batch_data = []
    binary_retdec_file = ut.find_file(parent, "_retdec.ll")

    if binary_retdec_file:
        instructions = ut.find_file(parent, '_instructions.json')
        basic_blocks = ut.find_file(parent, '_basic_blocks.json')
        if not instructions and not basic_blocks:
            # call parse ir script
            ut.parse_ir_file(PATTERN_MATCHING_TOOL, binary_retdec_file, parent)
            instructions = ut.find_file(parent, '_instructions.json')
            basic_blocks = ut.find_file(parent, '_basic_blocks.json')
        if instructions and basic_blocks:
            # load data
            batch_data = ut.create_df_rows(ut.load_data(instructions, basic_blocks, unlabeled=True))
    return batch_data

def split_data(df, select_classes = None):
    if select_classes:
        # binary 
        new_df = df.copy() 
        mask = df['label'].isin(select_classes)
        if len(select_classes) == 1:
            print("Binary classification with class {} and 0 as the rest.".format(select_classes[0]))
            new_df.loc[mask, 'label'] = select_classes[0]
        else:
            print("Binary classification with 1 as classes {} and 0 as the rest.".format(select_classes))
            new_df.loc[mask, 'label'] = 1
        new_df.loc[~mask, 'label'] = 0
        df = new_df

    y = df['label']
    X = df.drop('label', axis=1)
    print("n_samples: %d, n_features: %d" % X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    print("Train: ", Counter(y_train))
    print("Test: ", Counter(y_test))

    return X_train, X_test, y_train, y_test

def save_scores(model_name, scores):
    print("{} scores: \n".format(model_name))
    print("Accuracy: ", scores['test_accuracy'])
    print("Recall: ", scores['test_recall_macro'])
    print("Precision: ", scores['test_precision_macro'])
    print("F1 score: ", scores['test_f1_micro'])

    results = {}
    results.update({
        "model" : model_name,
        "accuracy: " : np.mean(scores['test_accuracy']),
        "recall: ": np.mean(scores['test_recall_macro']),
        "precision: ":np.mean(scores['test_precision_macro']),
        "f1-score: ": np.mean(scores['test_f1_macro']),
        "f1-score-micro": np.mean(scores['test_f1_micro'])
    })
    
    #create results dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, model_name + '_scores.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def svc_clasiffier(df):

    X_train, X_test, y_train, y_test = split_data(df)

    clf = svm.SVC(kernel='linear', C=1, probability = True).fit(X_train, y_train)
    # print("Score: ", clf.score(X_test, y_test))

    result = clf.predict(X_test)
    print("Same output(TP): ", np.count_nonzero(result == y_test))

    results = clf.predict_proba(X_test)
    print("Predict probabilities:\n", results)

    # gets a dictionary of {'class_name': probability}
    prob_per_class_dictionary = dict(zip(clf.classes_, results))
    print(prob_per_class_dictionary)

    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = cross_validate(clf, X_train, y_train, scoring = scoring, cv=kf, n_jobs = -1)
    # print("Accuracy SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    save_scores("svc", scores)

def random_forest_clasiffier(df):
    X_train, X_test, y_train, y_test = split_data(df)

    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    # Stratified K-Fold ensures the same portion of each class in each split.
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    model = RandomForestClassifier(n_jobs=-1, random_state=0).fit(X_train, y_train)

    scores = cross_validate(model, X_train, y_train, scoring = scoring, cv=kf, n_jobs = -1)
    save_scores("random_forest", scores)


def get_optimal_number_of_components(X):
    cov = np.dot(X,X.transpose())/float(X.shape[0])
    U,s,v = np.linalg.svd(cov)

    S_nn = sum(s)

    for num_components in range(0,s.shape[0]):
        temp_s = s[0:num_components]
        S_ii = sum(temp_s)
        if (1 - S_ii/float(S_nn)) <= 0.05:
            return num_components

    return s.shape[0]

def pca_pipeline3(df):
    y = df['label']
    features = df.columns[:-1]
    X = df[features]

    # Z-score the features
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # nnn = get_optimal_number_of_components(X)
    # print("NNNN:", nnn)

    # plot explained variance ratio and choose a number of components that "capture" at least 95% of the variance.
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    # plt.show()

    # The PCA model
    # model = PCA(n_components = "mle", svd_solver ="full") # Minka’s MLE is used to guess the dimension
    model = PCA(n_components = 13) # estimate only 13 PCs as the plot shows

    X_new = model.fit_transform(X) # project the original data into the PCA space

    # number of components
    n_pcs= model.components_.shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [features[most_important[i]] for i in range(n_pcs)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    dff = pd.DataFrame(dic.items())
    print(dff)

    return most_important_names

def pca_pipeline2(df):
    X = df
    y = df['label']

    sc = StandardScaler()
    X = sc.fit_transform(X)

    #Applying the PCA
    fig = plt.figure(figsize=(12,6))
    pca = PCA()
    pca_all = pca.fit_transform(X)
    pca1 = pca_all[:, 0]
    pca2 = pca_all[:, 1]

    fig.add_subplot(1,2,1)
    plt.bar(np.arange(pca.n_components_), 100*pca.explained_variance_ratio_)
    plt.title('Relative information content of PCA components')
    plt.xlabel("PCA component number")
    plt.ylabel("PCA component variance % ")

    fig.add_subplot(1,2,2)
    plt.scatter(pca1, pca2, c=y, marker='x', cmap='jet')
    plt.title('Class distributions')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    classifier = LogisticRegression(max_iter=10000, tol=0.1, random_state = 0)

    # PCA 2D space
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data=pca_all).iloc[:,0:2], y, test_size = 0.25, random_state = 0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy_pca_2d = accuracy_score(y_test, y_pred)

    # PCA 3D space
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data=pca_all).iloc[:,0:3], y, test_size = 0.25, random_state = 0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy_pca_3d = accuracy_score(y_test, y_pred)

    # PCA 2D space
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy_original = accuracy_score(y_test, y_pred)

    plt.figure()
    sns.barplot(x=['pca 2D space', 'pca 3D space', 'original space'], y=[accuracy_pca_2d, accuracy_pca_3d, accuracy_original])
    plt.ylabel('accuracy')
    plt.show()

def pca_pipeline(df):
    sc = StandardScaler()
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic = LogisticRegression(max_iter=10000, tol=0.1)

    pipe = Pipeline(steps=[('scaler', sc), ('pca', pca), ('logistic', logistic)])

    X_train, X_test, y_train, y_test = split_data(df)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [2, 4, 6, 8, 10],
        'logistic__C': np.logspace(-4, 4, 4),
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    predictions = search.best_estimator_.predict(X_test)

    # average=None gives per label score
    print("F1 score per class: ", f1_score(y_test, predictions, average = None))
    print(classification_report(predictions, y_test))

    # Plot the PCA spectrum
    pca.fit(X_train)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
            pca.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(-1, 70)

    plt.tight_layout()
    plt.show()

def one_class_svm(df):
    X_train, X_test, y_train, y_test = split_data(df, [3])
    model = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)

    parameters = {
        # "estimator__C": [1,2,4],
        # "estimator__degree":[1, 2, 3, 4],
    }

    # model_tunning = GridSearchCV(model, param_grid=parameters,
    #                             scoring='f1_micro', n_jobs=-1, cv=10)
    # model_tunning.fit(X_train[y_train==0], y_train[y_train==0])

    # fit on majority class
    model.fit(X_train[y_train==0])
    classes = np.unique(y_train)

    # make prediction
    for i in classes:
        if X_test[y_test==i].empty:
            continue
        predictions = [int(a) for a in model.predict(X_test[y_test==i])]
        num_corr = sum(int(a==1) for a in predictions)
        print ("   %d   " % i)
        if i==0:
            print ("%d of %d" % (num_corr, len(predictions)))
        else:
            print ("%d of %d" % (len(predictions)-num_corr, len(predictions)))
    pass

def get_acc_single(clf, X_test, y_test, class_):
    # Return the mean accuracy on the given test data and labels. Mean accuracy of self.predict(X) wrt. y.
    pos = np.where(y_test == class_)[0]
    neg = np.where(y_test != class_)[0]
    y_trans = np.empty(X_test.shape[0], dtype=bool)
    y_trans[pos] = True
    y_trans[neg] = False
    return clf.score(X_test, y_trans)

def one_vs_rest_classifier(df):
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = split_data(df, [3])

    for base_clf in (MultinomialNB(), GaussianNB(), LinearSVC(dual = False, max_iter=2000, random_state=0),
                    RandomForestClassifier(criterion='entropy', max_features=None, n_estimators=300, n_jobs=-1, random_state=0), MLPClassifier(random_state=0, max_iter=300)):

        print ("---- %s ----" % base_clf.__class__.__name__)
        # Learn to predict each class against the other
        # clf = OneVsRestClassifier(LinearSVC(dual=False, max_iter=2000))
        clf = OneVsRestClassifier(base_clf)

        parameters = {
        # "estimator__C": [1,2,4],
        # "estimator__degree":[1, 2, 3, 4],
        }

        model_tunning = GridSearchCV(clf, param_grid=parameters,
                                    scoring='f1_micro', n_jobs=-1, cv=10)
        model_tunning.fit(X_train, y_train)

        # for some X_test testing set
        print(model_tunning.best_estimator_)
        joblib.dump(model_tunning.best_estimator_, os.path.join(RESULTS_DIR + '/one_vs_rest/3',  'one_vs_rest_' +  base_clf.__class__.__name__ + '.pkl'), compress=1)
        
        # Get all accuracies
        classes = np.unique(y_train)

        for class_index, est in enumerate(model_tunning.best_estimator_.estimators_):
            class_ = classes[class_index]
            print('class ' + str(class_))
            print(get_acc_single(est, X_test, y_test, class_))

        predictions = model_tunning.predict(X_test)

        # average=None gives per label score
        print("F1 score per class: ", f1_score(y_test, predictions, average = None))

        print(classification_report(predictions, y_test))

def run_grid(df):
    X_train, X_test, y_train, y_test = split_data(df)

    pipe = Pipeline([
        ('clf', PipelineHelper([
            # ('svm', LinearSVC(dual=False)),
            ('rf', RandomForestClassifier()),
            # ('mnb', BernoulliNB()), 
            # ('mlp', MLPClassifier(random_state=1, verbose=1)),
        ])),
    ])

    params = {
        'clf__selected_model': pipe.named_steps['clf'].generate({
            # 'svm__C': [0.1, 1.0],
            # 'svm__max_iter' : [2000, 3000, 5000],
            # 'svm__multi_class': ['ovr', 'crammer_singer'],
            'rf__n_estimators': [10, 100, 300, 500],
            'rf__criterion': ['gini', 'entropy'],
            # 'rf__class_weight': [None, 'balanced', 'balanced_subsample'],
            # 'rf__max_depth': [None, 5, 10, 20, 40, 80],
            # 'rf__min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
            # 'rf__min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
            # 'rf__max_leaf_nodes': [None, 10, 20],
            'rf__max_features': [None, 'sqrt', 'log2'], 
            # 'mlp__solver': ['sgd', 'adam'],
            # 'mlp__max_iter': [200, 300, 1000],
            # 'mlp__alpha': [0.001, 0.0001, 0.00001],
            # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            # 'mnb__alpha': [0.1, 1, 10],
        }),
    }

    grid = GridSearchCV(pipe, param_grid = params, n_jobs=-1, cv=10, scoring='f1_micro')
    grid.fit(X_train, y_train)

    print("Best parameter (CV score=%0.3f):" % grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
    print("Best params: ", grid.best_params_)

    joblib.dump(grid.best_estimator_, os.path.join(RESULTS_DIR, 'saved_model.pkl'), compress=1)
    # classifier = joblib.load(filepath) # path to .pkl file

    print("Detailed classification report: \n")
    y_true, y_pred = y_test, grid.best_estimator_.predict(X_test)
    print("F1 score per class: ", f1_score(y_true, y_pred, average = None))

    evaluate(y_pred, y_true, plot = False)

def run_classifiers(df, binary = False, pca = False):
    if pca:
        pca_features = pca_pipeline3(df)
        print("Most important features selected by PCA: ", pca_features)
        pca_features.append('label')
        df = df[np.intersect1d(df.columns, pca_features)]

    X_train, X_test, y_train, y_test = split_data(df, [3])

    binary_cls = [                
                LinearSVC(dual=False, max_iter=2000),
                # SVC(kernel='linear', C=1, probability = True),
                RandomForestClassifier(criterion='entropy', max_features=None, n_estimators=300, n_jobs=-1, random_state=0), 
                BernoulliNB(),
                GaussianNB(),
                MLPClassifier(random_state=1, max_iter=2000)
    ]
    multi_cls = [
                LinearSVC(dual=False, max_iter=2000),
                RandomForestClassifier(criterion='entropy', max_features=None, n_estimators=300, n_jobs=-1, random_state=0), 
                MultinomialNB(), 
                GaussianNB(),
                BernoulliNB(),
                MLPClassifier(random_state=1, max_iter=2000)
    ]

    classifiers = multi_cls

    if binary:
        classifiers = binary_cls

    for c in classifiers:
        print ("---- %s ----" % c.__class__.__name__)
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'f1_micro']
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # Array of scores of the estimator for each run of the cross validation.
        # clf = c.fit(X_train, y_train)
        # scores = cross_validate(clf, X_train, y_train, scoring = scoring, cv=kf, n_jobs = -1)
        # save_scores(c.__class__.__name__, scores)

        model = GridSearchCV(c, param_grid={}, scoring='f1_micro', n_jobs=-1, cv=kf)
        model.fit(X_train, y_train)

        joblib.dump(model.best_estimator_, os.path.join(RESULTS_DIR + '/3', c.__class__.__name__ + '.pkl'), compress=1)

        predictions = model.best_estimator_.predict(X_test)

        # average=None gives per label score
        print("F1 score per class: ", f1_score(y_test, predictions, average='micro'))

        print(classification_report(predictions, y_test))

def plot_data(df):
    features = df.columns[:20]
    plot_colums = []
    values = []

    # plot bool features
    for c in features:
        print ("---- %s ---" % c)
        counts = df[c].value_counts()
        print(counts)

        is_bool = df[c].isin([0,1]).all()
        if is_bool:
            plot_colums.append(c)
            value = len(df[df[c] == 1])
            values.append(value)
        else:
            counts.plot(kind='bar', title = c)

        plt.show()
    
    # plot the rest of data
    Data = {'Features': plot_colums,
        'count': values
       }
    
    df = pd.DataFrame(Data,columns=['Features','count'])
    df.plot(x ='Features', y='count', kind = 'bar', title = "Bool features")
    plt.show()

def evaluate(result, actual_result, plot):
    classes = np.arange(11).tolist()
    cm = confusion_matrix(result, actual_result, labels=classes)

    cr = classification_report(result, actual_result, labels = classes)
    print(cr)

    if plot:
        plot_confusion_matrix(cm, classes)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():

    # labeled data
    saved_data = os.path.join(RESULTS_DIR, 'saved_filter.pkl')
    dataset = pd.DataFrame()
    try:
        dataset = pd.read_pickle(saved_data)
    except FileNotFoundError as e: 
        print (e)

    '''
    initialize filters for basic block and instructions
    filters[0] = faults_per_bb
    filters[1] = scanners_per_instr
    filters[2] = different scanners_per_bb
    '''
    filters = [None, None, None]
    binary_classification = False
    if dataset.empty:
        raw_data = []
        for dirpath, dirnames, files in os.walk(DATA_DIR):
            if files and not dirnames:
                print(dirpath)
                batch = get_labeled_data(dirpath, filters, binary_classification)
                raw_data.extend(batch)  

        dataset = pd.DataFrame(raw_data).astype(int)

        # save data temporarily
        saved_data = os.path.join(RESULTS_DIR, 'saved_filter.pkl')
        dataset.to_pickle(saved_data)
    
    print(dataset)

    # unlabeled data
    # unlabeled_data = get_unlabeled_data('data/train_data/pifs')
    # unlabeled_dataset = pd.DataFrame(unlabeled_data).astype(int)
    # print(unlabeled_dataset)    

    # Decision Tree Clasiffier
    # svc_clasiffier(dataset)

    # Random Forest Clasiffier
    # random_forest_clasiffier(dataset)

    # run_classifiers(dataset, binary_classification, pca = False)
    # run_grid(dataset)
    # one_vs_rest_classifier(dataset)
    # one_class_svm(dataset)
    # pca_pipeline3(dataset)

    # Plot
    # plot_data(dataset)

    # Evaluation example with no-cross-validation
    # result, actual_result = random_forest_clasiffier_example(dataset, unlabeled_dataset)
    # evaluate(result, actual_result)

    # calculate_mean_scores()
    
if __name__ == "__main__":
    main()