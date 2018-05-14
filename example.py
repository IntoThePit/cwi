from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.advanced import Advanced
from utils.scorer import report_score

import numpy as np

def execute_demo(language):
    data = Dataset(language)
    
#    test_data = data.testset
    test_data = data.devset

    print("{}: {} training - {} test".format(language, len(data.trainset), len(test_data)))

#    for sent in data.trainset:
#        print(sent['sentence'], sent['target_word'], sent['gold_label'])

#    baseline = Baseline(language)
    advanced = Advanced(language)

#    baseline.train(data.trainset)
    trained = advanced.train(data.trainset)
    importances = trained.feature_importances_
    ordered_feature_list = advanced.ordered_feature_list
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print("{}. feature {:>30} ({})".format(f+1, ordered_feature_list[indices[f]], importances[indices[f]]))

#    predictions = baseline.test(data.devset)
    advanced_predictions = advanced.test(test_data)

    gold_labels = [sent['gold_label'] for sent in test_data]
    
    # Error analysis:
    sentences = [sent['sentence'] for sent in test_data]
    targets = [sent['target_word'] for sent in test_data]
    
    debug = False
    if debug == True:
        max_prints = 100
        for sent_i in range(max_prints):
            print("Sent: {}\n Word: {}\n Predict: {}\n Gold: {}\n".format(sentences[sent_i], targets[sent_i],advanced_predictions[sent_i], gold_labels[sent_i]))

    report_score(gold_labels, advanced_predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


