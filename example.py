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


    baseline = Baseline(language)
    advanced = Advanced(language)
    
    models_to_run = [baseline, advanced]
    model_mistakes = {}
    
    gold_labels = [sent['gold_label'] for sent in test_data]
            # Error analysis:
    sentences = [sent['sentence'] for sent in test_data]
    targets = [sent['target_word'] for sent in test_data]

    model_predictions = {}
    
    debug = False
    
    for model in models_to_run:

        model.train(data.trainset)
        trained = model.train(data.trainset)

        # Since only English uses RFC
        importances = False
        if importances == True:
            if language == 'english' and model == advanced:
                importances = trained.feature_importances_
                ordered_feature_list = model.ordered_feature_list
                indices = np.argsort(importances)[::-1]
                for f in range(20):
                    print("{}. & {} & ({:0.3}) \\\\ \hline".format(f+1, ordered_feature_list[indices[f]], importances[indices[f]]))
    #
        predictions = model.test(test_data)
        model_predictions[model.name] = predictions
        
        print(model.name)
        report_score(gold_labels, predictions)
          
        if debug == True:
            look_at = 500
            for sent_i in range(look_at):
                if predictions[sent_i] != gold_labels[sent_i]:
                    if sent_i in model_mistakes:
                        model_mistakes[sent_i].append(model.name)
                    else:
                        model_mistakes[sent_i] = [model.name]
                else:
                    if sent_i not in model_mistakes:
                        model_mistakes[sent_i] = []
                        
                    
    if debug == True:                    

        both_right = []
        advanced_right = []
        baseline_right = []
        both_wrong = []
        
        for key, value in model_mistakes.items():
            if len(value) == 2:
                both_wrong.append(key)
            elif len(value) == 0:
                both_right.append(key)
            elif value[0] == 'Baseline':
                advanced_right.append(key)
            else:
                baseline_right.append(key)
         
            # Finds an example of an incorrect word.
        max_wrong = 10
        for perm in [both_right, both_wrong, advanced_right, baseline_right]:
            curr_wrong = 0
            for item in perm:
                if curr_wrong == max_wrong:
                    break
                curr_wrong += 1
                
                sent = sentences[item]
                target = targets[item]
                gold = gold_labels[item]
                if perm == advanced_right:
                    predict = model_predictions['Advanced'][item]
                else:
                    predict = model_predictions['Baseline'][item]
                
                if perm == advanced_right:
                    perm_name = 'Advanced Correct, Baseline Incorrect'
                elif perm == baseline_right:
                    perm_name = 'Advanced Incorrect, Baseline Correct'
                elif perm == both_right:
                    perm_name = 'Both Correct'
                else:
                    perm_name = 'Both Incorrect'
                    
#                print("{}:\n Sent: {}\n Target: {}\n".format(perm_name, sent, target))
                print("{}:\n Sent: {}\n Target: {}\n Predicted: {}\n Gold: {}\n".format(perm_name, sent, target, predict, gold))

        
                
    


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


