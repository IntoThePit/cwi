from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.advanced import Advanced
from utils.scorer import report_score



def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

#    for sent in data.trainset:
#        print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)
    advanced = Advanced(language)

#    baseline.train(data.trainset)
    advanced.train(data.trainset)

#    predictions = baseline.test(data.devset)
    advanced_predictions = advanced.test(data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]
    
    # Error analysis:
    sentences = [sent['sentence'] for sent in data.testset]
    targets = [sent['target_word'] for sent in data.testset]
    
    debug = False
    if debug == True:
        max_prints = 100
        for sent_i in range(max_prints):
            print("Sent: {}\n Word: {}\n Predict: {}\n Gold: {}\n".format(sentences[sent_i], targets[sent_i],advanced_predictions[sent_i], gold_labels[sent_i]))

    report_score(gold_labels, advanced_predictions)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


