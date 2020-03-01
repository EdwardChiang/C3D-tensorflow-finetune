filepath = "predict_ret.txt"

with open(filepath) as f:
    line = f.readline()
    tp, fp, tn, fn = 0, 0, 0, 0
    acc, total = 0, 0
    while line:
        # true label, class prob for true label, predicted label, class prob for predicted label
        true_label, _, predict_label, _= line.split(",")
        true_label, predict_label = int(true_label), int(predict_label)

        if true_label == predict_label:
            acc += 1
        total += 1

        # if true_label == 0:
        #     if predict_label == true_label:
        #         tp += 1
        #     else:
        #         fn += 1
        # else:
        #     if predict_label == true_label:
        #         tn += 1
        #     else:
        #         fp += 1
       
        line = f.readline()

print(tp, fp, tn, fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 / ( 1 / precision + 1 / recall )

print("Precision: ", precision)
print("Recall: ", recall)
print("F1_score: ", f1_score)
