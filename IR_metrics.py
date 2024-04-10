actual = ["2", "4", "5", "7"]
predicted = ["1", "2", "3", "4", "5", "6", "7", "8"]

# recall@k function
def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])

    recall = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return recall

for k in range(1, 9):
    print(f"Recall@{k} = {recall(actual, predicted, k)}")
    
def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])

    precision = round(len(act_set & pred_set) / float(k), 2)
    return precision

for k in range(1, 9):
    print(f"Precision@{k} = {precision(actual, predicted, k)}")
    
  
actual_relevant = [ 
[ 2, 4, 5, 7 ], 
[ 1, 4, 5, 7], 
[5, 8]
]

Q = len(actual_relevant)
cumulative_reciprocal = 0 
for i in range(Q):
    first_result  = actual_relevant[i][0]
    reciprocal  = 1 / first_result
    cumulative_reciprocal += reciprocal
    print(f"query #{i+1} = 1/{first_result} = {reciprocal}")

mrr = 1/Q * cumulative_reciprocal
print("MRR = ", round(mrr, 2))

    
    

# initialize variables
actual = [
    [2, 4, 5, 7],
    [1, 4, 5, 7],
    [5, 8]
]
Q = len(actual)
predicted = [1, 2, 3, 4, 5, 6, 7, 8]
k = 8
ap = []

# loop through and calculate AP for each query q
for q in range(Q):
    ap_num = 0
    # loop through k values
    for x in range(k):
        # calculate precision@k
        act_set = set(actual[q])                                                                                                                                   
        pred_set = set(predicted[:x+1])
        precision_at_k = len(act_set & pred_set) / (x+1)
        # calculate rel_k values
        if predicted[x] in actual[q]:
            rel_k = 1
        else:
            rel_k = 0
        # calculate numerator value for ap
        ap_num += precision_at_k * rel_k
    # now we calculate the AP value as the average of AP
    # numerator values
    ap_q = ap_num / len(actual[q])
    print(f"AP@{k}_{q+1} = {round(ap_q,2)}")
    ap.append(ap_q)

# now we take the mean of all ap values to get mAP
map_at_k = sum(ap) / Q

# generate results
print(f"mAP@{k} = {round(map_at_k, 2)}")

