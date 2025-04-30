import pandas as pd
import numpy as np


classes = ['Class 1', 'Class 2', 'Class 3']
confusion_matrix = [[100,20,5],[15,80,10],[10,5,5]]


df_confusion = pd.DataFrame(
    confusion_matrix,
    index=['Actual ' + _ for _ in classes],
    columns=['Predicted ' + _ for _ in classes]
)

def conf_positions(df,i):
    for i in range(1,3):
        tp=df[i,i]
        fp=np.sum(df[:,i])-tp
        fn=np.sum(df[i,:])-tp
        tn=np.sum(df)-tp-fp-fn
    return tp,fp,fn,tn
    
def metric_scores(df,type='micro'):
    metrics=[]
    arr=df.values
    for i in range(len(arr)):
        tp_i,fp_i,fn_i,tn_i=conf_positions(arr,i)
        metrics.append((tp_i,fp_i,fn_i,tn_i))
    sum_tp=sum(m[0] for m in metrics)
    sum_fp=sum(m[1] for m in metrics)
    sum_fn=sum(m[2] for m in metrics)
    sum_tn=sum(m[3] for m in metrics)
    
    if type=='micro':
        precision=sum_tp/sum_tp+sum_fp
        recall=sum_tp/sum_fn+sum_tp
        accuracy=sum_tp+sum_tn/sum_tn+sum_fn+sum_fp+sum_tp
        f1=2*precision*recall/precision+recall
        return {
            'Micro Precision: ':precision,
            'Micro Recall':recall,
            'Micro Accuracy':accuracy,
            'Micro F1: ':f1
        }
    elif type=='macro':
        precision_list = []
        recall_list = []
        accuracy_list = []
        f1_list = []
        
        for tp, fp, fn, tn in metrics:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            a = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precision_list.append(p)
            recall_list.append(r)
            accuracy_list.append(a)
            f1_list.append(f)
        
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        accuracy = sum(accuracy_list) / len(accuracy_list)
        f1 = sum(f1_list) / len(f1_list)
        
        return{
            'Macro Precision: ':precision,
            'Macro Recall':recall,
            'Macro Accuracy':accuracy,
            'Macro F1: ':f1
        }
    else:
        return 0

# Display the generated confusion matrix
print(df_confusion)
print(metric_scores(df_confusion,type='micro'))

