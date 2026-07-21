import numpy as np


def two_bay_cls_access(reference,result):
    # for Hermiston dataset
    # reference:change_value=1;unchange_value=0
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m = reference.shape
    # if reference.shape != result.shape:
    #     print('the size of reference shoulf be equal to that of result')
    #     return oa_kappa
    # reference = np.reshape(reference, -1)
    # result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]

    # labe_change = label_1.shape[0]
    # label_unchange = label_0.shape[0]

    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_0).intersection(set(predict_0))  # True Positive
    tn = set(label_1).intersection(set(predict_1))  # False Positive
    fn = set(label_0).intersection(set(predict_1))  # False Positive
    fp = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 6)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    # ture_change = (len(tp))/labe_change
    # ture_unchange = (len(tn))/label_unchange

    oa = (len(tp)+len(tn))/(len(tp)+len(tn)+len(fp)+len(fn))      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/(len(tp)+len(tn)+len(fp)+len(fn))/(len(tp)+len(tn)+len(fp)+len(fn))
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    # oa_kappa.append('ture_change')
    # oa_kappa.append(ture_change)
    # oa_kappa.append('ture_unchange')
    # oa_kappa.append(ture_unchange)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('ture_change: ' + str(ture_change) + '     ' + 'ture_unchange: ' + str(ture_unchange))
    return F1


def two_cls_access(reference,result):
    # for Hermiston dataset
    # reference:change_value=1;unchange_value=0
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m = reference.shape
    # if reference.shape != result.shape:
    #     print('the size of reference shoulf be equal to that of result')
    #     return oa_kappa
    # reference = np.reshape(reference, -1)
    # result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]

    # labe_change = label_1.shape[0]
    # label_unchange = label_0.shape[0]

    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 6)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    # ture_change = (len(tp))/labe_change
    # ture_unchange = (len(tn))/label_unchange

    oa = (len(tp)+len(tn))/(len(tp)+len(tn)+len(fp)+len(fn))      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/(len(tp)+len(tn)+len(fp)+len(fn))/(len(tp)+len(tn)+len(fp)+len(fn))
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    # oa_kappa.append('ture_change')
    # oa_kappa.append(ture_change)
    # oa_kappa.append('ture_unchange')
    # oa_kappa.append(ture_unchange)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('ture_change: ' + str(ture_change) + '     ' + 'ture_unchange: ' + str(ture_unchange))
    return F1

