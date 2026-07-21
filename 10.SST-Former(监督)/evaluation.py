import numpy as np


# 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
def two_cls_access(reference, result):
    # for Hermiston dataset
    # reference:change_value=1; unchange_value=0
    # result: predicted map:change_value=1; unchange_value=0
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]

    oa_kappa = []
    m, n = reference.shape
    # 检查参考图像（reference）和结果图像（result）的尺寸是否相等
    if reference.shape != result.shape:
        print('the size of reference should be equal to that of result')  # 不等则返回空的oa_kappa列表
        return oa_kappa

    reference = np.reshape(reference, -1)   # 将参考图像和结果图像都重塑为一维数组
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 0)  # 条件索引，找出参考图像中变化和未变化的标签
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)   # 条件索引，找出预测结果中识别为变化和未变化的预测结果
    predict_1 = np.where(result == 1)

    label_0 = label_0[0]  # 从由np.where返回的元组中提取出实际的索引数组
    label_1 = label_1[0]
    labe_change = label_1.shape[0]  # 计算变化（标记为1）和未变化（标记为0）的总像素数
    label_unchange = label_0.shape[0]

    predict_0 = predict_0[0]
    predict_1 = predict_1[0]

    tp = set(label_1).intersection(set(predict_1))  # True Positive（label_1 和 predict_1 的交集）
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive（label_0 和 predict_1 的交集）
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    precision = round(precision, 4)  # round返回浮点数的四舍五入值
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    ture_change = (len(tp))/labe_change  # 正确预测为变化的像素占所有实际变化像素的比例
    ture_unchange = (len(tn))/label_unchange  # 正确预测为未变化的像素占所有实际未变化像素的比例

    oa = (len(tp)+len(tn))/(m*n)      # Overall precision
    # pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    pe = ((len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/(m*n)^2)
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
    oa_kappa.append('ture_change')
    oa_kappa.append(ture_change)
    oa_kappa.append('ture_unchange')
    oa_kappa.append(ture_unchange)

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    print('ture_change: ' + str(ture_change) + '     ' + 'ture_unchange: ' + str(ture_unchange))
    return oa_kappa


# 参考图：change=1; unchanged=2; uncertain=0，存在不确定的值
# 处理的数据集标签值不同，以及对未确定区域（uncertain areas）的处理方式不同
def two_cls_access_for_Bay_Barbara(reference, result):
    # for Bay & Barbra datasets
    # reference:change_value=1;unchange_value=2
    # result: predicted map:change_value=1;unchange_value=0
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W), change=1; unchanged=2; uncertain=0
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    # m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa

    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 2)  # Unchanged
    label_1 = np.where(reference == 1)  # Changed
    predict_0 = np.where(result == 0)  # Unchanged
    predict_1 = np.where(result == 1)  # Changed

    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]

    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    recall = len(tp) / (len(tp) + len(fn))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))

    total_num = len(label_0) + len(label_1)  # 使用确定的像素总数，而不是整个图像的像素总数
    oa = (len(tp) + len(tn)) / total_num
    pe = ((len(tp)+len(fn))*(len(tp)+len(fp)) + (len(fp)+len(tn))*(len(fn)+len(tn))) / (total_num*total_num)
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

    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('whole OA is' + str(oa))
    # print('whole kappa is' + str(kappa))
    return oa_kappa
