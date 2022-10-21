import json, pickle, statistics, sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np, sys
import pandas as pd
from helper_functions import value_normalization, node_prop
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

article_sentence_key_index = None

sentence_level_test_begin = -1


def flat_accuracy_new(preds, labels, print_full_report, already_converted_to_array=False, rfd_test=False):
    accuracy = 0
    if already_converted_to_array:
        pred_flat = list(preds)
        labels_flat = list(labels)
        for pred, label in zip(pred_flat, labels_flat):
            if pred == label:
                accuracy += 1
        accuracy = accuracy / len(pred_flat)
    else:
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        accuracy = np.sum(pred_flat == labels_flat) / len(pred_flat)

    print("computing article level result ", len(pred_flat), len(labels_flat))
    print("pred flat   ", pred_flat[:10])
    print("labels flat ", labels_flat[:10])

    # print("len of true labels ", len(labels_flat))
    p = precision_score(labels_flat, pred_flat, average="macro")
    r = recall_score(labels_flat, pred_flat, average="macro")
    data = {'y_Actual': list(labels_flat),
            'y_Predicted': list(pred_flat)
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    score_rfd = 2 * p * r / (p + r)
    # precision, recall, f_score, support = score(labels_flat, pred_flat)
    precision, recall, fscore, support = score(labels_flat, pred_flat)
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print("confusion matrix")
    print(confusion_matrix)
    if print_full_report:

        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        if rfd_test:
            print('fscore for 2 classes only : {}'.format(np.sum(fscore) / 2))
        # print('support: {}'.format(support))
        # print('macro recall: {}'.format(np.mean(recall)))
        # print('macro fscore: {}'.format(np.mean(fscore)))
        # print('macro scores: ', np.mean(precision), np.mean(recall), np.mean(fscore))
    print("accuracy -> ", accuracy)
    return np.mean(precision), np.mean(recall), np.mean(fscore)


def get_sentence_level_test_index():
    global sentence_level_test_begin, article_sentence_key_index
    for i, (_, index) in enumerate(article_sentence_key_index):
        if index != -1:
            sentence_level_test_begin = i
            break
    return


print("sentence_level_test_begin ", sentence_level_test_begin)


def preprocess_result(content):
    print(content[:10])
    # sys.exit(0)
    result = content.replace('"', '')
    # print("aa", gold_result[:5])

    result = result.replace('[', '')
    # print("bb", gold_result[:5])

    result = result.replace(']', '')
    # print("cc", gold_result[:5])

    result = result.split(",")
    result = [int(x) for x in result]

    return result


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def compute_test_accuracy_and_prepare_pipeline_input(eval_file_path, rfd_test, negation_check=False, key_value=None):
    # print("key value -> ", key_value)
    print("starting the compute function")
    if negation_check:
        neg_sentence_ids = pickle.load(open('cd_sentence_ids_with_negation_strict.p', 'rb'))

        # neg_sentence_ids_temp = pickle.load(open('cd_sentence_ids_with_negation.p', 'rb'))
        # print("neg sentence ids temp -> ", neg_sentence_ids_temp[:5])
        # neg_sentence_ids = []
        # for elem in neg_sentence_ids_temp:
        #     neg_sentence_ids.append(elem.split("_")[0] + "_" + str(int(elem.split("_")[1]) + 1))
        print("neg sentence ids -> ", len(neg_sentence_ids), neg_sentence_ids[:5])
    global sentence_level_test_begin, article_sentence_key_index
    print("loading rfd_article_sentence_key_index ")
    article_sentence_key_index = pickle.load(open('rfd_article_sentence_key_index.p', "rb"))
    get_sentence_level_test_index()
    print("sentence_level_test_begin -> ", sentence_level_test_begin)
    # sys.exit(0)
    # path = 'eval_results_createdebate_test_bert_large_with_confidence_list.json'
    path = eval_file_path
    sentence_level_results = []
    sentence_level_confidences = []
    gold_result_final = []
    final_article_level_result = {}
    instance_ids = []
    avg_p = []
    avg_r = []
    avg_f = []

    # 5 keys
    rows, cols = (sentence_level_test_begin, 5)
    final_article_level_result_list_1 = [[0 for i in range(cols)] for j in range(rows)]
    final_article_level_result_list_2 = [0] * rows
    with open(path, "r") as content:
        test_result = json.load(content)
        for key in test_result:
            if len(key) != 1:
                continue
            # print("key in loop ", key)
            if type(key_value) == int:
                if int(key) != key_value:
                    print("continuing ")
                    continue
            # key = 0, 1, 2
            if key_value:
                print("match found")
            # print(int(key))
            gold_result = test_result[key]['test']['golds']

            # gold_result = preprocess_result(gold_result)
            predictions = test_result[key]['test']['predictions']
            # predictions = preprocess_result(predictions)
            confidences = test_result[key]['test']['confidences']
            # confidences = preprocess_result(confidences)

            # compute_accuracy(gold_result[:sentence_level_test_begin], predictions[:sentence_level_test_begin])
            if len(final_article_level_result) == 0:
                gold_result_final = gold_result[:sentence_level_test_begin]

            for i, (id_index, sentence_index) in enumerate(article_sentence_key_index):
                if sentence_index == -1:
                    if id_index not in final_article_level_result:
                        instance_ids.append(id_index)
                        final_article_level_result[id_index] = [predictions[:sentence_level_test_begin][i]]
                    else:
                        final_article_level_result[id_index].append(predictions[:sentence_level_test_begin][i])
                    final_article_level_result_list_1[i][int(key)] = predictions[:sentence_level_test_begin][i]
                    # if i <= 3:
                    #     print("check print -> ", i, int(key), final_article_level_result_list_1[i])
                else:
                    break
            # print("for loop done ")
            # print(final_article_level_result_list_1[:5])
            # print(".........")
                # for i, (id_index, sentence_index) in enumerate(article_sentence_key_index):
                #     if sentence_index == -1:
                #         instance_ids.append(id_index)
                #         final_article_level_result[id_index] = predictions[:sentence_level_test_begin][i]
                #     else:
                #         break

            sentence_level_results.append(predictions[sentence_level_test_begin:])
            # print("sentence_level_results first index -> ", len(sentence_level_results[0]))
            sentence_level_confidences.append(confidences[sentence_level_test_begin:])
    iii = 0
    for id_index in final_article_level_result:
        if iii < 0:
            print("final_article_level_result -> ", final_article_level_result[id_index], int(statistics.mode(final_article_level_result[id_index])))
            iii += 1
        try:
            final_article_level_result[id_index] = int(statistics.mode(final_article_level_result[id_index]))
        except Exception as e:
            final_article_level_result[id_index] = min([p[0] for p in statistics._counts(final_article_level_result[id_index])])

    for ind in range(len(final_article_level_result_list_1)):
        # print(ind, final_article_level_result_list_1[ind], )
        try:

            final_article_level_result_list_2[ind] = int(statistics.mode(final_article_level_result_list_1[ind]))
        except Exception as e:
            final_article_level_result_list_2[ind] = min([p[0] for p in statistics._counts(final_article_level_result_list_1[ind])])

    # print("final_article_level_result_list_1 ")
    # print(final_article_level_result_list_1[:5])
    # print(".... ")
    # print("final_article_level_result_list_1=2 ")
    # print(final_article_level_result_list_2[:5])
    # print("....")
    p, r, f = flat_accuracy_new(final_article_level_result_list_2, gold_result_final, True,
                                True, rfd_test)
    avg_p.append(p)
    avg_r.append(r)
    avg_f.append(f)
    # sys.exit(0)

    # print(len(sentence_level_results), len(sentence_level_results[0]), len(sentence_level_results[1]), len(sentence_level_results[2]), len(sentence_level_results[3]), len(sentence_level_results[4]))
    # print("average precision ", statistics.mean(avg_p))
    # print("average recall ", statistics.mean(avg_r))
    print("average fscore ", statistics.mean(avg_f))

    final_sentence_level_results = {}

    print("..... -> ", len(sentence_level_results[0]), len(sentence_level_results[1]), len(sentence_level_results[2]), len(sentence_level_results[3]),
          len(sentence_level_results[4]))

    # if rfd_test:
    #     for v_1, v_2, v_3, v_4, v_5, c_1, c_2, c_3, c_4, c_5, elem in zip(sentence_level_results[0], sentence_level_results[1], sentence_level_results[2],
    #                                        sentence_level_results[3], sentence_level_results[4], sentence_level_confidences[0],
    #                                                                       sentence_level_confidences[1], sentence_level_confidences[2],
    #                                                                       sentence_level_confidences[3], sentence_level_confidences[4],
    #                                                                       article_sentence_key_index[sentence_level_test_begin:]):
    #         all_class_count = [0, 0, 0, 0]
    #         all_confidence_count = [0, 0, 0, 0]
    #
    #         all_class_count[v_1] += 1
    #         all_confidence_count[v_1] += c_1
    #         all_class_count[v_2] += 1
    #         all_confidence_count[v_2] += c_2
    #         all_class_count[v_3] += 1
    #         all_confidence_count[v_3] += c_3
    #         all_class_count[v_4] += 1
    #         all_confidence_count[v_4] += c_4
    #         all_class_count[v_5] += 1
    #         all_confidence_count[v_5] += c_5
    #
    #         final_class = argmax(all_class_count)
    #         final_class_confidence = all_confidence_count[final_class] / all_class_count[final_class]
    #
    #         # print("all class count ", all_class_count)
    #         # print("all confidence count ", all_confidence_count)
    #         # print("final class ", final_class)
    #         # print("final class confidence ", final_class_confidence)
    #         # sys.exit(0)
    #         if final_class == 0:
    #             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 's' + '_' + str(round(value_normalization(final_class_confidence), 2))
    #         elif final_class == 1:
    #             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 'c' + '_' + str(round(value_normalization(final_class_confidence), 2))
    #         elif final_class == 2:
    #             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 'd' + '_' + str(round(value_normalization(final_class_confidence), 2))
    #         else:
    #             print("unknown prediction category ", v_1)
    #             sys.exit(0)


            ######

    if rfd_test:
        for v_1, v_2, v_3, v_4, v_5, c_1, c_2, c_3, c_4, c_5, elem in zip(sentence_level_results[0], sentence_level_results[1], sentence_level_results[2],
                                           sentence_level_results[3], sentence_level_results[4], sentence_level_confidences[0],
                                                                          sentence_level_confidences[1], sentence_level_confidences[2],
                                                                          sentence_level_confidences[3], sentence_level_confidences[4],
                                                                          article_sentence_key_index[sentence_level_test_begin:]):
            pro_vote = 0
            con_vote = 0
            pro_confidence = []
            con_confidence = []
            if v_1 == 0:
                pro_vote += 1
                pro_confidence.append(c_1)
            else:
                con_vote += 1
                con_confidence.append(c_1)
            if v_2 == 0:
                pro_vote += 1
                pro_confidence.append(c_2)
            else:
                con_vote += 1
                con_confidence.append(c_2)

            if v_3 == 0:
                pro_vote += 1
                pro_confidence.append(c_3)
            else:
                con_vote += 1
                con_confidence.append(c_3)

            if v_4 == 0:
                pro_vote += 1
                pro_confidence.append(c_4)
            else:
                con_vote += 1
                con_confidence.append(c_4)

            if v_5 == 0:
                pro_vote += 1
                pro_confidence.append(c_5)
            else:
                con_vote += 1
                con_confidence.append(c_5)
            if len(pro_confidence) > 0:

                pro_confidence_avg = statistics.mean(pro_confidence)
            else:
                pro_confidence_avg = 0
            if len(con_confidence) > 0:
                con_confidence_avg = statistics.mean(con_confidence)
            else:
                con_confidence_avg = 0
            # print("all v -> ", v_1, v_2, v_3, v_4, v_5, "pro vote -> ", pro_vote, "con vote -> ", con_vote)
            if pro_vote > con_vote:
                final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 's' + '_' + str(round(value_normalization(pro_confidence_avg), 2))
            else:
                final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 'c' + '_' + str(round(value_normalization(con_confidence_avg), 2))

    # for key in final_sentence_level_results:
    #     print(final_sentence_level_results[key])
    return instance_ids, final_article_level_result, final_sentence_level_results, gold_result_final


# def compute_test_accuracy_and_prepare_pipeline_input_per_key(eval_file_path, rfd_test, negation_check=False, key_value=None):
#     print("key value  --------> ", key_value)
#     if negation_check:
#         neg_sentence_ids = pickle.load(open('cd_sentence_ids_with_negation_strict.p', 'rb'))
#
#         # neg_sentence_ids_temp = pickle.load(open('cd_sentence_ids_with_negation.p', 'rb'))
#         # print("neg sentence ids temp -> ", neg_sentence_ids_temp[:5])
#         # neg_sentence_ids = []
#         # for elem in neg_sentence_ids_temp:
#         #     neg_sentence_ids.append(elem.split("_")[0] + "_" + str(int(elem.split("_")[1]) + 1))
#         print("neg sentence ids -> ", len(neg_sentence_ids), neg_sentence_ids[:5])
#     global sentence_level_test_begin, article_sentence_key_index
#
#     article_sentence_key_index = pickle.load(open('rfd_article_sentence_key_index.p', "rb"))
#
#     get_sentence_level_test_index()
#     # print("sentence_level_test_begin -> ", sentence_level_test_begin)
#     # path = 'eval_results_createdebate_test_bert_large_with_confidence_list.json'
#     path = eval_file_path
#     sentence_level_results = []
#     sentence_level_confidences = []
#     gold_result_final = []
#     final_article_level_result = {}
#     instance_ids = []
#     avg_p = []
#     avg_r = []
#     avg_f = []
#     with open(path, "r") as content:
#         test_result = json.load(content)
#         for key in test_result:
#             if len(key) != 1:
#                 continue
#             # print("key in loop ", key)
#             if type(key_value) == int:
#                 if int(key) != key_value:
#                     # print("continuing ")
#                     continue
#             # key = 0, 1, 2
#             # if key_value:
#             #     print("match found")
#             # print(int(key))
#             gold_result = test_result[key]['test']['golds']
#
#             # gold_result = preprocess_result(gold_result)
#             predictions = test_result[key]['test']['predictions']
#             # predictions = preprocess_result(predictions)
#             confidences = test_result[key]['test']['confidences']
#             # confidences = preprocess_result(confidences)
#
#             # compute_accuracy(gold_result[:sentence_level_test_begin], predictions[:sentence_level_test_begin])
#             p, r, f = flat_accuracy_new(predictions[:sentence_level_test_begin], gold_result[:sentence_level_test_begin], True, True, rfd_test)
#             avg_p.append(p)
#             avg_r.append(r)
#             avg_f.append(f)
#             if len(final_article_level_result) == 0:
#                 gold_result_final = gold_result[:sentence_level_test_begin]
#
#             for i, (id_index, sentence_index) in enumerate(article_sentence_key_index):
#                 if sentence_index == -1:
#                     if id_index not in final_article_level_result:
#                         instance_ids.append(id_index)
#                         final_article_level_result[id_index] = [predictions[:sentence_level_test_begin][i]]
#                     else:
#                         final_article_level_result[id_index].append(predictions[:sentence_level_test_begin][i])
#                 else:
#                     break
#
#                 # for i, (id_index, sentence_index) in enumerate(article_sentence_key_index):
#                 #     if sentence_index == -1:
#                 #         instance_ids.append(id_index)
#                 #         final_article_level_result[id_index] = predictions[:sentence_level_test_begin][i]
#                 #     else:
#                 #         break
#
#             sentence_level_results.append(predictions[sentence_level_test_begin:])
#             # print("sentence_level_results first index -> ", len(sentence_level_results[0]))
#             sentence_level_confidences.append(confidences[sentence_level_test_begin:])
#     iii = 0
#     for id_index in final_article_level_result:
#         if iii == 0:
#             # print(final_article_level_result[id_index], int(statistics.mode(final_article_level_result[id_index])))
#             iii += 1
#         final_article_level_result[id_index] = int(statistics.mode(final_article_level_result[id_index]))
#     # print(len(sentence_level_results), len(sentence_level_results[0]), len(sentence_level_results[1]), len(sentence_level_results[2]), len(sentence_level_results[3]), len(sentence_level_results[4]))
#     # print("average precision ", statistics.mean(avg_p))
#     # print("average recall ", statistics.mean(avg_r))
#     print("average fscore ", statistics.mean(avg_f))
#
#     final_sentence_level_results = {}
#     for v_1, c_1, elem in zip(sentence_level_results[0], sentence_level_confidences[0],
#                               article_sentence_key_index[sentence_level_test_begin:]):
#         if v_1 == 0:
#             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 's' + '_' + str(
#                 round(value_normalization(c_1), 2))
#         elif v_1 == 1:
#             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 'c' + '_' + str(
#                 round(value_normalization(c_1), 2))
#         elif v_1 == 2:
#             final_sentence_level_results[elem[0] + "_" + str(elem[1])] = 'd' + '_' + str(
#                 round(value_normalization(c_1), 2))
#         else:
#             print("unknown prediction category ", v_1)
#             sys.exit(0)
#
#     return instance_ids, final_article_level_result, final_sentence_level_results, gold_result_final
#

if __name__ == '__main__':
    pass
    # compute_test_accuracy_and_prepare_pipeline_input()