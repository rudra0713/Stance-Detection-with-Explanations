import pandas as pd, os, json
import numpy as np, argparse, pickle
import string
from nltk.corpus import stopwords
import nltk
import xlsxwriter
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from helper_functions import value_normalization
from make_prediction_with_mtdnn_rfd import return_prediction_rfd
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
from prepare_data_from_mtdnn import compute_test_accuracy_and_prepare_pipeline_input
# from prepare_data_from_mtdnn import compute_test_accuracy_and_prepare_pipeline_input_per_key
global tokenizer, model, stop_words
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


import logging

# now we will Create and configure # logger
# logging.basicConfig(filename="test_log.log", format='%(asctime)s %(message)s',	filemode='w')

# Let us Create an object
# logger=logging.get# logger()

# Now we are going to Set the threshold of # logger to DEBUG
# logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser("""BERT Pytorch for Stance Detection in Createdebate""")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cross_validation_count", type=int, default=2)
    parser.add_argument("--number_of_splits", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_sheet", type=str, default="clean_train")
    # parser.add_argument("--dev_sheet", type=str, default="dev")
    parser.add_argument("--test_sheet", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="/Users/rudra/PycharmProjects/pytorch-transformers/data/data_stance_createdebate_complete/Createdebate_complete_similarity.xlsx")
    parser.add_argument("--rm_punct", type=bool, default=False)
    parser.add_argument("--rm_sw", type=bool, default=False)
    parser.add_argument("--model_info", type=str, default="")
    parser.add_argument("--model_load_path", type=str, default="")
    parser.add_argument("--tokenizer_load_path", type=str, default="")
    parser.add_argument("--model_store_path", type=str, default="weight")
    parser.add_argument("--stop_words_path", type=str, default="")
    parser.add_argument("--show_attention_output", type=bool, default=False)
    parser.add_argument("--article_category", type=str, default="")
    parser.add_argument("--claim_category", type=str, default="claim")
    parser.add_argument("--num_of_labels", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--prediction_file_name", type=str, default="eval_results_rfd_test_mt_dnn.json")
    parser.add_argument("--negation_check", type=bool, default=False)
    parser.add_argument("--config_label", type=str, default='no_prune_agr_dst')
    parser.add_argument("--t_test", type=bool, default=False)
    parser.add_argument("--rfd_test", type=bool, default=False)
    parser.add_argument("--from_summary_side", type=bool, default=False)
    parser.add_argument("--survey_side", type=bool, default=False)

    args = parser.parse_args()
    return args


def flat_accuracy_new(preds, labels, print_full_report, already_converted_to_array=False, rfd_test=False):
    accuracy = 0
    if already_converted_to_array:
        pred_flat = list(preds)
        labels_flat = list(labels)
        # print("len of preds ", len(preds))
        # print("len of labels  ", len(labels))
        for pred, label in zip(pred_flat, labels_flat):
            if pred == label:
                accuracy += 1
        accuracy = accuracy / len(pred_flat)
    else:
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        accuracy = np.sum(pred_flat == labels_flat) / len(pred_flat)

    # print("pred flat ", pred_flat[:10])
    # print("labels flat ", labels_flat[:10])
    #
    # print("len of true labels ", len(labels_flat))
    p = precision_score(labels_flat, pred_flat, average="macro")
    r = recall_score(labels_flat, pred_flat, average="macro")
    data = {'y_Actual': list(labels_flat),
            'y_Predicted': list(pred_flat)
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

    score_rfd = 2 * p * r / (p + r)
    precision, recall, fscore, support = score(labels_flat, pred_flat)
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print("confusion matrix")
    print(confusion_matrix)

    if print_full_report:
        # pass
        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        if rfd_test:
            print('fscore for 2 classes only : {}'.format(np.sum(fscore) / 2))

        # print('support: {}'.format(support))
        # print('macro precision: {}'.format(np.mean(precision)))
        # print('macro recall: {}'.format(np.mean(recall)))
    print("accuracy ", accuracy)
    print('macro fscore: {}'.format(np.mean(fscore)))

    return accuracy, np.mean(fscore)


def det_stance(opt):
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)[opt.config_label]
        # print()
        # print("negation check -> ", opt.negation_check, type(opt.negation_check))

        b_instance_ids, article_level_results, sentence_level_results, gold_result = compute_test_accuracy_and_prepare_pipeline_input(opt.prediction_file_name,  opt.rfd_test, opt.negation_check)
        print("len of instance ids : ", len(b_instance_ids), 'test_583' in b_instance_ids)
        # prediction_per_sample = {}
        # for key in sentence_level_results:
        #     id_ind, pred = key.split("_")[0] + "_" + str(key.split("_")[1]), sentence_level_results[key].split("_")[0]
        #     if id_ind not in prediction_per_sample:
        #         prediction_per_sample[id_ind] = [pred]
        #     else:
        #         prediction_per_sample[id_ind].append(pred)
        # count = 0
        # for key in prediction_per_sample:
        #     print(key, prediction_per_sample[key])
        #     if count == 5:
        #         break
        #     count += 1
        if opt.rfd_test:
            print("gold result set ", set(gold_result))
            dst_predictions, dst_predictions_ob = return_prediction_rfd(b_instance_ids, sentence_level_results,
                                                                        all_failed_ids,
                                                                        article_level_results,
                                                                        category='test', from_summary_side=opt.from_summary_side,
                                                                        should_do_similarity_test=config[
                                                                            'should_do_similarity_test'],
                                                                        should_do_negation_test=config[
                                                                            'should_do_negation_test'],
                                                                        should_do_agreement_test=config[
                                                                            'should_do_agreement_test'],
                                                                        which_agr=config['agg_scheme'])
            # pickle.dump(dst_predictions_ob, open('rfd_article_sentence_level_prediction_sim_pruning.p', 'wb'))
        # if config['should_do_similarity_test'] and len(dst_predictions_ob) == len(gold_result):
        #     pickle.dump(dst_predictions_ob, open('article_sentence_level_prediction_sim_pruning.p', 'wb'))
            print("dst result ... ", len(dst_predictions_ob))
            # flat_accuracy_new(dst_predictions, gold_result[:len(dst_predictions)], True, True, opt.rfd_test)

            if len(dst_predictions) == len(gold_result):
                flat_accuracy_new(dst_predictions, gold_result, True, True, opt.rfd_test)
    return


def det_stance_generate_summary(prediction_file_name, instance_ids):
        b_instance_ids, article_level_results, sentence_level_results, gold_result = compute_test_accuracy_and_prepare_pipeline_input(
            prediction_file_name, opt.rfd_test)
        # for key in sentence_level_results:
        #     if 'test1_' in key:
        #         print(key)
        print("len of b instance ids ", len(b_instance_ids))
        all_failed_ids = pickle.load(open("sentence_dt/all/failed_ids.p", "rb"))
        all_failed_ids = [id_index for (id_index, _) in all_failed_ids]
        # instance_ids = ['test7022']
        dst_predictions, dst_predictions_ob = return_prediction(instance_ids, sentence_level_results, all_failed_ids,
                                                                article_level_results, 0,
                                                                category='test', from_summary_side=True, should_do_similarity_test=True,
                                 should_do_negation_test=False, should_do_agreement_test=False)
        return


def det_stance_t_test(opt):
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)[opt.config_label]
        # print()
        # print("negation check -> ", opt.negation_check, type(opt.negation_check))
        key_range = [t for t in range(5)]
        for key_value in key_range:
            b_instance_ids, article_level_results, sentence_level_results, gold_result = compute_test_accuracy_and_prepare_pipeline_input_per_key(opt.prediction_file_name, opt.rfd_test, opt.negation_check, key_value=key_value)
            # for key in sentence_level_results:
            #     if 'test1_' in key:
            #         print(key)
            # print("len of b instance ids ", len(b_instance_ids))
            if opt.rfd_test:
                # print("gold result set ", set(gold_result))
                dst_predictions, dst_predictions_ob = return_prediction_rfd(b_instance_ids, sentence_level_results,
                                                                        all_failed_ids,
                                                                        article_level_results, 0,
                                                                        category='test', from_summary_side=False,
                                                                        should_do_similarity_test=config[
                                                                            'should_do_similarity_test'],
                                                                        should_do_negation_test=config[
                                                                            'should_do_negation_test'],
                                                                        should_do_agreement_test=config[
                                                                            'should_do_agreement_test'],
                                                                        which_agr=config['agg_scheme'])
            else:
                dst_predictions, dst_predictions_ob = return_prediction(b_instance_ids, sentence_level_results, all_failed_ids,
                                                                    article_level_results, 0,
                                                                    category='test', from_summary_side=False, should_do_similarity_test=config['should_do_similarity_test'],
                                         should_do_negation_test=config['should_do_negation_test'], should_do_agreement_test=config['should_do_agreement_test'], which_agr=config['agg_scheme'])
            # if config['should_do_similarity_test'] and len(dst_predictions_ob) == len(gold_result):
            #     pickle.dump(dst_predictions_ob, open('article_sentence_level_prediction_sim_pruning.p', 'wb'))
            print("dst result ... ")
            if len(dst_predictions) == len(gold_result):
                flat_accuracy_new(dst_predictions, gold_result, True, True, opt.rfd_test)
    return


if __name__ == "__main__":
    # # logger.info("loading train files starts")

    opt = get_args()
    discourse_extra_info_path = '/scratch/rrs99/Stance_Detection_LST/rfd_sentence_level/rfd_discourse_extra_info/all/'
    print("path known")
    all_failed_ids = pickle.load(
        open("/scratch/rrs99/Stance_Detection_LST/rfd_sentence_level/rfd_sentence_dt/all/failed_ids.p", "rb"))
    all_failed_ids = [id_index for (id_index, _) in all_failed_ids]
    print("all failed ids : ", all_failed_ids)
    if opt.t_test:
        det_stance_t_test(opt)
    else:
        det_stance(opt)

                # attention explanation [-1][0][-1][0]
# [-1] -> not sure (out of total 12)
# [0] -> first example in the batch
# [-1] -> last layer (total 12 layer)
# [0] -> take the cls token