import networkx as nx
import re, sys
import os
import argparse, pickle
import sys, xlsxwriter
import pandas as pd, spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from termcolor import colored
from draw_graph import drawing

stop_words = set(stopwords.words('english'))

negation_words = ["no", "not", "n't", "nt", "none", "nobody", "nothing", "never", "hardly", "seldom", "without"]
contrast_words = ["but", "however", "yet", "unfortunately", "thought", "although", "nevertheless"]
sp = spacy.load('en_core_web_sm')


def replace(x):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'URL ', x)


def sibling(gr, node):
    parent_node = list(gr.predecessors(node))[0]
    children = list(gr.successors(parent_node))
    children_res = [child for child in children if child != node]
    return children_res


def add_neg_cont_node(graph, selected_nodes, check_sibling=True):
    rest_nodes = []

    if check_sibling:
        for node in selected_nodes:
            sibs = sibling(graph, node)
            if not sibs:
                print("no sibling found")
                sys.exit(0)
            for sib in sibs:
                if sib not in selected_nodes and sib != '0':
                    rest_nodes.append(sib)
    else:
        for node in selected_nodes:
            children = list(graph.successors(node))
            # print("node ", node)
            # print("children ", children)
            for child in children:
                if child not in selected_nodes and child != '0':
                    rest_nodes.append(child)
        # print("selected nodes ", selected_nodes)
        # print("rest nodes -> ", rest_nodes)
    add_nodes = []
    for node in rest_nodes:
        text = graph.nodes[node]['text'].split()
        # print("text ", text)
        for word in text:
            if word in negation_words or word in contrast_words:
                # print("found word ", word)
                add_nodes.append(node)
                break
    result_nodes = selected_nodes + add_nodes
    result_nodes = [int(node) for node in result_nodes]
    result_nodes.sort()
    result_nodes = [str(node) for node in result_nodes]
    return result_nodes


def left_most_nucleus_child(graph, node):
    # l_nodes = list(nx.dfs_edges(graph, source=node))
    # for (s_node, e_node) in l_nodes:
    #     if graph.node[e_node]['type'] == 'single_edu' and graph.node[e_node]['nuclearity'] == 'N':
    #         print("l m n c ", node, e_node)
    #
    #         return e_node
    successors = graph.successors(node)
    for successor in successors:
        if graph.nodes[successor]['nuclearity'] == 'N':
            # print("l m n c ", node, successor)
            return successor
    return


def parent(graph, node):
    return list(graph.predecessors(node))[0]  # only return the id of the parent node


def is_root(graph, node):
    # print("node ", node)
    # print(graph.node[node]['type'])
    return graph.nodes[node]['type'] == 'root'


def is_leaf(graph, node):
    # print("is leaf check")
    # print(node)
    # print(graph.nodes[node]['type'])
    return graph.nodes[node]['type'] == 'single_edu'


def is_nucleus(graph, node):
    return graph.nodes[node]['nuclearity'] == 'N'


def find_my_top_node(graph, node, print_all=False):
    if print_all:
        print("inside find_my_top_node")
        print("node -> ", node)
    C = node
    P = parent(graph, node)

    while left_most_nucleus_child(graph, P) == C and not is_root(graph, P):
        C = P
        P = parent(graph, P)
    if is_root(graph, P):
        C = P
    if print_all:
        print("returning ", C)
    return C


def find_nearest_s_ancestor(graph, node):
    if not is_nucleus(graph, node):
        return node
    P = parent(graph, node)
    while is_nucleus(graph, P) and not is_root(graph, P):
        P = parent(graph, P)
    return P


def find_head_edu(graph, node):
    while not is_leaf(graph, node):
        node = left_most_nucleus_child(graph, node)
    return node


def build_graph(edus, tree_infos):
    print("going to building graph ")
    G = nx.DiGraph()

    edus_ob = {}

    for i, edu in enumerate(edus):
        edus_ob[str(i + 1)] = re.sub('\s+', ' ', edu.strip())
    edus_ob_id = [int(key) for key in edus_ob]
    print("edus_ob_id ", edus_ob_id)
    print("len edus , edus_ob ", len(edus), len(edus_ob))
    for j, tree_info in enumerate(tree_infos):
        try:
            ids, nuclearity, relation = tree_info
            # print("ids, nuclearity, relation ", ids, nuclearity, relation)
        except:
            print(tree_info)
            print(tree_infos)
            sys.exit(0)
        start_edu_id = str(ids[0])
        end_edu_id = str(ids[1])

        if 'Nucleus' in nuclearity:
            nuclearity = 'N'
        elif 'Satellite' in nuclearity:
            nuclearity = 'S'
        else:
            print("no nuclearity found ", j)
            sys.exit(0)
        if start_edu_id not in edus_ob or end_edu_id not in edus_ob:

            print("returning false ", start_edu_id, end_edu_id)
            # sys.exit(0)
            return False
        if start_edu_id == end_edu_id:
            G.add_node(start_edu_id, type='single_edu', nuclearity=nuclearity, participants=[start_edu_id],
                       head=start_edu_id, text=edus_ob[start_edu_id], relation=relation)
        else:
            try:
                start_predecessor = list(nx.edge_dfs(G, start_edu_id, orientation='reverse'))[-1][0]
            except:
                start_predecessor = start_edu_id
            try:
                end_predecessor = list(nx.edge_dfs(G, end_edu_id, orientation='reverse'))[-1][0]
            except:
                end_predecessor = end_edu_id
            new_node_id = start_edu_id + "_" + end_edu_id
            G.add_node(new_node_id, type='combined_edu', nuclearity=nuclearity, participants=[start_predecessor, end_predecessor], head='', text='', relation=relation)
            G.add_edge(new_node_id, start_predecessor)
            G.add_edge(new_node_id, end_predecessor)
            # if j == 3:
            #     break
    subtree_root = 0
    root_start_edu_id = None
    root_end_edu_id = None
    root_start = None
    root_end = None
    for (node, in_deg) in G.in_degree:
        if in_deg == 0 and subtree_root == 0:
            root_start = node
            root_start_edu_id = root_start.split("_")[0]
            subtree_root += 1
        elif in_deg == 0 and subtree_root == 1:
            root_end = node
            root_end_edu_id = root_end.split("_")[-1]
            subtree_root += 1
        elif in_deg == 0 and subtree_root == 2:
            print("More than two disconnected component")
            sys.exit(0)
    # print("root start ", root_start)
    # print("root end ", root_end)
    if root_start_edu_id and root_end_edu_id:
        root_node_id = root_start_edu_id + "_" + root_end_edu_id
        # print("root node id ", root_node_id)
        G.add_node(root_node_id, type='root', nuclearity='', participants=[root_start, root_end], head='', text='', relation='')
        G.add_edge(root_node_id, root_start)
        G.add_edge(root_node_id, root_end)
    print("going to draw graph .. ")
    # drawing(G, 'test_2597', '', 'error_check')

    return G


def extract_all_nucleus(graph):
    only_nucleus_edus = [node for node in graph.nodes if graph.nodes[node]['type'] == 'single_edu' and graph.nodes[node]['nuclearity'] == 'N']
    # print("all nucleus before ", only_nucleus_edus)

    only_nucleus_edus = add_neg_cont_node(graph, only_nucleus_edus, check_sibling=True)
    # print("all nucleus after ", only_nucleus_edus)
    only_nucleus_text = " ".join(graph.nodes[str(val)]['text'] for val in only_nucleus_edus)
    return only_nucleus_text


def extract_only_left_nucleus(graph):
    result_nodes = []
    root_id = None
    for node in graph.nodes:
        if len(list(graph.predecessors(node))) == 0:
            root_id = node
    if not root_id:
        print("NO ROOT ")
        sys.exit(0)
    dfs_result = list(nx.dfs_edges(graph, root_id))
    visited_parent = []
    for (s, e) in dfs_result:
        if graph.nodes[e]['type'] == 'single_edu':
            if s in visited_parent:
                pass
            else:
                if graph.nodes[e]['nuclearity'] == 'N':
                    result_nodes.append(e)
                    visited_parent.append(s)

    result_nodes = add_neg_cont_node(graph, result_nodes, check_sibling=True)
    # print("only left nucleus ", result_nodes)

    only_left_nucleus_text = " ".join(graph.nodes[str(val)]['text'] for val in result_nodes)
    # print("result nodes ln ->  ", result_nodes)

    return result_nodes, only_left_nucleus_text


def extract_only_left_nucleus_plus_concession_nucleus(graph, bracket):
    contrast_participants = []
    all_left, _ = extract_only_left_nucleus(graph)
    for val in bracket:
        edu_ids, _, relation = val
        if relation == 'Contrast' and edu_ids[0] != edu_ids[1]:
            edu_id = str(edu_ids[0]) + "_" + str(edu_ids[1])
            dfs_result = list(nx.dfs_edges(graph, edu_id))
            for (s, e) in dfs_result:
                if graph.nodes[e]['type'] == 'single_edu' and graph.nodes[e]['nuclearity'] == 'N':
                    contrast_participants.append(e)
    result_nodes = all_left + contrast_participants
    result_nodes = add_neg_cont_node(graph, result_nodes, check_sibling=True)
    # print("result nodes lnpc ", result_nodes)
    # print("only left nucleus with concession ", result_nodes)

    only_left_nucleus_text = " ".join(graph.nodes[str(val)]['text'] for val in result_nodes)
    return only_left_nucleus_text


def extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=None, source='0'):

    bfs_res = list(nx.bfs_edges(ddt, source, depth_limit=max_depth))
    bfs_result = [e for (s, e) in bfs_res]
    # print("extract_top_nodes_using_bfs level before", max_depth, bfs_result)

    bfs_result = add_neg_cont_node(ddt, bfs_result, check_sibling=False)
    # print("extract_top_nodes_using_bfs level after ", max_depth, bfs_result)
    bfs_result = [node for node in bfs_result if node not in claim_edus]
    # print("extract_top_nodes_using_bfs level after ", max_depth, bfs_result)

    result = " ".join(ddt.nodes[e]['text'] for e in bfs_result)
    return result, bfs_result


def extract_claim_dependent_top_nodes_using_bfs(ddt, claim_edus, depth):

    # if all the claim edus are in one subtree of root,
        # check if there are other edus associated in that subtree
            # if not, go for the usual bfs
            # if yes, start from the lowest claim edu, do the usual bfs from there
                # if no edu found, then the other edus are associated with some intermediate
                # claim edus, so go to the top claim edu recursively
    # else do the usual bfs

    children_of_root = list(ddt.successors('0'))
    # first check, if any claim edu is a direct child of a root
        # if not, go for the usual bfs

    if children_of_root[0] not in claim_edus and children_of_root[1] not in claim_edus:
        temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
        return temp_res

    dfs_res = list(nx.dfs_edges(ddt, '0'))

    dfs_result = [e for (s, e) in dfs_res]

    right_child_pos = dfs_result.index(children_of_root[1])
    dfs_left = dfs_result[:right_child_pos]
    dfs_right = dfs_result[right_child_pos:]

    # print("dfs left 1 -> ", dfs_left)

    claim_edu_count_in_subtree = 0
    for edu in claim_edus:
        if edu in dfs_left:
            claim_edu_count_in_subtree += 1
    subtree_choice = ''
    if claim_edu_count_in_subtree == len(claim_edus):
        # all claim edus are in the left subtree
        subtree_choice = 'left'
        if claim_edu_count_in_subtree == len(dfs_left):
            # only claim edus are present in the left subtree
            temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
            return temp_res
        else:
            # sort the claim edus based on the their depth, reversely
            # compute distance from root to each claim edu
            distance_cedu_root = [(node, nx.shortest_path_length(ddt, source='0', target=str(node))) for node in claim_edus]
            distance_cedu_root.sort(key=lambda x : x[1], reverse=True)
            temp_res = []
            for (node, dist) in distance_cedu_root:
                # print("left subtree, starting with node ", node)
                temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth, source=node)
                if len(temp_res[1]) > 0:
                    return temp_res
            if len(temp_res) == 0:
                # last resort
                temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
                return temp_res
    elif claim_edu_count_in_subtree == 0:
        # all claim edus are in the right subtree
        subtree_choice = 'right'
        if len(dfs_right) == len(claim_edus):
            # only claim edus are present in the right subtree
            temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
            return temp_res
        else:
            distance_cedu_root = [(node, nx.shortest_path_length(ddt, source='0', target=str(node))) for node in claim_edus]
            distance_cedu_root.sort(key=lambda x : x[1], reverse=True)
            temp_res = []
            for (node, dist) in distance_cedu_root:
                # print("right subtree, starting with node ", node)
                temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth, source=node)
                if len(temp_res[1]) > 0:
                    return temp_res
            if len(temp_res) == 0:
                # last resort
                temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
                return temp_res
    else:
        temp_res = extract_top_nodes_using_bfs(ddt, claim_edus, max_depth=depth)
        return temp_res
    return


def extract_all_nodes_using_dfs(graph, source='0'):
    dfs_res = list(nx.dfs_edges(graph, source))
    print("extract_all_nodes_using_dfs ", dfs_res)
    dfs_result = []
    for s, e in dfs_res:
        if s not in dfs_result:
            dfs_result.append(s)
        if e not in dfs_result:
            dfs_result.append(e)
    result_nodes = [int(node) for node in dfs_result]
    result_nodes.sort()
    result_nodes = [str(node) for node in result_nodes]
    print("dfs result ", result_nodes)
    result = " ".join(graph.nodes[e]['text'] for e in result_nodes if e != '0')
    return result


def edu_to_article_mapping(edu_id_list, article, graph, print_all=False):
    # print("article here ", article)
    # print("article here ", type(article))

    doc = sp(article)
    article_sentences = list(doc.sents)
    article_sentences = [s for s in article_sentences if not str(s.text).isspace()]
    edu_start_count = 0

    edu_str = ''
    ignore_sentence_indices = []
    jac_score = 0
    edu_to_sentence_list = [0 for _ in range(int(edu_id_list[-1]) + 1)]  # sentence count will start from 1
    # print(" edu_to_sentence_list ", edu_to_sentence_list, len(edu_to_sentence_list))
    article_to_edu_mapping = {}
    print(" len(edu_id_list) inside edu_to_article_mapping -> ", len(edu_id_list))
    for i, article_sentence in enumerate(article_sentences):
        sent_token = [token.text for token in sp(article_sentence.text)]
        if print_all:
            print(colored("article sentence -> ", 'red'), article_sentence)
        current_edu_list = []
        print("starting for loop .... ", edu_start_count, len(edu_id_list))
        for j in range(edu_start_count, len(edu_id_list)):

            current_edu = graph.nodes[edu_id_list[j]]['text']
            print("current edu ", current_edu)
            edu_str += current_edu + " "
            edu_tokens = [token.text for token in sp(edu_str)]
            edu_tokens = [w for w in edu_tokens]
            print("cardinality computing .. ")
            intersection_cardinality = len(set.intersection(*[set(sent_token), set(edu_tokens)]))
            union_cardinality = len(set.union(*[set(sent_token), set(edu_tokens)]))
            new_jac_score = intersection_cardinality / float(union_cardinality)
            if print_all:
                print("edu str with score ", j, edu_id_list[j], edu_str, new_jac_score)
            if new_jac_score >= jac_score:
                jac_score = new_jac_score
                current_edu_list.append(edu_id_list[j])
            else:
                if print_all:
                    print("edu matched with extra ", edu_str)
                edu_start_count = j
                # edus_to_article += article_sentence.text
                # print("edus to article -> ", edus_to_article)
                jac_score = 0
                edu_str = ''
                if print_all:
                    print("adding current edu list for sentence inside else -> ", i, current_edu_list)
                for edu_id in current_edu_list:
                    edu_to_sentence_list[int(edu_id)] = i + 1
                article_to_edu_mapping[i] = current_edu_list
                current_edu_list = []

                break
        if len(current_edu_list) > 0:
            # print("adding current edu list for sentence inside if-> ", i, current_edu_list)

            for edu_id in current_edu_list:
                edu_to_sentence_list[int(edu_id)] = i + 1
            article_to_edu_mapping[i] = current_edu_list
            if print_all:
                print("adding current edu list for sentence -> ", i, current_edu_list)

    # return article_to_edu_mapping
    # print(" edu_to_sentence_list ", edu_to_sentence_list, len(edu_to_sentence_list))
    # print("---------")
    # print("article_to_edu_mapping ", article_to_edu_mapping)
    print("Returning from edu_to_article_mapping")
    return edu_to_sentence_list, article_to_edu_mapping, [str(s) for s in list(article_sentences)]


def map_edus_to_article_sentences(mapping, edu_list, article):
    already_used_sentences = []
    doc = sp(article)
    article_sentences = list(doc.sents)
    article_sentences = [sent.text for sent in article_sentences]
    result = ''
    for edu_id in edu_list:
        corresponding_sentnece_id = mapping[int(edu_id)] - 1
        if corresponding_sentnece_id in already_used_sentences:
            continue
        else:
            result += article_sentences[corresponding_sentnece_id]
            already_used_sentences.append(corresponding_sentnece_id)
    # print("map_edus_to_article_sentences..")
    # print(result, already_used_sentences)
    return result, already_used_sentences


def li_structure(edu_file, bracket_file, article_text, id_ind, claim_edus_list):
    # print("LIST of EDUS")
    # for j, edu in enumerate(edu_file):
    #     print(j + 1, edu)

    if len(bracket_file) == 1:
        print("bracket length 1")
        return edu_file[0], edu_file[0], edu_file[0], edu_file[0]
    ddt = nx.DiGraph()

    G = build_graph(edu_file, bracket_file)
    if not G:
        print("G does not exist")
        return None, None, None, None
    # drawing(G, id_ind, '../Discourse_images/development_recheck')
    edu_nodes = [node for node in G.nodes if G.nodes[node]['type'] == 'single_edu']
    ddt.add_node('0', type='super_root', nuclearity='')

    for edu in edu_nodes:

        P = find_my_top_node(G, edu)
        if is_root(G, P):
            ddt.add_node(edu, text=G.nodes[edu]['text'], nuclearity=G.nodes[edu]['nuclearity'])

            ddt.add_edge('0', edu, relation='')
            # ddt.add_node(edu, type='root')
        else:
            rel = G.nodes[P]['relation']
            P = parent(G, P)
            ddt.add_node(edu, text=G.nodes[edu]['text'], nuclearity=G.nodes[edu]['nuclearity'])

            edu_2 = find_head_edu(G, P)
            # if edu == '15':
            #     print("edu 2 ", edu_2)

            ddt.add_edge(edu_2, edu, relation=rel)

    # drawing_ddt(ddt, id_ind, '../Discourse_images/development_recheck')

    # print("Li DDT edges ", list(nx.bfs_edges(ddt, source='0')))
    # print("Shortest path -> ", nx.shortest_path_length(ddt, '0'))
    top_edu = extract_top_nodes_using_bfs(ddt, claim_edus_list, 1)
    # sys.exit(0)
    # all_nucleus = extract_all_nucleus(G)
    # _, all_left_nucleus = extract_only_left_nucleus(G)
    # all_left_nucleus_plus_more = extract_only_left_nucleus_plus_concession_nucleus(G, bracket_file)
    edus_ddt_level_2 = extract_top_nodes_using_bfs(ddt, claim_edus_list, 2)
    edus_ddt_level_3 = extract_top_nodes_using_bfs(ddt, claim_edus_list, 3)

    # the following part is for cases when claim is merged with article
    # claim edu should be 1 (in some case, can be 1,2)

    # edus_ddt_level_inf = extract_top_nodes_using_bfs(ddt)
    # edus_ddt_dfs = extract_all_nodes_using_dfs(ddt)
    #
    # print("top edu... ")
    # print(top_edu)
    # print("all nucleus ... ")
    # print(all_nucleus)
    # print("all left nucleus .. ")
    # print(all_left_nucleus)
    # print("all left nucleus plus more .. ")
    # print(all_left_nucleus_plus_more)
    # print("edus upto 2 level..")
    # print(edus_ddt_level_2)
    # print("edus upto 3 level..")
    # print(edus_ddt_level_3)
    # print("edus upto inf level..")
    # print(edus_ddt_level_inf)


    # print("top node -> ", find_my_top_node(G, '15'))
    # print("head edu -> ", find_head_edu(G, find_my_top_node(G, '15')))
    # print()
    # print()
    # print()
    return G, ddt, top_edu, edus_ddt_level_2, edus_ddt_level_3

    # return top_edu, all_nucleus, all_left_nucleus, all_left_nucleus_plus_more, edus_ddt_level_2, edus_ddt_level_3, edus_ddt_level_inf, edus_ddt_dfs


def test():
    X = nx.DiGraph()

    # X.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 7), (6, 8)])
    X.add_edges_from([(0, 1), (0, 2), (2, 5), (2, 6), (6, 7), (6, 8)])
    root = [x for x in X.nodes() if X.out_degree(x) == 0]
    print(root)
    print("total nodes ", len(X.nodes()))
    # X.add_node(5, weight=4)
    print(list(X.successors(0)))

    # print(X.nodes[5]['weight'])
    # print(list(nx.bfs_edges(X, 0, depth_limit=None)))
    # print(sibling(X, 2))
    # for node in X.nodes:
    #     print(node, list(X.predecessors(node)))
    return


def compute_edu_count(claim, edu_list):

    word_tokens_1 = word_tokenize(text=claim)
    filtered_claim = [w for w in word_tokens_1]
    # filtered_claim = [w for w in word_tokens_1 if w not in stop_words]

    jac_score = 0
    edu_str = ''
    for i, edu in enumerate(edu_list):
        edu_str += edu + " "
        edu_tokens = word_tokenize(text=edu_str)
        edu_tokens = [w for w in edu_tokens]
        # edu_tokens = [w for w in edu_tokens if w not in stop_words]
        print("sets of data ", set(filtered_claim), set(edu_tokens))
        intersection_cardinality = len(set.intersection(*[set(filtered_claim), set(edu_tokens)]))
        union_cardinality = len(set.union(*[set(filtered_claim), set(edu_tokens)]))
        new_jac_score = intersection_cardinality / float(union_cardinality)
        print("edu str with score ", edu_str, new_jac_score)
        if new_jac_score > jac_score:
            jac_score = new_jac_score
        else:
            print("edu matched with extra ", edu_str)
            return i
    return len(edu_list)


if __name__ == '__main__':
    # test()
    # bracket_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_brackets.p'
    # article_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_only.p'
    # edu_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_edus.p'
    # article_sentence_count_path = '../data/data_stance_createdebate_complete/sentence_counts_article.p'
    #
    # path_to_read = '../data/data_stance_createdebate_complete/Createdebate_complete.xlsx'
    # path_to_write = '../data/data_stance_createdebate_complete/Createdebate_quick_discourse_syntax_2.xlsx'
    # sheets_to_read_from = ['clean_train', 'test']

    # write_syntax_file(path_to_read, path_to_write, sheets_to_read_from, bracket_file_path, edu_file_path, article_file_path, article_sentence_count_path)

    bracket_file_path = '../data/data_old/claim_with_articles_brackets.p'
    edu_file_path = '../data/data_old/claim_with_articles_edus.p'
    article_file_path = '../data/data_old/claim_with_articles_only.p'


    article_sentence_count_path = '../data/data_stance_createdebate_development/sentence_counts_article_development.p'
    path_to_read = '../data/data_stance_createdebate_development/Createdebate_development_set_split.xlsx'
    path_to_write = '../data/data_stance_createdebate_development/Createdebate_development_discourse_relation.xlsx'
    sheets_to_read_from = ['development']

    # generate_image(path_to_read, sheets_to_read_from, bracket_file_path, edu_file_path, article_file_path)
    # write_syntax_file_development(path_to_read, path_to_write, sheets_to_read_from, bracket_file_path, edu_file_path, article_file_path, article_sentence_count_path)

    # bracket_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_brackets.p'
    # article_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_only.p'
    # edu_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_edus.p'
    # article_sentence_count_path = '../data/data_stance_createdebate_complete/sentence_counts_article.p'
    #
    # path_to_read = '../data/data_stance_createdebate_complete_claim_cv/Createdebate_complete_head_tail_edu_claim_cv.xlsx'
    # path_to_write = '../data/data_stance_createdebate_complete_claim_cv/Createdebate_complete_head_tail_edu_claim_cv_discourse_syntax.xlsx'
    # sheets_to_read_from = ['test_head', 'test_tail', 'clean_train_head', 'clean_train_tail']
    #
    # write_syntax_file_claim_cv(path_to_read, path_to_write, sheets_to_read_from, bracket_file_path, edu_file_path, article_file_path, article_sentence_count_path)
    #
    # print("done ... ")
    # bracket_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_brackets.p'
    # article_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_only.p'
    # edu_file_path = '../data/data_stance_createdebate_complete/claim_with_articles_edus.p'
    # article_sentence_count_path = '../data/data_stance_createdebate_complete/sentence_counts_article.p'
    #
    # path_to_read = '../data/data_stance_createdebate_complete_claim_cv/Createdebate_complete_head_tail_sentence_claim_cv.xlsx'
    # path_to_write = '../data/data_stance_createdebate_complete_claim_cv/Createdebate_complete_head_tail_sentence_claim_cv_discourse_syntax.xlsx'
    # sheets_to_read_from = ['test_head', 'test_tail', 'clean_train_head', 'clean_train_tail']
    #
    # write_syntax_file_claim_cv(path_to_read, path_to_write, sheets_to_read_from, bracket_file_path, edu_file_path, article_file_path, article_sentence_count_path)
    #
