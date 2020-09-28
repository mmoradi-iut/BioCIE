'''
Created on Oct 29, 2019

@author: milad moradi
'''

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from builtins import str

import csv
import sys, getopt

import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, Node
import xml.dom.minidom       
from numpy import double
            

#----------------------------------

class Instance:
    
    def __init__(self, input_question, input_class, input_tokens):
        self.question = input_question
        self.label = input_class
        self.question_tokens = input_tokens
        
        self.possible_predictions = []
        self.predicted_label = ''
        
    
    #----------------------------
    def add_concept_to_token_list(self, input_concept):
        concept_appears = False
        for i in range(0, len(self.question_tokens)):
            if (input_concept == self.question_tokens[i]):
                concept_appears = True
                
        if (concept_appears == False):
            self.question_tokens.append(input_concept)
    #----------------------------
    
    
    def does_item_appear(self, input_item):
        item_appear = False
        for i in range(0, len(self.question_tokens)):
            if (input_item.item == self.question_tokens[i]):
                item_appear = True
        
        if (item_appear == True):
            return 1
        else:
            return 0
        
    def does_itemset_appear(self, input_itemset):
        how_many_appear = 0
        for i in range(0, len(input_itemset)):
            item_appear = False
            for j in range(0, len(self.question_tokens)):
                if (input_itemset[i] == self.question_tokens[j]):
                    item_appear = True
            if (item_appear == True):
                how_many_appear += 1
        
        if (how_many_appear == len(input_itemset)):
            return 1
        else:
            return 0
        
    def does_itemset_appear_class(self, input_itemset, input_class):
        how_many_appear = 0
        
        if (input_class == self.label):
            for i in range(0, len(input_itemset)):
                item_appear = False
                for j in range(0, len(self.question_tokens)):
                    if (input_itemset[i] == self.question_tokens[j]):
                        item_appear = True
                if (item_appear == True):
                    how_many_appear += 1
        
        if (how_many_appear == len(input_itemset)):
            return 1
        else:
            return 0

#----------------------------------

class Item:
    
    def __init__(self, input_item, input_class):
        self.item = input_item
        self.label = input_class
        self.class_frequency = 1
        self.overall_frequency = 0
        self.support = 0
        self.class_support = 0
        self.confidence = 0
        self.f_score = 0
        self.lift = 0

#----------------------------------

class Itemlist:
    
    def __init__(self, input_class):
        self.label = input_class
        self.num_instances = 0
        self.items = []
        
    def add_item(self, input_item):
        item_exists = False
        
        for i in range(0, len(self.items)):
            if (input_item.item == self.items[i].item) and (input_item.label == self.items[i].label):
                item_exists = True
                self.items[i].class_frequency += 1
        
        if (item_exists == False):
            self.items.append(input_item)
        
    def return_frequent_items(self, min_supp, min_conf, min_fscore):
        frequent_items = []
        for i in range(0, len(self.items)):
            if(self.items[i].confidence >= min_conf and self.items[i].class_support >= min_supp):
                frequent_items.append(self.items[i])
                
        return frequent_items
                
#----------------------------------

class Itemset:
    
    def __init__(self, input_class):
        self.label = input_class
        self.items = []
        self.class_frequency = 0
        self.overall_frequency = 0
        self.support = 0
        self.class_support = 0
        self.confidence = 0
        self.f_score = 0
        self.lift = 0
        
    def is_itemset_frequent(self, input_supp, input_conf, input_fscore):
        if (self.confidence >= input_conf and self.class_support >= input_supp):
            return True
        else:
            return False
        
    def add_from_itemsets(self, first_itemset, second_itemset):
        for i in range(0, len(first_itemset.items)):
            self.items.append(first_itemset.items[i])
        for j in range(0, len(second_itemset.items)):
            if(second_itemset.items[j] not in(self.items)):
                self.items.append(second_itemset.items[j])
                
    def add_from_items(self, input_item, input_itemset):
        for i in range(0, len(input_itemset.items)):
            self.items.append(input_itemset.items[i])
        
        if(input_item not in(self.items)):
            self.items.append(input_item)
            
            
    def how_many_common_items(self, input_itemset):
        common_itemsets = 0
        for i in range(0, len(self.items)):
            for j in range(0, len(input_itemset.items)):
                if(self.items[i] == input_itemset.items[j]):
                    common_itemsets += 1
        
        return common_itemsets
    
    def is_same_itemsets(self, input_itemset):
        same_items = 0
        
        for i in range(0, len(self.items)):
            for j in range(0, len(input_itemset.items)):
                if (self.items[i] == input_itemset.items[j]):
                    same_items += 1
                    
        if (same_items == len(self.items)):
            return True
        else:
            return False
    
#----------------------------------

class Itemsetlist:
    
    def __init__(self, input_class):
        self.label = input_class
        self.num_instances = 0
        self.itemsets = []
        
    def itemset_already_exist(self, input_itemset):
        already_exist = False
        for i in range(0, len(self.itemsets)):
            if (len(self.itemsets[i].items) == len(input_itemset.items)):
                if (self.itemsets[i].is_same_itemsets(input_itemset) == True):
                    already_exist = True
                    
        return already_exist
    
#----------------------------------

class Prediction:
    
    def __init__(self, input_class, input_num_instances):
        self.class_label = input_class
        self.num_instances = input_num_instances
        self.class_score = 0
        self.itemsets = []
        
    def add_itemset(self, input_itemset):
        self.itemsets.append(input_itemset)
        self.class_score += input_itemset.confidence
        
#----------------------------------

class Stats:
    
    def __init__(self, input_class):
        self.class_label = input_class
        self.num_instances = 0
        self.correctly_classified = 0
        self.uncorrectly_classified = 0
        self.class_accuracy = 0
        
#----------------------------------

class Overallstats:
    
    def __init__(self):
        self.statslist = []
    
    def add_to_list(self, input_stats):
        already_exist = False
        for i in range(0, len(self.statslist)):
            if (input_stats.class_label == self.statslist[i].class_label):
                already_exist = True
        
        if (already_exist == False):
            self.statslist.append(input_stats)
            
    def update_stats(self, real_class, predicted_class):
        for i in range(0, len(self.statslist)):
            
            if (self.statslist[i].class_label == real_class):
                self.statslist[i].num_instances += 1
                if (predicted_class == real_class):
                    self.statslist[i].correctly_classified += 1
                else:
                    self.statslist[i].uncorrectly_classified += 1
                self.statslist[i].class_accuracy = self.statslist[i].correctly_classified / self.statslist[i].num_instances
                
             
        
#----------------------------------

            
            
            
def main(argv):

    supp_threshold = 0.5
    conf_threshold = 0.7
    fscore_threshold = 0.5
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = set(stopwords.words("english"))
    
    input_address = 'Train\\Train.txt'
    
    instances = []
    
    with open(input_address) as f:
        input_instances = f.readlines()
        
        sample_num = 1
        
        for line in input_instances:
            print(line)
            
            tokenized_text = nltk.word_tokenize(line)
            print(tokenized_text)
            instance_class = tokenized_text[0]
            instance_text = ''
            token_list = []
            
            for i in range(2, len(tokenized_text)):
                if i != 2:
                    instance_text += ' '
                instance_text += tokenized_text[i]
                
                        
            print('Class:', instance_class)
            print('Text:', instance_text)
            
            
            temp_instance = Instance(instance_text, instance_class, token_list)
            
            
            input_address_concept = 'ConceptsTXT\\Sample-' + str(sample_num) + '-concept.txt'
            with open(input_address_concept) as f:
                input_concepts = f.readlines()
                for line in input_concepts:
                    temp_concept = line[:len(line)-1]
                    print(temp_concept)
                    temp_instance.add_concept_to_token_list(temp_concept)
            
            
            instances.append(temp_instance)
            
            sample_num += 1
            print('------------------------------------------------------------------------------------------------')
            
    
    
    for i in range(0, len(instances)):
        print('Text:', instances[i].question, '----- Class:', instances[i].label, '----- Tokens:', len(instances[i].question_tokens))
        print(instances[i].question_tokens)
        
        
        
    print('---------- Total instances:', len(instances), '----------')
    
    
    #---------------------------------------------------------------------------------------------------------------
    
    #-------------- Create all items
    
    list_itemlist = []
    
    for i in range(0, len(instances)):
        instance_class = instances[i].label
        
        #----- Check if there is an itemlist for the current class
        a_list_exists = False
        list_index = -1
        for j in range(0, len(list_itemlist)):
            if (instance_class == list_itemlist[j].label):
                a_list_exists = True
                list_index = j
                
        #----- If a list exists: add the items of the current instance to the list
        if (a_list_exists == True):
            added_items = []
            for j in range(0, len(instances[i].question_tokens)):
                current_token = instances[i].question_tokens[j]
                if (current_token not in added_items):
                    temp_item = Item(current_token, instance_class)
                    list_itemlist[list_index].add_item(temp_item)
                    added_items.append(current_token)
            list_itemlist[list_index].num_instances += 1
        
        #----- If a list does not exists: create a list and add the items of the current instance to the list
        if (a_list_exists == False):
            temp_itemlist = Itemlist(instance_class)
            
            added_items = []
            for j in range(0, len(instances[i].question_tokens)):
                current_token = instances[i].question_tokens[j]
                if (current_token not in added_items):
                    temp_item = Item(current_token, instance_class)
                    temp_itemlist.add_item(temp_item)
                    added_items.append(current_token)
            
            temp_itemlist.num_instances += 1
            list_itemlist.append(temp_itemlist)
    
    
    for i in range(0, len(list_itemlist)):
        print('Itemlist:', i, '----- Class:', list_itemlist[i].label, '----- Instances:', list_itemlist[i].num_instances, '----- Items:', len(list_itemlist[i].items))
            
    #----- Just to print items within a given class
    print('-------------------------------------------------------------------------------------------------------')
    current_index = 1
    for i in range(0, len(list_itemlist[current_index].items)):
        current_item = list_itemlist[current_index].items[i]
        print(i, 'Item:', current_item.item, '----- Class:', current_item.label, '----- Frequency:', current_item.class_frequency, '----- Support:', current_item.support, '----- Confidence:', current_item.confidence)
        
        
    #----- Calculate support and confidence for all items
    
    num_all_instances = len(instances)
    
    for i in range(0, len(list_itemlist)):
        for j in range(0, len(list_itemlist[i].items)):
            for k in range(0, len(instances)):
                list_itemlist[i].items[j].overall_frequency += instances[k].does_item_appear(list_itemlist[i].items[j])
                
            list_itemlist[i].items[j].support = list_itemlist[i].items[j].overall_frequency / num_all_instances
            list_itemlist[i].items[j].class_support = list_itemlist[i].items[j].class_frequency / list_itemlist[i].num_instances
            list_itemlist[i].items[j].confidence = list_itemlist[i].items[j].class_frequency / list_itemlist[i].items[j].overall_frequency
            list_itemlist[i].items[j].lift = list_itemlist[i].items[j].confidence / (list_itemlist[i].num_instances / num_all_instances)
            
            list_itemlist[i].items[j].f_score = 2 * (list_itemlist[i].items[j].class_support * list_itemlist[i].items[j].confidence)
            list_itemlist[i].items[j].f_score = list_itemlist[i].items[j].f_score / (list_itemlist[i].items[j].class_support + list_itemlist[i].items[j].confidence)
    
    
    #----- Just to print items within a given class
    print('-------------------------------------------------------------------------------------------------------')
    for i in range(0, len(list_itemlist[current_index].items)):
        current_item = list_itemlist[current_index].items[i]
        print(i, 'Item:', current_item.item, '----- Class:', current_item.label, '----- Frequency:', current_item.overall_frequency, '----- Support:', current_item.support, '----- Confidence:', current_item.confidence, '----- F-score:', current_item.f_score)
    
    
    #----- Calculate frequent 1-itemsets
    list_itemsetlist = []
    for i in range(0, len(list_itemlist)):
        current_class = list_itemlist[i].label
        temp_itemsetlist = Itemsetlist (current_class)
        temp_itemsetlist.num_instances = list_itemlist[i].num_instances
        
        temp_freq_items = list_itemlist[i].return_frequent_items(supp_threshold, conf_threshold, fscore_threshold)
        for j in range(0, len(temp_freq_items)):
            temp_itemset = Itemset(current_class)
            temp_itemset.class_frequency = temp_freq_items[j].class_frequency
            temp_itemset.overall_frequency = temp_freq_items[j].overall_frequency
            temp_itemset.support = temp_freq_items[j].support
            temp_itemset.class_support = temp_freq_items[j].class_support
            temp_itemset.confidence = temp_freq_items[j].confidence
            temp_itemset.f_score = temp_freq_items[j].f_score
            temp_itemset.lift = temp_freq_items[j].lift
            temp_itemset.items.append(temp_freq_items[j].item)
            
            temp_itemsetlist.itemsets.append(temp_itemset)
            
        list_itemsetlist.append(temp_itemsetlist)
    
    
    #----- Just to print frequent 1-itemsets
    for i in range(0, len(list_itemsetlist)):
        print('---------- ItemsetList Class:', list_itemsetlist[i].label, '----------')
        for j in range(0, len(list_itemsetlist[i].itemsets)):
            print('Itemset:', list_itemsetlist[i].itemsets[j].items, '----- Support:', list_itemsetlist[i].itemsets[j].support, '----- Confidence:', list_itemsetlist[i].itemsets[j].confidence)
    
    
    #----- Frequent itemset mining algorithm begins
    
    k = 2
    algorithm_continue = True
    
    while(algorithm_continue == True):
        
        how_many_added = 0
        
        if (k == 2):
            for i in range(0, len(list_itemsetlist)): #----- Repeat for every class
                
                current_class = list_itemsetlist[i].label
                
                print('Class:', i)
                
                #----- Create candidate 2-itemsets
                candidate_itemsets = []
                candidate_num = 0
                for j in range(0, len(list_itemsetlist[i].itemsets)):
                    for n in range(j + 1, len(list_itemsetlist[i].itemsets)):
                        
                        temp_itemset = Itemset(current_class)
                        temp_itemset.add_from_itemsets(list_itemsetlist[i].itemsets[j], list_itemsetlist[i].itemsets[n])
                        
                        for m in range(0, len(instances)):
                            temp_itemset.overall_frequency += instances[m].does_itemset_appear(temp_itemset.items)
                            temp_itemset.class_frequency += instances[m].does_itemset_appear_class(temp_itemset.items, current_class)
                            
                        if(temp_itemset.overall_frequency > 0):
                            temp_itemset.support = temp_itemset.overall_frequency / num_all_instances
                            temp_itemset.class_support = temp_itemset.class_frequency / list_itemsetlist[i].num_instances
                            temp_itemset.confidence = temp_itemset.class_frequency / temp_itemset.overall_frequency
                            temp_itemset.lift = temp_itemset.confidence / (list_itemsetlist[i].num_instances / num_all_instances)
                            if(temp_itemset.class_support > 0):
                                temp_itemset.f_score = 2 * (temp_itemset.class_support * temp_itemset.confidence)
                                temp_itemset.f_score = temp_itemset.f_score / (temp_itemset.class_support + temp_itemset.confidence)
                        
                        candidate_itemsets.append(temp_itemset)
                        
                        candidate_num += 1
                        print('Class:', i, ' Candidate:', candidate_num)
                        #----- Candidate 2-itemsets were created
                
                #----- Add frequent 2-itemsets to the frequent itemsets list of the current class
                for j in range(0, len(candidate_itemsets)):
                    if (candidate_itemsets[j].is_itemset_frequent(supp_threshold, conf_threshold, fscore_threshold) == True):
                        list_itemsetlist[i].itemsets.append(candidate_itemsets[j])
                        how_many_added += 1
                        
        #--------------------------------------------------
        
        if (k > 2):
            
            for i in range(0, len(list_itemsetlist)):
                
                current_class = list_itemsetlist[i].label
                #print('Class:', i)
                
                #----- Create candidate k-itemsets
                candidate_itemsets = []
                for j in range(0, len(list_itemsetlist[i].itemsets)):
                    for n in range(j + 1, len(list_itemsetlist[i].itemsets)):
                        
                        if (len(list_itemsetlist[i].itemsets[j].items) == k-1 and len(list_itemsetlist[i].itemsets[n].items) == k-1):
                            if (list_itemsetlist[i].itemsets[j].how_many_common_items(list_itemsetlist[i].itemsets[n]) == k-2):
                                
                                temp_itemset = Itemset(current_class)
                                temp_itemset.add_from_itemsets(list_itemsetlist[i].itemsets[j], list_itemsetlist[i].itemsets[n])
                                
                                for m in range(0, len(instances)):
                                    temp_itemset.overall_frequency += instances[m].does_itemset_appear(temp_itemset.items)
                                    temp_itemset.class_frequency += instances[m].does_itemset_appear_class(temp_itemset.items, current_class)
                            
                                if(temp_itemset.overall_frequency > 0):
                                    temp_itemset.support = temp_itemset.overall_frequency / num_all_instances
                                    temp_itemset.class_support = temp_itemset.class_frequency / list_itemsetlist[i].num_instances
                                    temp_itemset.confidence = temp_itemset.class_frequency / temp_itemset.overall_frequency
                                    temp_itemset.lift = temp_itemset.confidence / (list_itemsetlist[i].num_instances / num_all_instances)
                                    if(temp_itemset.class_support > 0):
                                        temp_itemset.f_score = 2 * (temp_itemset.class_support * temp_itemset.confidence)
                                        temp_itemset.f_score = temp_itemset.f_score / (temp_itemset.class_support + temp_itemset.confidence)
                        
                                candidate_itemsets.append(temp_itemset)
                                #----- Candidate k-itemsets were created

                #----- Add frequent k-itemsets to the frequent itemsets list of the current class
                for j in range(0, len(candidate_itemsets)):
                    
                    #----- Check if the current candidate itemset is frequent
                    if (candidate_itemsets[j].is_itemset_frequent(supp_threshold, conf_threshold, fscore_threshold) == True):
                        
                        #----- Create subsets of the current candidate itemset
                        
                        is_any_infrequent_subset = False
                        subsets = []
                        #----- Create subsets with size 1
                        for p in range(0, len(candidate_itemsets[j].items)):
                            temp_itemset = Itemset(current_class)
                            temp_itemset.items.append(candidate_itemsets[j].items[p])
                            subsets.append(temp_itemset)
                            
                        #----- Create subsets with size > 1
                        for size in range(2, k):
                            temp_subsets = []
                            for q in range(0, len(candidate_itemsets[j].items)):
                                for r in range(0, len(subsets)):
                                    if (len(subsets[r].items) == size-1):
                                        temp_itemset = Itemset(current_class)
                                        temp_itemset.add_from_items(candidate_itemsets[j].items[q], subsets[r])
                                    
                                        if (len(temp_itemset.items) == size):
                                            temp_subsets.append(temp_itemset)
                            
                            for s in range(0, len(temp_subsets)):
                                subsets.append(temp_subsets[s])
                    
                        #if (k>=3 and i==46):
                            print('Class:', i, 'Candidate itemset:', j, 'All candidates:', len(candidate_itemsets), 'Number of subsets: ', len(subsets))
                            
                        #----- Check if every subset of the current candidate itemset is frequent
                        for p in range(0, len(subsets)):
                            
                            for m in range(0, len(instances)):
                                subsets[p].overall_frequency += instances[m].does_itemset_appear(subsets[p].items)
                                subsets[p].class_frequency += instances[m].does_itemset_appear_class(subsets[p].items, current_class)
                                
                            if(subsets[p].overall_frequency > 0):
                                subsets[p].support = subsets[p].overall_frequency / num_all_instances
                                subsets[p].class_support = subsets[p].class_frequency / list_itemsetlist[i].num_instances
                                subsets[p].confidence = subsets[p].class_frequency / subsets[p].overall_frequency
                                subsets[p].lift = subsets[p].confidence / (list_itemsetlist[i].num_instances / num_all_instances)
                                if(subsets[p].class_support > 0):
                                    subsets[p].f_score = 2 * (subsets[p].class_support * subsets[p].confidence)
                                    subsets[p].f_score = subsets[p].f_score / (subsets[p].class_support + subsets[p].confidence)
                                    
                            #----- Check
                            if (subsets[p].is_itemset_frequent(supp_threshold, conf_threshold, fscore_threshold) == False):
                                is_any_infrequent_subset = True
                                
                        if (is_any_infrequent_subset == False):
                            if (list_itemsetlist[i].itemset_already_exist(candidate_itemsets[j]) == False):
                                list_itemsetlist[i].itemsets.append(candidate_itemsets[j])
                                how_many_added += 1
                            
            
        #--------------------------------------------------
        
        print (how_many_added, ' ', k, '-itemsets were added')
        
        k += 1
        
        if (how_many_added == 0 or k > 3):
            algorithm_continue = False
            
    #----- End of the frequent itemset mining algorithm
    
    #----- Sort itemsets based on class frequency
    for i in range(0, len(list_itemsetlist)):
        for j in range(0, len(list_itemsetlist[i].itemsets)):
            for m in range(j+1, len(list_itemsetlist[i].itemsets)):
                if (list_itemsetlist[i].itemsets[j].class_support < list_itemsetlist[i].itemsets[m].class_support):
                    temp_itemset = list_itemsetlist[i].itemsets[j]
                    list_itemsetlist[i].itemsets[j] = list_itemsetlist[i].itemsets[m]
                    list_itemsetlist[i].itemsets[m] = temp_itemset
    
    
    #----- Output the frequent itemsets in a file
    Output = ''
    for i in range(0, len(list_itemsetlist)):
        #print('---------- ItemsetList Class:', list_itemsetlist[i].label, '----------')
        Output += '\n---------- ItemsetList Class:' + str(list_itemsetlist[i].label) + '----- Instances:' + str(list_itemsetlist[i].num_instances) + '----------' + '\n'
        for j in range(0, len(list_itemsetlist[i].itemsets)):
            #print('Itemset:', list_itemsetlist[i].itemsets[j].items, '----- Support:', list_itemsetlist[i].itemsets[j].support, '----- Confidence:', list_itemsetlist[i].itemsets[j].confidence)
            Output += 'Itemset:' + str(list_itemsetlist[i].itemsets[j].items) + '----- OverallFreq:' + str(list_itemsetlist[i].itemsets[j].overall_frequency) + '----- Support:' + str(list_itemsetlist[i].itemsets[j].support) + '----- ClassFreq:' + str(list_itemsetlist[i].itemsets[j].class_frequency) + '----- Class_support:' + str(list_itemsetlist[i].itemsets[j].class_support) + '----- Confidence:' + str(list_itemsetlist[i].itemsets[j].confidence) + '----- Lift:' + str(list_itemsetlist[i].itemsets[j].lift) + '----- F-score:' + str(list_itemsetlist[i].itemsets[j].f_score) + '\n'
    
    output_file = open('XAI model\\Output(conf0.7-supp0.5)-concept.txt', 'w')
    output_file.write(Output)
    output_file.close()
    
    
    #-------------------- Writing the model in a XML file
    
    
    model_xml = '<model>' + '\n'
    
    for i in range(0, len(list_itemsetlist)): #---- Repeat for every class
        current_class = list_itemsetlist[i].label
        model_xml += '<class>' + '\n'
        model_xml += '<class_label>' + current_class + '</class_label>' + '\n'
        model_xml += '<class_instances>' + str(list_itemsetlist[i].num_instances) + '</class_instances>' + '\n'
        model_xml += '<itemsets>' + '\n'
        
        for j in range(0, len(list_itemsetlist[i].itemsets)): #---- Repeat for every itemset
            model_xml += '<itemset>' + '\n'
            model_xml += '<items>' + '\n'
            
            for m in range(0, len(list_itemsetlist[i].itemsets[j].items)): #---- Repeat for every item
                model_xml += '<item>' + list_itemsetlist[i].itemsets[j].items[m] + '</item>' + '\n'
                
            model_xml += '</items>' + '\n'
            model_xml += '<class_frequency>' + str(list_itemsetlist[i].itemsets[j].class_frequency) + '</class_frequency>' + '\n'
            model_xml += '<overall_frequency>' + str(list_itemsetlist[i].itemsets[j].overall_frequency) + '</overall_frequency>' + '\n'
            model_xml += '<support>' + str(list_itemsetlist[i].itemsets[j].support) + '</support>' + '\n'
            model_xml += '<class_support>' + str(list_itemsetlist[i].itemsets[j].class_support) + '</class_support>' + '\n'
            model_xml += '<confidence>' + str(list_itemsetlist[i].itemsets[j].confidence) + '</confidence>' + '\n'
            model_xml += '<f_score>' + str(list_itemsetlist[i].itemsets[j].f_score) + '</f_score>' + '\n'
            model_xml += '<lift>' + str(list_itemsetlist[i].itemsets[j].lift) + '</lift>' + '\n'
            
            model_xml += '</itemset>' + '\n'
            
        model_xml += '</itemsets>' + '\n'
        model_xml += '</class>' + '\n'
        
    model_xml += '</model>' + '\n'
        
    output_file = open('XAI model\\Model(conf0.7-supp0.5)-concept.xml', 'w')
    output_file.write(model_xml)
    output_file.close()
            
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
                
    #---------------------------------------- Testing the model ----------------------------------------
    
    #----- Read the test dataset
    
    input_address = 'Test\\Test.txt'
    
    instances = []
    
    with open(input_address) as f:
        input_instances = f.readlines()
        
        sample_num = 1
        
        for line in input_instances:
            print(line)
            
            tokenized_text = nltk.word_tokenize(line)
            print(tokenized_text)
            instance_class = tokenized_text[0]
            instance_text = ''
            token_list = []
            
            for i in range(2, len(tokenized_text)):
                if i != 2:
                    instance_text += ' '
                instance_text += tokenized_text[i]
                
                        
            print('Class:', instance_class)
            print('Text:', instance_text)
            
            temp_instance = Instance(instance_text, instance_class, token_list)
            
            
            input_address_concept = 'ConceptsTXT\\Sample-' + str(sample_num) + '-concept.txt'
            with open(input_address_concept) as f:
                input_concepts = f.readlines()
                for line in input_concepts:
                    temp_concept = line[:len(line)-1]
                    print(temp_concept)
                    temp_instance.add_concept_to_token_list(temp_concept)
            
            
            instances.append(temp_instance)
            
            sample_num += 1
            print('------------------------------------------------------------------------------------------------')
            
            
    for i in range(0, len(instances)):
        print('Text:', instances[i].question, '----- Class:', instances[i].label, '----- Tokens:', instances[i].question_tokens)
        
    print('---------- Total instances:', len(instances), '----------')
    
    #----- Read the model
    
    model = xml.dom.minidom.parse('XAI model\\Model(conf0.7-supp0.5)-concept.xml')
    
    list_itemsetlist = []
    
    for cls in model.getElementsByTagName("class"):
        
        current_class = ''
        for cls_label in cls.getElementsByTagName("class_label"):
            for value in cls_label.childNodes:
                current_class = value.data
        
        temp_itemsetlist = Itemsetlist(current_class)
        
        class_instances = 0
        for cls_inst in cls.getElementsByTagName("class_instances"):
            for value in cls_inst.childNodes:
                class_instances = int(value.data)
                
        temp_itemsetlist.num_instances = class_instances
        print('\n')
        print('---------- Class:', temp_itemsetlist.label, '---------- Instances:', temp_itemsetlist.num_instances)
        
        for itemset in cls.getElementsByTagName("itemset"):
            temp_itemset = Itemset(current_class)
            
            for item in itemset.getElementsByTagName("item"):
                for value in item.childNodes:
                    temp_itemset.items.append(value.data)
            
            for class_freq in itemset.getElementsByTagName("class_frequency"):
                for value in class_freq.childNodes:
                    temp_itemset.class_frequency = double(value.data)
                    
            for overall_freq in itemset.getElementsByTagName("overall_frequency"):
                for value in overall_freq.childNodes:
                    temp_itemset.overall_frequency = double(value.data)
                    
            for supp in itemset.getElementsByTagName("support"):
                for value in supp.childNodes:
                    temp_itemset.support = double(value.data)
                    
            for class_supp in itemset.getElementsByTagName("class_support"):
                for value in class_supp.childNodes:
                    temp_itemset.class_support = double(value.data)
                    
            for conf in itemset.getElementsByTagName("confidence"):
                for value in conf.childNodes:
                    temp_itemset.confidence = double(value.data)
                    
            for fscore in itemset.getElementsByTagName("f_score"):
                for value in fscore.childNodes:
                    temp_itemset.f_score = double(value.data)
                    
            for lft in itemset.getElementsByTagName("lift"):
                for value in lft.childNodes:
                    temp_itemset.lift = double(value.data)
                    
            print('Itemset:', temp_itemset.items, '----- Class_freq:', temp_itemset.class_frequency, '----- Overal_freq:', temp_itemset.overall_frequency, '----- Support:', temp_itemset.support, '----- Class_support:', temp_itemset.class_support, '----- Confidence:', temp_itemset.confidence, '----- F_score:', temp_itemset.f_score, '----- Lift', temp_itemset.lift)
            temp_itemsetlist.itemsets.append(temp_itemset)
            
        list_itemsetlist.append(temp_itemsetlist)
        
    print('\n-------------------- Model has been loaded -------------------- All classes:', len(list_itemsetlist))
    
    
    #----- Sort the classe based on the number of instances that every class has
    
    for i in range(0, len(list_itemsetlist)):
        for j in range(i+1, len(list_itemsetlist)):
            if (list_itemsetlist[i].num_instances < list_itemsetlist[j].num_instances):
                temp_itemsetlist = list_itemsetlist[i]
                list_itemsetlist[i] = list_itemsetlist[j]
                list_itemsetlist[j] = temp_itemsetlist
                
    overall_stats = Overallstats()
    
    for i in range(0, len(list_itemsetlist)):
        
        temp_stats = Stats(list_itemsetlist[i].label)
        overall_stats.add_to_list(temp_stats)
        
        print(i, '----- Class:', list_itemsetlist[i].label, '----- Instances:', list_itemsetlist[i].num_instances)
    
    
    #----- Start classification
    
    correctly_predicted = 0
    
    for i in range(0, len(instances)):
        
        for j in range(0, len(list_itemsetlist)):
            
            current_class = list_itemsetlist[j].label
            num_instances = list_itemsetlist[j].num_instances
            temp_prediction = Prediction(current_class, num_instances)
            
            for m in range(0, len(list_itemsetlist[j].itemsets)):
                if (instances[i].does_itemset_appear(list_itemsetlist[j].itemsets[m].items) == 1):
                    temp_prediction.add_itemset(list_itemsetlist[j].itemsets[m])
                    
            if (len(temp_prediction.itemsets) > 0):
                instances[i].possible_predictions.append(temp_prediction)
                
        #----- Sort possible predictions based on the number of instances each one has
        for j in range(0, len(instances[i].possible_predictions)):
            for m in range(j+1, len(instances[i].possible_predictions)):
                if (instances[i].possible_predictions[j].num_instances < instances[i].possible_predictions[m].num_instances):
                    temp_prediction = instances[i].possible_predictions[j]
                    instances[i].possible_predictions[j] = instances[i].possible_predictions[m]
                    instances[i].possible_predictions[m] = temp_prediction
                    
        #----- Sort possible predictions based on their score
        for j in range(0, len(instances[i].possible_predictions)):
            for m in range(j+1, len(instances[i].possible_predictions)):
                if (instances[i].possible_predictions[j].class_score < instances[i].possible_predictions[m].class_score):
                    temp_prediction = instances[i].possible_predictions[j]
                    instances[i].possible_predictions[j] = instances[i].possible_predictions[m]
                    instances[i].possible_predictions[m] = temp_prediction
                    
        #----- Select the first possible prediction as the final prediction
        if (len(instances[i].possible_predictions) > 0):
            instances[i].predicted_label = instances[i].possible_predictions[0].class_label
        else:
            instances[i].predicted_label = list_itemsetlist[0].label
        
        if (instances[i].predicted_label == instances[i].label):
            correctly_predicted += 1
            
        overall_stats.update_stats(instances[i].label, instances[i].predicted_label)
            
        print('---------- Instance', i, 'was predicted ----------')
        
    
    #---------------------------------------------------------------------------------------------
    
                        
    print('\n-------------------- Sorting based on Confidence --------------------\n')
    for i in range(0, len(instances)):
        for j in range(0, len(instances[i].possible_predictions)):
            for m in range(0, len(instances[i].possible_predictions[j].itemsets)):
                for n in range(m+1, len(instances[i].possible_predictions[j].itemsets)):
                    if (instances[i].possible_predictions[j].itemsets[m].confidence < instances[i].possible_predictions[j].itemsets[n].confidence):
                        temp_itemset = instances[i].possible_predictions[j].itemsets[m]
                        instances[i].possible_predictions[j].itemsets[m] = instances[i].possible_predictions[j].itemsets[n]
                        instances[i].possible_predictions[j].itemsets[n] = temp_itemset
                        
    print('\n-------------------- Sorting based on Class_support --------------------\n')
    for i in range(0, len(instances)):
        for j in range(0, len(instances[i].possible_predictions)):
            for m in range(0, len(instances[i].possible_predictions[j].itemsets)):
                for n in range(m+1, len(instances[i].possible_predictions[j].itemsets)):
                    if (instances[i].possible_predictions[j].itemsets[m].class_support < instances[i].possible_predictions[j].itemsets[n].class_support):
                        temp_itemset = instances[i].possible_predictions[j].itemsets[m]
                        instances[i].possible_predictions[j].itemsets[m] = instances[i].possible_predictions[j].itemsets[n]
                        instances[i].possible_predictions[j].itemsets[n] = temp_itemset
                        
    print('\n-------------------- Sorting based on Support --------------------\n')
    for i in range(0, len(instances)):
        for j in range(0, len(instances[i].possible_predictions)):
            for m in range(0, len(instances[i].possible_predictions[j].itemsets)):
                for n in range(m+1, len(instances[i].possible_predictions[j].itemsets)):
                    if (instances[i].possible_predictions[j].itemsets[m].support < instances[i].possible_predictions[j].itemsets[n].support):
                        temp_itemset = instances[i].possible_predictions[j].itemsets[m]
                        instances[i].possible_predictions[j].itemsets[m] = instances[i].possible_predictions[j].itemsets[n]
                        instances[i].possible_predictions[j].itemsets[n] = temp_itemset
    
    #----- Results
    
    result = 'Total number of test instances: ' + str(len(instances)) + '\n'
    result += 'Correctly predicted instances: ' + str(correctly_predicted) + '\n'
    result += 'Uncorrectly predicted instances: ' + str(len(instances) - correctly_predicted) + '\n'
    result += '\n' + 'Accuracy: ' + str(correctly_predicted / len(instances))
    
    
    
    result += '\n\n------------------------------ Stitistics per class ------------------------------'
    for i in range(0, len(overall_stats.statslist)):
        result += '\n\n' + '---------- Class: ' + overall_stats.statslist[i].class_label + ' ----------'
        result += '\n' + '    Total instances: ' + str(overall_stats.statslist[i].num_instances)
        result += '\n' + '    Correctly classified instances: ' + str(overall_stats.statslist[i].correctly_classified)
        result += '\n' + '    Uncorrectly classified instances: ' + str(overall_stats.statslist[i].uncorrectly_classified)
        result += '\n' + '    Classification accuracy: ' + str(overall_stats.statslist[i].class_accuracy)
    
    for i in range(0, len(instances)):
        result += '\n\n********************************************************************************'
        result += '\n' + str(i) + '----- Instance:' + instances[i].question + '----- Real class:'+ instances[i].label + '----- Predicted class:' + instances[i].predicted_label
        print('\n', i, '----- Instance:', instances[i].question, '----- Real class:',instances[i].label, '----- Predicted class:',instances[i].predicted_label)
        for j in range(0, len(instances[i].possible_predictions)):
            result += '\n\n' + '    Possible prediction class:' + instances[i].possible_predictions[j].class_label + '----- Score:' + str(instances[i].possible_predictions[j].class_score)
            print('Possible prediction class:', instances[i].possible_predictions[j].class_label, '----- Score:', instances[i].possible_predictions[j].class_score)
            for m in range(0, len(instances[i].possible_predictions[j].itemsets)):
                if (len(instances[i].possible_predictions[j].itemsets[m].items) <= 2):
                    result += '\n' + '        Itemset:' + str(instances[i].possible_predictions[j].itemsets[m].items) + '----- Confidence:' + str(instances[i].possible_predictions[j].itemsets[m].confidence) + '----- Class_support:' + str(instances[i].possible_predictions[j].itemsets[m].class_support) + '----- Support:' + str(instances[i].possible_predictions[j].itemsets[m].support) + '----- F_score:' + str(instances[i].possible_predictions[j].itemsets[m].f_score)
                    print('    Itemset:', instances[i].possible_predictions[j].itemsets[m].items, '----- F_score:', instances[i].possible_predictions[j].itemsets[m].f_score, '----- Confidence:', instances[i].possible_predictions[j].itemsets[m].confidence, '----- Class_support:', instances[i].possible_predictions[j].itemsets[m].class_support, '----- Support:', instances[i].possible_predictions[j].itemsets[m].support)
                
    output_file = open('Test\\Result(conf0.7-supp0.5)-concept.txt', 'w')
    output_file.write(result)
    output_file.close()
            
        
            
    
            
            
            
            
            

if __name__ == '__main__':
    main(sys.argv[1:])