from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf
import numpy as np
import os
import multiprocessing as mp
#saved_model_path='saved_model'
import regex as re


def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

def sertis_tokenizer(text, saved_model_path):
    #print(text)
    text = text.strip().split('||')
    #print(len(text))
    #print(text)
    for i in range(len(text)):
        if text[i] == '':
            text[i] = ' '
    inputs = [[ThaiWordSegmentLabeller.get_input_labels(i)] for i in text]
    #print(len(inputs))
    #print(inputs)
    lengths = [[len(i)] for i in text]
    #print(lengths)
    with tf.Session() as session:
        model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph = tf.get_default_graph()
        g_inputs = graph.get_tensor_by_name(signature.inputs['inputs'].name)
        g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
        g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
        g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)
        label = []
        all_words = []
        for i, j in enumerate(inputs):
            # if j == [ThaiWordSegmentLabeller.get_input_labels('')]:
                # print('YES')
            #print(j)
            #print(lengths[i])
            y = session.run(g_outputs, feed_dict = {g_inputs: j, g_lengths: lengths[i], g_training: False})
            words = split(text[i], nonzero(y))
            words = [word.strip() for word in words if word.strip() != '']
            #print(i)
            if i % 2:
                print('label')
                label = label + ['1']*len(words)
            else:
                print('not label')
                label = label + ['0']*len(words)
            all_words = all_words + words
    print(len(words))
    #print(all_words)
    #print(label)
    #print('finished!!!')
    return ('||'.join(all_words), '||'.join(label), len(label))

def main():
    os.chdir('E:\\min\\Tokyo_tech_exchange\\NER\\data')
    saved_model_path = 'E:\\Coding_projects\\Name-Entity-Recognition-at-tokyo-tech\\saved_model'
    num_process = os.cpu_count()
    total_num_line = {} 
    for i in ['mea', 'org', 'dat', 'loc', 'per', 'oth']:
        with open(i+'.txt', 'r', encoding='utf8') as f:
            num_line = 0
            for line in f:
                num_line = num_line + 1
            total_num_line[i] = num_line
    print(num_process)
    all_types = ['mea', 'org', 'dat', 'loc', 'per', 'oth']
    all_word_cnt = []
    all_article_cnt = []
    all_max = []
    all_min = []
    batch_size = 300
    for cur_type in all_types:
        print(cur_type)
        print('now reach: ' + str((all_types.index(cur_type)+1)/len(all_types)))
        cnt_word = 0
        cnt_sentence = 0
        word_min = 100000000
        word_max = 0
        with open(cur_type+'.txt', 'r', encoding='utf8') as f:
            article = []
            last_batch = 0
            for line_ind, line in enumerate(f):
                if line_ind%2==0:
                    article.append(line)
                    print('reach: ' + str(line_ind) + 'from' + str(total_num_line[cur_type]))
                    if line_ind % batch_size == 0:
                        if line_ind !=0:
                            pool = mp.Pool(processes=num_process)
                            results = [pool.apply(sertis_tokenizer, args=(re.sub(r'[^ก-๙A-Za-z0-9.-/%:|]','  ',x),saved_model_path)) for x in article]
                            pool.close()
                            pool.join()
                            with open('.\\sertis_data\\'+ cur_type + '_sertis.txt', 'a', encoding = 'utf8') as f_out:
                                with open('.\\sertis_data\\'+ cur_type + '_label_sertis.txt', 'a', encoding = 'utf8') as f_out_label:
                                    for Words, Label, num_word in results:
                                        if Words == -1:
                                            continue
                                        cnt_sentence = cnt_sentence + 1
                                        cnt_word = cnt_word + num_word
                                        word_min = min(word_min,num_word)
                                        word_max = max(word_max,num_word)
                                        f_out.write(Words)
                                        f_out.write('\n')
                                        f_out_label.write(Label)
                                        f_out_label.write('\n')
                                    article = []
            pool = mp.Pool(processes=num_process)
            results = [pool.apply(sertis_tokenizer, args=(re.sub(r'[^ก-๙A-Za-z0-9.-/%:|]','  ',x),saved_model_path)) for x in article]
            pool.close()
            pool.join()
            with open('.\\sertis_data\\'+ cur_type + '_sertis.txt', 'a', encoding = 'utf8') as f_out:
                with open('.\\sertis_data\\'+ cur_type + '_label_sertis.txt', 'a', encoding = 'utf8') as f_out_label:
                    for Words, Label, num_word in results:
                        if Words == -1:
                            continue
                        cnt_sentence = cnt_sentence + 1
                        cnt_word = cnt_word + num_word
                        word_min = min(word_min,num_word)
                        word_max = max(word_max,num_word)
                        f_out.write(Words)
                        f_out.write('\n')
                        f_out_label.write(Label)
                        f_out_label.write('\n')
                    article = []            
                    
        all_word_cnt.append(cnt_word)
        all_article_cnt.append(cnt_sentence)
        all_max.append(word_max)
        all_min.append(word_min)
    all_word_cnt = [str(i) for i in all_word_cnt]
    all_article_cnt = [str(i) for i in all_article_cnt]
    all_max = [str(i) for i in all_max]
    all_min = [str(i) for i in all_min]
    
    with open('.\\sertis_data\\report_sertis.txt', 'w') as f:
        f.write('all_type: ' + ','.join(all_types) + '\n')
        f.write('word_cnt: '+ ','.join(all_word_cnt) + '\n')
        f.write('article_cnt: '+ ','.join(all_article_cnt)+'\n')
        f.write('max: '+ ','.join(all_max) + '\n')
        f.write('min: '+ ','.join(all_min)+'\n')

if __name__ == '__main__':
    main()
# w = sertis_tokenizer(['ฉันกินข้าวปลาอาหาร', 'ฉันไม่กินข้าว ปลา อาหาร'], saved_model_path)
# print(w)