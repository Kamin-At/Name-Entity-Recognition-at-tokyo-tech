from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf
import numpy as np
import os
import multiprocessing as mp
#saved_model_path='saved_model'



def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

def sertis_tokenizer(text, ind, saved_model_path):
    if ind%2:
        return (-1,-1,-1)
    text = text.strip().split('||')
    inputs = [[ThaiWordSegmentLabeller.get_input_labels(i)] for i in text]
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
            #print(j)
            #print(lengths[i])
            y = session.run(g_outputs, feed_dict = {g_inputs: j, g_lengths: lengths[i], g_training: False})
            words = split(text[i], nonzero(y))
            if i % 2:
                label = label + ['1']*len(words)
            else:
                label = label + ['0']*len(words)
            all_words = all_words + words
    print(words)
    #print(all_words)
    #print(label)
    print('finished!!!')
    return ('||'.join(all_words), '||'.join(label), len(label))

def main():
    os.chdir('E:\\min\\Tokyo_tech_exchange\\NER\\data')
    saved_model_path = 'E:\\Coding_projects\\Name-Entity-Recognition-at-tokyo-tech\\saved_model'
    num_process = 4#os.cpu_count()#-1
    
    print(num_process)
    with mp.Pool(processes=num_process) as pool:
        all_types = ['mea', 'org', 'dat', 'loc', 'per', 'oth']
        all_word_cnt = []
        all_article_cnt = []
        all_max = []
        all_min = []
        for cur_type in all_types:
            print(cur_type)
            print('now reach: ' + str((all_types.index(cur_type)+1)/len(all_types)))
            cnt_word = 0
            word_min = 100000000
            word_max = 0
            with open(cur_type+'.txt', 'r', encoding='utf8') as f:
                with open('.\\sertis_data\\'+ cur_type + '_sertis.txt', 'a', encoding = 'utf8') as f_out:
                    with open('.\\sertis_data\\'+ cur_type + '_label_sertis.txt', 'a', encoding = 'utf8') as f_out_label:
                        results = [pool.apply(sertis_tokenizer, args=(x,ind,saved_model_path)) for ind, x in enumerate(f)]
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
                        all_word_cnt.append(cnt_word)
                        all_article_cnt.append(cnt_sentence)
                        all_max.append(word_max)
                        all_min.append(word_min)
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