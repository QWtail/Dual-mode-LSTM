'''
Dataloader
Author: Wei QIU
Date: 20/07/2021
'''
import torch
import gc
import os
import pickle
import numpy as np
import pandas as pd
import scipy.linalg as sl
import random
import scipy.stats
import copy
import math
import smogn



class DataLoader():
    def __init__(self, args): #

        #for test
        # self.course = 'EE2010'
 
        self.args=args

        self.course = self.args.dataset

        self.train_file_path = os.path.join('./data/Sample/trainX.csv')
        self.train_data_file = os.path.join('./data/Sample/train_data.cpkl')
        self.train_batch_cache = os.path.join('./data/Sample//train_batch_cache.cpkl')
        
        self.test_file_path = os.path.join('./data/Sample/testX.csv')
        self.test_data_file = os.path.join('./data/Sample/test_data.cpkl')
        self.test_batch_cache = os.path.join('./data/Sample/test_batch_cache.cpkl')

        if not os.path.exists(self.train_batch_cache):

            print("Creating pre-processed data from raw data.")
            self.data_preprocess('train')
            self.data_preprocess('test')
            print("Done.")

            # Load the processed data from the pickle file
            print("Preparing data batches.")
            self.StudGrade_dict = self.load_dict(self.train_data_file)
            self.dataProcess('train')
        if not os.path.exists(self.test_batch_cache):
            self.test_StudGrade_dict = self.load_dict(self.test_data_file)
            self.dataProcess('test')

        self.trainbatch, self.trainbatchnums, self.train_batch_student = self.load_cache(self.train_batch_cache)
        self.testbatch, self.testbatchnums,self.test_batch_student = self.load_cache(self.test_batch_cache)
        print("Done.")

        print(self.args.program, self.course,' Total number of training batches:', self.trainbatchnums)
        print(self.args.program, self.course,' Total number of test batches:', self.testbatchnums)


    def data_preprocess(self,setname):
        if setname=='train':
            data_file=self.train_data_file
            file_path = self.train_file_path
            label_ = 'train'
        else:
            data_file=self.test_data_file
            file_path = self.test_file_path
            label_ = 'test'
        # Load the data from the csv file
        if not os.path.exists(data_file):
            # Load the data from the csv file
            data = pd.read_csv(file_path,index_col=0)
            Studlist = data.columns.tolist()

            StudGrade_dict = {}
            for stud_ in Studlist:

                StudGrade_dict[stud_] = {}
                grade = [[int(stud__.split('/')[-1])/100]]
                stud_data = data[stud__]
                StudGrade_dict[stud_]['raw_grade'] = stud_data
                StudGrade_dict[stud_]['grade'] = grade


            f = open(data_file, "wb")
            pickle.dump((StudGrade_dict), f, protocol=2)
            f.close()

    def load_dict(self,data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        Neig_dict=raw_data[0]
        StudGrade_dict=raw_data[1]
        return Neig_dict,StudGrade_dict
    
    def load_cache(self,data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data
    
    def dataProcess(self,setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        '''
        if setname=='train':
            StudGrade_dict=self.StudGrade_dict
            cachefile=self.train_batch_cache

            val_index = []
            train_index = [*StudGrade_dict]

        else:
            StudGrade_dict=self.test_StudGrade_dict
            cachefile = self.test_batch_cache

            val_index = []
            train_index = [*StudGrade_dict]

        trainbatch,train_batch_student = self.generate_batch_data(StudGrade_dict,train_index)

        trainbatchnums=len(train_batch_student)

        f = open(cachefile, "wb")
        pickle.dump(( trainbatch, trainbatchnums,train_batch_student), f, protocol=2)
        f.close()

    def generate_batch_data(self,StudGrade_dict,data_index):

        num_batch = int(len(data_index)//self.args.batch_size)

        batch_student = []
        for i in range(num_batch):
            batch_student.append(data_index[i*self.args.batch_size:(i+1)*self.args.batch_size])
        if len(data_index)%self.args.batch_size != 0:
            batch_student.append(data_index[num_batch * self.args.batch_size:])

        trainbatch = {}
        for stud_ in data_index:

            trainbatch[stud_] = {}
            trainbatch[stud_]['Grade'] = StudGrade_dict[stud_]['grade']
            trainbatch[stud_]['raw_grade'] = StudGrade_dict[stud_]['raw_grade']
        return trainbatch,batch_student

    def get_train_batch(self,idx):
        # key_ = [*self.trainbatch][idx]
        batch_student = self.train_batch_student[idx]
        Raw_grade = []
        grade = []
        for i in range(self.args.seq_length):
            raw_grade = []
            for std_ in batch_student:
                batch_data = self.trainbatch[std_]
                stud_org = np.reshape(np.array(batch_data['raw_grade']), (-1, self.args.input_size))
                raw_grade.append(stud_org[i])
            Raw_grade.append(np.nan_to_num(raw_grade))

        for std_ in batch_student:
            batch_data = self.trainbatch[std_]
            grade.append(batch_data['Grade'])

        return batch_student, Raw_grade, grade

    def get_test_batch(self,idx):
        # key_ = [*self.testbatch][idx]
        batch_student = self.test_batch_student[idx]
        Raw_grade = []
        grade = []
        for i in range(self.args.seq_length):
            raw_grade = []
            for std_ in batch_student:
                batch_data = self.testbatch[std_]
                stud_org = np.reshape(np.array(batch_data['raw_grade']), (-1, self.args.input_size))
                raw_grade.append(stud_org[i])
            Raw_grade.append(np.nan_to_num(raw_grade))
            
        for std_ in batch_student:
            batch_data = self.testbatch[std_]
            grade.append(batch_data['Grade'])
            
        return batch_student, Raw_grade, grade

def import_class(name):
    mod = __import__('models')
    print(' model mode:', name)
    mod = getattr(mod, name)
    return mod
