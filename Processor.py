
from utils_prog import *

import time
import datetime
import torch.nn as nn
import yaml
import torchviz
import math
import matplotlib.pyplot as plt


class Processor():
    def __init__(self, args):
        self.args=args
        torch.manual_seed(self.args.seed)

        Dataloader=DataLoader
        self.dataloader = Dataloader(self.args)

        if self.args.phase == 'train':
            self.train_2LSTM()
        elif self.args.phase == 'test':
            self.test_2LSTM()


    def set_optimizer(self,net):
        self.optimizer = torch.optim.Adam(net.parameters(),lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduce=False)

    def train_2LSTM(self):
        model = import_class(self.args.first_train_model)
        self.net = model(self.args)

        self.set_optimizer(self.net)
        self.load_model(self.args.first_train_model,self.net)
        print(self.net)

        Train_loss = []
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):
            train_error, self.train_student_index = self.train_1LSTM_epoch(epoch)
            Train_loss.append(train_error)
            test_error,self.test_student_index = self.test_1LSTM_epoch(epoch)
            self.save_model(epoch,self.args.first_train_model,self.net)

        second_model = import_class(self.args.second_train_model)
        self.second_net = second_model(self.args)
        self.set_optimizer(self.second_net)
        self.load_model(self.args.second_train_model,self.second_net)

        for epoch in range(self.args.num_epochs):
            train_error = self.train_2LSTM_epoch(epoch,self.train_student_index)
            Train_loss.append(train_error)
            test_error = self.test_2LSTM_epoch(epoch,self.test_student_index)
            self.save_model(epoch,self.args.second_train_model,self.second_net)

    def train_1LSTM_epoch(self, epoch):
        start = time.time()
        loss_epoch = 0
        grades = []
        preds = []
        student_labels = []
        student_index = []

        for batch in range(self.dataloader.trainbatchnums):
            batch_student, Raw_grade, grade = self.dataloader.get_train_batch(batch)
            inputs = Raw_grade, grade
            inputs = tuple([torch.Tensor(i) for i in inputs])

            loss = torch.zeros(1)
            Raw_grade, grade = inputs
            inputs_fw = Raw_grade

            self.net.zero_grad()
            outputs = self.net.forward(inputs_fw)

            grades += [[i.item()] for i in grade[:, 0]]
            preds += [[i.item()] for i in outputs]
            student_labels += [[i] for i in batch_student]
            student_index.append([1 if i[0]<0.4 else 0 for i in outputs])


            loss = torch.sum(self.criterion(outputs, grade[:, 0]))
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()

            error = torch.sum(self.criterion(outputs, grade[:, 0]))
            loss_epoch = loss_epoch + error.item()
            if batch % 10 == 0:
                print(self.args.program, self.args.dataset, ' 1st model ', self.args.first_train_model,
                      ' train-{}/{} (epoch {}), train_loss = {:.5f}'
                      .format(batch, self.dataloader.trainbatchnums, epoch, loss.item()))
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        end = time.time()

        res = pd.DataFrame(data=np.concatenate([student_labels, grades, preds], axis=1),columns=['HashID', 'tar', 'out'])
        res.set_index('HashID', inplace=True)

        res.to_csv(os.path.join(self.args.save_dir,'train_res_csv')

        return train_loss_epoch,student_index

    def train_2LSTM_epoch(self, epoch, student_index):
        start = time.time()
        loss_epoch = 0
        grades = []
        preds = []
        student_labels = []
        sel_student = []
        for batch in range(self.dataloader.trainbatchnums):
            batch_student, Raw_grade, grade = self.dataloader.get_train_batch(batch)
            inputs = Raw_grade, grade
            inputs = tuple([torch.Tensor(i) for i in inputs])
            batch_student_index = torch.Tensor(student_index[batch])

            loss = torch.zeros(1)
            Raw_grade, grade = inputs
            inputs_fw = Raw_grade, Nei_raw_grade

            self.second_net.zero_grad()
            outputs = self.second_net.forward(inputs_fw)
            grades += [[i.item()] for i in grade[:, 0]]
            preds += [[i.item()] for i in outputs]
            student_labels += [[i] for i in batch_student]
            sel_student += [i.item() for i in batch_student_index]

            selected_pred = torch.mul(outputs, batch_student_index)
            selected_grade = torch.mul(grade[:, 0], batch_student_index)
 
            ar = torch.Tensor([[0 if i > 0.4 else 1 for i in j] for j in selected_grade])
            weight = torch.add(torch.mul(ar, self.args.lossweight), 1)
            error = self.criterion(selected_pred, selected_grade)
            loss = torch.sum(torch.mul(error, weight))

            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.second_net.parameters(), 1)
            self.optimizer.step()

            error = torch.sum(self.criterion(outputs, grade[:, 0]))
            loss_epoch = loss_epoch + error.item()
            if batch % 10 == 0:
                print(self.args.program, self.args.dataset, self.args.second_train_model,
                      ' train-{}/{} (epoch {}), train_loss = {:.5f}'
                      .format(batch, self.dataloader.trainbatchnums, epoch, loss.item()))
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        end = time.time()


        res_path = os.path.join(self.args.save_dir,'train_res.csv')
        res = pd.read_csv(res_path,index_col=0)
        res['out2'] = [i[0] for i in preds]
        res['labels'] = sel_student

        res.to_csv(res_path)
        return train_loss_epoch

    def test_1LSTM_epoch(self, epoch):

        start0 = time.time()
        loss_epoch = 0
        grades = []
        preds = []
        student_labels = []
        student_index = []
        for batch in range(self.dataloader.testbatchnums):
            batch_student, Raw_grade, grade = self.dataloader.get_test_batch(batch)
            inputs = Raw_grade, Nei_raw_grade, grade
            inputs = tuple([torch.Tensor(i) for i in inputs])

            loss = torch.zeros(1)
            Raw_grade, grade = inputs
            inputs_fw = Raw_grade

            forward = self.net.forward
            outputs = forward(inputs_fw)

            grades += [[i.item()] for i in grade[:, 0]]
            preds += [[i.item()] for i in outputs]
            student_labels += [[i] for i in batch_student]
            student_index.append([1 if i[0] < 0.4 else 0 for i in outputs])

            error = torch.sum(torch.sum(self.criterion(outputs, grade[:, 0])))
            loss_epoch = loss_epoch + error.item()
        final_error = loss_epoch / self.dataloader.testbatchnums

        res = pd.DataFrame(data=np.concatenate([student_labels, grades, preds], axis=1),
                           columns=['HashID', 'tar', 'out'])
        res.set_index('HashID', inplace=True)

        res.to_csv(os.path.join(self.args.save_dir,'test_res.csv')

        return final_error,student_index

    def test_2LSTM_epoch(self, epoch,student_index):

        start0 = time.time()
        loss_epoch = 0
        grades = []
        preds = []
        student_labels = []
        sel_students = []
        for batch in range(self.dataloader.testbatchnums):
            batch_student, Raw_grade, grade = self.dataloader.get_test_batch(batch)
            inputs = Raw_grade, Nei_raw_grade, grade
            inputs = tuple([torch.Tensor(i) for i in inputs])
            
            loss = torch.zeros(1)
            Raw_grade, grade = inputs
            inputs_fw = Raw_grade

            forward = self.second_net.forward

            outputs = forward(inputs_fw)

            grades += [[i.item()] for i in grade[:, 0]]
            preds += [[i.item()] for i in outputs]
            student_labels += [[i] for i in batch_student]
            sel_students += [i for i in student_index[batch]]

            gates[str(batch)]['student_labels'] = batch_student
            gates[str(batch)]['it'] = it
            gates[str(batch)]['ft'] = ft

            error = torch.sum(torch.sum(self.criterion(outputs, grade[:, 0])))
            loss_epoch = loss_epoch + error.item()
        final_error = loss_epoch / self.dataloader.testbatchnums
        
        res_path = os.path.join(self.args.save_dir,'test_res.csv')
        res = pd.read_csv(res_path,index_col=0)
        res['out2'] = [i[0] for i in preds]
        res['labels'] = sel_students
        res.to_csv(res_path)

        return final_error

