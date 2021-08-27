import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision
import torch.nn as nn
import math
import tqdm
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler
from bs4 import BeautifulSoup
device = 'cuda'

#Function to make word2vec dict. for our dataset
def find_word2vec_read_xml(path):
    with open(path,'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, 'xml')
    file_name = Bs_data.filename.contents[0]
    whole_class = []
    for i in Bs_data.findAll('object'):
        class_ = Bs_data.object.find('name').contents[0]
        whole_class.append(class_)
    return whole_class

#Function to read xml file
def read_xml(path,word2vec):
    with open(path,'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, 'xml')
    file_name = Bs_data.filename.contents[0]
    whole_class = []
    whole_bbox = []
    whole_dif = []
    for i in Bs_data.findAll('object'):
        class_ = word2vec[Bs_data.object.find('name').contents[0]]
        x1 = int(Bs_data.object.xmin.contents[0])
        y1 = int(Bs_data.object.ymin.contents[0])
        x2 = int(Bs_data.object.xmax.contents[0])
        y2 = int(Bs_data.object.ymax.contents[0])
        difficulty = int(Bs_data.object.difficult.contents[0])
        bbox = [x1,y1,x2,y2]
        whole_class.append(class_)
        whole_bbox.append(bbox)
        whole_dif.append(difficulty)
    return file_name,whole_class,whole_bbox, whole_dif

#Function to read image
def image_reader(img_path, bbox, train_mode, transform):
    img = cv2.imread(img_path)
    w,h,_ = img.shape
    x1,y1,x2,y2 = bbox
    #img = cv2.resize(img,(224,224))
    #x1 = int(x1/h*224)
    #y1 = int(y1/w*224)
    #x2 = int(x2/h*224)
    #y2 = int(y2/w*224)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    img = img[y1:y2,x1:x2,:]
    if train_mode:
        img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    img = img.transpose(2,0,1)
    img = torch.tensor(img).float()
    img = transform(img)
    return img

#Class to build custom dataset for dataloader
class face_dataset(Dataset):
    def __init__(self, X, y, train_mode):
        super(face_dataset,self).__init__()
        self.X = X
        self.y = y
        assert len(X) == len(y)
        self.train_mode = train_mode
        if train_mode:
            self.transform = transforms.Compose([
                 transforms.RandomCrop((224,224)),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
        else:
            self.transform = transforms.Compose([
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        img_path, bbox = self.X[idx]
        image = image_reader(img_path, bbox, self.train_mode,self.transform)
        label = self.y[idx]
        return image, label
    
#Our model using pretrained mobilenetv2
class EmotionClassifier(nn.Module):
    def __init__(self,class_number):
        super(EmotionClassifier,self).__init__()
        self.base_model = torchvision.models.mobilenet_v2(pretrained = True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.gap = nn.AvgPool2d((7,7))
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(1280,class_number)
    def forward(self,image):
        x = self.base_model(image)
        x = self.gap(x)
        x = self.fc_layer(self.flatten(x))
        return x
    
#Schduler
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
#Function to train our model
def train(model,trainloader,valloader,optimizer,scheduler,criteria, save_after,epochs=100,best_accuracy = 0, best_model = None):
    model.train()
    loader_len = len(trainloader)
    gradient_accumulation_steps = loader_len
    loss_curve = []
    for epoch in range(epochs):
        total_loss = 0
        for i,data in tqdm.notebook.tqdm(enumerate(trainloader,1)):
            input_,target = data
            input_ = input_.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred_y = model(input_.float())
            loss = criteria(pred_y,target)
            loss = loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1) % gradient_accumulation_steps ==0 and epoch > save_after:
                print(f'{epoch}th epoch {i} iteration loss avg :',total_loss)
                loss_curve.append(total_loss)
                accuracy = test(model,valloader)
                print(accuracy)
                if accuracy > best_accuracy:
                    #print(accuracy)
                    best_accuracy = accuracy
                    best_model = deepcopy(model)
                model.train()
        scheduler.step()
    return best_accuracy, best_model, loss_curve

#Function to evaluate our model using testset(valset)
def test(model,testloader):
    model.eval()
    total_example = 0
    correct_example = 0
    for _,data in enumerate(testloader):
        input_,target = data
        input_ = input_.to(device)
        target = target.to(device)
        pred_y = model(input_.float())
        #print((pred_y.argmax(axis=1)))
        correct_example += int((pred_y.argmax(axis=1)==target).type(torch.uint8).sum())
        total_example += len(target)
        del input_,target,pred_y
    accuracy = correct_example/total_example
    return accuracy
