import sys,os,argparse,time
import numpy as np
import torch
import utils
from datetime import datetime
from torchvision import datasets,transforms
import random
from tensorflow.keras.datasets import cifar10
tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
parser.add_argument('--device',             default='cuda:0',       type=str,   help='gpu id')
parser.add_argument('--experiment',         default='cifar',       type=str,  
                                            choices=['mnist2','mnist5','pmnist','cifar','mixture'])
parser.add_argument('--approach',           default='ucb',            type=str,   help='acl')
parser.add_argument('--data_path',          default='../data/',            type=str,   help='gpu id')

# Training parameters
parser.add_argument('--output',             default='',                     type=str,   help='')
parser.add_argument('--checkpoint_dir',     default='../checkpoints/',    type=str,   help='')
parser.add_argument('--nepochs',            default=200,            type=int,   help='')
parser.add_argument('--sbatch',             default=64,             type=int,   help='')
parser.add_argument('--lr',                 default=0.3,           type=float, help='')  # use 0.3 for non-mnist datasets
parser.add_argument('--nlayers',            default=1,              type=int,   help='')
parser.add_argument('--nhid',               default=1200,           type=int, help='')
parser.add_argument('--parameter',          default='',             type=str,   help='')

# UCB HYPER-PARAMETERS
parser.add_argument('--samples',            default='10',           type=int,     help='Number of Monte Carlo samples')
parser.add_argument('--rho',                default='-3',           type=float,   help='Initial rho')
parser.add_argument('--sig1',               default='0.0',          type=float,   help='STD foor the 1st prior pdf in scaled mixture Gaussian')
parser.add_argument('--sig2',               default='6.0',          type=float,   help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
parser.add_argument('--pi',                 default='0.25',         type=float,   help='weighting factor for prior')
parser.add_argument('--arch',               default='mlp',          type=str,     help='Bayesian Neural Network architecture')

parser.add_argument('--resume',          default='no',            type=str,   help='resume?')
parser.add_argument('--sti',             default=0,               type=int,   help='starting task?')

args=parser.parse_args()
utils.print_arguments(args)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print('Using device:', args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach=='ucb':
    from approaches import ucb as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5':
    from networks import mlp_ucb as network
else:
    from networks import resnet_ucb as network


########################################################################################################################
print()
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

taskcla = [(0,10)]
# taskcla = []
# for i in range(12):
#     taskcla.append((i, 10))

inputsize=[3,32,32]
# Load
print('Load data...')
# data,taskcla,inputsize=dataloader.get(data_path=args.data_path, seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)
args.num_tasks=len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

# Inits
print('Inits...')
model=network.Net(args).to(args.device)


print('-'*100)
appr=approach.Appr(model,args=args)
print('-'*100)

# args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print('-'*100)

if args.resume == 'yes':
    checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=args.device)
else:
    args.sti = 0

(trainX, trainY), (testX, testY) =  cifar10.load_data()
trainY = trainY.flatten()
trainY = [int(v) for v in trainY]
init_ids = np.loadtxt("drive/My Drive/marketplace/UCB/src/init.csv")
init_ids = [int(v) for v in init_ids]
label_ids = [[] for _ in range(10)]
for i in range(len(trainY)):
    if i not in init_ids:
        label_ids[trainY[i]].append(i)

scores = [0]*10

# read all data
train_x = []
train_y = []
test_x = []
test_y = []

path = os.path.join(args.data_path, 'binary_cifar')
if not os.path.isdir(path):
    os.makedirs(path)

mean=[x/255 for x in [125.3,123.0,113.9]]
std=[x/255 for x in [63.0,62.1,66.7]]


train_dataset=datasets.CIFAR10(args.data_path,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False)
for image,target in train_loader:
    train_x.append(image)
    train_y.append(target)

test_dataset=datasets.CIFAR10(args.data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
for image,target in test_loader:
    test_x.append(image)
    test_y.append(target)
test_x = torch.stack(test_x).view(-1,inputsize[0],inputsize[1],inputsize[2])
test_y = torch.LongTensor(np.array(test_y,dtype=int)).view(-1)

acquired_ids = init_ids.copy()

partial_train_x = [train_x[i] for i in acquired_ids]
partial_train_y = [train_y[i] for i in acquired_ids]
partial_train_x = torch.stack(partial_train_x).view(-1,inputsize[0],inputsize[1],inputsize[2])
partial_train_y = torch.LongTensor(np.array(partial_train_y,dtype=int)).view(-1)

t=0
print("initial training...")
appr.train(0,partial_train_x,partial_train_y,test_x,test_y,is_first_itr=True)
test_loss,init_acc=appr.eval(t,test_x,test_y)
t+=1
torch.save(appr.model, './init_model')

cost_file = open("drive/My Drive/marketplace/UCB/time_cost.txt", 'a')
batch_size = 500
budget = 40000
while len(acquired_ids) < budget:
    
    if t == 1:
        for i in range(10):
            
            appr.model = torch.load('./init_model')
            newly_acquired_ids = random.sample(label_ids[i], batch_size)
            acquired_ids.extend(newly_acquired_ids)
            previous_samples = random.sample(init_ids, batch_size)
            newly_acquired_ids.extend(previous_samples)
            label_ids[i] = [v for v in label_ids[i] if v not in newly_acquired_ids]
            
            temp_train_x = [train_x[i] for i in newly_acquired_ids]
            temp_train_y = [train_y[i] for i in newly_acquired_ids]
            temp_train_x=torch.stack(temp_train_x).view(-1,inputsize[0],inputsize[1],inputsize[2])
            temp_train_y=torch.LongTensor(np.array(temp_train_y,dtype=int)).view(-1)
            appr.train(0,temp_train_x,temp_train_y,test_x,test_y)
            
            test_loss,test_acc=appr.eval(0,test_x,test_y)
            t+=1
            scores[i] = test_acc - init_acc
            print(str(t) + ": " + str(test_acc))
            
        partial_train_x = [train_x[i] for i in acquired_ids]
        partial_train_y = [train_y[i] for i in acquired_ids]
        partial_train_x=torch.stack(partial_train_x).view(-1,inputsize[0],inputsize[1],inputsize[2])
        partial_train_y=torch.LongTensor(np.array(partial_train_y,dtype=int)).view(-1)
        appr.train(0,partial_train_x,partial_train_y,test_x,test_y)
        # t+=1
        
        test_loss,init_acc=appr.eval(0,test_x,test_y)
        print(str(t) + ": " + str(test_acc))

    selected_label = scores.index(max(scores))
    if len(label_ids[selected_label]) < batch_size:
        newly_acquired_ids = label_ids[selected_label]
        scores[selected_label] = -2
    else:
        newly_acquired_ids = random.sample(label_ids[selected_label], batch_size)
    label_ids[selected_label] = [v for v in label_ids[selected_label] if v not in newly_acquired_ids]
    previous_samples = random.sample(acquired_ids, batch_size)
    acquired_ids.extend(newly_acquired_ids)
    newly_acquired_ids.extend(previous_samples)
    temp_train_x = [train_x[i] for i in newly_acquired_ids]
    temp_train_y = [train_y[i] for i in newly_acquired_ids]
    temp_train_x=torch.stack(temp_train_x).view(-1,inputsize[0],inputsize[1],inputsize[2])
    temp_train_y=torch.LongTensor(np.array(temp_train_y,dtype=int)).view(-1)
    appr.train(0,temp_train_x,temp_train_y,test_x,test_y)
    t+=1
    test_loss,test_acc=appr.eval(0,test_x,test_y)
    print(str(t) + ": " + str(test_acc))
    scores[selected_label] = test_acc - init_acc
    init_acc = test_acc
    total_time = time.time()-tstart
    cost_file.write(str(t) + ',' + str(total_time) + "\n")
    write_file = open("drive/My Drive/marketplace/UCB/budget-" + str(t) + ".txt", 'w')
    write_file.write(' '.join([str(v) for v in acquired_ids]))
    write_file.close()
    cost_file.flush()
cost_file.close()

# Loop tasks
# acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
# lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
# for t,ncla in taskcla[args.sti:]:
#     print('*'*100)
#     print('Task {:2d} ({:s})'.format(t,data[t]['name']))
#     print('*'*100)

#     if args.approach == 'joint':
#         # Get data. We do not put it to GPU
#         if t==0:
#             xtrain=data[t]['train']['x']
#             ytrain=data[t]['train']['y']
#             xvalid=data[t]['valid']['x']
#             yvalid=data[t]['valid']['y']
#             task_t=t*torch.ones(xtrain.size(0)).int()
#             task_v=t*torch.ones(xvalid.size(0)).int()
#             task=[task_t,task_v]
#         else:
#             xtrain=torch.cat((xtrain,data[t]['train']['x']))
#             ytrain=torch.cat((ytrain,data[t]['train']['y']))
#             xvalid=torch.cat((xvalid,data[t]['valid']['x']))
#             yvalid=torch.cat((yvalid,data[t]['valid']['y']))
#             task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
#             task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
#             task=[task_t,task_v]
#     else:
#         # Get data
#         xtrain=data[t]['train']['x'].to(args.device)
#         ytrain=data[t]['train']['y'].to(args.device)
#         xvalid=data[t]['valid']['x'].to(args.device)
#         yvalid=data[t]['valid']['y'].to(args.device)
#         task=t

#     # Train
#     appr.train(task,xtrain,ytrain,xvalid,yvalid)
#     print('-'*100)

#     # Test
#     for u in range(t+1):
#         xtest=data[u]['test']['x'].to(args.device)
#         ytest=data[u]['test']['y'].to(args.device)
#         test_loss,test_acc=appr.eval(u,xtest,ytest,debug=True)
#         print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
#         acc[t,u]=test_acc
#         lss[t,u]=test_loss

#     # Save
#     print('Save at '+args.checkpoint)
#     np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}.txt'.format(args.experiment,args.approach,args.seed)),acc,'%.5f')


# utils.print_log_acc_bwt(args, acc, lss)
# print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

