from model import LSTMModel
from loss import pinball_regloss, pinball_loss
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
 
def train(train_loader,valX,valy,model,optimizer,epochs,mdl_name): 
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        trainloss = []
        valloss = []
        for x, targets in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(x)
            loss,_ = criterion(outputs,targets) 
            trainloss.append(loss.detach().cpu().item())
            loss.backward() 
            optimizer.step()
            
            with torch.no_grad():
                model.eval()
                val_out = model(valX)
                vloss,_ = criterion(val_out,valy) 
                valloss.append(vloss.detach().cpu().item())
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {sum(trainloss)/len(trainloss):.4f}')
        print(f'Validation Loss: {sum(valloss)/len(valloss):.4f}')
        train_loss.append(sum(trainloss)/len(trainloss)) 
        val_loss.append(sum(valloss)/len(valloss)) 
        ### save model
        if (epoch+1)%50==0:
            model_path = f'experiments/{mdl_name}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'model saved as {model_path}') 
    print("Training finished!")
    
    return train_loss, val_loss, model
 
if __name__=='__main__':   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected device: {device}') 
    
    experiment_name = "NSRDB_lag0_lt0_pred1d_daytime_only"
    
    ### load data
    trainData = np.load('data/'+experiment_name+'/2017_NSRDB_lag0_lt0_pred1d_daytime_only_TRAIN.npz')
    testData = np.load('data/'+experiment_name+'/2018_NSRDB_lag0_lt0_pred1d_daytime_only_TEST.npz')
    
    trainX = trainData['X']
    trainy = trainData['y']
    testX = testData['X']
    testy = testData['y']
    
    trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.2, random_state=42) 
    batch_size = 32
    train_dataset = TensorDataset(torch.Tensor(trainX).to(device),torch.Tensor(trainy).to(device)) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valX = torch.Tensor(valX).to(device)
    valy = torch.Tensor(valy).to(device)
    
    strategies = ['none','narrow','wide']
    for strategy in strategies:
        ### define model parameters
        n_features = trainX.shape[2]
        time_steps = trainX.shape[1]
        num_layers = 2
        output_size = trainy.shape[1]
        dropout = 0.3
        n_percentiles = 11
        model = LSTMModel(n_features, time_steps, num_layers, output_size,dropout,n_percentiles)
        
        model.to(device)
        
        ### training
        epochs = 400
        learning_rate = 0.001
        criterion  = pinball_loss(strategy=strategy)
        optimizer = optim.Adam(model.parameters(),learning_rate,weight_decay=0.01) 
        mdl_name = f'{experiment_name}_{strategy}_pinball_loss_hidden_size_32'
        train_loss, val_loss, model = train(train_loader,valX,valy,model, optimizer, epochs,mdl_name)
        
        ### evaluation
        model.eval()
        with torch.no_grad():
            testX = torch.Tensor(testX).to(device)
            predictions = model(testX) 
        savepath = 'experiments/predictions_'+mdl_name+'.npz'
        np.savez(savepath, predictions=predictions.detach().cpu().numpy(),
                 targets=testy)
        print(f'predictions saved in {savepath}')
         
        ### plot training and validation loss
        plt.plot(list(range(1,epochs+1)),train_loss,label='Training loss')
        plt.plot(list(range(1,epochs+1)),val_loss,label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Pinball regularized Loss')
        plt.legend()
        plt.savefig(f'experiments/{mdl_name}.png',dpi=300)
        plt.show() 
        plt.close()
