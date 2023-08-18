import torch.nn as nn
import torch
 
class PercentileOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.PReLU() 
        self.output_lyr = nn.Linear(1,1)
    def forward(self, x):   
        output = self.output_lyr(self.act(x))
        return output

class LSTMModel(nn.Module):
    def __init__(self, n_features, time_steps, num_layers, output_size,dropout,n_percentiles):
        super(LSTMModel, self).__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = 32
        
        # self.lstm = nn.LSTM(n_features, time_steps, num_layers, batch_first=True,dropout=dropout) 
        self.embedding = nn.Linear(time_steps,self.hidden_size)  
        self.lstm = nn.LSTM(n_features, self.hidden_size, num_layers, batch_first=True,dropout=dropout) 
        # self.fc = nn.Linear(time_steps, output_size)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.OutputLayer = nn.ModuleList(PercentileOutputLayer() for _ in range(n_percentiles))
    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.time_steps).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.time_steps).to(x.device) 
        x = x.permute(0,2,1)
        x = self.embedding(x)
        x = x.permute(0,2,1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.fc(out[:, -1, :])  # Using the last time step's output for prediction 
        out = torch.reshape(out,(-1,self.output_size,1))
        
        output = []
        for i, lyr in enumerate(self.OutputLayer): 
            if i==0:
                output = lyr(out)
            else:
                output = torch.cat([output,lyr(out)], axis=2) 
        
        return output

if __name__=='__main__':
    # Example usage
    n_features = 12
    time_steps = 13
    num_layers = 2
    output_size = 24
    n_percentiles = 11
    dropout=0.3
    model = LSTMModel(n_features, time_steps, num_layers, output_size,dropout,n_percentiles)
    print(model) 
     
    input1 = torch.randn([10,13,12]) ### batch size * time steps * n_featurues
    out = model(input1)
    print(f"output shape: {out.shape}")