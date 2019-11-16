"""
What should serve as a baseline model. Going to keep all of the code in here for now. 

This is going to be how the data is loaded, and then modeled using Pytorch. 

"""

#import dependencies
import numpy as np 
import torch
import torch.nn as nn
import data_loader
from models import CNN


#define model hyperparameters
epochs = 15
batch_size = 32 
shuffle = False #Since this data is very obviously time dependent, training and validation data should not be shuffled. 
learning_rate = 1e-3
weight_decay  = 1e-3
fout = 'train_val_loss.txt' #where to output training and validation loss. 


f_best_model = 'BestModel_placeholder_%.1e_%.1e_%d.pt'\
               %(learning_rate, weight_decay, batch_size)  #save best model here. 


#import data
train_loader, val_loader = data_loader.load_trainval_data(batch_size = 32)




#instantiate the model and quickly state how many parameters it has 
model =  CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print('Total number of parameters in the network: %d'%total_params)


# define the loss and optimizer
criterion = nn.MSELoss()  #mean squared error is good for regression :)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)


########################################################################################

# create the training, validation and testing datasets


print("data is loaded in ! ")



#for weight decay 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, 
                                                       patience=10, verbose=True)


model.eval()   #what's this do 

for epoch in range(epochs):

    # TRAIN
    model.train()
    count, loss_train = 0, 0.0
    #loader contains dark matter cube, dm subhalo cube, subhalo + galaxy cube, dm + galaxy mask, and coords. 
    for maps,  params_true in train_loader: #currently ignoring subhalo structure. 
        maps = maps.to(device).float()
        params_true = params_true.to(device).float()
        # Forward Pass
        optimizer.zero_grad()
        params_pred = model(maps)
        loss = criterion(params_pred, params_true)
        loss_train += loss.cpu().detach().numpy()
        
        # Backward Prop
        loss.backward()
        optimizer.step()
        
        count += 1
    loss_train /= count
    
    # VALID
    model.eval() 
    count, loss_valid = 0, 0.0
    with torch.no_grad():
        for maps, params_true  in val_loader:
            
            maps = maps.to(device).float()
            params_true = params_true.to(device).float()
            params_pred = model(maps)
            error    = criterion(params_pred, params_true)   
            loss_valid += error.cpu().numpy()
            count += 1
    loss_valid /= count
    # Save Best Model
    print("Best loss: ", best_loss) 
    if loss_valid<best_loss:
        best_loss = loss_valid
        torch.save(model.state_dict(), f_best_model)
        print("Saving best model placeholder")
        print('%03d %.4e %.4e (saving)'\
              %(epoch, loss_valid, loss_test))
    else:
        print('%03d %.4e %.4e'%(epoch, loss_train, loss_valid))
    
    # update learning rate
    scheduler.step(loss_valid)
    # save results to file
    f = open(fout, 'a')
    f.write('%d %.4e %.4e\n'%(epoch, loss_train, loss_valid))
    f.close()
