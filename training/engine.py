
def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    train_labels, train_preds = [], []
    train_loss = 0.0
    
    model.train()
    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        y_preds = model(images)
        
        loss = criterion(y_preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        
        train_preds.append(y_preds.softmax(1).to('cpu').detach().numpy())
        train_labels.append(labels.to('cpu').numpy())
        
    return train_loss / len(data_loader), np.concatenate(train_preds), np.concatenate(train_labels)
    

def valid_epoch(model, data_loader, criterion, device):
    valid_loss = 0.0
    
    model.eval()
    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            y_preds = model(images)
        
        loss = criterion(y_preds, labels)
        valid_loss += loss.item()
        
        #valid_preds.append(y_preds.softmax(1).to('cpu').detach().numpy())
        
    return valid_loss / len(data_loader)#, np.concatenate(valid_preds)
