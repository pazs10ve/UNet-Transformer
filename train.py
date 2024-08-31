import matplotlib.pyplot as plt
import torch
import time
import logging
import os


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    training_loss = 0
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        #print(f'{batch_idx+1}/{len(train_loader)}')
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss /= len(train_loader)
    return training_loss



def eval_one_epoch(model, val_loader, criterion, device):
    validation_loss = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            #print(f'{batch_idx+1}/{len(val_loader)}')
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            validation_loss += loss.item()
        validation_loss /= len(val_loader)
    return validation_loss


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir='model_checkpoints'):
    train_losses = []
    val_losses = []

    log_file = 'training_log.log'
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        log_message = (f'Epoch {epoch+1}/{num_epochs} --> Training Loss : {train_loss:.4f}, '
                       f'Validation Loss : {val_loss:.4f}, Time: {epoch_time:.2f} seconds')
        print(log_message)
        
        logging.info(log_message)
        
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            logging.info(f'Model saved at {model_save_path}')

    return train_losses, val_losses, num_epochs


    


def plot_history(train_losses, val_losses):
    plt.figure(figsize=(10, 20))
    plt.plot(train_losses, colour = 'red')
    plt.plot(val_losses, color = 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Soft Dice Loss')
    plt.tight_layout()
    plt.show()