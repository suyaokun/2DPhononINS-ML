# -*- coding: utf-8 -*-
# Author: Yaokun Su
# Date  : 1/27/2023
# Machine Learning project

from utils_models import *
import os


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    label_file = os.path.join('dataset', 'x_record')
    autoencoder_dir = os.path.join('models', 'customized')  # folder to store model

    # model
    autoencoder = FCAE(latent_dim=30)  # choose the antoencoder model (FCAE, FCVAE, CNNAE, CNNVAE)
    is_read_model = True  # used only for continued training
    read_model = '2000'  # only valid if is_read_model==True

    # training parameters
    batch = 64
    learning_rate = 0.1
    momentum = 0.9
    epoch = 500
    weight = 1e-6  # only used for variational models
    save_model_interval = 100 if is_read_model else 200

    # transform the input data for the model
    exp = torch.load(os.path.join(root_dir, 'data', '1.pt')); exp = exp.to(device)
    mask = ~exp.isnan()  # mask is the places where data is valid
    if autoencoder.isfc:
        transform = MaskFlatten(mask)
    else:
        transform = None
    mask_flatten = MaskFlatten(mask)

    train_dataset = Mydata(root_dir, os.path.join('dataset', 'train'), label_file, transform=transform)
    validation_dataset = Mydata(root_dir, os.path.join('dataset', 'validation'), label_file, transform=transform)
    # test_dataset = Mydata(root_dir, 'dataset/test/', label_file, transform=transform)
    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Length of validation dataset: {len(validation_dataset)}')
    # print(f'Length of test dataset: {len(test_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True, drop_last=True)

    os.mkdirs(os.path.join(root_dir, autoencoder_dir), exist_ok=True)
    if is_read_model:
        autoencoder.load_state_dict(torch.load(os.path.join(root_dir, autoencoder_dir, f'model_epoch{read_model}.pth')))
    autoencoder = autoencoder.to(device)
    is_vae = True if hasattr(autoencoder, 'reparameterization') else False

    loss_mse = nn.MSELoss(reduction='mean')
    loss_mse = loss_mse.to(device)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=momentum)

    print('\n',autoencoder,'\n')
    print(f'Batch size: {batch}, Epoch: {epoch}, MSELoss')
    print(f'SGD optim with lr: {learning_rate}, momentum: {momentum}')
    print(f'Tuned from previous model: {read_model if is_read_model else "False"}')
    print(f'KLD weight: {weight}','\n') if is_vae else print('\n')

    total_train_step = 0
    start = int(read_model) if is_read_model else 0
    for i in range(start, epoch + start):
        print(f"---------- Epoch: {i+1} ----------")
        autoencoder.train()
        for data, _ in train_dataloader:
            data = data.to(device)
            code, output = autoencoder(data)
            if autoencoder.isfc:
                loss = loss_mse(output, data)
            else:
                loss = loss_mse(mask_flatten(output), mask_flatten(data))
            if is_vae:
                mse_loss=loss.item()
                kld_loss=kld(code[0],code[1])
                loss = (1-weight) * loss + weight * kld(code[0],code[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 5 == 0:
                if is_vae:
                    print(f"Steps: {total_train_step}, MSE Loss: {(1 - weight) * mse_loss}, KLD: {weight * kld_loss}")
                else:
                    print(f"Steps: {total_train_step}, Loss: {loss.item()}")

        autoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            num_batch=0
            for data, _ in validation_dataloader:
                data = data.to(device)
                code, output = autoencoder(data)
                if autoencoder.isfc:
                    loss = loss_mse(output, data)
                else:
                    loss = loss_mse(mask_flatten(output), mask_flatten(data))
                if is_vae:
                    loss = (1-weight) * loss + weight * kld(code[0],code[1])
                total_val_loss += loss.item()
                num_batch += 1
            total_val_loss = total_val_loss / num_batch
        print(f'Total validation loss: {total_val_loss}')
        if (i+1) % save_model_interval == 0:
            torch.save(autoencoder.state_dict(), os.path.join(root_dir, autoencoder_dir, f'model_epoch{i + 1}.pth'))

