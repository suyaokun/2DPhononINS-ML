# -*- coding: utf-8 -*-
# Author: Yaokun Su
# Date  : 2/2/2023
# Machine Learning project

import torchvision
from utils_models import *
import os


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    label_file = os.path.join('dataset', 'x_record')
    autoencoder_dir = os.path.join('models', 'customized')  # folder to read the autoencoder model
    predictor_dir = os.path.join(autoencoder_dir, 'predictor', '1')  # folder to store the regressor model

    # model
    autoencoder = FCAE(latent_dim=30)  # choose the antoencoder model
    fc_predictor = CodeToFCs_NNN(latent_dim=30)
    autoencoder_model = '2500'  # remember to change this
    is_read_model = False  # used only for continued training of regressor
    read_model = '2000'  # only used if is_read_model==True
    mean_train = False  # use mean value or sampled value as the latent code for regressor in variational models. True: mean value; False: sampled value

    # training parameters
    batch = 64
    learning_rate = 0.05
    momentum = 0.9
    epoch = 2000
    save_model_interval = 200

    # transform the input data for the model
    exp = torch.load(os.path.join(root_dir, 'data', '1.pt')); exp = exp.to(device)
    mask = ~exp.isnan()  # mask is the places where data is valid
    if autoencoder.isfc:
        transform = MaskFlatten(mask)
    else:
        transform = torchvision.transforms.Compose([])

    train_dataset = Mydata(root_dir, os.path.join('dataset', 'train'), label_file, transform=transform)
    validation_dataset = Mydata(root_dir, os.path.join('dataset', 'validation'), label_file, transform=transform)
    # test_dataset = Mydata(root_dir, 'dataset/test/', label_file, transform=transform)
    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Length of validation dataset: {len(validation_dataset)}')
    # print(f'Length of test dataset: {len(test_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True, drop_last=True)

    # load feature extractor
    autoencoder.load_state_dict(torch.load(os.path.join(root_dir, autoencoder_dir, f'model_epoch{autoencoder_model}.pth')))
    autoencoder = autoencoder.to(device)
    is_vae = True if hasattr(autoencoder, 'reparameterization') else False
    autoencoder.eval()

    os.mkdirs(os.path.join(root_dir, predictor_dir), exist_ok=True)
    if is_read_model:
        fc_predictor.load_state_dict(torch.load(os.path.join(root_dir, predictor_dir, f'predictor_epoch{read_model}.pth')))
    fc_predictor = fc_predictor.to(device)

    loss_predictor_mse = nn.MSELoss(reduction='mean')
    loss_predictor_mse = loss_predictor_mse.to(device)
    optimizer_predictor = torch.optim.SGD(fc_predictor.parameters(), lr=learning_rate, momentum=momentum)

    print('\n',fc_predictor,'\n')
    print(f'Batch size: {batch}, Epoch: {epoch}, MSELoss')
    print(f'SGD optim with lr: {learning_rate}, momentum: {momentum}')
    print(f'Use autoencoder model {autoencoder_model}\n')

    total_train_step = 0
    start = int(read_model) if is_read_model else 0
    for i in range(start,epoch+start):
        print(f"---------- Epoch: {i+1} ----------")
        fc_predictor.train()
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)
            if is_vae:
                mu, logvar = autoencoder.encoder(data)
                code = autoencoder.reparameterization(mu, logvar) if not mean_train else mu
            else:
                code = autoencoder.encoder(data)
            code = code.clone().detach()
            fcs = fc_predictor(code)
            loss_predictor = loss_predictor_mse(fcs, label)

            optimizer_predictor.zero_grad()
            loss_predictor.backward()
            optimizer_predictor.step()

            total_train_step += 1
            if total_train_step % 5 == 0:
                print(f"Steps: {total_train_step}, Loss: {loss_predictor.item()}")

        fc_predictor.eval()
        total_val_loss = 0
        with torch.no_grad():
            num_batch = 0
            for data, label in validation_dataloader:
                data = data.to(device)
                label = label.to(device)
                if is_vae:
                    mu, logvar = autoencoder.encoder(data)
                    code = autoencoder.reparameterization(mu, logvar) if not mean_train else mu
                else:
                    code = autoencoder.encoder(data)
                code = code.clone().detach()
                fcs = fc_predictor(code)
                loss = loss_predictor_mse(fcs, label)
                total_val_loss += loss.item()
                num_batch += 1
            total_val_loss = total_val_loss / num_batch
        print(f'Total validation loss: {total_val_loss}')
        if (i + 1) % save_model_interval == 0:
            torch.save(fc_predictor.state_dict(), os.path.join(root_dir, predictor_dir, f'predictor_epoch{i + 1}.pth'))

