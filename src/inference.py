# -*- coding: utf-8 -*-
# Author: Yaokun Su
# Date  : 10/30/2023

# Machine Learning project


import numpy as np
import torchvision.transforms as T
from utils_models import *


def predict_fcs(input_data, return_var='fcs'):
    pre_list = []
    if is_vae:
        mu, logvar = autoencoder.encoder(input_data)
        mean_fcs = fc_predictor(mu) if autoencoder.isfc else fc_predictor(mu)[0]
        mean_fcs = mean_fcs.reshape(-1)
        if is_random:
            for i in range(num_sample):  # can change this to get the average one ########################
                code = autoencoder.reparameterization(mu, logvar)
                fcs = fc_predictor(code)
                if not autoencoder.isfc:  # not yet dig in why there is different for fully connect and CNN, but it works here
                    fcs = fcs[0]
                fcs = fcs.reshape(-1)
                pre_list.append(fcs)
            fcs = np.sum(pre_list) / num_sample
        else:
            fcs = mean_fcs
    else:
        code = autoencoder.encoder(input_data)
        fcs = fc_predictor(code)
        fcs = fcs.reshape(-1)

    if is_vae and return_var == 'both':
        return fcs, mean_fcs
    elif is_vae and is_random and return_var == 'all':
        return fcs, mean_fcs, pre_list
    elif is_vae and return_var == 'std':
        return fcs, np.std([list(x) for x in pre_list], axis=0)
    else:
        return fcs


if __name__ == '__main__':
    root_dir = '/../' 
    label_file = 'dataset/x_record'
    autoencoder_dir = 'autoencoder/FCAE_FT/'  # folder to read the autoencoder model
    predictor_dir = autoencoder_dir + 'predictor/1/'  # folder to read the regressor model

    # model
    autoencoder = FCAE(latent_dim=30)  # choose the antoencoder model
    fc_predictor = CodeToFCs_NNN(latent_dim=30)
    autoencoder_model = '2500'  # autoencoder model index
    read_model = '2000'  # regressor model index
    mean_train = False  # use mean value or sampled value as the latent code for regressor in variational models. True: mean value; False: sampled value.

    # transform the input data for the model
    mask = torch.load(root_dir + 'data/1.pt')
    mask = ~mask.isnan()  # mask is the places where data is valid
    if autoencoder.isfc:
        transform = MaskFlatten(mask)
        retransform = MaskUnflatten(mask)
    else:
        transform = T.Compose([])
        retransform = T.Compose([])
    mask_flatten = MaskFlatten(mask)

    autoencoder.load_state_dict(torch.load(root_dir + autoencoder_dir + f'model_epoch{autoencoder_model}.pth', map_location="cpu"))
    is_vae = True if hasattr(autoencoder, 'reparameterization') else False
    autoencoder.eval()
    fc_predictor.load_state_dict(torch.load(root_dir + predictor_dir + f'predictor_epoch{read_model}.pth', map_location="cpu"))
    fc_predictor.eval()

    # Start testing

    # test params
    is_random = False if (mean_train or not is_vae) else True  # can mannually set here
    num_sample = 10000  # the number of sampled latent code for VAE models
    is_std = True  # whether to print standard deviation of the predicted force constants for variational models

    # test on exp test set
    with torch.no_grad():
        for t in [50,150,280,350,450,540,640]:
            exp = torch.load(root_dir + f'dataset/exp_23379_sub/exp_chi_{t}K_mean.pt')
            exp = exp.float()
            data = transform(exp).reshape(1,-1) if autoencoder.isfc else transform(exp).reshape(1,1,70,101)
            if is_vae and is_std:
                fcs, fcs_std = predict_fcs(data, 'std')
                print(f'{t} K, Force Constants: {fcs}, std: {fcs_std}')
            else:
                fcs = predict_fcs(data)
                print(f'{t} K, Force Constants: {fcs}')

    # # test on simulation test set. You can uncomment it to run the code. Please note that there are 1000 test examples, so it will take some time
    # test_dataset = Mydata(root_dir, 'dataset/test/', label_file, transform=transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    # with torch.no_grad():
    #     for data, label in test_dataloader:
    #         if is_vae and is_std:
    #             fcs, fcs_std = predict_fcs(data, 'std')
    #             print(f'Force Constants: {fcs}, std: {fcs_std}')
    #         else:
    #             fcs = predict_fcs(data)
    #             print(f'Force Constants: {fcs}')
