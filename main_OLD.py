import torch
import pandas as pd
import os
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional
import skimage.transform
import torch.optim as optim


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #torch.nn.Conv2d(1, 1, (101, 101), (1, 1)) #Input: 1000x1000 Output: 900x900
        #torch.nn.MaxPool2d(2, 2) #Input: 900x900 Output: 450x450
        self.conv_1 = torch.nn.Conv2d(1, 1, kernel_size=(64, 64), stride=(2, 2))
        self.maxPool_1 = torch.nn.MaxPool2d(64, stride=1)
        self.conv_2 = torch.nn.Conv2d(1, 1, kernel_size=(32, 32), stride=(2, 2))
        self.maxPool_2 = torch.nn.MaxPool2d(32,stride=1)
        self.conv_3 = torch.nn.Conv2d(1, 1, kernel_size=(16, 16), stride=(2, 2))
        self.maxPool_3 = torch.nn.MaxPool2d(16, stride=1)
        self.conv_4 = torch.nn.Conv2d(1, 1, kernel_size=(8, 8), stride=(2, 2))
        self.maxPool_4 = torch.nn.MaxPool2d(8, stride=1)
        self.fc_1 = torch.nn.Linear(361, 4)
        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x):
        conv_output = self.conv_1(x)
        layer_1 = self.maxPool_1(conv_output)
        out_layer_1 = torch.nn.functional.relu(layer_1)
        layer_2 = self.maxPool_2(self.conv_2(out_layer_1))
        out_layer_2 = torch.nn.functional.relu(layer_2)
        layer_3 = self.maxPool_3(self.conv_3(out_layer_2))
        out_layer_3 = torch.nn.functional.relu(layer_3)
        layer_4 = self.maxPool_4(self.conv_4(out_layer_3))
        out_layer_4 = torch.nn.functional.relu(layer_4)
        dropout_flatten = torch.flatten(self.dropout(out_layer_4), start_dim=1)
        layer_5 = self.fc_1(dropout_flatten)
        final_out = torch.nn.functional.softmax(layer_5)
        return final_out



class customDataset(Dataset):
    def __init__(self, csv_location, image_directory, transform=None):
        self.dataframe = pd.read_csv(csv_location)
        self.dataframe.dropna(axis=0, inplace=True, thresh=2)
        for name in self.dataframe[self.dataframe.columns[11]].values:
            new_name = name.replace('000000.dcm', '1-1.dcm')
            self.dataframe["image file path"].replace(name, new_name, regex=True, inplace=True)
        self.root_dir = image_directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.normpath(os.path.join(self.root_dir, self.dataframe.iloc[idx, 11]))
        image = dicom.dcmread(image_path)
        breast_density = self.dataframe.iloc[idx, 1]
        breast_side = self.dataframe.iloc[idx, 2]
        image_view = self.dataframe.iloc[idx, 3]
        abnormality_id = self.dataframe.iloc[idx, 4]
        mass_shape = self.dataframe.iloc[idx, 6]
        mass_margin = self.dataframe.iloc[idx, 7]
        assessment = self.dataframe.iloc[idx, 8]
        pathology = self.dataframe.iloc[idx, 9]

        if (pathology == 'MALIGNANT'):
            pathology = 0
        elif(pathology == 'BENIGN'):
            pathology = 1
        else:
            pathology = 2

        subtlety = self.dataframe.iloc[idx, 10]

        sample = (
            skimage.transform.resize(image.pixel_array, (1024, 1024)),
            pathology
            # 'breast_density': breast_density,
            # 'breast_side'   : breast_side,
            # 'image_view'    : image_view,
            # 'abnormality_id': abnormality_id,
            # 'mass_shape'    : mass_shape,
            # 'mass_margin'   : mass_margin,
            # 'assessment'    : assessment,
            # 'subtlety'      : subtlety
        )

        if self.transform:
            sample = self.transform(sample)

        return sample



# Dataset is too big to use in its entirity. So maybe we could split the dataset and
# create new trainloader and testloaders on the fly??
#testing_dataset = customDataset(r"C:\Users\aaron\Desktop\Coding\FPGA AI Project\mass_case_description_test_set.csv", r"C:\Breast-Mass-Images\CBIS-DDSM\Mass-Test")

batch_size = 4


#testloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork()

criteria = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for dataset in range(14):
    path_to_dataset = rf"massTrain{dataset}.csv"
    training_dataset = customDataset(path_to_dataset, r"C:\Breast-Mass-Images\CBIS-DDSM\Mass-Training")
    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, real_pathologies = data
            inputs = torch.unsqueeze(inputs, 1)
            print(inputs.shape)
            optimizer.zero_grad()
            model_guessed_pathologies = model(inputs.float())
            loss = criteria(model_guessed_pathologies, real_pathologies)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(i)

            if i % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

print('Training Done')



# f, ax = plt.subplots(1, 2)
# test_sample = training_dataset[100]
# test_sample_2 = training_dataset[101]
# print(test_sample['image_pixels'].shape)
#
# ax[0].imshow(test_sample['image_pixels'], cmap=plt.cm.gray)
# ax[1].imshow(test_sample_2['image_pixels'], cmap=plt.cm.gray)
# plt.show()