import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import Batch

class GraphPSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of samples in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1
        
    def __getitem__(self, index):
       
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        else:
            data =self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)

class GraphMSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # # Return as a graph (Data object)
        # graph = Data(x=x, edge_index=edge_index)
        
        # return graph, torch.tensor(label)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)
class GraphSMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index)
        
        return graph, torch.tensor(label)

class GraphSMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index)
        
        return graph, torch.tensor(label)

class GraphServiceSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index)
        
        return graph, torch.tensor(label)
        
class GraphSWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy", allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy", allow_pickle=True)
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy", allow_pickle=True)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        else:
            data =self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)
    
class GraphWADISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/WADI_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/WADI_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/WADI_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)

class GraphNIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-2), min(self.win_size, i+3)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)     

class GraphNIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)


def custom_collate_fn(batch):
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    batch_data = Batch.from_data_list(data_list)
    batch_labels = torch.stack(label_list)
    return batch_data, batch_labels


def get_graph_loader(data_path, batch_size, win_size=100, step=1, mode='train',dataset='KDD'):
    """
    Returns a DataLoader for graph-based time-series data.
    """
    # dataset = GraphTimeSeriesLoader(data_path, win_size, step, mode)
    if (dataset == 'SMD'):
        dataset = GraphSMDSegLoader(data_path, win_size, 1, mode)
    elif 'machine' in dataset:
        dataset = GraphServiceSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = GraphMSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = GraphSMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = GraphSWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI'):
        dataset = GraphWADISegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = GraphPSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = GraphNIPS_TS_WaterSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = GraphNIPS_TS_SwanSegLoader(data_path, win_size, 1, mode)


    shuffle = mode == 'train'
    
    # data_loader = torch_geometric.data.DataLoader(dataset=dataset,
    #                          batch_size=batch_size,
    #                          shuffle=shuffle,
    #                          num_workers=8)  
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=custom_collate_fn
    )

    return data_loader
