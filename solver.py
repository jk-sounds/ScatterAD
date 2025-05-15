import torch 
import numpy as np 
import os 
from tqdm import tqdm 
from model.ScatterAD import Model 
from data_factory.data_loader  import get_graph_loader 
from metrics.metrics  import combine_all_evaluation_scores 
import time 
class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_graph_loader(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_graph_loader(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_graph_loader(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_graph_loader(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.build_model()

    def build_model(self):
        self.model = Model(input_dim=self.input_c,hidden_dim=self.d_model,num_layers=self.e_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')

    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i, (input_data,_) in enumerate(vali_loader):
                batch = input_data.to(self.device)
                h_pred, h_target = self.model(batch)
                loss = self.model.model.compute_loss(batch.x, h_pred, h_target, batch.edge_index)
                loss_list.append(loss.detach().cpu())  
        return np.average(loss_list)

    def train(self):
        print("======================TRAIN MODE======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        for epoch in tqdm(range(self.num_epochs)):
            loss_list = []
            epoch_time = time.time()
            self.model.train()
            total_loss = 0
            with tqdm(self.train_loader, total=len(self.train_loader)) as pbar:
                for i, (input_data, _) in enumerate(pbar):
                    self.optimizer.zero_grad()
                    batch = input_data.to(self.device)
                    model = self.model
                    h_pred, h_target = self.model(batch)
                    loss = self.model.model.compute_loss(batch.x, h_pred, h_target, batch.edge_index)
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    self.model.model.update_target()
                    total_loss += loss.item()
                    loss_list.append(loss.item()) 
                    pbar.set_description(f"Batch loss: {loss.item():.4f}")
                    pbar.update(1)            
            train_loss = np.mean(loss_list)           
            vali_loss = self.vali(self.vali_loader)
            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))       
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}")
            print(
                "Epoch: {0}, Cost time: {1:.3f}s, Train Loss: {2:.8f} Vali Loss: {3:.8f}".format(
                    epoch + 1, time.time() - epoch_time, train_loss, vali_loss))



    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        # (1) find the threshold
        attens_energy = []
        with torch.no_grad():
            for i, (input_data,_) in enumerate(self.thre_loader):
                batch = input_data.to(self.device)
                scores = self.model.model.get_anomaly_scores(batch.x, batch.edge_index).cpu().numpy()
                attens_energy.append(scores)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (2) evaluation on the test set
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.test_loader):
                batch = input_data.to(self.device)
                scores = self.model.model.get_anomaly_scores(batch.x, batch.edge_index).cpu().numpy()
                attens_energy.append(scores)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        print(combine_all_evaluation_scores(pred,gt))
        pass
