import numpy as np

class Buffer():
    def __init__(self):
        self.features = []
        self.labels = []
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
    
    def __init__from_pt(self, pt_file):
        print('Buffer init from pt file')
        self.features, self.labels = torch.load(pt_file)
        self.features = self.features.cpu().numpy()
        self.labels = self.labels.cpu().numpy()

    def add_row(self, feat, label):
        if len(self.features) == 0:
            self.features = np.array([feat])
            self.labels = np.array([label])
        else:
            feat = np.array([feat])
            self.features = np.vstack((self.features, feat))
            self.labels = np.vstack((self.labels, label))
 
    def add_rows(self, feat_lt, label_lt):
        assert len(feat_lt) == len(label_lt)
        if len(self.features) == 0:
            self.features = feat_lt
            self.labels = label_lt
        else:
            self.features = np.concatenate((self.features, feat_lt), axis=0)
            self.labels = np.concatenate((self.label_lt, label_lt), axis=0)
        '''
        for i in range(len(feat_lt)):
            feat = feat_lt[i]
            label = label_lt[i]
            self.add_row(feat, label)
        '''

