import torch

class acc():

    def accuracy_score (true_labels, pred_labels):

        accuracy = torch.sum(true_labels.view(-1).float() == 
                            pred_labels.float()).item() / true_labels.size(0)
        return accuracy
