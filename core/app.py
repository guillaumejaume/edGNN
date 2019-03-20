import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint

from core.data.constants import LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, GRAPH
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)


class App:

    def __init__(self, model, early_stopping=True):
        self.model = model
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=100, verbose=True)

    def train(self, data, config, save_path='', mode=NODE_CLASSIFICATION):

        loss_fcn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=config['lr'],
                                     weight_decay=config['weight_decay'])

        labels = data[LABELS]
        # initialize graph
        if mode == NODE_CLASSIFICATION:
            train_mask = data[TRAIN_MASK]
            val_mask = data[VAL_MASK]
            dur = []
            for epoch in range(config['n_epochs']):
                self.model.train()
                if epoch >= 3:
                    t0 = time.time()
                # forward
                logits = self.model(None)
                loss = loss_fcn(logits[train_mask], labels[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                val_acc, val_loss = self.model.eval_node_classification(labels, val_mask)
                print("Epoch {:05d} | Time(s) {:.4f} | Train loss {:.4f} | Val accuracy {:.4f} | "
                      "Val loss {:.4f}".format(epoch,
                                               np.mean(dur),
                                               loss.item(),
                                               val_acc,
                                               val_loss))

                self.early_stopping(val_loss, self.model, save_path)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

        elif mode == GRAPH_CLASSIFICATION:
            self.accuracies = np.zeros(10)
            graphs = data[GRAPH]                 # load all the graphs
            for k in range(10):                  # 10-fold cross validation
                start = int(len(graphs)/10) * k
                end = int(len(graphs)/10) * (k+1)

                # testing batch
                testing_graphs = graphs[start:end]
                self.testing_labels = labels[start:end]
                self.testing_batch = dgl.batch(testing_graphs)

                # training batch
                training_graphs = graphs[:start] + graphs[end+1:]
                training_labels = labels[list(range(0, start)) + list(range(end+1, len(graphs)))]
                training_samples = list(map(list, zip(training_graphs, training_labels)))
                training_batches = DataLoader(training_samples,
                                              batch_size=config['batch_size'],
                                              shuffle=True,
                                              collate_fn=collate)

                dur = []
                for epoch in range(config['n_epochs']):
                    self.model.train()
                    if epoch >= 3:
                        t0 = time.time()
                    losses = []
                    training_accuracies = []
                    for iter, (bg, label) in enumerate(training_batches):
                        logits = self.model(bg)
                        loss = loss_fcn(logits, label)
                        losses.append(loss.item())
                        _, indices = torch.max(logits, dim=1)
                        correct = torch.sum(indices == label)
                        training_accuracies.append(correct.item() * 1.0 / len(label))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if epoch >= 3:
                        dur.append(time.time() - t0)
                    val_acc, val_loss = self.model.eval_graph_classification(self.testing_labels, self.testing_batch)
                    print("Epoch {:05d} | Time(s) {:.4f} | Train acc {:.4f} | Train loss {:.4f} "
                          "| Val accuracy {:.4f} | Val loss {:.4f}".format(epoch,
                                                                           np.mean(dur) if dur else 0,
                                                                           np.mean(training_accuracies),
                                                                           np.mean(losses),
                                                                           val_acc,
                                                                           val_loss))

                    is_better = self.early_stopping(val_loss, self.model, save_path)
                    if is_better:
                        self.accuracies[k] = val_acc

                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
                self.early_stopping.reset()
        else:
            raise RuntimeError

    def test(self, data, load_path='', mode=NODE_CLASSIFICATION):

        try:
            print('*** Load pre-trained model ***')
            self.model = load_checkpoint(self.model, load_path)
        except ValueError as e:
            print('Error while loading the model.', e)

        if mode == NODE_CLASSIFICATION:
            test_mask = data[TEST_MASK]
            labels = data[LABELS]
            acc, _ = self.model.eval_node_classification(labels, test_mask)
        else:
            acc = np.mean(self.accuracies)

        print("\nTest Accuracy {:.4f}".format(acc))

        return acc
