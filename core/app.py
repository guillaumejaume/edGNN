import time
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import KFold

from utils.early_stopping import EarlyStopping
from utils.io import load_checkpoint

from core.data.constants import LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, GRAPH
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION
from core.models.model import Model
from core.data.constants import GRAPH, N_RELS, N_CLASSES, N_ENTITIES


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).cuda() if labels[0].is_cuda else torch.tensor(labels)


class App:

    def __init__(self, early_stopping=True):
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=100, verbose=True)

    def train(self, data, model_config, learning_config, save_path='', mode=NODE_CLASSIFICATION):

        loss_fcn = torch.nn.CrossEntropyLoss()

        labels = data[LABELS]
        # initialize graph
        if mode == NODE_CLASSIFICATION:
            train_mask = data[TRAIN_MASK]
            val_mask = data[VAL_MASK]
            dur = []

            # create GNN model
            self.model = Model(g=data[GRAPH],
                               config_params=model_config,
                               n_classes=data[N_CLASSES],
                               n_rels=data[N_RELS] if N_RELS in data else None,
                               n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                               is_cuda=learning_config['cuda'],
                               mode=mode)

            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=learning_config['lr'],
                                         weight_decay=learning_config['weight_decay'])

            for epoch in range(learning_config['n_epochs']):
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

            # @TODO leave one out fold for testing.

            self.accuracies = np.zeros(10)
            graphs = data[GRAPH]                 # load all the graphs
            num_samples = len(graphs)
            num_folds = 10
            kf = KFold(n_splits=num_folds)

            for k, (train_index, test_index) in enumerate(kf.split(graphs)):

                # create GNN model
                self.model = Model(g=data[GRAPH],
                                   config_params=model_config,
                                   n_classes=data[N_CLASSES],
                                   n_rels=data[N_RELS] if N_RELS in data else None,
                                   n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                                   is_cuda=learning_config['cuda'],
                                   mode=mode)

                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=learning_config['lr'],
                                             weight_decay=learning_config['weight_decay'])

                if learning_config['cuda']:
                    self.model.cuda()

                print('\n\n\nProcess new k')

                # testing batch
                testing_graphs = [graphs[i] for i in test_index]
                self.testing_labels = labels[test_index]
                self.testing_batch = dgl.batch(testing_graphs)

                # all training batch (val + train)
                train_val_graphs = [graphs[i] for i in train_index]
                trai_val_labels = labels[train_index]

                # extract indices to split train and val
                random_indices = list(range(len(train_val_graphs)))
                random.shuffle(random_indices)
                val_indices = random_indices[:int(num_samples/num_folds)]
                train_indices = random_indices[int(num_samples/num_folds):]

                # train batch
                training_graphs = [train_val_graphs[i] for i in train_indices]
                training_labels = trai_val_labels[train_indices]

                # validation batch
                validation_graphs = [train_val_graphs[i] for i in val_indices]
                self.validation_labels = trai_val_labels[val_indices]
                self.validation_batch = dgl.batch(validation_graphs)

                training_samples = list(map(list, zip(training_graphs, training_labels)))
                training_batches = DataLoader(training_samples,
                                              batch_size=learning_config['batch_size'],
                                              shuffle=True,
                                              collate_fn=collate)

                dur = []
                for epoch in range(learning_config['n_epochs']):
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
                    val_acc, val_loss = self.model.eval_graph_classification(self.validation_labels, self.validation_batch)
                    print("Epoch {:05d} | Time(s) {:.4f} | Train acc {:.4f} | Train loss {:.4f} "
                          "| Val accuracy {:.4f} | Val loss {:.4f}".format(epoch,
                                                                           np.mean(dur) if dur else 0,
                                                                           np.mean(training_accuracies),
                                                                           np.mean(losses),
                                                                           val_acc,
                                                                           val_loss))

                    is_better = self.early_stopping(val_loss, self.model, save_path)
                    if is_better:
                        test_acc, _ = self.model.eval_graph_classification(self.testing_labels, self.testing_batch)
                        self.accuracies[k] = test_acc

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
