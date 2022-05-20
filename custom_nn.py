import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def deriv_sigmoid(s):
    """Derivative of the sigmoid given the output s of the sigmoid"""
    return s * (1 - s)

def relu(x):
    return x * (x > 0)

def heaviside_step(x):
    return np.array(x > 0, dtype=float)


class MultilayerPerceptronNN():
    
    def __init__(self, dim=(784, 64, 10), activ='sigmoid'):
        self.dim = dim
        self.w = []
        for i in range(len(dim)-1):
            lim = 1 / np.sqrt(dim[i])
            self.w.append(np.random.uniform(-lim, lim, (dim[i+1], dim[i]+1)))
        self.best_w = self.w
        if activ == 'sigmoid':
            self.activ = sigmoid
            self.d_activ = deriv_sigmoid
        elif activ == 'relu':
            self.activ = relu
            self.d_activ = heaviside_step
        else:
            raise Exception(f"Activation function '{activ}' not supported")
        
    def forward(self, x):
        _, batch_size = x.shape
        ones_row = np.ones((1, batch_size))
        hidden = []
        for i in range(len(self.dim) - 2):
            x = np.concatenate((x, ones_row))
            hidden.append(x)
            x = self.w[i] @ x
            hidden.append(x)
            x = self.activ(x)
        x = np.concatenate((x, ones_row))
        hidden.append(x)
        x = self.w[-1] @ x
        y = softmax(x)
        return y, hidden

    def backprop(self, hidden, y, target):
        delta = y - target
        dw = []
        for i in range(len(self.dim) - 1):
            deriv = delta @ hidden[2*len(self.dim)-4-2*i].T
            dw.append(deriv)
            if i != len(self.dim) - 2:
                d_activ = self.d_activ(hidden[2*len(self.dim)-4-2*i])
                delta = self.w[len(self.dim)-2-i].T @ delta * d_activ
                delta = delta[:-1]
        dw.reverse()
        return dw
    

def one_hot(labels):
    """Build one-hot vectors of the labels"""
    t = np.zeros((10, len(labels)))
    t[labels, range(len(labels))] = 1
    return t

def accuracy(y, labels):
    """Proportion of outputs y which match the labels"""
    guess = np.argmax(y, axis=0)
    nb_correct = np.sum(guess == labels)
    return nb_correct / len(labels)

def ce_loss(y, target):
    """Cross-entropy loss"""
    return -np.mean(np.sum(np.log(y)*target, axis=0))


def train(model, data, n_epoch, batch_size, lr0, decay_rate=0):
    """Train Neural Net model"""
    
    inp, labels, inp_val, labels_val = data
    n_examples = len(labels)
    assert batch_size <= n_examples
    n_itr = int(np.ceil(n_examples*n_epoch/batch_size))
    print_itr_step = 100
    print('Total iterarions:', n_itr)

    idx_permut = np.concatenate([np.random.permutation(n_examples)
                                 for _ in range(n_epoch+2)])
    idx_permut = idx_permut[:(n_itr+1)*batch_size].reshape((n_itr+1, -1))

    labels_one_hot = one_hot(labels)
    labels_val_one_hot = one_hot(labels_val)

    y, hidden = model.forward(inp[:, idx_permut[0]])
    y_val, _ = model.forward(inp_val)

    # Loss and accuracy of the training batch and validation set
    log = {'loss': [ce_loss(y, labels_one_hot[:, idx_permut[0]])],
           'acc': [accuracy(y, labels[idx_permut[0]])],
           'vloss': [ce_loss(y_val, labels_val_one_hot)],
           'vacc': [accuracy(y_val, labels_val)],}
    best_vloss = log['vloss'][0]

    for itr in range(n_itr):
        epoch = int(itr*batch_size/n_examples)
        dw = model.backprop(hidden, y, labels_one_hot[:, idx_permut[itr]])
        lr = lr0 / (1 + decay_rate*epoch)
        for i in range(len(model.dim) - 1):
            model.w[i] -= lr * dw[i]
        y, hidden = model.forward(inp[:, idx_permut[itr+1]])
        y_val, _ = model.forward(inp_val)

        log['loss'].append(ce_loss(y, labels_one_hot[:, idx_permut[itr+1]]))
        log['acc'].append(accuracy(y, labels[idx_permut[itr+1]]))
        log['vloss'].append(ce_loss(y_val, labels_val_one_hot))
        log['vacc'].append(accuracy(y_val, labels_val))
        
        # Store the weights yielding the best validation loss
        if log['vloss'][-1] < best_vloss:
            for i in range(len(model.dim) - 1):
                model.best_w[i] = model.w[i].copy()

        # Keep track of the loss
        if itr%print_itr_step == 0 or itr == n_itr-1:
            info = f"Iteration {itr}/{n_itr} (epoch {epoch})"
            info += f" ; loss={log['loss'][itr]} ; vloss={log['vloss'][itr]}"
            print(info)

    for i in range(len(model.dim) - 1):
        model.w[i] = model.best_w[i].copy()

    return log


def normalize(data):
    """Min-Max normalization: rescale to [0,1]"""
    data_min = data.min(axis=1).reshape((-1, 1))
    data_max = data.max(axis=1).reshape((-1, 1))
    data_range = (data_max - data_min) + (data_max == data_min)
    return (data - data_min) / data_range

def prepare(images, labels, p_validation=10):
    """Normalize and split train/validation sets"""
    n_examples = len(images)
    inputs = images.reshape((n_examples, -1))
    normalized_inputs = normalize(inputs)
    
    permutations = np.random.permutation(n_examples)
    n_validation = round(p_validation * n_examples)
    validation_ids = permutations[:n_validation]
    train_ids = permutations[n_validation:]
    
    inputs_valid = normalized_inputs[validation_ids]
    labels_valid = labels[validation_ids]
    
    inputs_train = normalized_inputs[train_ids]
    labels_train = labels[train_ids]
    
    return inputs_train.T, labels_train, inputs_valid.T, labels_valid