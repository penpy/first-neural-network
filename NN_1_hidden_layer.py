import struct
import array
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

# ===================
# Dataset
# ===================


DATA_TYPES = {0x08: 'B',  # unsigned byte
              0x09: 'b',  # signed byte
              0x0b: 'h',  # short (2 bytes)
              0x0c: 'i',  # int (4 bytes)
              0x0d: 'f',  # float (4 bytes)
              0x0e: 'd'}  # double (8 bytes)

FILE_NAMES = ['train-images.idx3-ubyte',
              'train-labels.idx1-ubyte',
              't10k-images.idx3-ubyte',
              't10k-labels.idx1-ubyte']

def mnist_data(i):
    filename = 'database//' + FILE_NAMES[i]
    fd = open(filename, 'rb')
    header = fd.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))
    data = array.array(data_type, fd.read())
    data.byteswap()
    
    return np.array(data).reshape(dimension_sizes)


TRAIN_IMAGES = mnist_data(0)
TRAIN_LABELS = mnist_data(1)
TEST_IMAGES = mnist_data(2)
TEST_LABELS = mnist_data(3)

LEN_TRAIN = len(TRAIN_LABELS)
LEN_TEST = len(TEST_LABELS)


# ===================
# Neural network
# ===================


class NN:
    
    def __init__(self):
        # 3 layers: input x, hidden h, output y
        self.n_x = 28 * 28
        self.n_h = 16
        self.n_y = 10
        
        self.shape_w1 = (self.n_x, self.n_h)
        self.shape_w2 = (self.n_h, self.n_y)
        
        self.w1 = np.random.uniform(0, 1, self.shape_w1) / self.n_h
        self.w2 = np.random.uniform(0, 1, self.shape_w2) / self.n_y
        
        self.b1 = np.full(self.n_h, 0.0)
        self.b2 = np.full(self.n_y, 0.0)
        
        self.n_iter = 1875
        self.iter = 0
        self.batch_size = 32
        self.learn_r = 0.05
        self.n_tests = LEN_TEST
        self.train_indices = np.arange(LEN_TRAIN)
        
        if self.n_x != 28 * 28:
            raise ValueError('input vector size must be equal to the numbers of pixels in one image')
        if self.n_y != 10:
            raise ValueError('output vector size must be equal to 10')
        if LEN_TRAIN < self.n_iter * self.batch_size:
            raise ValueError('Too many iterations for 60 000 training images')
    
    def forward_propagation(self, image):
        # Activation: sigmoid for ha, softmax for ya
        x = image.flatten() / image.max()
        h = np.dot(x, self.w1) + self.b1
        ha = 1 / (1 + np.exp(-h))
        y = np.dot(ha, self.w2) + self.b2
        exp_y = np.exp(y)
        ya = exp_y / exp_y.sum()
        return x, h, ha, y, ya
    
    def backpropagation(self, x, ha, ya, t):
        # Derivatives d_u of the cross-entropy loss with respect to each parameter u
        d_b2 = ya
        d_b2[t] -= 1
        d_w2 = np.outer(ha, d_b2)
        d_b1 = np.dot(self.w2, d_b2) * ha * (1 - ha)
        d_w1 = np.outer(x, d_b1)
        return d_w1, d_w2, d_b1, d_b2
    
    def train(self):
        
        np.random.shuffle(self.train_indices)
        
        for k in range(self.n_iter):
            
            # Initialization of the derivatives for the batch
            sum_d_w1 = np.zeros(self.shape_w1)
            sum_d_w2 = np.zeros(self.shape_w2)
            sum_d_b1 = np.zeros(self.n_h)
            sum_d_b2 = np.zeros(self.n_y)
            
            for i in range(self.batch_size):
                index = self.train_indices[k * self.batch_size + i]
                image = TRAIN_IMAGES[index]
                t = TRAIN_LABELS[index]
                
                x, h, ha, y, ya = self.forward_propagation(image)
                d_w1, d_w2, d_b1, d_b2 = self.backpropagation(x, ha, ya, t)
                
                sum_d_w1 += d_w1
                sum_d_w2 += d_w2
                sum_d_b1 += d_b1
                sum_d_b2 += d_b2
            
            self.w1 -= self.learn_r * sum_d_w1
            self.w2 -= self.learn_r * sum_d_w2
            self.b1 -= self.learn_r * sum_d_b1
            self.b2 -= self.learn_r * sum_d_b2
    
    def test(self, test_index):
        image = TEST_IMAGES[test_index]
        label = TEST_LABELS[test_index]
        x, h, ha, y, ya = self.forward_propagation(image)
        result = ya.argmax()
        return result == label
    
    def accuracy(self):
        # Returns the proportion of correctly guessed digits by the network
        acc = 0
        for i in range(self.n_tests):
            if self.test(i):
                acc += 1
        return acc / self.n_tests
    
    def irl(self, image):
        x, h, ha, y, ya = self.forward_propagation(255 - image)
        plt.figure('Test IRL')
        plt.subplot(211)
        plt.imshow(255 - image, 'Greys')
        plt.subplot(212)
        plt.bar(range(10), ya)


# ===================
# Plot performance
# ===================

def perf(nn=None):
    if nn == None:
        nn = NN()
    nn.n_iter = 5
    y = [nn.accuracy()]
    x = [0]
    tot = 50
    string = (tot - 12) * ' ' + '| end'
    print('Progression:' + string)
    for i in range(tot):
        print('=', end='')
        nn.train()
        y.append(nn.accuracy())
        x.append(nn.n_iter + x[-1])
    print('\nFinished.')
    plt.plot(x, y, 'o-', lw=1)
    plt.grid()
    plt.ylabel('Proportion of correct guesses')
    plt.xlabel('Number of iterations')
    plt.title('Accuracy during training process')


class Check:
    
    def __init__(self, nn=None):
        self.n_epochs = 3
        if nn is None:
            self.nn = NN()
        else:
            self.nn = nn
    
    def train_nn(self):
        print('Starting training process...')
        str_1 = '      0/' + str(self.n_epochs) + ' completed'
        str_2 = '|' + ' ' * self.n_epochs + '|'
        print(str_1, str_2)
        for i in range(self.n_epochs):
            self.nn.train()
            i += 1
            str_1 = 'Epoch ' + str(i) + '/' + str(self.n_epochs) + ' completed'
            str_2 = '|' + '=' * i + ' ' * (self.n_epochs - i) + '|'
            print(str_1, str_2)
        print('Training process finished.')
    
    def example(self, image_id=None):
        if image_id == None:
            image_id = np.random.randint(10000)
        image = TEST_IMAGES[image_id]
        label = TEST_LABELS[image_id]
        x, h, ha, y, ya = self.nn.forward_propagation(image)
        plt.figure(image_id)
        if label == ya.argmax():
            plt.suptitle('CORRECT', color='g', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('INCORRECT', color='r', fontsize=16, fontweight='bold')
        plt.subplot(121)
        plt.imshow(image, 'Greys')
        plt.title('Label = ' + str(label))
        plt.subplot(122)
        plt.bar(range(10), ya)
        plt.xticks(range(10), range(10))
        plt.ylim(0, 1)
        plt.xlim(-.5, 9.5)
        plt.grid(axis='y')
        plt.title('Guessed digit = ' + str(ya.argmax()))


# ===================
# TEST WITH PERSONAL IMAGE
# ===================

class Img:
    
    def __init__(self, name):
        self.name = name
        self.my_image = img.imread(self.name, format='jpg')
        self.my_image = np.array(self.my_image)
        self.height, self.length, self.n_colors = self.my_image.shape
        self.new_dim()
        self.convert()
    
    def new_dim(self):
        self.new_size = 28
        if self.height < self.length:
            self.big_pixel_size = self.height // self.new_size
            self.i_start = 0
            self.j_start = int((self.length - self.height) / 2)
        else:
            self.big_pixel_size = self.length // self.new_size
            self.i_start = int((self.height - self.length) / 2)
            self.j_start = 0
    
    def convert(self):
        self.new_image = np.zeros((self.new_size, self.new_size))

        for index in range(784):
            sum_for_new_p = 0
            new_I = index // self.new_size
            new_J = index % self.new_size
            for i in range(self.new_size):
                for j in range(self.new_size):
                    I = i + self.i_start + self.big_pixel_size * (new_I)
                    J = j + self.j_start + self.big_pixel_size * (new_J)
                    for K in range(self.n_colors):
                        sum_for_new_p += self.my_image[I][J][K]
    
            self.new_image[new_I][new_J] = sum_for_new_p // (3*784)
        