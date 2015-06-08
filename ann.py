import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def print_shapes(v, varname):
    print '\n=======[%s]======= ' % varname
    print '\n'.join(['%d: %s' % (i, str(e.shape)) for i,e in enumerate(v)])

class Network(object):
    def __init__(self, sizes, eta=0.1):
        self._eta = eta
        self._sizes = sizes
        self._num_layers = len(sizes)

        # Weights are going to be W * x, which is sz_out x sz_in.
        self._weights = [np.random.randn(so, si)
                for so, si in zip(sizes[1:], sizes[:-1])]
        self._biases = [np.random.randn(so, 1) for so in sizes[1:]]

        # print_shapes(self._weights, 'W0')
        # print_shapes(self._biases, 'b0')

    def _cost_derivative(self, a, y):
        return a - y

    def feedforward(self, a, z_out=None, a_out=None):
        a = np.vstack(a)

        for w, b in zip(self._weights, self._biases):
            z = np.dot(w, a) + b
            a = sigmoid_vec(z)

            assert(a.shape == b.shape)

            if z_out is not None:
                z_out.append(z)
            if a_out is not None:
                a_out.append(a)
        return a

    def backpropagate(self, x, y):
        x = np.vstack(x)
        y = np.vstack(y)

        # Feedforward x
        zl = []
        al = [x]
        self.feedforward(x, z_out=zl, a_out=al)

        # print_shapes(zl, 'zl')
        # print_shapes(al, 'al')
        # print_shapes(self._weights, 'W')

        # Initialize with output layer error
        err = [np.multiply(self._cost_derivative(al[-1], y),
                           sigmoid_prime_vec(zl[-1]))]


        for i in xrange(2, self._num_layers):
            w = self._weights[-i + 1]
            nabla = np.multiply(np.dot(np.transpose(w), err[-1]),
                                sigmoid_prime_vec(zl[-i]))
            err.append(nabla)
        err = list(reversed(err))

        # print_shapes(err, 'err')
        dC_db = err
        dC_dw = [np.dot(n, np.transpose(a))
                 for n, a in zip(err, al[:-1])]
        return dC_db, dC_dw

    def train(self, x, y):
        x = np.vstack(x)
        y = np.vstack(y)

        db, dw = self.backpropagate(x, y)
        self._biases = [b - self._eta * d for b, d  in zip(self._biases, db)]
        self._weights = [b - self._eta * d for b, d  in zip(self._weights, dw)]

def main():
    pos_label = [1]
    neg_label = [0]
    samples = []
    labels = []
    for i in xrange(20000):
        x, y = np.random.randint(0, high=2, size=2)
        samples.append([x, y])
        if x ^ y == 1:
            labels.append(pos_label)
        else:
            labels.append(neg_label)

    s = list(zip(samples, labels))
    network = Network([2, 2, 1])
    for x, y in s:
        network.train(x, y)

    correct = 0
    total = 0
    for x, y in s:
        gt_label = y[0]
        pred_label = 1 if network.feedforward(x) > 0.5 else 0
        total += 1
        if gt_label == pred_label:
            correct += 1
        # print '%s%d == %d' % (' ' if gt_label == pred_label else '*',
        #                      gt_label, pred_label)

    print 'Eval: %0.5f (%d/%d)' % (float(correct)/total, correct, total)


def main_quad_problem():
    quad1 = [1, 0, 0, 0]
    quad2 = [0, 1, 0, 0]
    quad3 = [0, 0, 1, 0]
    quad4 = [0, 0, 0, 1]

    q1_samples = [[x, y] for x, y in zip(np.random.uniform(0, 1, 250),
                                         np.random.uniform(0, 1, 250))]

    q2_samples = [[-x, y] for x, y in zip(np.random.uniform(0, 1, 250),
                                          np.random.uniform(0, 1, 250))]

    q3_samples = [[-x, -y] for x, y in zip(np.random.uniform(0, 1, 250),
                                           np.random.uniform(0, 1, 250))]
    q4_samples = [[x, -y] for x, y in zip(np.random.uniform(0, 1, 250),
                                          np.random.uniform(0, 1, 250))]

    q1 = list(zip(q1_samples, [quad1]*len(q1_samples)))
    q2 = list(zip(q2_samples, [quad2]*len(q2_samples)))
    q3 = list(zip(q3_samples, [quad3]*len(q3_samples)))
    q4 = list(zip(q4_samples, [quad4]*len(q4_samples)))

    q_samples = np.concatenate([q1, q2, q3, q4])
    np.random.shuffle(q_samples)

    network = Network([2, 4])
    for x, y in q_samples:
        network.train(x, y)

    correct = 0
    total = 0
    for x, y in q_samples:
        gt_label = np.argmax(y)
        pred_label = np.argmax(network.feedforward(x))
        total += 1
        if gt_label == pred_label:
            correct += 1
        # print '%s%d == %d' % (' ' if gt_label == pred_label else '*',
        #                      gt_label, pred_label)

    print 'Eval: %0.5f (%d/%d)' % (float(correct)/total, correct, total)

# Toy problem to divide quandrant.
if __name__ == '__main__':
    main()

