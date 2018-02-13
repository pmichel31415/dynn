from __future__ import print_function, division

import numpy as np
import dynet as dy

import layers


class NaryTreeLSTMCell(layers.Layer):
    """N-ary TreeLSTM as introduced in Tai et al, 2015"""
    def __init__(self, N, di, dh, pc, dropout=0.0, diagonal=False):
        super(NaryTreeLSTMCell, self).__init__(pc, '%d-treelstm' % N)
        # Arity
        self.N = N
        assert self.N>=0, 'Arity %d not supported' % self.N
        # Output dimension
        self.dh = dh
        # Input dimension
        self.di = di
        # Whether to use diagonal forget gates (ie no connections between h_i and f_j if i!=j)
        self.diagonal = diagonal
        # Dimensions for the parameters
        self.incoming_dim = self.di + self.dh * self.N
        self.diagonal_dim = (self.di + self.dh) if self.diagonal else self.incoming_dim
        # Dropout
        self.dropout = dropout
        # Parameters
        self.Wi_p = self.pc.add_parameters((self.dh, self.incoming_dim), name='Wi')
        self.Wo_p = self.pc.add_parameters((self.dh, self.incoming_dim), name='Wo')
        self.Wg_p = self.pc.add_parameters((self.dh, self.incoming_dim), name='Wg')
        self.Wf_p = [self.pc.add_parameters((self.dh, self.diagonal_dim), name='Wf%d' % i) for i in range(self.N)]
        # Biases
        self.bi_p = self.pc.add_parameters(self.dh, name='bi', init=layers.ZeroInit)
        self.bo_p = self.pc.add_parameters(self.dh, name='bo', init=layers.ZeroInit)
        self.bg_p = self.pc.add_parameters(self.dh, name='bg', init=layers.ZeroInit)
        self.bf_p = [self.pc.add_parameters(self.dh, name='bf%d' % i, init=layers.OneInit) for i in range(self.N)]


    def init(self, test=False, update=True):
        # Load weights in computation graph
        self.Wi = self.Wi_p.expr(update)
        self.Wo = self.Wo_p.expr(update)
        self.Wg = self.Wg_p.expr(update)
        self.Wf = [w.expr(update) for w in self.Wf_p]
        # Load biases in computation graph
        self.bi = self.bi_p.expr(update)
        self.bo = self.bo_p.expr(update)
        self.bg = self.bg_p.expr(update)
        self.bf = [w.expr(update) for w in self.bf_p]
        # Initialize dropout mask
        self.test=test
        if not test and self.dropout>0:
            self.dropout_mask = dy.dropout(dy.ones(self.incoming_dim), self.dropout)

    def __call__(self, *args):
        """Arguments are of the form h1, c1, h2, c2,...,h_N, c_N[, x]"""
        if not (len(args) == (2 * self.N + 1) or (len(args) == (2 * self.N) and self.di==0)):
            error_msg = '%d arguments for %d-ary treeLSTM (should be 2 * arity + 1)' % (len(args), self.N)
            raise ValueError(error_msg)
        # Concatenate inputs
        h_m1 = args[::2]
        c_m1 = args[1::2]
        h_m1 = dy.concatenate(list(h_m1))
        # Dropout
        if not self.test and self.dropout>0:
            h = dy.cmult(self.dropout_mask, h_m1)
        # Compute gates
        self.i = dy.logistic(dy.affine_transform([self.bi, self.Wi, h_m1]))
        self.o = dy.logistic(dy.affine_transform([self.bo, self.Wo, h_m1]))
        self.g = dy.tanh(dy.affine_transform([self.bg, self.Wg, h_m1]))
        # Forget gates are computed differently if the connections are diagonal
        if self.diagonal:
            self.f = [dy.logistic(dy.affine_transform([self.bf[i], self.Wf[i], h_m1[i]])) for i in range(self.N)]
        else:
            self.f = [dy.logistic(dy.affine_transform([self.bf[i], self.Wf[i], h_m1])) for i in range(self.N)]
        # Update c with gating
        self.c = dy.cmult(self.i, self.g)
        if self.N>0:
            self.c += dy.esum([dy.cmult(self.f[i], c_m1[i]) for i in range(self.N)])
        # Output
        self.h = dy.cmult(self.o, dy.tanh(self.c))
        # Return h and c
        return self.h, self.c

def LSTM(di, dh, pc, dropout=0.0):
    """Standard LSTM"""
    return NaryTreeLSTMCell(1, di, dh, pc, dropout=dropout, diagonal=False)

def BinaryTreeLSTM(di, dh, pc, dropout=0.0, diagonal=False):
    """Binary tree-lstm"""
    return NaryTreeLSTMCell(2, di, dh, pc, dropout=dropout, diagonal=diagonal)

class CompactLSTM(layers.Layer):
    """standard LSTM using dynet's memory efficient LSTM operations"""

    def __init__(self, di, dh, pc, dropout=0.0):
        super(CompactLSTM, self).__init__(pc, 'compact-lstm')
        self.di = di
        self.dh = dh
        self.dropout = dropout

        # Parameters
        scale = np.sqrt(6/(2*self.dh + self.di))
        self.Whx_p = self.pc.add_parameters((self.dh * 4, self.di), name='Whx', init=dy.UniformInitializer(scale))
        self.Whh_p = self.pc.add_parameters((self.dh * 4, self.dh), name='Whh', init=dy.UniformInitializer(scale))
        self.bh_p = self.pc.add_parameters((self.dh * 4,), name='bh', init=layers.ZeroInit)

    def init(self, test=False, update=True):
        # Load weights in computation graph
        self.Whx = self.Whx_p.expr(update)
        self.Whh = self.Whh_p.expr(update)
        self.bh = self.bh_p.expr(update)
        # Initialize dropout mask
        self.test=test
        if not test and self.dropout>0:
            self.dropout_mask_x = dy.dropout(dy.ones(self.di), self.dropout)
            self.dropout_mask_h = dy.dropout(dy.ones(self.dh), self.dropout)

    def __call__(self, h, c, x):
        if not self.test and self.dropout>0:
            gates = dy.vanilla_lstm_gates_dropout(x,h,self.Whx, self.Whh,self.bh, self.dropout_mask_x, self.dropout_mask_h)
        else:
            gates = dy.vanilla_lstm_gates(x,h,self.Whx, self.Whh,self.bh)
        new_c = dy.vanilla_lstm_c(c, gates)
        new_h = dy.vanilla_lstm_h(new_c, gates)
        return new_h, new_c


def transduce(lstm, xs, h0, c0, masks=None):
    """Helper function for LSTM transduction with masking"""
    h, c = h0, c0
    hs = []
    if masks is not None:
        for x, m in zip(xs, masks):
            h, c = lstm(h, c, x)
            hs.append(h)
            m_e = dy.inputTensor(m, batched=True)
            h, c = dy.cmult(h, m_e), dy.cmult(c, m_e)
    else:
        for x in xs:
            h, c = lstm(h, c, x)
            hs.append(h)
    return hs