import argparse


class Options:
    def initialize(self, parser):
        parser.add_argument('-data_path', type=str, default='./data/', help='path to dataset')
        parser.add_argument('-vgg19', action='store_true', help='if specified, use VGG19 features')

        # train
        parser.add_argument('-model_name', type=str, default='model.txt', help='name of the model to save')
        parser.add_argument('-hidden_layer_num', type=int, default=1, help='number of hidden layers')
        parser.add_argument('-hidden_unit_num', type=int, default=300, help='number of hidden units in hidden layers')
        parser.add_argument('-epoch_num', type=int, default=50, help='number of epochs')
        parser.add_argument('-batch_size', type=int, default=32, help='batch size for mini-batch gradient descent')
        parser.add_argument('-learning_rate', type=float, default=0.1, help='learning rate for gradient descent')
        parser.add_argument('-reduce_lr', action='store_true', help='if specified, reduce learning rate')
        parser.add_argument('-activation_func', type=str, default='relu', help='sigmoid | tanh | relu')
        parser.add_argument('-objective_func', type=str, default='log', help='log | sse | mse')

        # test
        parser.add_argument('-model_path', type=str, default='./model/sl_nn_', help='path to saved model')

        self.parser = parser
        return parser

    def print_options(self, opt):
        options = ''
        options += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            options += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        options += '----------------- End -------------------'
        print(options)

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = self.initialize(parser).parse_args()
        self.print_options(self.opt)
        return self.opt
