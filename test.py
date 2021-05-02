from data_loader import DataLoader
from options import Options
from neural_network import NeuralNetwork
from utils import validate

if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
    data_loader.load_test()
    nn = NeuralNetwork(model_import=True, model_path=opt.model_path)
    test_acc = validate(nn, data_loader.X_test, data_loader.y_test)
    print('Test acc: %0.2f' % test_acc)
