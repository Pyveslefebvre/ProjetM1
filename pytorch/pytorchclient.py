from collections import OrderedDict
import torch
import flwr as fl
from centralized import load_data, load_model, train, test

net = load_model()
trainloader, testloader = load_data()

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print("Client: get_parameters called")
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        print("Client: fit called")
        set_parameters(net, parameters)
        train(net, trainloader,epochs=1)
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client: evaluate called")
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {'accuracy': accuracy}


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)

