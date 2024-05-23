import flwr as fl

strategy = fl.server.strategy.FedAvg()

# Start the flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy
)