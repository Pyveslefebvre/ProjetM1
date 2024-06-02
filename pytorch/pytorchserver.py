import time
import flwr as fl

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def start_server_with_delay():
    print("Waiting for clients to connect...")
    time.sleep(10)  # Wait for 10 seconds to allow clients to connect
    print("Starting the server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average
        )
    )

if __name__ == "__main__":
    start_server_with_delay()


