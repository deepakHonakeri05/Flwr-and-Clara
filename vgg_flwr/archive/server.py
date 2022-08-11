import flwr as fl
from pathlib import Path

# Start Flower server
fl.server.start_server(
  "[::]:8080",
  config={"num_rounds": 3},
  certificates=(
        Path(".cache/certificates/ca.crt").read_bytes(),
        Path(".cache/certificates/server.pem").read_bytes(),
        Path(".cache/certificates/server.key").read_bytes(),
    )
)
