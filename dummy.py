import os
import subprocess
import threading

# Function to trigger dummy fine-tuning remotely
def fine_tune_remote_client(client_index, id, ssh_config):
    try:
        # Extract SSH configuration
        host = ssh_config["host"]
        username = ssh_config["username"]
        private_key = ssh_config["private_key"]

        # Dummy command simulating fine-tuning
        remote_command = f"python3 ~/Documents/Research/SafeDrive/main.py {client_index} {id}"
        ssh_command = (
            f"ssh -i {private_key} {username}@{host} '{remote_command}'"
        )

        # Execute the SSH command
        print(f"Starting fine-tuning on client {client_index}...")
        subprocess.run(ssh_command, shell=True, check=True)

        # Fetch the dummy result from the client
        remote_result_path = f"~/client{client_index}/output_{id}.txt"
        local_result_path = f"output_client{client_index}_{id}.txt"
        scp_command = (
            f"scp -i {private_key} {username}@{host}:{remote_result_path} {local_result_path}"
        )
        subprocess.run(scp_command, shell=True, check=True)

        print(f"Result saved at {local_result_path}")

    except Exception as e:
        print(f"Error for client {client_index}: {e}")

# Main function to orchestrate testing
def main():
    num_clients = 2
    id = "test_run"
    ssh_configs = [
        {"host": "192.168.55.1", "username": "nvidia", "private_key": "~/.ssh/id_rsa"},
        # {"host": "192.168.1.102", "username": "user2", "private_key": "~/.ssh/id_rsa"},
    ]

    threads = []
    for i in range(num_clients):
        thread = threading.Thread(
            target=fine_tune_remote_client, args=(i + 1, id, ssh_configs[i])
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All clients have completed their tasks.")

if __name__ == "__main__":
    main()
