# Used to test random ideas.
# Send file through putty

import paramiko
from scp import SCPClient   
import os

cwd = os.getcwd() 

ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.load_system_host_keys()
# Connect to the remote device
ssh.connect(hostname='192.168.55.1', username='nvidia', password='nvidia')

# Create SFTP client
# sftp = ssh.open_sftp()
scp = SCPClient(ssh.get_transport())

local_path = os.path.join(cwd, 'client1')
print(local_path)
remote_path = os.path.dirname('/Documents/Research/SafeDrive/client1')
scp.put(local_path, remote_path)

# Close the SFTP session
scp.close()
ssh.close()