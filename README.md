# Adaptive Heterogeneous Client Sampling for Federated Learning over Wireless Networks

<!-- Bunch of Icon -->
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi) ![Socket.io](https://img.shields.io/badge/Socket.io-black?style=for-the-badge&logo=socket.io&badgeColor=010101)

This repo demonstrates our WirelessFL[^1], an Internet of Things platform for Federated Learning. We constructed this platform to validate the performance of our proposed FL algorithm in real-world environment.

1. [Installation](#installation)
2. [Network Configuration](#network-configuration)
3. [Clients Setup](#clients-setup)
4. [Run Experiment](#run-experiment)
5. [Workflow in Paper](#workflow-in-paper)


## Installation

### 1.  Install Anaconda and required packages

The preferred approach for installing all the dependencies is to use [Anaconda](https://www.anaconda.com/products/individual), which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis. Once you install Anaconda you can run the following command inside the working directory to install the required packages for FL server:

```bash
conda env create -f wirelessFL.yml
```

Once you have all the packages installed, run the following command to activate the environment when you work on the server.

```bash
conda activate wirelessFL
```

### 2. Install and Config Matlab Engine

Our proposed algorithm will call MATLAB's `fmincon()` function to help solve optimization problem. Thus, you need to install necessary MATLAB package according to [Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

### 3. Download data and generate FL dataset

We used `EMNIST` dataset for prototype environment. To download data and generate non-i.i.d. dataset, run the following command from the `data/EMNIST` directory:

```bash
python generate_random_niid.py
```

The program will generate 40-client niid dataset by default. You can modify `NUM_USER` vairabke in `generate_random_niid.py` to change the number of users.

### 4. Create necessary folders

You could create necessary helper folders by running the following command:

```bash
./setup.sh
```

## Network Configuration

### 1. Connect all equipments to the same WiFi

The wireless communication is achieved in WiFi environment. A powerful WiFi router is required to support 40 or more clients' data transition. Before running experiment, you need to:

- Turn on the WiFi router;
- Connect the server (a laptop) to that WiFi;
- Connect all 40 Raspberry Pis to that WiFi.

**[OPTIONAL]** To save time, we provide two tips to quickly configure WiFi for Raspberry Pi :

1. Automatically connect the Raspberry Pi to WiFi[^2]
2. Bind Raspberry Pi to fixed IP address by following steps:
   - Open WiFi configuration page in browser (URL: `admin:admin@[IP OF ROUTER]`)
   - In `PORT Management / DHCP Setting`
        - Scan devices under this WiFi
        - Check device's connect by identity
   - In `VLAN: address binding`
        - Bind address of server and Pis to a fixed ip

### 2. Config IP Address in program

The end-to-end communication is implemented based on TCP protocol. We used socket programing in Python to achieve low-level speed monitoring. To ensure clients can connect to the server, you need to change `IP` variable in `src/communication/comm_config.py` to the IP address of server. The following shows the default configuration:

```Python

# Connection
IP="192.168.43.2"
```

## Clients Setup

Since the procedure of configuring IoT devices is tedious, we prepared two bash scripts to help automatically set up the clients and execute programs on Raspberry Pi remotely.

### 1. Install packages on Raspberry Pi

To run programs on Raspberry Pi, you need to install necessary packages and send the Python code to them. We prepared a `prepare.sh` bash script to automatically do that for all 40 clients. First, you need to write IP addresses of clients in the script like following:

```bash
HOSTS="192.168.XX.01 192.168.XX.02"
```

Then, you can run the following command to install package and send the latest version of code to clients:

```bash
sh prepare.sh
```

### 2. Set up `run.sh` script

We also prepared a script for you to remotely run client programs on Raspberry Pi for federated learning training. Like previous step, you need to first add IP address in the `run.sh` script:

```bash
HOSTS="192.168.XX.01 192.168.XX.02"
```

## Run Experiment

### 1. Set hyperparameters in `app.py` in `server_main()`



### 2. Run Experiment

First, run server by

```bash
python app.py --model server
```

Then, run clients by

```bash
./run.sh
```

or [optional] `python app.py --model client`

## Workflow in Paper

### 1. Pre-Training

To estimate $\frac{\alpha}{\beta}$ for our proposed algorithm, you have to first run few experiments on two benchmark sampling schems:

```python
args['algo'] = 'weighted'
fedServer.start_experiment()
fl_experiment(args, fedServer)

args['algo'] = 'uniform'
fedServer.start_experiment()
fl_experiment(args, fedServer)
```

### 2. Estimation and store gradient

Then, esitmate by execute cells in `estimate.ipynb`

### 3. Run experiments for comparation

After that, you can run experiments on proposed and statistical sampling to compare efficiency of shcemes:

```python
args['algo'] = 'proposed'
fedServer.start_experiment()
fl_experiment(args, fedServer)

args['algo'] = 'statistical'
fedServer.start_experiment()
fl_experiment(args, fedServer)
```

[^1]: The platform is constructed is based on the works in [fedavgpy](https://github.com/lx10077/fedavgpy) and [FedProx](https://github.com/litian96/FedProx).

[^2]: The following two links are helpful for auto connect Pis to WiFi: 1. [How To: Connect your Raspberry Pi to WiFi](https://raspberrypihq.com/how-to-connect-your-raspberry-pi-to-wifi/); 2. [Automatically connect a Raspberry Pi to a Wifi network](https://weworkweplay.com/play/automatically-connect-a-raspberry-pi-to-a-wifi-network/).