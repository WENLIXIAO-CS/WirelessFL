#!/bin/bash

## prepare Raspberry Pi

USERNAME="pi"

HOSTS="192.168.XX.01 192.168.XX.02"

SCRIPT="pwd; ls"

# Transfer new version
# cd ..
DIR="$( cd "$( dirname "$0" )" && pwd )"
echo ${DIR}

# copy ssh-key
for HOSTNAME in ${HOSTS} ; do
    echo ${HOSTNAME}
    # copy ssh key 
    ssh-copy-id pi@${HOSTNAME}

    # remove old version code 
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/wirelessFL"

    # Upload new code 
    rsync -r -avu -e ssh --stats --exclude=.git --exclude=data/synthetic --exclude=log/* --exclude=cache/* "${DIR}" pi@${HOSTNAME}:/home/pi/fed-iot
    scp -rp "${DIR}" pi@${HOSTNAME}:/home/pi/fed-iot/

    # Install necessary library
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "sudo su - <<'EOF' 
    python -m pip install pickle5
    python -m pip install tensorboardX"
    echo ${HOSTNAME}
done
