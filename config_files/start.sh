#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/robot.env

# check docker login on git.uwaterloo.ca:5050
if ! grep -q "git.uwaterloo.ca:5050" ~/.docker/config.json ; then
    echo "Enter your UW login to log in to git.uwaterloo.ca docker registry"
    echo "If this is a public machine you can log out using docker logout git.uwaterloo.ca:5050"
    docker login git.uwaterloo.ca:5050
fi

# install files in config if they are not there
searchpath=${DIR}/configs/
searchpath_len=$((${#searchpath}+1))
for filename in $(find "${searchpath}"); do
    entry=$(echo $filename | cut --bytes=${searchpath_len}-)
    dest=${DIR}/../${entry}
    if [ ! -e ${dest} ]; then
    	echo "Installing config file $(basename ${filename}) in $dest"
        cp -R ${filename} ${dest}
    fi
done
if [ -f $DIR/../.ssh/.id_rsa_robohub ]; then
    chmod go-rwx $DIR/../.ssh/id_rsa_robohub
fi

# copy gitconfig from home
if [ ! -f ${DIR}/../.gitconfig ] && [ -f $HOME/.gitconfig ]; then
   cp $HOME/.gitconfig ${DIR}/../
   # install some default git config elements
   git config --file ${DIR}/../.gitconfig alias.lg "log --pretty=oneline --abbrev-commit --graph --decorate --all"
fi

# Variables required for logging as a user with the same id as the user running this script
export LOCAL_USER_NAME=$USER
export LOCAL_USER_ID=`id -u $USER`
export LOCAL_GROUP_ID=`id -g $USER`
export LOCAL_GROUP_NAME=`id -gn $USER`
DOCKER_USER_ARGS="--env LOCAL_USER_NAME --env LOCAL_USER_ID --env LOCAL_GROUP_ID --env LOCAL_GROUP_NAME"

# Variables for forwarding ssh agent into docker container
SSH_AUTH_ARGS=""
if [ ! -z $SSH_AUTH_SOCK ]; then
    DOCKER_SSH_AUTH_ARGS="-v $(dirname $SSH_AUTH_SOCK):$(dirname $SSH_AUTH_SOCK) -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
fi

# Settings required for having nvidia GPU acceleration inside the docker
DOCKER_GPU_ARGS="--env DISPLAY --ipc=host --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"

which nvidia-docker > /dev/null 2> /dev/null
HAS_NVIDIA_DOCKER=$?
if [ $HAS_NVIDIA_DOCKER -eq 0 ]; then
  DOCKER_COMMAND=nvidia-docker
  DOCKER_GPU_ARGS="$DOCKER_GPU_ARGS --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all"
else
  #echo "Running without nvidia-docker, if you have an NVidia card you may need it"\
  #"to have GPU acceleration"
  DOCKER_COMMAND=docker
fi


xhost + 

#ADDITIONAL_FLAGS="--detach"
ADDITIONAL_FLAGS="--rm --interactive --tty"
ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --device /dev/dri:/dev/dri --volume=/run/udev:/run/udev"

# forward spacemouse input device
if [ -e /dev/input/js0 ]; then
	ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --device /dev/input/js0"
	readlink -f /dev/input/js0 > /dev/null
	if [ 0 == "$?" ]; then
		ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS --device $(readlink -f /dev/input/js0)"
	fi
fi


if [ -z "${IMAGE_NAME}" ]; then
	IMAGE_NAME=git.uwaterloo.ca:5050/robohub/docker:${ROBOTNAME}_stable
	if [ ! -z "${1}" ]; then
		IMAGE_NAME=git.uwaterloo.ca:5050/robohub/docker:${1}
	fi
fi
echo Starting container: $IMAGE_NAME

if [ ! -z "${DOCKER_ROBOT_FLAGS}" ]; then
    ADDITIONAL_FLAGS="${ADDITIONAL_FLAGS} ${DOCKER_ROBOT_FLAGS}"
fi

CONTAINER_NAME=uw_${ROBOTNAME}_${USER}
if ! docker container ps | grep -q ${CONTAINER_NAME}; then
	echo "Starting new container with name: ${CONTAINER_NAME}"
	$DOCKER_COMMAND run \
	$DOCKER_USER_ARGS \
	$DOCKER_GPU_ARGS \
	$DOCKER_SSH_AUTH_ARGS \
	-v "$DIR/..:/home/${USER}" \
	$ADDITIONAL_FLAGS --user root \
	--name ${CONTAINER_NAME} --workdir /home/$USER \
	--cap-add=SYS_PTRACE \
	--cap-add=SYS_NICE \
	--net host \
	--device /dev/bus/usb \
	$IMAGE_NAME
else
	echo "Starting shell in running container"
	docker exec -it --workdir /home/${USER} --user $(whoami) ${CONTAINER_NAME} bash -l -c "stty cols $(tput cols); stty rows $(tput lines); bash"
fi

