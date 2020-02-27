docker build --file Dockerfile -t $(basename $PWD) .
nvidia-docker run --tty --interactive --volume "$PWD:/home"  $(basename $PWD)