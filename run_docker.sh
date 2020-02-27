docker build --file Dockerfile -t $(basename $PWD) .
nvidia-docker run --tty --interactive  $(basename $PWD)