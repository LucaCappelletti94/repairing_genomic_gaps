docker build --file Dockerfile -t $(basename $PWD) .
docker run --gpus all --tty --interactive -v "$PWD:/home"  $(basename $PWD)