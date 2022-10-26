#!/bin/bash

function help_msg () {
  echo 'Usage: ./start.sh [<options>]'
  echo
  echo -e '  -h, --help\t\t Show this message.'
  echo -e '  -m, --model-name\t Set classification NN model.'
  echo -e '  -0, --from-scratch\t Rebuild all components before the launch.'
  echo -e '  -c, --topk-guesses\t Number of guessed classes.'
}


function main () {
  short_opts=h,0,c:,m:
  long_opts=help,from-scratch,topk-guesses:,model-name:
  params=$(getopt -o $short_opts -l $long_opts  --name "$0" -- "$@")

  eval set -- "$params"
  unset params

  while [[ $1 != -- ]]; do
    case $1 in
      -h|--help)           help_msg; return ;;
      -0|--from-scratch)   from_scratch=true; shift 1 ;;
      -c|--topk-guesses)   topk=$2; shift 2 ;;
      -m|--model-name)     model=$2; shift 2 ;;

      *) echo Impl.error; return 1 ;;
    esac
  done

  server_img=nvcr.io/nvidia/tritonserver:21.09-py3
  client_img=triton-client:dev

  models="$(realpath "$(dirname "$0")/models")"
  workdir="$(realpath "$(dirname "$0")")"

  if [[ ! -d $models || -z $(ls -A "$models") ]]; then
    echo "You don't have any models."
    echo "But you can fetch some with 'fetch_models.sh' script."
    return
  fi

  if ${from_scratch:=false}; then
    from_scratch=true
    docker stop server 2> /dev/null
    docker stop client 2> /dev/null
  fi

  if $from_scratch || [[ -z $(docker images -q "$server_img") ]]; then
    docker pull "$server_img"
  fi

  if $from_scratch || [[ -z $(docker images -q "$client_img") ]]; then
    docker build -t "$client_img" docker/
  fi


  docker network create backend 2> /dev/null

  if [[ ! $(docker ps -q -f name=server) ]]; then
    docker run --rm -d --name server --net=backend -v"$models":/models \
      $server_img tritonserver --model-repository=/models
  fi

  if [[ ! $(docker ps -q -f name=client) ]]; then
    docker run --rm -d --name client \
      -e CLASSIFIER_TOPK="${topk:-1}" \
      -e CLASSIFIER_MODEL="${model:-inception_graphdef}" \
      --net=backend -p8000:80 -v"$workdir":/workspace $client_img
  fi
}


main "$@"
