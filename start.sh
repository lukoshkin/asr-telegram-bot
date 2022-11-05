#!/bin/bash

function help_msg () {
  echo 'Usage: ./start.sh [<options>]'
  echo
  echo -e '  -h, --help\t\t Show this message.'
  echo -e '  -0, --from-scratch\t Rebuild all components before the launch.'
  echo
  printf "  -p\t\t\t Publish client's port to the host\n"
  printf '  -t,--tag\t\t Tag of the Triton server image.'
  printf ' Default: 21.09-py3\n'
  printf '  -n, --names\t\t Names for server and client separated by comma.'
  printf ' Default: server,client\n\n'
  printf '  -m, --models\t\t Set classification and/or ASR NN model(s).'
  printf ' Default: inception_graphdef,quartznet15x5\n'
  printf '  -c, --topk-guesses\t Number of guessed classes'
  printf ' (classification task). Default: 1\n'
}


function container_is_running () {
  [[ $# -eq 0 ]] && { echo Impl.error1: pass an argument.; exit 1; }
  [[ -n $(docker ps -q -fname="$1") ]] && return 0 || return 1
}


function launch_and_check () {
  local name=$1
  local cmd_args=$2
  local stop_word=$3

  ## NOTE: if the stop_word is the last line in the stream,
  ## the `docker logs -f ...` will get you into an infinite loop.
  if [[ -z $stop_word ]]; then
    echo Impl.error2: No 3rd arg given
  fi

  local run_cmd="docker run --rm -d --name $name --net=backend"
  local logfile=/tmp/nn-powered-tg-bot.log

  printf "Starting %s.." "$name"
  if ! container_is_running "$name"; then
    eval "$run_cmd $cmd_args > /dev/null 2>> $logfile"
    (docker logs -f "$name" 2>&1 | sed "/$stop_word/ q") >> $logfile

    ## The following does not help
    # (docker logs -f "$name" 2>&1 | sed 'a
    # ' | sed "/$stop_word/ q") >> $logfile

    sleep 1
    if container_is_running "$name"; then
      printf '\b\b [✓]\n'
      return 0
    else
      printf '\b Unable to launch [X]\nSee logs at %s\n' $logfile
      return 1
    fi
  fi
  printf '\b Running already.\n'
}


function main () {
  short_opts=h,0,p:,t:,n:,m:,c:
  long_opts=help,from-scratch,tag:,names:,models:,topk-guesses:

  params=$(getopt -o $short_opts -l $long_opts  --name "$0" -- "$@")
  eval set -- "$params"
  unset params

  while [[ $1 != -- ]]; do
    case $1 in
      -h|--help)            help_msg; return ;;
      -0|--from-scratch)    from_scratch=true; shift 1 ;;

      -p)                   port=$2; shift 2 ;;
      -t|tag)               tag=$2; shift 2 ;;
      -n|--names)           names=$2; shift 2 ;;

      -c|--topk-guesses)    topk=$2; shift 2 ;;
      -m|--models)          models=$2; shift 2 ;;

      *) echo Impl.error3: Infinite args parsing; return 1 ;;
    esac
  done

  server_img=nvcr.io/nvidia/tritonserver:${tag:-21.09-py3}
  client_img=triton-client:dev

  model_depo="$(realpath "$(dirname "$0")/models")"
  workdir="$(realpath "$(dirname "$0")")"

  server=$(cut -d, -f1 <<< "$names")
  client=$(cut -d, -f2 <<< "$names")
  server=${server:-server}
  client=${client:-client}

  visual_model=$(cut -d, -f1 <<< "$models")
  audio_model=$(cut -d, -f2 <<< "$models")

  if [[ ! -d $model_depo || -z $(ls -A "$model_depo") ]]; then
    echo "You don't have any models."
    echo "But you can fetch some with 'fetch_models.sh' script."
    return
  fi

  if ${from_scratch:=false}; then
    docker stop "$server" 2> /dev/null
    docker stop "$client" 2> /dev/null
  fi

  if $from_scratch || [[ -z $(docker images -q "$server_img") ]]; then
    docker pull "$server_img"
  fi

  if $from_scratch || [[ -z $(docker images -q "$client_img") ]]; then
    docker build -t "$client_img" docker/
  fi

  docker network create backend 2> /dev/null
  launch_and_check "$server" \
    "-v$model_depo:/models $server_img \
    tritonserver --model-repository=/models" \
    'Started GRPCInferenceService at' || return 1

  launch_and_check "$client" \
    "-e ASR_MODEL=${audio_model:-quartznet15x5} \
     -e CLASSIFIER_MODEL=${visual_model:-inception_graphdef} \
     -e CLASSIFIER_TOPK=${topk:-1} \
     -p${port:-8000}:80 -v$workdir:/workspace $client_img" \
     'Application startup complete' || return 1
}


main "$@"
