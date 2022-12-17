#!/bin/bash

set -e
export DOCKER_BUILDKIT=1
choices='efficientnet-lite4 inception-graphdef quartznet15x5'

help_msg () {
  echo 'Usage: ./start.sh [<options>]'
  echo
  printf '  %-20s %s\n' '-h, --help' 'Show this message.'
  printf '  %-20s %s\n' '-0, --from-scratch' 'Rebuild all components before the launch.'
  echo
  printf '  %-20s %s\n' '-p' "Publish client's port to the host"
  printf '  %-20s %s' '-t,--tag' 'Tag of the Triton server image.'
  printf ' Default: 22.11-py3\n'
  printf '  %-20s %s' '-n, --names' 'Names for server and client separated by comma.'
  printf ' Default: server,client\n\n'
  printf '  %-20s %s' '-m, --models' 'Set classification and/or ASR NN model(s).'
  printf ' Default: inception-graphdef,quartznet15x5\n'
  printf '  %-20s Default choices: %s\n' '' "$choices"
  printf '  %-20s %s' '-c, --topk-guesses' 'Number of guessed classes'
  printf ' (classification task). Default: 1\n'
}


container_is_running () {
  [[ $# -eq 0 ]] && { echo Impl.error1: pass an argument.; exit 1; }
  [[ -n $(docker ps -q -fname="$1") ]] && return 0 || return 1
}


launch_and_check () {
  local name=$1
  local cmd_args=$2
  local stop_word=$3

  ## NOTE: if the stop_word is the last line in the stream,
  ## the `docker logs -f ...` will get you into an infinite loop.
  if [[ -z $stop_word ]]; then
    echo Impl.error2: No 3rd arg given
  fi

  local run_cmd="docker run --rm -d --name $name --net=backend"
  local logfile=/tmp/asr-telegram-bot.log
  ## Clear previous sessions logs, keep only relevant.
  > $logfile

  printf "Starting %s.." "$name"
  if ! container_is_running "$name"; then
    eval "$run_cmd $cmd_args > /dev/null 2>> $logfile"
    (docker logs -f "$name" 2>&1 | sed "/$stop_word/ q") >> $logfile

    ## The following does not help
    # (docker logs -f "$name" 2>&1 | sed 'a
    # ' | sed "/$stop_word/ q") >> $logfile

    sleep 2  # Unreliable: hard-coded and should be adjusted manually
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


init_variables () {
  local short_opts long_opts params names models
  long_opts=help,from-scratch,tag:,names:,models:,topk-guesses:
  short_opts=h,0,p:,t:,n:,m:,c:

  params=$(getopt -o $short_opts -l $long_opts  --name "$0" -- "$@")
  eval set -- "$params"

  while [[ $1 != -- ]]; do
    case $1 in
      -h|--help)            help_msg; exit ;;
      -0|--from-scratch)    from_scratch=true; shift 1 ;;

      -p)                   port=$2; shift 2 ;;
      -t|tag)               tag=$2; shift 2 ;;
      -n|--names)           names=$2; shift 2 ;;

      -c|--topk-guesses)    topk=$2; shift 2 ;;
      -m|--models)          models=$2; shift 2 ;;

      *) echo Impl.error3: Infinite args parsing; exit 1 ;;
    esac
  done

  workdir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
  server_img=nvcr.io/nvidia/tritonserver:${tag:-22.11-py3}
  client_img=triton-client:tg-bot

  server=$(cut -d, -f1 <<< "$names")
  client=$(cut -d, -f2 <<< "$names")
  server=${server:-server}
  client=${client:-client}

  [[ $models =~ , ]] || models+=,
  ## if there is no delimiter, the fields will be the same.
  visual_model=$(cut -d, -f1 <<< "$models")
  acoustic_model=$(cut -d, -f2 <<< "$models")
}


main () {
  init_variables "$@"

  if ${from_scratch:=false}; then
    docker stop "$server" &> /dev/null || :  # discard return value; this way,
    docker stop "$client" &> /dev/null || :  # the cmd is ignored by `set -e`
  fi

  if $from_scratch || [[ -z $(docker images -q "$server_img") ]]; then
    docker pull "$server_img"
  fi

  if $from_scratch || [[ -z $(docker images -q "$client_img") ]]; then
    docker build -t "$client_img" "$workdir/client/"
  fi

  # Ignore the exit code of network creation if the network already exists.
  docker network create backend 2> /dev/null || :

  launch_and_check "$client" \
    "-e ASR_MODEL=${acoustic_model:-quartznet15x5} \
     -e CLASSIFIER_MODEL=${visual_model:-inception-graphdef} \
     -e CLASSIFIER_TOPK=${topk:-1} \
     -p${port:-8000}:80 -v$workdir:/workspace $client_img" \
     'Application startup complete'

  if  [[ -d $workdir/server/models ]]; then
    folders=$(ls "$workdir/server/models")
    skip=true
  fi

  for model in $choices; do
    [[ $folders =~ $model ]] || skip=false
  done

  if ! $skip; then
    printf 'Copying default models..'
    mkdir -p "$workdir/server"
    docker cp "$client":/models "$workdir/server"
    printf '\033[1K\r'
  fi

  launch_and_check "$server" \
    "-v$workdir/server/models:/models $server_img \
    tritonserver --model-repository=/models" \
    'Started GRPCInferenceService at'
}


main "$@"
