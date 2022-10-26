# Image Classification Telegram Bot

## Usage

1. Create a Telegram bot.
1. Obtain its token (can be done via BotFather) and assign it to  
   `API_TOKEN` string variable in `ImageClassifierBot.py` file.
1. Use `./start.sh` to launch triton-server and `fastapi` 'dummy client'.  
   You can also check the script options with `./start.sh -h`.
1. Launch the bot with `python ImageClassifierBot.py` command.

## References
[triton-inference-server](https://github.com/triton-inference-server/server.git)
─ some files are taken from there
