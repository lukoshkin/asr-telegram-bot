# ASR/CV Telegram Bot

## Usage

0. Prepare the environment. Note that parameters in  
   `grpc_audio_client.py` are adjusted to `quartznet15x5` model.
   ```bash
   ./fetch_models.sh  # CV models; ASR ─ you load manually (onnx)
   pip install requests aiogram  # telegram bot API dependencies
   ```
1. Create a Telegram bot, obtain its token (all can be done via BotFather).
1. Use `./start.sh` to launch triton-server and `fastapi` "dummy client".  
   You can also check the script options with `./start.sh -h` command.
1. Pass the token to the script launching the telegram bot
   ```bash
   python voice_transcription_bot.py -t YOUR_API_TOKEN
   ```

## References
- [triton-inference-server](https://github.com/triton-inference-server/server.git)
   ─ some files from there (`grpc_image_client.py` and `fetch_models.sh`)  
   are adjusted to the use in this repository.
