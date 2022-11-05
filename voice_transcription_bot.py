"""
This is VoiceTranscriptionBot (in fact, TranscribeVoiceBot ─
since all the other options were occupied). It transcribes any
incoming voice messages. But only those in English are good :)
"""
import logging
import argparse

import requests
from aiogram import Bot, Dispatcher, executor, types


parser = argparse.ArgumentParser()
parser.add_argument(
        '-p',
        dest='port',
        default=8000,
        type=int,
        help='Port number to the host')
parser.add_argument(
        '-t',
        '--api-token',
        required=True,
        help="Your telegram bot's token")

args = parser.parse_args()
HOST = f'http://127.0.0.1:{args.port}'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=args.api_token)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'restart'])
async def send_welcome(message: types.Message):
    """
    The handler of `/start` or `/restart` commands.
    """
    await message.reply(
            f"Hello, {message.from_user.first_name}!\n\n"
            "I'm TranscribeVoiceBot! In narrow circles, "
            "I'm also known as ImageClassifierBot, but never mind\n\n"
            "Send me a voice (lang: eng) ─ I will transcibe it for you.")


@dp.message_handler()
async def chat(message: types.Message):
    """
    Tell that you are not really into chatting.
    """
    await message.answer(
            "I don't speak much. Send me an audio and "
            'I will transcribe it for you! If it is in English..')


@dp.message_handler(content_types=['voice'])
async def transcibe(message: types.Message):
    """
    Transcribe voice message to English text.
    """
    filename = 'user_voice.ogg'
    file_info = await bot.get_file(message.voice.file_id)
    await bot.download_file(file_info.file_path, filename)

    req = requests.get(f'{HOST}/voice')
    transcription = req.json()['message']

    reply = 'Check if the following is on the record?:'
    reply += transcription

    await bot.send_message(message.from_user.id, f'{reply}')


@dp.message_handler(content_types=['photo'])
async def classify(message: types.Message):
    """
    Classify an image similar to those from ImageNet dataset.
    """
    filename = 'input_image.jpeg'
    file_info = await bot.get_file(message.photo[0].file_id)
    await bot.download_file(file_info.file_path, filename)

    req = requests.get(f'{HOST}/image')
    subjects = req.json()['message'][filename]

    if len(subjects) > 1:
        reply = 'It might be one of the following: '
        reply += ', '.join(map(str.lower, subjects))
        reply += f'. I will say it is a/an {subjects[0].lower()}'

    else:
        reply = f'I think it is a/an {subjects[0].lower()}'

    await bot.send_message(message.from_user.id, f'{reply}')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
