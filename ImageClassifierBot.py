"""
This is ImageClassifierBot (in fact, TranscribeVoiceBot ─
I should have come up with a more general name than using the one
for different tasks). It classifies an image of an animal.
"""

import logging
import requests

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = 'PASTE YOUR TOKEN HERE'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'restart'])
async def send_welcome(message: types.Message):
    """
    Handle `/start` or `/restart` commands
    """
    await message.reply(
            f"Hello, {message.from_user.first_name}!\n"
            "I'm ImageClassifierBot!\n"
            "Send me an image of an animal"
            " ─ I will guess who is on it.")


@dp.message_handler(content_types=['photo'])
async def classify(message: types.Message):
    """
    Classify an image similar to those from ImageNet dataset.
    """
    filename = 'input_image.jpeg'
    file_info = await bot.get_file(message.photo[0].file_id)
    await bot.download_file(file_info.file_path, filename)
    req = requests.get(' http://127.0.0.1:8000/')
    subjects = req.json()['message'][filename]

    if len(subjects) > 1:
        reply = 'It might be one of the following: '
        reply += ', '.join(map(str.lower, subjects))
        reply += f'. I will say it is a/an {subjects[0].lower()}'

    else:
        reply = f'I think it is a/an {subjects[0].lower()}'

    await bot.send_message(message.from_user.id, f'{reply}')


@dp.message_handler()
async def chat(message: types.Message):
    """
    Tell that you are not really into chatting.
    """
    await message.answer(
            "I don't speak much. Send me an image and"
            'I will try to guess what it contains!')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
