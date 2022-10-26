"""
This is VoiceTranscriptionBot (in fact, TranscribeVoiceBot ─
since all the other options were occupied). It transcribes any
incoming voice messages.
"""

import logging

from aiogram import Bot, Dispatcher, executor, types

API_TOKEN = 'PASTE YOUR TOKEN HERE'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'restart'])
async def send_welcome(message: types.Message):
    """
    The handler of `/start` or `/restart` commands
    """
    await message.reply(
            f"Hello, {message.from_user.first_name}!\n"
            "I'm VoiceTranscriptionBot!\n"
            "Send me a voice ─ I will transcibe it.")


@dp.message_handler(content_types=['voice'])
async def echo(message: types.Voice):
    file_info = await bot.get_file(message.voice.file_id)
    await bot.download_file(file_info.file_path, 'users_voice.ogg')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
