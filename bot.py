"""Telegram bot: every photo in a group gets the Wol Smoth treatment."""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, InputMediaPhoto, Message

import smoth

logging.basicConfig(level=logging.INFO)
bot = Bot(os.environ["BOT_TOKEN"])
dp = Dispatcher()
pool = ThreadPoolExecutor(1)  # mediapipe graphs are not thread-safe
MAX_SIDE = 2560


def _image_of(m: Message):
    if m.photo:
        return m.photo[-1].file_id
    if m.document and (m.document.mime_type or "").startswith("image/"):
        return m.document.file_id
    return None


def _render(data: bytes, effects):
    """{effect: jpeg bytes}; empty when the image is broken or has no faces."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    k = MAX_SIDE / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    found = smoth.faces(img)
    if not found:
        return {}
    return {e: cv2.imencode(".jpg", smoth.apply(img.copy(), e, found)[0], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            for e in effects}


async def smothify(m: Message, src: Message, effects):
    file_id = _image_of(src)
    if not file_id:
        return
    data = (await bot.download(file_id)).read()
    outs = await asyncio.get_running_loop().run_in_executor(pool, _render, data, effects)
    if len(outs) == 1:
        await m.reply_photo(BufferedInputFile(next(iter(outs.values())), "smoth.jpg"))
    elif outs:  # no face -> stay quiet, it's a group chat
        await m.reply_media_group([InputMediaPhoto(media=BufferedInputFile(b, f"{e}.jpg"), caption=f"/{e}")
                                   for e, b in outs.items()])


@dp.message(Command("start", "help"))
async def help_(m: Message):
    await m.reply("Кидай фотки — пришлю альбом со всеми эффектами. Ответь на фото командой эффекта "
                  "(" + ", ".join("/" + e for e in smoth.EFFECTS) + "), чтобы получить только его.")


@dp.message(Command(*smoth.EFFECTS))
async def effect_cmd(m: Message):
    src = m.reply_to_message or m
    await smothify(m, src, [m.text.split()[0].lstrip("/").split("@")[0]])


@dp.message(F.photo | F.document)
async def any_photo(m: Message):
    await smothify(m, m, list(smoth.EFFECTS))


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    asyncio.run(main())
