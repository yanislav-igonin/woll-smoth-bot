# Woll Smoth Bot

Telegram bot that gives every face in a group photo the *Wol Smoth* (tiny face) treatment. No generative AI: YuNet (OpenCV) finds faces, MediaPipe Face Mesh places landmarks, then plain OpenCV does the rest.

- **tiny** (default for any photo): liquify-style mesh warp. Features shrink to 50% as one block and the real skin of the face stretches in around them, keeping its texture; the face outline does not move.
- **eyes**: tiny eyes, via a radial pinch warp.
- **squint**: eyes squashed vertically.

In a group, any photo (or image file) gets a reply with the tiny-face version. Photos without faces are ignored. Reply to a photo with `/tiny`, `/eyes` or `/squint` to apply a different filter. The bot needs Group Privacy **off** (BotFather → /setprivacy), or admin rights, to see photos.

## Run

```sh
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python test_smoth.py
BOT_TOKEN=... .venv/bin/python bot.py
python smoth.py photo.jpg   # writes photo_tiny.jpg, photo_eyes.jpg, photo_squint.jpg
```

Docker: `docker build -t woll-smoth . && docker run -e BOT_TOKEN=... woll-smoth`. Uses long polling, so it needs no domain.

Credits: `models/yunet.onnx` is from [opencv_zoo](https://github.com/opencv/opencv_zoo) (MIT). `samples/will.jpg` is by Gage Skidmore, CC BY-SA 3.0, via Wikimedia Commons.
