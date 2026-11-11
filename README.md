# Woll Smoth Bot

Telegram bot that gives every face in a group photo the *Wol Smoth* (tiny face) treatment. No generative AI: YuNet (OpenCV) finds faces, MediaPipe Face Mesh places landmarks, then plain OpenCV does the rest.

- **tiny** (default for any photo): liquify-style pinch. Features shrink to 55% as one block toward their centre and the skin around them stretches in, fading out over an ellipse 2x the feature size. It can't fold, so turned heads don't tear.
- **eyes**: tiny eyes, via a radial pinch warp.
- **squint**: eyes squashed vertically.
- **thatcher**: eyes and mouth flipped upside down in place.
- **moutheyes**: the mouth pasted over both eyes.
- **mirror**: the face rebuilt from its more visible half, mirrored across the midline.
- **bigeyes** / **bigmouth**: inverse pinch, eyes or mouth blown up.
- **cas**: liquid rescale (seam carving) of the face only, down to 55% and stretched back.

In a group, any photo (or image file) gets a reply with an album of every effect. Photos without faces are ignored. Reply to a photo with an effect command (`/tiny`, `/thatcher`, ...) to get just that one. The bot needs Group Privacy **off** (BotFather → /setprivacy), or admin rights, to see photos.

## Run

```sh
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python test_smoth.py
BOT_TOKEN=... .venv/bin/python bot.py
python smoth.py photo.jpg   # writes photo_<effect>.jpg for every effect
```

Docker: `docker build -t woll-smoth . && docker run -e BOT_TOKEN=... woll-smoth`. Uses long polling, so it needs no domain.

Credits: `models/yunet.onnx` is from [opencv_zoo](https://github.com/opencv/opencv_zoo) (MIT). `samples/will.jpg` is by Gage Skidmore, CC BY-SA 3.0, via Wikimedia Commons.
