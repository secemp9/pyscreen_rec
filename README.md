# screen_rec
Screen Recorder supporting various libraries

# Features

It mostly just record the screen, with a broken support for capturing the cursor...it doesn't yet fully support DPI scaling, but will add this later, hopefully.
It uses the following for capturing the screen (optional, but not really since there no if confitions yet):
- Pillow
- Pyautogui
- mss

Pillow is the default one, since it's the fastest and easiest to work with. mss is faster, but I didn't find a way yet to include the broken mouse support. Pyautogui...it works in a similar way as Pillow does, but it uses too much cpu. Still kept it though.

# Why

OBS (among other ones) are too slow on my laptop (intel only). So I thought of making my own thing instead. I guess it's decently fast, but need to really benchmark this.

# TODO
- Add the rest of the attribution and credit for certain functions
- Fix the mouse since it's only present most of the time instead of all the time
- Fix the DPI scaling support (it's not really working right, at least according to the resulting video/frame)
- Fix the fps count, since it's really confusing to see 14 fps when it's actually much faster than that, seriously
- Fix mss so it support displaying the cursor too
- Add ctypes support for capturing frames
- Add other ways beside existing ones...maybe Directx/Opengl?
- Add benchmark to really see if it's as fast as I think it is
- Add sampling, antialiasing, bells and all the whistles :o
- Add more features?
