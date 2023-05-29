# The Baby Sleep Coach

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---
## [As seen on YouTube](https://www.youtube.com/channel/UCzxiOKO3vX1ER_U3Z_eY_yw)

This repo contains code to run the AI Baby Sleep Tracking service as well as a web application which provides the user with analysis and charts based on the recorded sleep data.

## Pre-requisites

- Camera which supports RTSP
- Compute accessible via HTTP requests (I used a Raspberry Pi, but you can use any computer)
- Python 3.10 or lower (3.11 is not supported by MediaPipe [[see](https://github.com/google/mediapipe/issues/1325)])

## Setup

There are two components to configure:
1) A sleep tracking python script
2) A web application

### Warning: jank ahead

I don't treat these projects as I would in industry. The result is a very monolithic and duct-tapey project.

If there is a desire, I will refactor and make this more repeatable and easy to use.

### Part 1: Sleep Tracking Script

Install requirements: `pip install -r requirements.txt`

Run: `python main.py`

That's it. Except this is where the fun starts. 

Most of the dependencies are self explanatory, the only issue I had was installing [MediaPipe](https://google.github.io/mediapipe/) on Raspbian. I believe I used https://pypi.org/project/mediapipe-rpi4/, but I ran into a number of other issues I won't document here. glhf

There are number of environment variables and holes you'll need to fill with info about your environment. Instead of fixing things, I left a lot of comments.

Alternatively you can `touch .env` and then copy and paste the contents of `.env_sample` into it. Then fill in the blanks.

The sleep data is written to `sleep_logs.csv`. I primed this file with a few rows as an example. Feel free to remove these and start from scratch.

### Part 2: The Web App

This one is more straight forward. Just make sure you have [`yarn`](https://yarnpkg.com/getting-started/install).

Execute the following commands:

`cd webapp; yarn install; yarn start;`

And you'll probably get a warning about the app trying to boot on port `80`. You can change it to whatever you want in the package.json.

You'll need to update some paths and IPs in the code.
<br/><br/>
## Someone send me proof you got it all running.
