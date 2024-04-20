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

This repo contains code to run the The Baby Sleep Coach. The Baby Sleep Coach provides sleep analysis of your baby, using only a video feed. You can access the sleep analysis via a locally hosted web app, which runs on launch.

## Pre-requisites

- Docker
- USB webcam or camera which supports RTSP
- Computer accessible via HTTP requests (I used a Raspberry Pi, but you can use any computer)

## tl;dr Run it with Docker

Copy the .env_sample template into .env

```cp .env_sample .env```

### Configure .env file:

`CAM_URL` URL of your camera. Likely `rtsp://admin:password@192.168.CAMERA_IP:554/h264Preview_01_sub` or `/dev/video0` for a webcam

`PORT` Port for accessing web app

`REACT_APP_BACKEND_IP` IP of backend/api layer. Likely 192.168.COMPUTER_IP:8001

`REACT_APP_RESOURCE_SERVER_IP` IP of resource server (runs on launch) Likely 192.168.COMPUTER_IP:8000

`HATCH_IP` (optional) IP of your hatch for wake light

`VIDEO_PATH` (optional) use to set path to recorded footage for debugging

*Remarks:* 
- instead of `192.168.COMPUTER_IP` you can use `raspberrypi.local` (or whatever name you gave when setting up your Raspberry Pi)
- when using `/dev/video0` (for USB webcam), also uncomment corresponding lines in `docker-compose.yml`

### Run it
`docker compose up`
