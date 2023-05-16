#Deriving the latest base image
FROM node:slim

WORKDIR /usr/app/babysleepcoach

RUN mkdir -p video

#Copy all files in the container
COPY . .

# Install required packages
RUN apt-get update && apt-get install python3-pip libgl1 libglib2.0-0  -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN cd webapp && yarn install

ENV SLEEP_DATA_PATH=/usr/app/babysleepcoach
ENV VIDEO_PATH=/usr/app/babysleepcoach/video

CMD ["./start_docker.sh"]