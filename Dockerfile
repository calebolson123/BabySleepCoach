FROM node:slim

WORKDIR /usr/app/babysleepcoach
EXPOSE 80

#Copy the pip build files in first
COPY ./requirements.txt .

# Install required packages
ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN apt-get update && apt-get install python3-pip libgl1 libglib2.0-0  -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy in the rest of the files
COPY . .

RUN cd webapp && yarn install && cd ..

WORKDIR /usr/app/babysleepcoach

ENTRYPOINT ["bash", "start_docker.sh"]
