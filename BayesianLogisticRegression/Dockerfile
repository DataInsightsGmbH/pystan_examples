FROM ubuntu:18.04

# update apt-get and install Python (version 3.X) and Pip for Python3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN mkdir -p /home/di_user/bayes_log_reg/

# set the working directory (target-directory for copying-operations)
WORKDIR /home/di_user/bayes_log_reg

# copy everything
COPY . .

# install the required Python-libraries
RUN pip3 install -r requirements.txt
