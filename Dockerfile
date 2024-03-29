FROM ubuntu:latest

# Install necessary dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get upgrade -y && apt-get install -y git && apt-get install -y libmagic1
RUN git clone https://github.com/tamu-edu-students/toucanstrike_webserver.git
# RUN cd toucanstrike

# Expose the port
EXPOSE 5000

# Install Python dependencies
RUN pip install flask colorama && pip install tqdm && pip install -U "ipython>=7.20"
# RUN PYTHONPATH=/usr/bin/python pip install -r requirements.txt
# RUN cd toucanstrike && git checkout webserver && pip install -r requirements.txt && python3 app.py
RUN cd toucanstrike_webserver && git pull && pip install -r requirements.txt


# Set the entrypoint command
CMD ["python3", "toucanstrike_webserver/app.py"]
