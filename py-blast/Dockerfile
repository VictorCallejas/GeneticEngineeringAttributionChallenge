FROM python:3.8.5-buster

RUN wget https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.10.1+-x64-linux.tar.gz
RUN tar zxvpf ncbi-blast-2.10.1+-x64-linux.tar.gz
ENV PATH="/ncbi-blast-2.10.1+/bin:${PATH}"

COPY requirements.txt .
RUN python -m pip install -r requirements.txt