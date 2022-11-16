FROM python:3.8

WORKDIR /api
COPY . /api

RUN export PYTHONPATH=/usr/bin/python  \
    && pip install --upgrade pip       \
    && pip install -r requirements.txt \
    && pip3 install --use-deprecated=legacy-resolver torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


EXPOSE 5003

# CMD ["python", "./main.py", "&&", "python", "./download_hf_models.py"]
CMD ["python",  "-u", "./main.py"]
