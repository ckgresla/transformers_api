FROM python:3.8

COPY requirements.txt ./requirements.txt
COPY controllers ./controllers
COPY main.py ./main.py
COPY download_hf_models.py ./download_hf_models.py
COPY models/ ./models

RUN export PYTHONPATH=/usr/bin/python  \
    && pip install -r requirements.txt \
    && pip install --upgrade pip       \
    && pip3 install --use-deprecated=legacy-resolver torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html



# CMD ["python", "./main.py", "&&", "python", "./download_hf_models.py"]
CMD ["python",  "-u", "./main.py"]
