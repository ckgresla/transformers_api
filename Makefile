# MAKE your life easier

SHELL := /bin/bash
DOCKER_CONTAINER ?= transformers_api

.PHONY: build start test


build:
	docker build -t $(DOCKER_CONTAINER) -f Dockerfile .

start:
	docker run -v ${PWD}:/api -p 5003:5003 $(DOCKER_CONTAINER)

test:
	curl localhost:5003/health




