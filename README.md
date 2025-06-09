## setup

```sh
pyenv virtualenv 3.10.0 snakeagent
pyenv local snakeagent
poetry install
```

## train

```sh
poetry run python -m src.train
```
