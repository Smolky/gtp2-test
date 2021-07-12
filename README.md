# gtp2-test
1. Instalar Python y PiP
2. Crear y activar entorno virtual https://docs.python.org/3/library/venv.html
3. Instalar requisitos ```pip install -r requirements.txt```


# Uso
```test.py``` permite probar el modelo. Hay que cambiar la línea:

La línea 
```
huggingface_model = 'dccuchile/bert-base-spanish-wwm-cased'
```

permite usar el modelo por defecto, y
```
huggingface_model = 'finetune'
```

el modelo ajustado.

El texto para probar, está en la variable ```text```


Para generar el modelo ajustado 
```
python fine-tune.py
```
