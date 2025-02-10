# Непрерывное дообучение без потерь
## Использование Lora и Mergekit

В этом документе мы рассмотрим, как выполнять непрерывное дообучение (Fine-tuning) моделей с открытым кодом, используя адаптер Lora, методы слияния (Merge) и библиотеку Mergekit. Для демонстрации мы будем использовать Unsloth, так как это самая простая библиотека для дообучения моделей с Lora.

### Введение
Меня зовут Ramas Uzkurys, но большинство знает меня как Rombodawg. Я начал с объединения моделей, потому что не мог позволить себе полноценное дообучение, пока не получил спонсорство от TensorDock. Это дало мне возможность обучать собственные модели, стремясь создать бесплатные AI-модели, которые могут конкурировать с закрытыми.

Однако при обсуждении с такими опытными специалистами в области Open Source AI, как Teknium (Nous-Research) и Eric Hartford (Cognitive Computations), выяснилось, что повторное дообучение уже обученной модели приводит к значительным потерям знаний (катастрофическому забыванию). Причина в том, что при каждом обучении веса модели обновляются, что приводит к утрате части информации, полученной на предыдущих этапах. Таким образом, даже если модель проходит этап инструкционного обучения, она может потерять часть знаний, полученных при предобучении.

### Решение: Метод (Lora + Ties-Merge)
Чтобы избежать потерь, мы используем методику "Lora + Ties-Merge". Lora позволяет выбрать, когда и где применять дообученные веса, а слияние помогает сохранить все изменения без потери предыдущих данных.

## Алгоритм работы

### 1. Выбираем базовую модель
```
Base Model: Qwen/Qwen2-7B
```

### 2. Дообучаем на инструкционном датасете с таким же форматом чата, как у целевой модели:
```
Dataset: rombodawg/Everything_Instruct_8k_context_filtered
Target Merge Model: Qwen/Qwen2-7B-Instruct
```

Формат чата (ChatML):
```xml
<|im_start|>system

{}<|im_end|>

<|im_start|>user

{}<|im_end|>

<|im_start|>assistant

{}
```

Для обучения используем Unsloth:
[Colab-ноутбук](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)

### 3. Сохраняем Lora
```python
model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

### 4. Применяем Lora к целевой модели, а не к базовой
Это важный момент! Обучение производится на базовой модели, а Lora применяется на целевой.

### 5. Объединяем веса с помощью Mergekit
[Mergekit GitHub](https://github.com/arcee-ai/mergekit)

Пример конфигурации Mergekit:
```yaml
models:
  - model: Lora_model_7b
    parameters:
      weight: 1
  - model: Qwen_Qwen2-7B-Instruct
    parameters:
      weight: 1
merge_method: ties
base_model: Qwen_Qwen2-7B
parameters:
  normalize: true
  int8_mask: true
dtype: bfloat16
```

После слияния все веса объединяются в одну модель без потерь параметров.

---

## ВАЖНОЕ ОБНОВЛЕНИЕ
Лучше объединять все на одном этапе, даже если дообучение продолжается:
```yaml
models:
  - model: Qwen2.5-7B-Instruct-Adapted-Finetune-1
    parameters:
      weight: 1
  - model: Qwen2.5-7B-Instruct-Adapted-Finetune-2
    parameters:
      weight: 1
  - model: Qwen2.5-7B-Instruct-Adapted-Finetune-3
    parameters:
      weight: 1
  - model: Qwen2.5-7B-Instruct-Adapted-Finetune-4
    parameters:
      weight: 1
  - model: Qwen2.5-7B-Instruct
    parameters:
      weight: 1
merge_method: ties
base_model: Qwen2.5-7B-Base
parameters:
  normalize: true
  int8_mask: true
  tokenizer_source: Qwen2.5-7B-Instruct-Adapted-Finetune-1
  dtype: bfloat16
```

Сохранение адаптеров и их одновременное объединение повышает качество итоговой модели.

---

## Еще одно обновление: добавление веса и плотности
```yaml
models:
  - model: Qwen2.5-7B-Instruct-Adapted-Finetune-1
    parameters:
      weight: 1
      density: 1
  - model: Qwen2.5-7B-Instruct
    parameters:
      weight: 1
      density: 1
merge_method: ties
base_model: Qwen2.5-7B-Base
parameters:
  weight: 1
  density: 1
  normalize: true
  int8_mask: true
  tokenizer_source: Qwen2.5-7B-Instruct-Adapted-Finetune-1
  dtype: bfloat16
```

Добавление этих параметров значительно улучшает итоговую модель.

---

**Спасибо за внимание! Надеюсь, этот метод поможет вам создавать мощные модели.**

*Автор: Rombodawg / Ramas Uzkurys*
