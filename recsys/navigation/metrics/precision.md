# Precision

## Что такое Precision?

**Precision@k** — это метрика, которая показывает, какая часть из первых N рекомендованных элементов действительно релевантна для пользователя. Проще говоря, она измеряет **СКОЛЬКО РЕЛЕВАТНЫХ ТОВАРОВ** купил пользователь среди **ВСЕХ ТОП-N ТОВАРОВ**.

***

## Пример

Представьте что у нас есть пользователь, он заказал в нашем ритейле некоторые продукты. Мы показали ему пиццу, конфеты, **шоколадку, пончик**, рыбу и вок, и из предложенных 6 товаров он купил только 2. Тогда что бы посчитать Precision@6 мы посчитаем 2/6 = 0.33

***

## Реализация на Python

```python
import numpy as np
import pandas as pd

# Создание примера данных
data = {
    'user': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    'item': [101, 102, 103, 104, 105, 106, 101, 102, 103, 104],
    'score': [4.5, 4.0, 3.0, 5.0, 2.0, 1.0, 3.5, 3.0, 4.0, 5.0],
    'target': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]  # 1 - релевантный элемент, 0 - нерелевантный элемент
}

dataframe = pd.DataFrame(data)

def precision(recommendation, targets):
    """Computes the precision at k for a single user."""
    if isinstance(recommendation, (list, np.ndarray)) == False:
        raise TypeError(f'recommendation must be a list or numpy.array, not {type(recommendation)}')
    if isinstance(targets, (list, np.ndarray)) == False:
        raise TypeError(f'targets must be a list or numpy.array, not {type(targets)}')

    flags = np.isin(recommendation, targets)
    precision = float(np.sum(flags)) / len(recommendation)
    return precision

def precision_at_k(dataframe, k=5, user_col='user', item_col='item', score_col='score', target_col='target'):
    """Computes the precision at k for each user and returns the average."""
    grouped = dataframe.groupby(user_col)
    
    precisions = grouped.apply(lambda user_data: precision(
        user_data.sort_values(score_col, ascending=False)[item_col].values[:k],
        user_data[user_data[target_col] == 1][item_col].values
    ))
    
    return precisions.mean()
```

***

## Дополнительные ссылки

[https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
