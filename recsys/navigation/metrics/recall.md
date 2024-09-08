# Recall

## Что такое Recall?

**Recall@k** — это метрика, которая показывает, какая часть всех релевантных элементов была рекомендована пользователю в первых N рекомендованных элементах. Проще говоря, она измеряет СКОЛЬКО РЕЛЕВАТНЫХ ТОВАРОВ из ТОП-N выдачи купил пользователь из ВСЕХ ВОЗМОЖНЫХ РЕЛЕВАТНЫХ ТОВАРОВ.

***

## Пример

У нас есть 10 товаров, которые мы показали пользователю. Тогда что бы посчитать полноту — recall@10 мы просто возьмем все релеватные товары в выдаче, а их у нас 5 _(под индексами 1, 3, 4, 6, 8)_ и поделим на все релевантные товары, а их у нас 8 _(под индексами 1, 3, 4, 6, 8, 11, 13, 14)_. recall@10 = 5/8 = 0.625

Грубо говоря это единички таргета в выдаче, по сравнению ко всем единичкам в таргетах.

Давайте теперь посмотрим в топ 5 рекомендаций. В этом коротком листе мы имеем только 3 релевантных элемента которые мы предложили. Тогда Recall@5 будет 37.5% (3 из 8). Это означает что система уловила менее половины элементов в топ-5 рекомендациях.

***

## Реализация на Python

```python
def recall(recommendation, targets):
    """Computes the recall at k for a single user."""
    if not isinstance(recommendation, (list, np.ndarray)):
        raise TypeError(f'recommendation must be a list or numpy.array, not {type(recommendation)}')
    if not isinstance(targets, (list, np.ndarray)):
        raise TypeError(f'targets must be a list or numpy.array, not {type(targets)}')

    if len(targets) == 0:
        return 0.0

    flags = np.isin(recommendation, targets)
    recall = float(np.sum(flags)) / len(targets)
    return recall

def recall_at_k(dataframe, k=5, user_col='user', item_col='item', score_col='score', target_col='target'):
    """Computes the recall at k for each user and returns the average."""
    grouped = dataframe.groupby(user_col)
    
    recalls = grouped.apply(lambda user_data: recall(
        user_data.sort_values(score_col, ascending=False)[item_col].values[:k],
        user_data[user_data[target_col] == 1][item_col].values
    ))
    
    return recalls.mean()
```

***

## Дополнительные ссылки

[https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
