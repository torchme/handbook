# DSSM \[WIP]

### TODO LIST:

* [ ] Добавить интуицию
  * [ ] Представление текстовых данных
  * [ ] Абстрактная архитектура
  * [ ] Дот продукт
* [ ] Подробнее расписать про архитектуру
  * [ ] Разобраться про каждый слой
  * [ ] Можно ли добавлять другие слои (attention, менять структуру, могут ли быть разные головы, …)
* [ ] Добавить пример реализациия Two-Tower DSSM на PyTorch
* [ ] Модификации DSSM (Multi-head DSSM?) и доп ссылки его применения (WB, LAMODA, Yandex Market, Yandex Dzen, …)

***

## Интуицию

### BoW

### Embedding

***

## DSSM

Цель обучения состоит в том, чтобы увеличить вероятность того, что модель предскажет документ, на который пользователь кликнет, исходя из запроса. То есть, если пользователь кликнул на документ, этот документ считается релевантным для данного запроса, и модель должна обучиться предсказывать такие результаты.

* Входной слой: 30000 нейронов
  * Это соответствует размеру хешированного вектора признаков после обработки текста N-граммами.
* Первый скрытый слой: 300 нейронов
* Второй скрытый слой: 300 нейронов
* Выходной слой: 128 нейронов
  * Это финальное семантическое представление запроса или документа

текстовые данные обрабатываются через bag of words с применением n-грамм (обычно от 1 до 3) и после применяется хеширование

В оригинальной статье используется 3 полносвязных слоя с нелинейными функциями активациями tanh

*   **CODE EXAMPLE**

    ```python
    # loss func

    - L = -log P(d+|q) # -torch.log(prob)

    - P(d+|q) = exp(γ R(q,d+)) / Σ exp(γ R(q,di)) # prob
    - q - вектор запроса # query_vec
    - d+ - вектор релевантного документа # pos_doc_vec
    - di - векторы всех документов в наборе (включая релевантный) # all_cosines
    - R(q,d) - косинусное сходство между векторами # pos_cosine
    - γ - коэффициент масштабирования (обычно 10) # gamma

    import torch
    import torch.nn.functional as F

    def loss_function(query_vec, pos_doc_vec, neg_doc_vecs, gamma=10):
        # Вычисляем косинусное сходство между запросом и позитивным документом
        pos_cosine = F.cosine_similarity(query_vec, pos_doc_vec)
        
        # Вычисляем косинусное сходство между запросом и каждым негативным документом
        neg_cosines = [F.cosine_similarity(query_vec, neg_doc_vec) for neg_doc_vec in neg_doc_vecs]
        neg_cosines = torch.stack(neg_cosines)
        all_cosines = torch.cat([pos_cosine.unsqueeze(0)] + neg_cosines)
        
        # Применяем экспоненту с масштабированием
        exp_pos = torch.exp(gamma * pos_cosine)
        exp_all = torch.sum(torch.exp(gamma * all_cosines))
        
        # Вычисляем вероятность позитивного документа
        prob = exp_pos / exp_all
        
        # Вычисляем отрицательный логарифм вероятности (функция потерь)
        loss = -torch.log(prob) # 
        
        return loss
    ```

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class DSSM(nn.Module):
        def __init__(self, vocab_size=30000, hidden_size=300, semantic_size=128):
            super(DSSM, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(vocab_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, semantic_size)
            )
        
        def forward(self, x):
            return self.model(x)

    def cosine_similarity(x1, x2):
        return F.cosine_similarity(x1, x2)

    def loss_function(pos_cosine, neg_cosines, gamma=10):
        all_cosines = torch.cat([pos_cosine.unsqueeze(0), neg_cosines])
        exp_all = torch.sum(torch.exp(gamma * all_cosines))
        exp_pos = torch.exp(gamma * pos_cosine)
        return -torch.log(exp_pos / exp_all)
        
        return -torch.log(exp_pos / (exp_pos + exp_negs))

    class DSSMWrapper:
        def __init__(self, vocab_size=30000, hidden_size=300, semantic_size=128, lr= 1e-3):
            self.model = DSSM(vocab_size, hidden_size, semantic_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        def train_step(self, query, pos_doc, neg_docs):
            self.model.train()
            self.optimizer.zero_grad()
            
            query_vec = self.model(query)
            pos_doc_vec = self.model(pos_doc)
            neg_doc_vecs = [self.model(neg_doc) for neg_doc in neg_docs]
            
            pos_cosine = cosine_similarity(query_vec, pos_doc_vec)
            neg_cosines = torch.stack([cosine_similarity(query_vec, neg_doc_vec) for neg_doc_vec in neg_doc_vecs])
            
            loss = loss_function(pos_cosine, neg_cosines)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        def get_semantic_vector(self, x):
            self.model.eval()
            with torch.no_grad():
                return self.model(x)
    ```

    ```python
    # input example
    {
        "query": "лучшие рестораны в Москве",
        "positive_doc": "Top 10 ресторанов Москвы: обзор лучших заведений столицы",
        "negative_docs": [
            "Как приготовить борщ: пошаговый рецепт",
            "Достопримечательности Санкт-Петербурга",
            "Прогноз погоды на завтра",
            "Расписание поездов Москва-Петербург"
        ]
    }
    ```

    ```python
    # output exmaple
    {
        "query": [0.34, -0.12, 0.56, -0.78, 0.23, ...],
        "positive_doc": [0.31, -0.15, 0.52, -0.75, 0.26, ...] # similarity: 0.7823
        "negative_docs": [
            [0.11, 0.45, -0.23, 0.67, -0.34, ...], # similarity: 0.2145
            [0.22, -0.08, 0.41, -0.53, 0.18, ...], # similarity: 0.3012
            [-0.15, 0.33, 0.09, -0.42, -0.61, ...], # similarity: 0.1587
            [0.19, -0.07, 0.38, -0.62, 0.14, ...] # similarity: 0.2789
        ]
    }
    ```

***

## Two tower

### Recommender DSSM

DSSM — это нейросеть из двух башен. Каждая башня строит свой эмбеддинг, затем между эмбеддингами считается косинусное расстояние, это число — выход сети. То есть сеть учится оценивать близость объектов в левой и правой башне. Подобные нейросети используются, например, в [веб-поиске](https://habr.com/ru/company/yandex/blog/314222/), чтобы находить релевантные запросу документы. Для задачи поиска в одну из башен подаётся запрос, в другую — документ. Для нашей сети роль запроса играет пользователь, а в качестве документов выступают фильмы.

Башня фильма строит эмбеддинг на основе данных о фильме: это заголовок, описание, жанр, страна, актёры и т. д. Эта часть сети достаточно сильно похожа на поисковую. Однако для зрителя мы хотим использовать его историю. Чтобы это сделать, мы агрегируем эмбеддинги фильмов из истории с затуханием по времени с момента события. Затем поверх суммарного эмбеддинга применяем несколько слоёв сети и в итоге получаем эмбеддинг размера 400.

***

## CODE

[https://github.com/insdout/RecSys-Core-Algorithms/blob/main/4. DSSM.ipynb](https://github.com/insdout/RecSys-Core-Algorithms/blob/main/4.%20DSSM.ipynb)
