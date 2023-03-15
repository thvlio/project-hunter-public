# Research - Project Hunter

In this research the problem of classifying projects into categories based on title and description was approached and techniques to solve the problem were investigated. The main technology used was NLP -- natural language processing.

Also, we use results from the previous [experiments](old/README.md). Mainly:
- Using the project's title and description combined gave the best results.
- Using fine-tuned pre-trained neural networks gave the best results.
- To showcase the model's potential, the dataset should be as balanced as possible.

## Methodology

First, an ample research was made to answer certain questions such as:
- What techniques, using AI or not, are adequate to solve the problem?
- Is the dataset sufficiently large for AI applications?
- What types of input would a model need to achieve good performance?

These questions will be detailed and answered later on.

Then, some AI-based approaches that could solve the problem were selected. For each one, cross-validation was used to determined the performance of certain model and parameter combinations. Cross-validation consists in splitting the dataset in `k` folds and using `k-1` folds as the training set and `1` fold as a test set. Then a different fold is picked as a test set until all folds were test sets only once.

After that, a model was trained using the best parameters and tested against a test split of the dataset.

Finally, results were registered on tables and possible future research was discussed.

## Experiments and Results

Below are the questions made (and their answers):

1. What are the viable solutions, with or without AI?
- Solutions without AI would involve plenty of manual processing of the texts and a statistical modelling of the problema, and we realized it would not be a viable path. There are various possible solutions with AI, but only some were considered: 1) fine-tuning a pre-trained model for text embedding extraction and 2) manual splitting of text into tokens for embedding generation and 3) usage of ChatGPT in some way. Please note number 2 was adressed in the previous experiment so it won't be approached here due to it's ineffieciency.

2. What tools would be needed? Which are already built? What's the cost? Can a whole page be used as input?
- For a complete solution, tools such as Snowflake can be used to host the database/dataset, Streamlit can be used to present results, AWS can be used to host models and crawling and scraping services and TensorFlow or PyTorch can be used as frameworks to train and deploy models. Snowflake and AWS are paid services. Future experiments that are out of the scope of this research would have to be made to try and use a whole page as an input to a model; we only experimented with titles and descriptions of projects that are already in a database.

3. What statistical methods would be important to reduce sample space?
- The less redundant and more significant words and phrases that describe the project the better.  If tags are to be used, something between 5 and 50 could be generated initially and then linearly depend variables could be eliminated using Pearson's correlation, for example.

4. What is the minimum number of samples a database needs to allow experiments in: finance/impact/area/city/year/subarea?
- This is the most complex question, seen as there is no simple answer. Deep down, the number of samples depends on the problem's underlying statistical model. We don't know this model, and it certainly is of high complexity. Is the model were to be simple, we could use few features and relatively few samples, but being a complex problem, and making use of AI, a large database is necessary. Two empirical guesses were picked: 1) 5000 samples per class (based on the book [Deep Learning](https://www.deeplearningbook.org/)) or 2) 10x the degrees of freedom (trainable parameters) of the model. The latter is more incertain because the model that ends up being used depends on the statistical complexity of the problem, as was mentioned before. A conclusion was reached that a database of at least 1000 samples would be adequate to run performance tests on AI tools.

5. How should the database be constructed (number of tags, number of areas) to maximize performance?
- See answer above.

6. What outputs are easier to obtain?
- Some variables are binary and therefore of less complexity, such as high impact project and social/green/climate financing. Other outputs that are easily obtained with scraping are city and year of the project. Area and subarea are outputs of higher complexity.

With these answers, our research went deeper in investigating NLP techniques for project area prediction.

### Exploring and Cleaning the Dataset

Samples lacking data in the title, description or area columns were removed. Also, preliminary tests showed that using all available areas hurt performance due to the severe imbalance of classes. So, for our tests, only areas with more than 100 samples were used. Samples with title or description too short were removed. The cleaning resulted in the following sample count for each area:

| area             |   sample_count |
|:-----------------|---------------:|
| transport        |            347 |
| waste management |            315 |
| water management |            269 |
| energy effiency  |            228 |
| renewable energy |            224 |
| buildings        |            177 |

Thsi results in a dataset with `1560` samples. We chose to feed the model text in english, so a language classifier was used to determine the non-english samples and Google Cloud Translation API was used to translate the texts to english.

The code for processing data can be found in [data_processing_cdp.ipynb](data_processing_cdp.ipynb) for `2022 Full Cities` and [data_processing_es.ipynb](data_processing_es.ipynb) for `2021 Full Cities`. There are 2 different codes because the column and row format is different for the `2021 Full Cities` and `2022 Full Cities` files provided.

### Fine-tuning Pre-trained Models for Embeddings Extraction

#### Cross-validation

The methods used are based of the following tutorials:
- https://www.tensorflow.org/tutorials/keras/text_classification
- https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

The cross-validation code can be found in [cross_validation.ipynb](cross_validation.ipynb). Running the code and searching for optimal parameters, the following table was obtained:

| input_column   | model_type   |   hidden_neurons |   cv_score |   cv_time |
|:---------------|:-------------|-----------------:|-----------:|----------:|
| title+desc     | nnlm128      |                0 |     0.8314 |       224 |
| title+desc     | nnlm128      |               64 |     0.8237 |       223 |
| title+desc     | use          |                0 |     0.8282 |      1209 |
| title+desc     | use          |               64 |     0.8128 |      1242 |

The time is given in seconds and is the time necessary to run the cross-validation algorithm with that set of parameters. As was mentioned before, for these experiments, we only used the combined title and descriptions columns as input and only used pre-trained models + fine-tuning as our method. Variations on the number of hidden neurons in the last fully-connected layer and the model type were kept. The number of epochs for each combination was set empirically. There are also two smaller tables for the model type:

| model_type   |   cv_score |   cv_time |
|:-------------|-----------:|----------:|
| nnlm128      |     0.8276 |     223.5 |
| use          |     0.8205 |    1225.5 |

And hidden neurons:

|   hidden_neurons |   cv_score |   cv_time |
|-----------------:|-----------:|----------:|
|                0 |     0.8298 |     716.5 |
|               64 |     0.8183 |     732.5 |

Notice that the models, with or without neurons in the last fully-connected layer, offer similar performance, but with the [NNLM](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2) architecture offering much faster training (and inference) times than the [USE](https://tfhub.dev/google/universal-sentence-encoder/4) architecture. It's difficult to make other conclusions since the accuracy difference is so small and the dataset is also small.

In the end, on average and on a imbalanced dataset, the `NNLM` model achieved `83.14%` cross-validation accuracy, indicating promising results and that tests with a larger dataset and perhaps a different model could achieve even better results. What also needs to be address to make sure the model performs well is create a separate test set that is independent from the ones used in training. This could evaluate better the model generalization skills.

#### Training and Testing

To be able to conclude with more certainty that a trained model has good performance, instead of cross-validation, a model was trained using the best parameter combination and was tested against a test split of the dataset. The code can be found in [train_model.ipynb](train_model.ipynb). The proportion used to split the dataset into train, test and validation was `70:15:15`.

The same data processing is used and an `NNLM` pre-trained model with no intermediate FC layer at the end was used. The model performance can seen on the table below:

|       |    acc |
|:------|-------:|
| train | 0.9955 |
| val   | 0.8232 |
| test  | 0.7778 |

Since the dataset is small, it would be hard for a model to achieve similar train, validation and test accuracy, especially since overfitting can easily happen, and it leads to lack of generalization skills for the model. That said, while the model got close to overfitting with `99.55%` train accuracy and a lower `82.62%` validation accuracy, good generalization was achieved with `77.78%` test accuracy, which is reasonably close to the cross-validation accuracy of `83.14%`.

In the previous research project, the best cross-validation accuracy achieved with a dataset with 291 samples was `71.81%`. Now, with 1560 samples, an accuracy of `83.14%` was achieved using the same parameters. The higher cross-validation and model training scores indicate that dataset and model improvements could be made in a future research project and with a larger dataset, more accuracy and more generalization skills can be achieved.

### ChatGPT for Project Web Page Classification

ChatGPT's use was not deeply research seen as the most relevant parts of the tool are paid. Still, the following queries show that there is great potential in investigating this approach.

![Query ChatGPT 1](imgs/chatgptquery1.jpeg)
![Query ChatGPT 2](imgs/chatgptquery2.jpeg)

## Conclusion and Future Work

Possibilities:

- Expand the dataset (which could include data augmentation techniques). To get more solid conclusions about the use of AI on the problem it is imperative to enlarge the database. When using AI, one is be affected by the "curse of dimensionality": a large amount of data is needed to apply AI to a given problem. [Possible data augmentation techniques](https://neptune.ai/blog/data-augmentation-nlp) include translate the text back and forth to another language so that phrases are slightly different. Also, words can be chosen at random to be replaced by synonyms.

- Pre-trained models as embeddings extractors. As seen on the tables previously, models such as `NNLM` and `USE` are good feature extractors, allowing for 83% accuracy in cross-validation before overfitting.

- Free or paid solutions involving ChatGPT. As noted on the queries shown above, ChatGPT from OpenAI offers good potential, but before diving in it's dvised to search for other free and paid services that could offer better cost benefit. There are options such as [extracting embeddings from text (or tokens)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) or [fine-tuning available models](https://platform.openai.com/docs/guides/fine-tuning) but both are paid options. It should also be investigated if ChatGPT can work with whole web pages (or even only the URL) and from the results create meaningful classifications.
