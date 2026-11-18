"""Builder for unit5_nlp_databases.ipynb (stdlib only, run with system python3)."""

from nbhelper import md, code, write_notebook

cells = []

cells.append(md(r'''
# Unit V, Part 3 — NLTK, Gensim, TextBlob, WordCloud and Databases

This reference notebook covers two families of Python libraries used in data science. The first half is **natural language processing (NLP)**: NLTK for classic text work (tokenization, stopwords, stemming, lemmatization, POS tagging, named entities, corpora, frequency analysis, VADER sentiment and WordNet), Gensim for vector-space topic and word models, plus TextBlob and WordCloud for quick practical tasks. The second half is **database access**: SQLAlchemy 2.0 (ORM and Core), DuckDB analytics over pandas, and the standard-library `sqlite3`, with connection-URL templates for MySQL and PostgreSQL. Everything runs offline: NLTK data downloads appear only as commented examples, and every call that needs downloaded data is wrapped in `try/except`, while a tiny built-in corpus keeps all examples runnable.

**Learning objectives**

- Download NLTK data with `nltk.download(..., quiet=True)` and know the resource names needed (punkt, stopwords, wordnet, vader_lexicon, taggers, chunkers).
- Tokenize text into words and sentences with `word_tokenize`, `sent_tokenize`, `WordPunctTokenizer` and `RegexpTokenizer`.
- Remove stopwords, stem with `PorterStemmer` and `SnowballStemmer`, and lemmatize with `WordNetLemmatizer` using POS tags.
- Perform POS tagging and named entity recognition, and explain the NLTK data each requires.
- Explore the Gutenberg, movie_reviews and Brown corpora and use `FreqDist`, `Text.concordance` and `Text.collocations`.
- Compute VADER sentiment scores and look up WordNet synsets, definitions and similarity.
- Build bag-of-words and TF-IDF matrices with scikit-learn `CountVectorizer` and `TfidfVectorizer`.
- Use Gensim 4 concepts: `Dictionary`, `doc2bow`, `TfidfModel`, `LsiModel`, `LdaModel`, `Word2Vec`, similarity queries and save/load.
- Apply TextBlob for sentiment and WordCloud for a quick visualization.
- Create an engine, declarative ORM model and `Session` with SQLAlchemy 2.0; run Core `text()` queries and load results into pandas.
- Run in-memory SQL with DuckDB over a pandas DataFrame and use `sqlite3` for a local database.
- Write connection URLs for MySQL and PostgreSQL and explain SQLAlchemy transactions and reflection.
'''))

cells.append(md(r'''
## Official documentation & references

- [NLTK Book](https://www.nltk.org/book/) — free online textbook; chapters 1–5 cover the tools used here.
- [NLTK tokenize API](https://www.nltk.org/api/nltk.tokenize.html) — word and sentence tokenizers.
- [NLTK stem API](https://www.nltk.org/api/nltk.stem.html) — `PorterStemmer`, `SnowballStemmer`, `WordNetLemmatizer`.
- [NLTK corpus API](https://www.nltk.org/api/nltk.corpus.html) — `stopwords`, `wordnet`, `gutenberg`, `movie_reviews`, `brown`.
- [NLTK tag API](https://www.nltk.org/api/nltk.tag.html) — `pos_tag` and the taggers.
- [Gensim tutorials](https://radimrehurek.com/gensim/auto_examples/index.html) — runnable examples for corpora, TF-IDF, LSI, LDA and word2vec.
- [Gensim API reference](https://radimrehurek.com/gensim/apidocs.html) — full signatures for the Gensim 4 API.
- [scikit-learn feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) — `CountVectorizer` and `TfidfVectorizer`.
- [TextBlob documentation](https://textblob.readthedocs.io/) — sentiment, noun phrases and spelling.
- [wordcloud documentation](https://wordcloud.readthedocs.io/) — `WordCloud.generate()` and options.
- [SQLAlchemy 2.0 ORM quickstart](https://docs.sqlalchemy.org/en/20/orm/quickstart.html) — engines, declarative models and sessions.
- [SQLAlchemy 2.0 SQL / Core tutorial](https://docs.sqlalchemy.org/en/20/core/tutorial.html) — `text()`, `select()` and transactions.
- [DuckDB Python overview](https://duckdb.org/docs/stable/clients/python/overview.html) — `duckdb.connect`, `.sql()`, `.df()`.
- [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) — the standard-library SQLite driver.
'''))

cells.append(md(r'''
## 0. Setup

The first code cell imports everything used in this notebook. `DATA` points two levels above this notebook at `../../data`, so the project CSVs are always reachable; none of the examples below actually need them, but the pattern is included for reuse.

NLTK separates **package code** from **data files** (corpora, models, lexicons). Installing with pip gives you only the code; a one-time download gives you the data. Run the following **once, with internet available, outside this offline notebook**:

```python
import nltk
nltk.download("punkt", quiet=True)                           # tokenizer model (older NLTK)
nltk.download("punkt_tab", quiet=True)                       # tokenizer model (NLTK 3.9+)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)                         # Open Multilingual WordNet
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)  # POS tagger, NLTK 3.9+
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)           # NE chunker, NLTK 3.9+
nltk.download("words", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("gutenberg", quiet=True)
nltk.download("movie_reviews", quiet=True)
nltk.download("brown", quiet=True)
```

`quiet=True` suppresses the download progress output. Newer NLTK releases renamed several resources (the `_eng` / `_tab` suffixes above); downloading both spellings is harmless. The `nltk_need_data` helper defined in the first code cell catches `LookupError`, so this notebook still runs when a resource is absent.
'''))

cells.append(code(r'''
"""
Imports and shared small data.

`nltk_need_data` wraps any NLTK call that depends on downloaded data: it returns the
real result when the data is present, otherwise it prints a short SKIP message naming
the resource to download. This keeps the notebook runnable offline without hiding errors.
"""

from pathlib import Path

import nltk
from nltk.tokenize import (
    word_tokenize,
    sent_tokenize,
    wordpunct_tokenize,
    WordPunctTokenizer,
    RegexpTokenizer,
)
from nltk.corpus import stopwords, wordnet, gutenberg, movie_reviews, brown
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk import FreqDist, Text
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, LdaModel, Word2Vec
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

from textblob import TextBlob, Word
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import duckdb
import sqlite3

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    select,
    text,
    inspect,
    MetaData,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

# Project data directory (two levels above this notebook's folder).
DATA = Path("../../data")

# Tiny self-contained corpus: every example works with zero downloads.
TOY_SENTENCE = "Data science uses Python to clean data, model data, and explain data."
TOY_DOCS = [
    "python data science uses python for data analysis",
    "machine learning models learn patterns from data",
    "python pandas cleans messy data before modelling",
    "data visualisation and statistics explain the data",
]
TOY_SENTENCES = [doc.split() for doc in TOY_DOCS]

# A longer toy text for wordcloud, Text and frequency examples.
TOY_TEXT = " ".join(TOY_DOCS + [
    "natural language processing helps computers read and understand text",
    "tokens words and sentences are the basic units of text",
    "topic models discover hidden themes inside a collection of documents",
])


def nltk_need_data(label, func, *args, **kwargs):
    """Call func; print a helpful message instead of crashing when data is missing."""
    try:
        return func(*args, **kwargs)
    except LookupError as exc:
        print("SKIP [" + label + "]: missing NLTK data -> " + str(exc))
        print("       Download it once, e.g. nltk.download('<name>', quiet=True).")
    except Exception as exc:
        print("SKIP [" + label + "]: " + type(exc).__name__ + ": " + str(exc))
    return None
'''))

cells.append(md(r'''
## 1. NLTK — tokenization

Tokenization splits raw text into **tokens**, usually words or sentences. Two families exist:

- **Rule-based tokenizers** need no downloaded data: `WordPunctTokenizer` (splits on non-alphanumeric characters, keeping punctuation as its own token), `RegexpTokenizer` (splits using a regex), and the function form `nltk.regexp_tokenize`.
- **Model-based tokenizers** need data: `word_tokenize` and `sent_tokenize` use a Punkt model (`punkt`, or `punkt_tab` in NLTK 3.9+).

With `RegexpTokenizer`, `gaps=False` (the default) returns the text that *matches* the pattern, while `gaps=True` returns the text *between* matches. That distinction lets the same regex either keep words or strip them.
'''))

cells.append(code(r'''
# Rule-based tokenizers: no NLTK data download required.

print("wordpunct_tokenize  :", wordpunct_tokenize(TOY_SENTENCE))
print("WordPunctTokenizer  :", WordPunctTokenizer().tokenize(TOY_SENTENCE))
print("RegexpTokenizer     :", RegexpTokenizer(r"\w+").tokenize(TOY_SENTENCE))
print("nltk.regexp_tokenize:", nltk.regexp_tokenize(TOY_SENTENCE, r"\w+"))
print("gaps=True           :", RegexpTokenizer(r"\d+", gaps=True).tokenize("a1 b22 c"))

# Keep words and dollar amounts such as $3.50 as single tokens.
money = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")
print("with $ amounts      :", money.tokenize("The model cost $3.50 last year."))

# A tokenizer can also be reused across many documents.
splitter = WordPunctTokenizer()
for doc in TOY_DOCS[:2]:
    print("doc:", splitter.tokenize(doc))
'''))

cells.append(code(r'''
# Model-based tokenizers need the Punkt data; the helper keeps this cell safe offline.

sents = nltk_need_data("sent_tokenize", sent_tokenize, TOY_TEXT)
if sents is not None:
    for index, sent in enumerate(sents, start=1):
        print(index, sent)

words = nltk_need_data("word_tokenize", word_tokenize, TOY_SENTENCE)
if words is not None:
    print("word_tokenize:", words)
    print("token count  :", len(words))

# word_tokenize(..., preserve_line=True) skips sentence splitting and is faster
# when you already have one sentence at a time.
'''))

cells.append(md(r'''
## 2. Stopwords

`nltk.corpus.stopwords.words("english")` returns a list of common function words. This is a **downloaded** corpus, so the cell below falls back to a tiny built-in list when it is missing. Stopword lists are not universal: removing them may hurt tasks like sentiment analysis, where words such as "not" carry meaning. The language argument supports many languages, for example `"french"`, `"german"`, `"spanish"`.
'''))

cells.append(code(r'''
# Stopwords: use NLTK's list when available, otherwise a small built-in fallback.

tokens = RegexpTokenizer(r"\w+").tokenize(TOY_SENTENCE.lower())

stop_list = nltk_need_data("stopwords", stopwords.words, "english")
if stop_list is None:
    stop_list = ["to", "and", "the", "a", "an", "uses", "for", "in", "of", "is"]

print("stopword sample:", stop_list[:12], "... total", len(stop_list))
content = [token for token in tokens if token not in stop_list]
print("original :", tokens)
print("filtered :", content)
'''))

cells.append(md(r'''
## 3. Stemming

A stemmer chops word endings with simple rules; the result need not be a real English word. `PorterStemmer` implements the classic Porter algorithm. `SnowballStemmer` (the "Porter2" family) supports many languages — pass a language name such as `"english"`, `"french"`, `"german"` or `"spanish"`, and list them with `SnowballStemmer.languages`. Both are rule-based and need **no** data download. Stemming is fast but can conflate unrelated words (e.g. "university" and "universe").
'''))

cells.append(code(r'''
# Stemming: rule-based, offline, and intentionally crude.

porter = PorterStemmer()
snowball_en = SnowballStemmer("english")
snowball_fr = SnowballStemmer("french")

sample = ["running", "runs", "runner", "easily", "fairly", "generously", "connection"]
print(f"{'word':<12}{'Porter':<12}{'Snowball(en)'}")
for word in sample:
    print(f"{word:<12}{porter.stem(word):<12}{snowball_en.stem(word)}")

print("Snowball languages:", SnowballStemmer.languages)
print("French 'mangeant':", snowball_fr.stem("mangeant"))
'''))

cells.append(md(r'''
## 4. Lemmatization — WordNetLemmatizer

Lemmatization reduces a word to its dictionary **lemma** using a vocabulary and morphological analysis, so the result is a real word. `WordNetLemmatizer` needs the `wordnet` corpus (and `omw-1.4` for some lookups).

The `pos` argument matters: its default is `"n"` (noun), which is why "running" stays "running". Pass `"v"` for verbs, `"a"` / `"s"` for adjectives and `"r"` for adverbs. POS tags come from `pos_tag` in Penn Treebank form and are converted with a small mapping (`J`→`a`, `V`→`v`, `N`→`n`, `R`→`r`).
'''))

cells.append(code(r'''
# Lemmatization needs the WordNet data; guard every call.

lemmatizer = WordNetLemmatizer()

def lemmatize_safe(token, pos="n"):
    result = nltk_need_data("wordnet lemmatize", lemmatizer.lemmatize, token, pos=pos)
    return result if result is not None else token

print("noun  'running' ->", lemmatize_safe("running", "n"))
print("verb  'running' ->", lemmatize_safe("running", "v"))
print("noun  'cars'    ->", lemmatize_safe("cars", "n"))
print("adj   'better'  ->", lemmatize_safe("better", "a"))
print("adv   'quickly' ->", lemmatize_safe("quickly", "r"))
'''))

cells.append(md(r'''
## 5. POS tagging

`nltk.pos_tag(tokens)` assigns a Penn Treebank part-of-speech tag to every token: `NN` noun, `NNS` plural noun, `VB` / `VBG` verb forms, `JJ` adjective, `RB` adverb, `IN` preposition, `DT` determiner, and so on. The default tagger is a averaged perceptron model that needs `averaged_perceptron_tagger` (older NLTK) or `averaged_perceptron_tagger_eng` (NLTK 3.9+). `nltk.pos_tag(tokens, tagset="universal")` maps the fine-grained tags to a coarse set (`NOUN`, `VERB`, `ADJ`, `ADV`, ...). POS tags are the bridge to POS-sensitive lemmatization and to named entity recognition.
'''))

cells.append(code(r'''
# POS tagging with the averaged perceptron tagger (needs downloaded tagger data).

PENN_TO_WORDNET = {"J": "a", "V": "v", "N": "n", "R": "r"}

def penn_to_wordnet(tag):
    """Map a Penn Treebank tag to the WordNet POS letter used by the lemmatizer."""
    return PENN_TO_WORDNET.get(tag[0].upper(), "n")

tagged = nltk_need_data(
    "pos_tag", pos_tag, RegexpTokenizer(r"\w+").tokenize(TOY_SENTENCE)
)
if tagged is not None:
    print("Penn tags:", tagged)
    lemmas = [(word, lemmatize_safe(word, penn_to_wordnet(tag))) for word, tag in tagged]
    print("Lemmas   :", lemmas)

    universal = nltk_need_data("pos_tag universal", pos_tag, [w for w, t in tagged],
                               tagset="universal")
    print("Universal tags:", universal)
'''))

cells.append(md(r'''
## 6. Named entity recognition (NER)

`nltk.ne_chunk(tagged_tokens)` groups POS-tagged tokens into an `nltk.Tree` whose subtrees are labelled entities such as `PERSON`, `ORGANISATION`, `GPE` (geo-political entity) and `DATE`. With `binary=True`, every entity is labelled simply `NE`. Print a tree with `.pretty_print()` (or `print(tree)`), and inspect each child subtree with `.label()` and `.leaves()`. The chunker needs `maxent_ne_chunker` plus the `words` corpus, and the POS tagger needs its own model.
'''))

cells.append(code(r'''
# NER: POS-tag first, then chunk. Both steps are guarded.

entity_text = "Ada Lovelace worked at the University of London in 1843."

tokens = RegexpTokenizer(r"\w+").tokenize(entity_text)
tagged = nltk_need_data("pos_tag", pos_tag, tokens)
if tagged is not None:
    tree = nltk_need_data("ne_chunk", ne_chunk, tagged, binary=False)
    if tree is not None:
        print(tree)
        print("Entities found:")
        for subtree in tree:
            if hasattr(subtree, "label"):                 # subtree is a named entity
                words = " ".join(word for word, tag in subtree.leaves())
                print(" ", subtree.label(), "->", words)
'''))

cells.append(md(r'''
## 7. NLTK corpora — Gutenberg, movie_reviews and Brown

`nltk.corpus` gives uniform access to many bundled corpora. All three below are **downloadable corpora**, not part of the pip package:

- `gutenberg` — public-domain books; `gutenberg.fileids()`, `gutenberg.raw(id)`, `gutenberg.words(id)`.
- `movie_reviews` — 2,000 reviews labelled `pos` / `neg`; `movie_reviews.categories()`, `movie_reviews.fileids("pos")`, `movie_reviews.words(fileid)`.
- `brown` — the Brown corpus of American English in 15 genres; `brown.categories()`, `brown.sents(categories=["news"])`, `brown.words()`.

Common reader methods are `fileids()`, `categories()`, `raw()`, `words()`, `sents()`, `paras()` and `open()`. Every call below is wrapped because the data may not have been downloaded.
'''))

cells.append(code(r'''
# Corpus readers: show file ids, categories and small samples when data exists.

def corpus_report(name, reader):
    file_ids = nltk_need_data(name + ".fileids", lambda: reader.fileids())
    if file_ids is None:
        return
    print(name, "file count:", len(file_ids), "sample:", file_ids[:3])
    categories = nltk_need_data(name + ".categories", lambda: reader.categories())
    if categories is not None:
        print(name, "categories:", categories)
    words = nltk_need_data(name + ".words", lambda: reader.words(file_ids[0]))
    if words is not None:
        print(name, "first 12 words:", list(words)[:12])
    print("-" * 60)

corpus_report("gutenberg", gutenberg)
corpus_report("movie_reviews", movie_reviews)
corpus_report("brown", brown)
'''))

cells.append(md(r'''
## 8. Frequency distributions and `Text`

`FreqDist` is a `Counter`-like container for token counts: `.most_common(n)`, `freq[token]`, `.freq(token)`, `.hapaxes()` (words seen once), plus `.plot()` and `.tabulate()` for display. `nltk.Text` wraps a list of tokens and offers concordance searches (`.concordance(word, width=..., lines=...)`), `.collocations(num=..., window_size=...)`, `.similar(word)`, `.count(word)` and `.plot()`. These `Text` methods work on the tokens you pass in, so they need **no** download. `Text.collocations()` finds frequent word pairs using a bigram association measure.
'''))

cells.append(code(r'''
# Frequency analysis on tokens from the built-in corpus (no downloads needed).

tokens = [word.lower() for word in RegexpTokenizer(r"\w+").tokenize(TOY_TEXT)]
freq = FreqDist(tokens)

print("top 10 tokens:", freq.most_common(10))
print("count('data'):", freq["data"], " relative freq:", round(freq.freq("data"), 4))
print("hapaxes:", freq.hapaxes()[:10])

text_obj = Text(tokens)
print("--- concordance for 'data' ---")
text_obj.concordance("data", width=70, lines=4)
print("--- collocations ---")
text_obj.collocations(num=8)
print("--- similar to 'data' ---")
text_obj.similar("data")
'''))

cells.append(md(r'''
## 9. Sentiment with VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is tuned for short, informal text such as reviews and tweets. `SentimentIntensityAnalyzer().polarity_scores(text)` returns a dict with `neg`, `neu`, `pos` and `compound` scores. `compound` is normalised to [-1, +1]; a common rule is `compound >= 0.05` → positive, `compound <= -0.05` → negative, otherwise neutral. VADER needs the `vader_lexicon` resource, so the cell below guards its construction.
'''))

cells.append(code(r'''
# VADER sentiment: guard the lexicon dependency.

analyzer = nltk_need_data("SentimentIntensityAnalyzer", SentimentIntensityAnalyzer)
if analyzer is not None:
    reviews = [
        "This library is absolutely fantastic and easy to use!",
        "The results were terrible and the documentation was confusing.",
        "The notebook is fine. It does what it says.",
    ]
    for review in reviews:
        scores = analyzer.polarity_scores(review)
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else (
            "negative" if compound <= -0.05 else "neutral"
        )
        print(f"{label:<9} compound={compound:+.4f}  {scores}  | {review}")
'''))

cells.append(md(r'''
## 10. WordNet synsets and similarity

A **synset** is a group of synonyms that share one meaning. `wordnet.synsets(word, pos=...)` returns synsets; each has `.name()`, `.definition()`, `.examples()`, `.lemmas()`, `.hypernyms()` (more general terms) and `.hyponyms()` (more specific terms). Similarity methods include `.path_similarity(other)` (shortest path in the hierarchy), `.wup_similarity(other)` (Wu–Palmer, based on the deepest common ancestor) and `.lch_similarity(other)` (Leacock–Chodorow, which requires the same POS). Cross-POS similarity is undefined and returns `None`. WordNet needs the `wordnet` download (and ideally `omw-1.4`).
'''))

cells.append(code(r'''
# WordNet: synset lookup, definitions and similarity (data-dependent).

def wordnet_demo():
    for syn in wordnet.synsets("car")[:3]:
        print(syn.name(), "|", syn.definition())
        print("   examples:", syn.examples()[:2])
        print("   hypernyms:", [h.name() for h in syn.hypernyms()[:3]])
    car = wordnet.synset("car.n.01")
    vehicle = wordnet.synset("vehicle.n.01")
    truck = wordnet.synset("truck.n.01")
    print("path_similarity(car, vehicle):", round(car.path_similarity(vehicle), 3))
    print("wup_similarity(car, truck)  :", round(car.wup_similarity(truck), 3))
    print("lch_similarity(car, truck)  :", round(car.lch_similarity(truck), 3))

nltk_need_data("wordnet", wordnet_demo)
'''))

cells.append(md(r'''
## 11. Bag-of-words and TF-IDF with scikit-learn

`CountVectorizer` converts a list of documents into a token-count matrix. `TfidfVectorizer` does the same but weights each count by inverse document frequency: tf-idf = tf × idf, with idf = ln((1+n)/(1+df)) + 1. Useful parameters for both:

- `lowercase=True`, `stop_words="english"` or a custom list, `token_pattern` (default `(?u)\b\w\w+\b`).
- `ngram_range=(1, 2)` for unigrams and bigrams, `max_df` / `min_df` to drop very common or rare terms, `max_features` to cap vocabulary size.
- `binary=True` for presence/absence; `norm="l2"` on `TfidfVectorizer` to length-normalise each row.

After `.fit_transform(docs)`, use `.get_feature_names_out()` for column names, `.toarray()` for a dense matrix, `.transform(new_docs)` for unseen data, `.inverse_transform(matrix)` to recover tokens, and `.idf_` to inspect the learned weights.
'''))

cells.append(code(r'''
# Count vectors (bag of words) for the tiny in-notebook corpus.

count_vec = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
bow = count_vec.fit_transform(TOY_DOCS)
feature_names = count_vec.get_feature_names_out()

print("shape (documents, features):", bow.shape)
print("features:", list(feature_names))
print("dense matrix:\n", bow.toarray())
print("vocabulary size:", len(count_vec.vocabulary_))
'''))

cells.append(code(r'''
# TF-IDF vectors, including transformation of a previously unseen document.

tfidf_vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 1))
tfidf = tfidf_vec.fit_transform(TOY_DOCS)
names = tfidf_vec.get_feature_names_out()

print("features:", list(names))
print("tf-idf matrix (rounded):\n", np.round(tfidf.toarray(), 3))
print("idf values:", np.round(tfidf_vec.idf_, 3))

new_doc = ["python models explain data"]
new_vec = tfidf_vec.transform(new_doc)
print("new doc tf-idf:", np.round(new_vec.toarray(), 3))
'''))

cells.append(md(r'''
### Try it yourself #1 — NLTK mini-pipeline

Using `RegexpTokenizer(r"\w+")`, `PorterStemmer`, `FreqDist` and a small manual stopword set, write a function `top_terms(text, n=5)` that lowercases the text, tokenizes it, drops stopwords and one-character tokens, stems each remaining token, and returns the `n` most common stems with their counts. Test it on `TOY_TEXT`.
'''))

cells.append(code(r'''
# Solution #1

def top_terms(text, n=5, extra_stop=None):
    stops = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "is", "are"}
    if extra_stop:
        stops |= set(extra_stop)
    stemmer = PorterStemmer()
    words = RegexpTokenizer(r"\w+").tokenize(text.lower())
    cleaned = [stemmer.stem(word) for word in words if word not in stops and len(word) > 2]
    return FreqDist(cleaned).most_common(n)

print(top_terms(TOY_TEXT, n=8))
'''))

cells.append(md(r'''
## 12. Gensim 4 — corpus, dictionary and vector-space models

Gensim uses a two-stage pipeline:

1. **Dictionary** — `Dictionary(documents)` maps every unique token to an integer id. Useful members: `.token2id`, `.id2token`, `.num_docs`, `.filter_extremes()`, `.filter_n_most_frequent()`, `.compactify()`.
2. **Corpus** — a list of bag-of-words vectors. `dictionary.doc2bow(doc)` returns `[(token_id, count), ...]` for one document. This sparse format is what every Gensim model consumes.

Models then transform that corpus: `TfidfModel` (weighting), `LsiModel` (Latent Semantic Indexing via truncated SVD), `LdaModel` (Latent Dirichlet Allocation topics) and `Word2Vec` (predictive word embeddings).

**Gensim 4 API notes** (the current major version): the old `size` parameter became `vector_size`, and `iter` became `epochs`; `window`, `min_count`, `sg` and `workers` are unchanged. Word vectors live on `model.wv` with `.similarity()`, `.most_similar()`, `.doesnt_match()` and `.index_to_key`. `LsiModel` and `LdaModel` take `id2word=dictionary`. Persist with `model.save(path)` and reload with `Word2Vec.load(path)`; this is local and needs no network.
'''))

cells.append(code(r'''
# Dictionary + bag-of-words corpus. simple_preprocess lowercases, strips accents
# and drops very short tokens; it is Gensim's recommended tokenizer.

docs_tokens = [simple_preprocess(doc) for doc in TOY_DOCS]
print("tokenized docs:", docs_tokens)

dictionary = Dictionary(docs_tokens)
print("size:", len(dictionary), " num_docs:", dictionary.num_docs)
print("sample token2id:", dict(list(dictionary.token2id.items())[:8]))

bow_corpus = [dictionary.doc2bow(doc) for doc in docs_tokens]
print("first doc as BOW:", bow_corpus[0])
print("human-readable  :", [(dictionary[id_], count) for id_, count in bow_corpus[0]])

# Filter rare tokens, renumber the ids, and rebuild the corpus to match.
dictionary.filter_extremes(no_below=1, no_above=1.0)
dictionary.compactify()
bow_corpus = [dictionary.doc2bow(doc) for doc in docs_tokens]
'''))

cells.append(md(r'''
## 13. TF-IDF, LSI and similarity search

`TfidfModel(corpus)` learns idf weights from a corpus; `model[bow_doc]` returns a sparse `[(id, weight), ...]` vector. `LsiModel(corpus, id2word=dictionary, num_topics=k)` projects documents into a k-dimensional latent space; `lsi[bow_doc]` gives the latent vector. `MatrixSimilarity(lsi_corpus)` builds an index so `index[query_vector]` returns one score per document; `SparseMatrixSimilarity` is the memory-efficient choice for sparse vectors.

With only four toy documents the scores are illustrative rather than meaningful — the goal is to show the API and the query pattern end to end.
'''))

cells.append(code(r'''
# TF-IDF weighting and LSI latent space, then a similarity query.

tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = [tfidf_model[doc] for doc in bow_corpus]
print("TF-IDF of first doc:", tfidf_corpus[0])

lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=2)
print("LSI topics:", lsi_model.print_topics(num_words=4))
lsi_corpus = [lsi_model[doc] for doc in tfidf_corpus]
print("latent vector doc 0:", np.round(lsi_corpus[0], 3))

index = MatrixSimilarity(lsi_corpus)
query_bow = dictionary.doc2bow(simple_preprocess("python data and models"))
query_lsi = lsi_model[tfidf_model[query_bow]]
scores = index[query_lsi]
for doc_id, score in sorted(enumerate(scores), key=lambda item: -item[1]):
    print(f"doc {doc_id}  score={score:.3f}  {TOY_DOCS[doc_id][:55]}")
'''))

cells.append(md(r'''
## 14. LDA topics with Gensim

`LdaModel(corpus, num_topics=k, id2word=dictionary, passes=..., random_state=...)` estimates a generative topic model. Important parameters: `num_topics`, `alpha="auto"` (learn the document-topic prior), `eta` (topic-word prior), `passes` (training passes), `iterations`, `random_state` for reproducible output, and `per_word_topics=True` to also return word-level assignments. Inspect topics with `.print_topics(num_topics, num_words)` or `.show_topic(topic_id, topn)`, and assign documents with `.get_document_topics(bow)` (equivalently `model[bow]`).
'''))

cells.append(code(r'''
# LDA topic model on the toy corpus (small corpus -> demonstration only).

lda_model = LdaModel(
    corpus=bow_corpus,
    num_topics=2,
    id2word=dictionary,
    passes=10,
    alpha="auto",
    random_state=42,
)
for topic_id, topic_text in lda_model.print_topics(num_topics=2, num_words=5):
    print("topic", topic_id, ":", topic_text)

print("document 0 topics:", lda_model.get_document_topics(bow_corpus[0]))
'''))

cells.append(md(r'''
## 15. Word2Vec and persistence

`Word2Vec(sentences, vector_size=8, window=2, min_count=1, sg=0, epochs=20, workers=1)` learns dense word vectors. `sg=0` uses CBOW, `sg=1` uses skip-gram; `window` is the context radius, `min_count` drops rare words, and `negative` sets the number of negative samples. Use `min_count=1` for a toy corpus so every word gets a vector. Results live in `model.wv`: `.similarity(w1, w2)`, `.most_similar(word, topn=...)`, `.doesnt_match(words)`, `.vector(word)` and `.index_to_key`. Export the classic text format with `.wv.save_word2vec_format("vec.txt", binary=False)`.
'''))

cells.append(code(r'''
# Word2Vec on the tokenized toy sentences, plus a save/load round-trip.

w2v = Word2Vec(
    sentences=docs_tokens,
    vector_size=8,
    window=2,
    min_count=1,
    sg=0,
    epochs=20,
    workers=1,
    seed=42,
)
print("vocabulary:", list(w2v.wv.index_to_key))
print("vector for 'data':", np.round(w2v.wv["data"], 3))

for first, second in [("data", "python"), ("data", "models"), ("clean", "messy")]:
    if first in w2v.wv and second in w2v.wv:
        print("similarity(%s, %s) = %.3f" % (
            first, second, w2v.wv.similarity(first, second)
        ))
print("most similar to 'data':", w2v.wv.most_similar("data", topn=3))

model_path = str(Path("/tmp") / "toy_word2vec.model")
w2v.save(model_path)
reloaded = Word2Vec.load(model_path)
print("reloaded vocabulary size:", len(reloaded.wv.index_to_key))
'''))

cells.append(md(r'''
### Try it yourself #2 — Gensim TF-IDF query

Reuse `dictionary` and `bow_corpus`. Build a `TfidfModel`, then define `query_docs(text, top=2)` that converts `text` to a bag of words with `dictionary.doc2bow(simple_preprocess(text))`, applies the TF-IDF model, ranks all documents with `SparseMatrixSimilarity` over the TF-IDF corpus, and returns the top `top` `(index, score)` pairs. Test it with `"statistics and visualisation explain data"`.
'''))

cells.append(code(r'''
# Solution #2

tfidf_model_2 = TfidfModel(bow_corpus)
tfidf_corpus_2 = [tfidf_model_2[doc] for doc in bow_corpus]
tfidf_index = SparseMatrixSimilarity(tfidf_corpus_2, num_features=len(dictionary))

def query_docs(text, top=2):
    bow = dictionary.doc2bow(simple_preprocess(text))
    vec = tfidf_model_2[bow]
    scores = tfidf_index[vec]
    ranked = sorted(enumerate(scores), key=lambda item: -item[1])
    return [(doc_id, round(float(score), 3)) for doc_id, score in ranked[:top]]

print(query_docs("statistics and visualisation explain data"))
'''))

cells.append(md(r'''
## 16. TextBlob

TextBlob is a friendly wrapper over NLTK and Pattern for common tasks. `TextBlob(text).sentiment` returns a `Sentiment(polarity, subjectivity)` object, both scores in [-1, 1]. `.words` and `.sentences` tokenize (and may require the `punkt` download), `.noun_phrases` extracts phrases, `.correct()` spell-checks (needs extra corpora), and `Word("cats").lemmatize()` uses WordNet. Pattern-based sentiment works offline; the tokenization helpers are guarded below. For a one-time corpus download use `textblob.download_corpora()` with internet access.
'''))

cells.append(code(r'''
# TextBlob: offline sentiment, plus guarded tokenization and lemmatization.

blob = TextBlob("Python is wonderful, but the missing documentation was frustrating.")

print("sentiment:", blob.sentiment)
print("polarity:", round(blob.sentiment.polarity, 3),
      "subjectivity:", round(blob.sentiment.subjectivity, 3))
print("words:", nltk_need_data("TextBlob words", lambda: list(blob.words)))
print("sentences:", nltk_need_data("TextBlob sentences", lambda: list(blob.sentences)))
print("noun_phrases:", nltk_need_data("TextBlob noun_phrases", lambda: list(blob.noun_phrases)))
print("lemmatize 'running':",
      nltk_need_data("TextBlob lemmatize", lambda: Word("running").lemmatize("v")))
'''))

cells.append(md(r'''
## 17. WordCloud

`WordCloud.generate(text)` builds a cloud from a single string; `.generate_from_frequencies(dict)` accepts precomputed word counts instead. `.to_image()` returns a Pillow `Image`, which we display with matplotlib using `plt.imshow(image, interpolation="bilinear")` and `plt.axis("off")`. Useful constructor parameters include `width`, `height`, `background_color`, `colormap`, `max_words`, `stopwords`, `min_font_size`, `prefer_horizontal`, `mask`, `contour_width` and `contour_color`. We deliberately skip `.to_file()` so no output file is written.
'''))

cells.append(code(r'''
# Word cloud from the toy corpus; display the in-memory image, write no files.

cloud = WordCloud(
    width=640,
    height=360,
    background_color="white",
    colormap="viridis",
    max_words=30,
    stopwords={"the", "and", "for", "with", "from", "are", "was"},
).generate(TOY_TEXT)

image = cloud.to_image()
plt.figure(figsize=(9, 5))
plt.imshow(image, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud of the toy corpus")
plt.tight_layout()
plt.show()

print("words placed:", len(cloud.words_))
'''))

cells.append(md(r'''
## 18. Database access — SQLAlchemy, connection URLs, transactions, reflection

Python reaches a database through a **DBAPI driver** (`sqlite3` is built in, `psycopg2` for PostgreSQL, `pymysql` for MySQL) or through **SQLAlchemy**, which wraps drivers behind one API.

Common SQLAlchemy 2.0 connection URLs (the segment after the third slash is the database name or file path):

- SQLite in memory: `sqlite:///:memory:`
- SQLite file: `sqlite:///my_database.db`
- PostgreSQL: `postgresql+psycopg2://user:pass@host:5432/db`
- MySQL: `mysql+pymysql://user:pass@host:3306/db`

The MySQL and PostgreSQL forms are shown for reference only — they need a **running server and valid credentials**, so this notebook does not connect to them. Never hard-code real passwords; read them from environment variables instead.

Reference only (do not run here):

```python
# from sqlalchemy import create_engine
# pg = create_engine("postgresql+psycopg2://user:pass@localhost:5432/mydb")
# my = create_engine("mysql+pymysql://user:pass@localhost:3306/mydb")
```

**Transactions.** In SQLAlchemy 2.0 nothing is committed automatically. `with engine.begin() as conn:` starts a transaction and commits on clean exit, rolling back on exception. `with engine.connect() as conn:` gives a connection on which you must call `conn.commit()` yourself (or `conn.rollback()`). An ORM `Session` commits with `session.commit()`; use `with Session(engine) as session:` so it closes reliably.

**Reflection** loads an existing schema instead of declaring models: `insp = inspect(engine); insp.get_table_names()` lists tables, and `metadata = MetaData(); metadata.reflect(bind=engine)` loads them into `metadata.tables`. The ORM equivalent is `from sqlalchemy.ext.automap import automap_base`.
'''))

cells.append(code(r'''
# SQLAlchemy 2.0: engine, declarative ORM model, Session, Core text() and pandas.

engine = create_engine("sqlite:///:memory:", echo=False)

class Base(DeclarativeBase):
    """Declarative base: classes inheriting from it become ORM models."""
    pass

class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self):
        return f"Student(id={self.id}, name={self.name!r}, score={self.score})"

Base.metadata.create_all(engine)          # create the table(s)
print("tables:", inspect(engine).get_table_names())

# --- ORM writes and reads ---
with Session(engine) as session:
    session.add_all([
        Student(name="Ada", score=91),
        Student(name="Linus", score=88),
        Student(name="Grace", score=95),
    ])
    session.commit()                      # persist the transaction

    stmt = select(Student).where(Student.score >= 90).order_by(Student.score.desc())
    for student in session.execute(stmt).scalars().all():
        print(student)

    # Core-style SQL inside the ORM session, with a bound parameter (:cutoff).
    rows = session.execute(
        text("SELECT name, score FROM students WHERE score >= :cutoff"),
        {"cutoff": 90},
    ).all()
    print("core rows:", rows)

# --- Core connection: engine.begin() commits automatically on clean exit ---
with engine.begin() as conn:
    conn.execute(
        text("INSERT INTO students (name, score) VALUES (:n, :s)"),
        {"n": "Barbara", "s": 93},
    )
    df = pd.read_sql(text("SELECT * FROM students ORDER BY score DESC"), conn)

print(df)

# --- Reflection: inspect the schema we just created ---
metadata = MetaData()
metadata.reflect(bind=engine)
print("reflected tables:", list(metadata.tables))
'''))

cells.append(md(r'''
## 19. DuckDB — SQL over a pandas DataFrame

DuckDB is an in-process analytical database. `duckdb.connect(":memory:")` opens an in-memory database; `con.execute(sql, params)` runs a statement, with `.fetchall()`, `.fetchone()`, `.fetchdf()` and `.df()` for results. Its standout feature is **replacement scans**: a pandas DataFrame in Python scope can be queried by name as if it were a table, with no copying or explicit load step. Top-level helpers `duckdb.sql(query)` and `duckdb.query(query)` work too. Everything is local and needs no server.
'''))

cells.append(code(r'''
# DuckDB: query a DataFrame directly with SQL, no server and no loading step.

grades = pd.DataFrame({
    "name": ["Ada", "Linus", "Grace", "Barbara", "Alan"],
    "team": ["A", "B", "A", "B", "A"],
    "score": [91, 88, 95, 93, 79],
})

con = duckdb.connect(":memory:")

# A DataFrame named `grades` in Python scope is visible to SQL automatically.
result = con.execute(
    "SELECT team, COUNT(*) AS n, ROUND(AVG(score), 1) AS avg_score "
    "FROM grades GROUP BY team ORDER BY avg_score DESC"
).fetchdf()
print(result)

# Parameterised query plus fetchall.
top = con.execute(
    "SELECT name, score FROM grades WHERE score >= ? ORDER BY score DESC", [90]
).fetchall()
print("top students:", top)

# Register an explicit named view, then query it with the relation API.
con.register("grades_view", grades)
print(con.sql("SELECT COUNT(*) AS total FROM grades_view").fetchall())
con.close()
'''))

cells.append(md(r'''
## 20. sqlite3 — standard-library SQL

`sqlite3` is built into Python, so no extra dependency is needed. `conn = sqlite3.connect(":memory:")` opens an in-memory database; `conn.execute(sql, params)` runs a statement; `cur.fetchall()` and `cur.fetchone()` read rows; `conn.commit()` is required for writes; and `conn.close()` releases the connection. Use `?` placeholders for parameters, never f-strings. Setting `conn.row_factory = sqlite3.Row` gives dict-like access to columns by name. A file-backed database uses a file path instead of `":memory:"`.
'''))

cells.append(code(r'''
# sqlite3: a self-contained, file-free database example.

conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row            # rows behave like dicts
cur = conn.cursor()

cur.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, year INTEGER)")
cur.executemany(
    "INSERT INTO books (title, year) VALUES (?, ?)",
    [("Python Crash Course", 2019), ("Automate the Boring Stuff", 2022)],
)
conn.commit()

for row in cur.execute("SELECT * FROM books ORDER BY year"):
    print(dict(row))

cur.execute("UPDATE books SET year = ? WHERE title = ?", (2023, "Python Crash Course"))
conn.commit()
print("after update:", [dict(row) for row in cur.execute("SELECT title, year FROM books")])
conn.close()
'''))

cells.append(md(r'''
### Try it yourself #3 — SQLAlchemy CRUD

Create a fresh in-memory SQLite engine with a `Task` ORM model (`id`, `title`, `done` as a boolean). Insert three tasks, mark one as `done=True`, delete another, then select the remaining titles ordered by `id` and load the full table into a pandas DataFrame with `pd.read_sql`.
'''))

cells.append(code(r'''
# Solution #3

crud_engine = create_engine("sqlite:///:memory:")

class CrudBase(DeclarativeBase):
    pass

class Task(CrudBase):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(80))
    done: Mapped[bool] = mapped_column(default=False)

CrudBase.metadata.create_all(crud_engine)

with Session(crud_engine) as session:
    session.add_all([
        Task(title="tokenize text"),
        Task(title="train a model"),
        Task(title="write tests"),
    ])
    session.commit()
    session.execute(text("UPDATE tasks SET done = 1 WHERE title = :t"),
                    {"t": "train a model"})
    session.execute(text("DELETE FROM tasks WHERE title = :t"), {"t": "write tests"})
    session.commit()
    remaining = session.execute(select(Task.title).order_by(Task.id)).scalars().all()
    print("remaining titles:", remaining)

with crud_engine.connect() as conn:
    tasks_df = pd.read_sql(text("SELECT id, title, done FROM tasks ORDER BY id"), conn)
print(tasks_df)
'''))

cells.append(md(r'''
### Try it yourself #4 — DuckDB analytics

Given a DataFrame `sales` with columns `region`, `product` and `amount`, write a DuckDB query that returns, for each region, the total `amount` and the number of rows, keeping only regions whose total exceeds a threshold passed as a parameter, ordered by total descending. Use `SUM(...)` in `HAVING` and a `?` placeholder for the threshold.
'''))

cells.append(code(r'''
# Solution #4

sales = pd.DataFrame({
    "region": ["north", "south", "north", "east", "south", "east"],
    "product": ["A", "B", "B", "A", "A", "B"],
    "amount": [40, 70, 80, 30, 60, 90],
})

con4 = duckdb.connect(":memory:")
summary = con4.execute(
    "SELECT region, SUM(amount) AS total, COUNT(*) AS n "
    "FROM sales GROUP BY region HAVING SUM(amount) > ? "
    "ORDER BY total DESC",
    [100],
).fetchdf()
con4.close()
print(summary)
'''))

cells.append(md(r'''
## Cheat sheet / key takeaways

| Area | Key call | Needs a download / server? |
|---|---|---|
| NLTK setup | `nltk.download("wordnet", quiet=True)` | one-time internet download |
| Word tokenize | `word_tokenize(text)` / `WordPunctTokenizer().tokenize(text)` | Punkt for `word_tokenize`; none for rule-based |
| Sentence split | `sent_tokenize(text)` | Punkt (`punkt` / `punkt_tab`) |
| Stem | `PorterStemmer().stem(w)`, `SnowballStemmer("english").stem(w)` | none |
| Lemmatize | `WordNetLemmatizer().lemmatize(w, pos="v")` | `wordnet` |
| POS / NER | `pos_tag(tokens)`, `ne_chunk(tagged)` | taggers, `maxent_ne_chunker`, `words` |
| Corpora | `gutenberg.words()`, `movie_reviews.words()`, `brown.sents()` | corpus downloads |
| Counts / text | `FreqDist(tokens).most_common()`, `Text(tokens).concordance(w)` | none |
| Sentiment | `SentimentIntensityAnalyzer().polarity_scores(t)["compound"]` | `vader_lexicon` |
| WordNet | `wordnet.synsets(w)`, `syn.wup_similarity(other)` | `wordnet`, `omw-1.4` |
| Vectorizers | `CountVectorizer()`, `TfidfVectorizer().fit_transform(docs)` | none |
| Gensim | `Dictionary(docs)`, `doc2bow`, `TfidfModel`, `LsiModel`, `LdaModel`, `Word2Vec` | none |
| TextBlob | `TextBlob(t).sentiment`, `Word(w).lemmatize("v")` | sentiment offline; tokenizers need Punkt |
| WordCloud | `WordCloud().generate(t).to_image()` | none |
| SQLAlchemy | `create_engine`, `DeclarativeBase`, `mapped_column`, `Session`, `select`, `text` | none for SQLite |
| DuckDB | `duckdb.connect(":memory:")`, `con.execute(sql).fetchdf()` | none |
| sqlite3 | `sqlite3.connect(":memory:")`, `conn.commit()` | none |
| MySQL / PostgreSQL | `mysql+pymysql://...`, `postgresql+psycopg2://...` | running server + credentials |

Rules of thumb: guard NLTK data calls with `try/except LookupError`; use `engine.begin()` for auto-committing Core writes and `session.commit()` for ORM writes; use parameter binding (`?` / `:name`) instead of string interpolation; and prefer `id2word`, `vector_size` and `epochs` for the Gensim 4 API.
'''))

cells.append(md(r'''
## Further reading

- [NLTK Book — Processing Raw Text](https://www.nltk.org/book/ch03.html)
- [NLTK Book — Categorising and Tagging](https://www.nltk.org/book/ch05.html)
- [NLTK HowTo — Sentiment](https://www.nltk.org/howto/sentiment.html)
- [Gensim — Word2Vec tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [Gensim — LDA tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html)
- [scikit-learn — TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [SQLAlchemy 2.0 — Working with Engines and Connections](https://docs.sqlalchemy.org/en/20/core/engines.html)
- [SQLAlchemy 2.0 — ORM Session basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [DuckDB — Python NumPy and pandas integration](https://duckdb.org/docs/stable/clients/python/overview.html)
- [pandas — read_sql](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html)
- [wordcloud — documentation home](https://wordcloud.readthedocs.io/)
'''))

write_notebook("notebooks/data-science/unit5_nlp_databases.ipynb", cells)
